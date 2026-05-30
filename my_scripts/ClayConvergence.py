"""
ClayConvergence.py  –  Block-split and cumulative WHAM convergence analysis
for clay–drug umbrella sampling.

Wraps ClayPMF to run WHAM on time-sliced subsets of the production
trajectory and assess whether the 1D PMF has converged as a function of
simulation length.

Two complementary analyses
--------------------------
1. **Cumulative** (`run_cumulative`): WHAM is run at n evenly-spaced
   fractions of total production data (e.g. 20 %, 40 %, 60 %, 80 %, 100 %
   for n=5). Shows how the PMF evolves as more simulation time is added.

2. **Block-split** (`run_block_split`): Each window's production data is
   divided into n_blocks equal, *independent* segments. WHAM is run
   separately on each block. The spread across block PMFs estimates
   statistical uncertainty from finite trajectory length.

Both analyses use a **common RC grid** (xi_max fixed from the full-data run)
so PMFs are directly comparable across tests and checkpoints.

Typical workflow
----------------
From the workspace root (or a Jupyter notebook)::

    from my_scripts.ClayConvergence import ClayConvergence

    cc = ClayConvergence(
        umbrella_dir='.../Umbrella',
        n_windows=31,
        k=1000.0,
        T=298.0,
        equil_skip_ps=1000.0,
        n_bins=200,
        bulk_fraction=0.2,
    )
    cc.run_full_wham()
    cc.run_cumulative(n_checkpoints=5)
    cc.run_block_split(n_blocks=5)
    cc.print_summary()
    cc.save('convergence_anionic_NaCl_11_rep1.npz')

    # Later – reload without re-running WHAM:
    cc2 = ClayConvergence.load('convergence_anionic_NaCl_11_rep1.npz')

Notes
-----
* ``equil_skip_ps`` is applied during the initial full-data load; all subset
  runs operate on the already-trimmed production data.
* PMFs are bulk-referenced: the mean PMF value over the outermost
  ``bulk_fraction`` of the r-range is set to zero so that all curves share a
  common reference and adsorption energies (PMF minimum) are directly
  comparable.
* Simulation parameters assumed: T = 298 K, k = 1000 kJ mol⁻¹ nm⁻²,
  nstenergy = 1000 (2 ps/frame).
"""

import os
import warnings

import numpy as np

# ---------------------------------------------------------------------------
# Import ClayPMF  (same package; try workspace-root path first)
# ---------------------------------------------------------------------------
try:
    from my_scripts.ClayPMF import ClayPMF
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ClayPMF import ClayPMF


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _block_slice(n: int, block_idx: int, n_blocks: int) -> tuple:
    """
    Return (start, end) indices for block ``block_idx`` of ``n`` total frames.

    The last block absorbs any remainder so that all frames are used.

    Parameters
    ----------
    n         : total number of frames
    block_idx : 0-based block index
    n_blocks  : total number of blocks

    Returns
    -------
    start, end : int, int
    """
    block_size = n // n_blocks
    start = block_idx * block_size
    end   = start + block_size if block_idx < n_blocks - 1 else n
    return start, end


# ---------------------------------------------------------------------------
# ClayConvergence class
# ---------------------------------------------------------------------------

class ClayConvergence:
    """
    Convergence analysis for 1D WHAM PMF via cumulative and block-split tests.

    Parameters
    ----------
    umbrella_dir : str
        Directory containing ``pullx*.xvg`` files (and optionally
        ``umbrella*.mdp`` for auto-detection of window count).
    n_windows : int
        Number of umbrella windows.
    k : float
        Harmonic spring constant in kJ mol⁻¹ nm⁻².  Default 1000.0.
    T : float
        Temperature in K.  Default 298.0 (with_salts simulations).
    equil_skip_ps : float
        Equilibration to discard per window in ps.  Default 1000.0.
    n_bins : int
        Number of histogram bins for WHAM.  Default 200.
    xi_min : float or None
        Lower bound of RC grid (nm).  Always 0 (|z|); kept for API symmetry.
    xi_max : float or None
        Upper bound of RC grid (nm).  Auto-detected from data if None.
    pullx_prefix : str
        Prefix of pull-position xvg files.  Default ``'pullx'``.
    tolerance : float
        WHAM self-consistency convergence criterion.  Default 1e-6.
    max_iter : int
        Maximum WHAM iterations.  Default 50000.
    bulk_fraction : float
        Fraction of the r-range (from the far end) used as the zero-reference
        for all PMF comparisons.  Default 0.2.
    """

    # Boltzmann constant (kJ mol⁻¹ K⁻¹) — kept for potential future use
    K_B = 8.314462618e-3

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        umbrella_dir,
        n_windows,
        k=1000.0,
        T=298.0,
        equil_skip_ps=1000.0,
        n_bins=200,
        xi_min=None,
        xi_max=None,
        pullx_prefix='pullx',
        tolerance=1e-6,
        max_iter=50000,
        bulk_fraction=0.2,
    ):
        self.umbrella_dir  = os.path.abspath(umbrella_dir)
        self.n_windows     = int(n_windows)
        self.k             = float(k)
        self.T             = float(T)
        self.equil_skip_ps = float(equil_skip_ps)
        self.n_bins        = int(n_bins)
        self.xi_min        = xi_min
        self.xi_max        = xi_max
        self.pullx_prefix  = pullx_prefix
        self.tolerance     = float(tolerance)
        self.max_iter      = int(max_iter)
        self.bulk_fraction = float(bulk_fraction)

        # Populated by run_full_wham()
        self._master        = None   # ClayPMF instance (full production data)
        self._xi_max_fixed  = None   # RC upper bound fixed after full-data run
        self.r_bins         = None   # (n_bins,) common bin centres
        self.pmf_full       = None   # (n_bins,) full PMF, bulk-referenced

        # Populated by run_cumulative()
        self.cumulative_fracs = None   # (n_checkpoints,)
        self.cumulative_pmfs  = None   # (n_checkpoints, n_bins)

        # Populated by run_block_split()
        self.n_blocks       = None
        self.block_pmfs     = None   # (n_blocks, n_bins)
        self.pmf_block_mean = None   # (n_bins,)
        self.pmf_block_std  = None   # (n_bins,)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_pmf_instance(self) -> 'ClayPMF':
        """
        Create a ClayPMF configured with the fixed RC grid and zero equil_skip
        (equil already stripped when the master was loaded).
        """
        return ClayPMF(
            umbrella_dir=self.umbrella_dir,
            n_windows=self.n_windows,
            k=self.k,
            T=self.T,
            equil_skip_ps=0.0,           # equil already stripped
            n_bins=self.n_bins,
            xi_min=None,                 # r_min = 0 always (|z|)
            xi_max=self._xi_max_fixed,   # fix grid from full-data run
            pullx_prefix=self.pullx_prefix,
            tolerance=self.tolerance,
            max_iter=self.max_iter,
            verbose=False,
        )

    def _run_wham_on_subset(self, z_data_subset: list) -> tuple:
        """
        Run WHAM on a given subset of z-trajectories.

        Parameters
        ----------
        z_data_subset : list of (z1_arr, z2_arr)
            Length must equal ``self.n_windows``.  Arrays may differ in
            length across windows (some blocks may be unequal).

        Returns
        -------
        r_bins : np.ndarray  (n_bins,)
        pmf    : np.ndarray  (n_bins,)  bulk-referenced, kJ/mol
        """
        tmp = self._make_pmf_instance()
        tmp.z_data         = z_data_subset
        tmp.window_centers = self._master.window_centers   # shared centres
        tmp._build_histograms()
        tmp.run_wham()
        return (
            tmp.bin_centers_abs,
            self._reference_to_bulk(tmp.pmf_abs, tmp.bin_centers_abs),
        )

    def _reference_to_bulk(self, pmf: np.ndarray, r: np.ndarray) -> np.ndarray:
        """
        Shift *pmf* so that its mean over the bulk region is zero.

        The bulk region is defined as r > (1 − bulk_fraction) × r_max.
        Falls back to min-referencing if no valid (non-NaN) bulk bins exist.

        Parameters
        ----------
        pmf : (n_bins,) array
        r   : (n_bins,) array of bin centres

        Returns
        -------
        pmf_shifted : (n_bins,) array
        """
        r_min    = float(np.nanmin(r))
        r_max    = float(np.nanmax(r))
        r_thresh = r_min + self.bulk_fraction * (r_max - r_min)
        bulk_mask = (r <= r_thresh) & ~np.isnan(pmf)
        if np.any(bulk_mask):
            ref = float(np.mean(pmf[bulk_mask]))
        else:
            ref = float(np.nanmin(pmf))
        return pmf - ref

    # ------------------------------------------------------------------
    # Full-data WHAM  (must be called first)
    # ------------------------------------------------------------------

    def run_full_wham(self, verbose=True):
        """
        Load all production data and run WHAM on the complete trajectory.

        This step **must** be called before ``run_cumulative()`` or
        ``run_block_split()`` because it fixes the shared RC grid used by
        all subset runs.

        Sets ``self.r_bins``, ``self.pmf_full``, and ``self._xi_max_fixed``.

        Parameters
        ----------
        verbose : bool
            Print progress to stdout.  Default True.

        Returns
        -------
        r_bins   : np.ndarray  (n_bins,)
        pmf_full : np.ndarray  (n_bins,)  bulk-referenced, kJ/mol
        """
        pmf_obj = ClayPMF(
            umbrella_dir=self.umbrella_dir,
            n_windows=self.n_windows,
            k=self.k,
            T=self.T,
            equil_skip_ps=self.equil_skip_ps,
            n_bins=self.n_bins,
            xi_min=self.xi_min,
            xi_max=self.xi_max,
            pullx_prefix=self.pullx_prefix,
            tolerance=self.tolerance,
            max_iter=self.max_iter,
            verbose=verbose,
        )
        pmf_obj.load_data()
        pmf_obj._build_histograms()
        pmf_obj.run_wham()

        self._master = pmf_obj

        # Fix the RC upper bound from the actual histogram grid so all
        # subsequent subset runs use identical bins.
        self._xi_max_fixed = float(pmf_obj.bins[-1])

        self.r_bins   = pmf_obj.bin_centers_abs.copy()
        self.pmf_full = self._reference_to_bulk(pmf_obj.pmf_abs, self.r_bins)

        if verbose:
            print(
                f"Full PMF: r = {self.r_bins[0]:.3f} … {self.r_bins[-1]:.3f} nm  |  "
                f"range = {np.nanmin(self.pmf_full):.2f} … "
                f"{np.nanmax(self.pmf_full):.2f} kJ/mol"
            )

        return self.r_bins, self.pmf_full

    # ------------------------------------------------------------------
    # Cumulative convergence test
    # ------------------------------------------------------------------

    def run_cumulative(self, n_checkpoints=5, verbose=True):
        """
        Run WHAM at ``n_checkpoints`` evenly-spaced fractions of the
        production trajectory.

        For ``n_checkpoints=5`` the fractions are 20 %, 40 %, 60 %, 80 %,
        100 %.  Each checkpoint uses the **first** ``frac`` of each window's
        production data, making it a true cumulative (growing-data) test.

        ``run_full_wham()`` is called automatically if not already done.

        Parameters
        ----------
        n_checkpoints : int
            Number of time checkpoints.  Default 5.
        verbose : bool
            Print per-checkpoint progress.  Default True.

        Returns
        -------
        fracs : np.ndarray  (n_checkpoints,)  fractions of data used
        pmfs  : np.ndarray  (n_checkpoints, n_bins)  bulk-referenced PMFs
        """
        if self._master is None:
            self.run_full_wham(verbose=verbose)

        fracs = np.linspace(1.0 / n_checkpoints, 1.0, n_checkpoints)
        pmfs  = np.full((n_checkpoints, self.n_bins), np.nan)

        if verbose:
            print(f"\nCumulative convergence: {n_checkpoints} checkpoints")

        for ci, frac in enumerate(fracs):
            # Take the first *frac* of each window's production data
            z_sub = []
            for z1, z2 in self._master.z_data:
                n1 = max(1, int(round(len(z1) * frac)))
                n2 = max(1, int(round(len(z2) * frac)))
                z_sub.append((z1[:n1], z2[:n2]))

            if verbose:
                n_frames_tot = sum(len(z1) + len(z2) for z1, z2 in z_sub)
                print(
                    f"  {frac:.0%}: {n_frames_tot:,} frames … ",
                    end='', flush=True,
                )

            try:
                _, pmf_i = self._run_wham_on_subset(z_sub)
                pmfs[ci] = pmf_i
                if verbose:
                    print(
                        f"done  "
                        f"(range {np.nanmin(pmf_i):.2f} … {np.nanmax(pmf_i):.2f} kJ/mol)"
                    )
            except Exception as exc:
                warnings.warn(
                    f"Cumulative checkpoint {frac:.0%} failed: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                if verbose:
                    print(f"FAILED: {exc}")

        self.cumulative_fracs = fracs
        self.cumulative_pmfs  = pmfs
        return fracs, pmfs

    # ------------------------------------------------------------------
    # Block-split convergence test
    # ------------------------------------------------------------------

    def run_block_split(self, n_blocks=5, verbose=True):
        """
        Split each window's production data into ``n_blocks`` equal,
        non-overlapping segments and run WHAM independently on each block.

        The spread (standard deviation) of block PMFs estimates the
        statistical uncertainty of the PMF arising from finite trajectory
        length, and can reveal slow degrees of freedom that cause block-to-
        block inconsistency.

        ``run_full_wham()`` is called automatically if not already done.

        Parameters
        ----------
        n_blocks : int
            Number of independent blocks per window.  Default 5.
        verbose : bool
            Print per-block progress.  Default True.

        Returns
        -------
        block_pmfs : np.ndarray  (n_blocks, n_bins)  bulk-referenced PMFs
        """
        if self._master is None:
            self.run_full_wham(verbose=verbose)

        self.n_blocks = int(n_blocks)
        block_pmfs    = np.full((n_blocks, self.n_bins), np.nan)

        if verbose:
            print(f"\nBlock-split convergence: {n_blocks} blocks")

        for bi in range(n_blocks):
            z_sub = []
            for z1, z2 in self._master.z_data:
                s1, e1 = _block_slice(len(z1), bi, n_blocks)
                s2, e2 = _block_slice(len(z2), bi, n_blocks)
                z_sub.append((z1[s1:e1], z2[s2:e2]))

            if verbose:
                n_frames_tot = sum(len(z1) + len(z2) for z1, z2 in z_sub)
                print(
                    f"  block {bi + 1}/{n_blocks}: {n_frames_tot:,} frames … ",
                    end='', flush=True,
                )

            try:
                _, pmf_i = self._run_wham_on_subset(z_sub)
                block_pmfs[bi] = pmf_i
                if verbose:
                    print(
                        f"done  "
                        f"(range {np.nanmin(pmf_i):.2f} … {np.nanmax(pmf_i):.2f} kJ/mol)"
                    )
            except Exception as exc:
                warnings.warn(
                    f"Block {bi + 1} WHAM failed: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                if verbose:
                    print(f"FAILED: {exc}")

        self.block_pmfs = block_pmfs

        # Summary statistics across valid (non-all-NaN) blocks
        valid = ~np.all(np.isnan(block_pmfs), axis=1)
        if np.any(valid):
            self.pmf_block_mean = np.nanmean(block_pmfs[valid], axis=0)
            n_valid = int(np.sum(valid))
            ddof = min(1, n_valid - 1)
            self.pmf_block_std  = np.nanstd(block_pmfs[valid], axis=0, ddof=ddof)
        else:
            self.pmf_block_mean = np.full(self.n_bins, np.nan)
            self.pmf_block_std  = np.full(self.n_bins, np.nan)

        if verbose:
            print(
                f"  Block statistics: max σ = "
                f"{np.nanmax(self.pmf_block_std):.3f} kJ/mol  "
                f"({int(np.sum(valid))}/{n_blocks} blocks succeeded)"
            )

        return block_pmfs

    # ------------------------------------------------------------------
    # Convenience: adsorption energy and drift metrics
    # ------------------------------------------------------------------

    def adsorption_energies(self) -> dict:
        """
        Return the adsorption energy (PMF minimum) for the full PMF, each
        cumulative checkpoint, and each block.

        Adsorption energy = min(PMF(r)) since PMFs are bulk-referenced
        (bulk → 0, adsorption well → negative value).

        Returns
        -------
        dict with keys:
          ``'full'``            : float  (kJ/mol)
          ``'cumulative'``      : np.ndarray or None  (n_checkpoints,)
          ``'blocks'``          : np.ndarray or None  (n_blocks,)
          ``'blocks_mean'``     : float or None
          ``'blocks_std'``      : float or None
        """
        result = {'full': None, 'cumulative': None,
                  'blocks': None, 'blocks_mean': None, 'blocks_std': None}

        if self.pmf_full is not None:
            result['full'] = float(np.nanmin(self.pmf_full))

        if self.cumulative_pmfs is not None:
            result['cumulative'] = np.array(
                [float(np.nanmin(p)) for p in self.cumulative_pmfs]
            )

        if self.block_pmfs is not None:
            vals = np.array([float(np.nanmin(p)) for p in self.block_pmfs])
            result['blocks']      = vals
            result['blocks_mean'] = float(np.nanmean(vals))
            result['blocks_std']  = float(np.nanstd(vals, ddof=min(1, len(vals) - 1)))

        return result

    def drift_metric(self) -> np.ndarray:
        """
        Per-bin absolute change in PMF from the first to the last cumulative
        checkpoint.  A small value indicates the PMF has converged.

        Returns
        -------
        drift : np.ndarray  (n_bins,)  kJ/mol  or None
        """
        if self.cumulative_pmfs is None or len(self.cumulative_pmfs) < 2:
            return None
        return np.abs(self.cumulative_pmfs[-1] - self.cumulative_pmfs[0])

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, filepath):
        """
        Save all PMF arrays and metadata to a compressed ``.npz`` file.

        Parameters
        ----------
        filepath : str
            Output path.  ``.npz`` extension is added if absent.
        """
        if self.pmf_full is None:
            raise RuntimeError(
                "Nothing to save – call run_full_wham() first."
            )

        # Metadata stored as 0-d arrays for easy retrieval
        arrays = {
            'meta_umbrella_dir':  np.array(str(self.umbrella_dir)),
            'meta_n_windows':     np.array(self.n_windows),
            'meta_k':             np.array(self.k),
            'meta_T':             np.array(self.T),
            'meta_equil_skip_ps': np.array(self.equil_skip_ps),
            'meta_n_bins':        np.array(self.n_bins),
            'meta_bulk_fraction': np.array(self.bulk_fraction),
            'meta_xi_max_fixed':  np.array(
                self._xi_max_fixed if self._xi_max_fixed is not None else np.nan
            ),
            'r_bins':   self.r_bins,
            'pmf_full': self.pmf_full,
        }

        if self.cumulative_pmfs is not None:
            arrays['cumulative_fracs'] = self.cumulative_fracs
            arrays['cumulative_pmfs']  = self.cumulative_pmfs

        if self.block_pmfs is not None:
            arrays['n_blocks']       = np.array(self.n_blocks)
            arrays['block_pmfs']     = self.block_pmfs
            arrays['pmf_block_mean'] = self.pmf_block_mean
            arrays['pmf_block_std']  = self.pmf_block_std

        np.savez_compressed(filepath, **arrays)
        print(f"Saved convergence data to: {filepath}")

    @classmethod
    def load(cls, filepath) -> 'ClayConvergence':
        """
        Reconstruct a ``ClayConvergence`` object from a saved ``.npz`` file.

        The ``_master`` ClayPMF instance is not restored (it requires file
        I/O); re-call ``run_full_wham()`` if further subset analysis is
        needed.

        Parameters
        ----------
        filepath : str

        Returns
        -------
        ClayConvergence
        """
        data = np.load(filepath, allow_pickle=True)

        def _scalar(key):
            v = data[key]
            return v.item() if v.ndim == 0 else v

        inst = object.__new__(cls)

        inst.umbrella_dir  = str(_scalar('meta_umbrella_dir'))
        inst.n_windows     = int(_scalar('meta_n_windows'))
        inst.k             = float(_scalar('meta_k'))
        inst.T             = float(_scalar('meta_T'))
        inst.equil_skip_ps = float(_scalar('meta_equil_skip_ps'))
        inst.n_bins        = int(_scalar('meta_n_bins'))
        inst.bulk_fraction = float(_scalar('meta_bulk_fraction'))
        inst.xi_min        = None
        inst.xi_max        = None
        inst.pullx_prefix  = 'pullx'
        inst.tolerance     = 1e-6
        inst.max_iter      = 50000

        xi_max_v            = float(_scalar('meta_xi_max_fixed'))
        inst._xi_max_fixed  = None if np.isnan(xi_max_v) else xi_max_v

        inst._master        = None   # not persisted
        inst.r_bins         = data['r_bins']
        inst.pmf_full       = data['pmf_full']

        inst.cumulative_fracs = (
            data['cumulative_fracs'] if 'cumulative_fracs' in data else None
        )
        inst.cumulative_pmfs  = (
            data['cumulative_pmfs']  if 'cumulative_pmfs'  in data else None
        )

        inst.n_blocks       = (
            int(_scalar('n_blocks')) if 'n_blocks' in data else None
        )
        inst.block_pmfs     = (
            data['block_pmfs']     if 'block_pmfs'     in data else None
        )
        inst.pmf_block_mean = (
            data['pmf_block_mean'] if 'pmf_block_mean' in data else None
        )
        inst.pmf_block_std  = (
            data['pmf_block_std']  if 'pmf_block_std'  in data else None
        )

        return inst

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def print_summary(self):
        """Print a concise table of convergence results to stdout."""
        sep = '─' * 62
        print(sep)
        print("  ClayConvergence Summary")
        print(f"  dir : {self.umbrella_dir}")
        print(
            f"  T = {self.T} K   k = {self.k} kJ mol⁻¹ nm⁻²   "
            f"n_windows = {self.n_windows}"
        )
        print(sep)

        if self.pmf_full is not None:
            ea = float(np.nanmin(self.pmf_full))
            r_ea = float(self.r_bins[np.nanargmin(self.pmf_full)])
            print(
                f"  Full PMF:  r = {self.r_bins[0]:.3f} … {self.r_bins[-1]:.3f} nm  |  "
                f"ΔG_ads = {ea:.2f} kJ/mol  at r = {r_ea:.3f} nm"
            )

        if self.cumulative_pmfs is not None:
            print(f"\n  Cumulative ({len(self.cumulative_fracs)} checkpoints):")
            print(f"  {'Fraction':>10}  {'ΔG_ads (kJ/mol)':>18}  {'PMF range':>22}")
            for frac, pmf in zip(self.cumulative_fracs, self.cumulative_pmfs):
                ea_i = float(np.nanmin(pmf))
                mn_i, mx_i = float(np.nanmin(pmf)), float(np.nanmax(pmf))
                print(
                    f"  {frac:>9.0%}  {ea_i:>+18.2f}  "
                    f"[{mn_i:+.2f}, {mx_i:+.2f}]"
                )

        if self.block_pmfs is not None:
            print(f"\n  Block-split ({self.n_blocks} blocks):")
            print(f"  {'Block':>8}  {'ΔG_ads (kJ/mol)':>18}  {'PMF range':>22}")
            for bi, pmf in enumerate(self.block_pmfs):
                ea_i = float(np.nanmin(pmf))
                mn_i, mx_i = float(np.nanmin(pmf)), float(np.nanmax(pmf))
                print(
                    f"  {bi + 1:>8d}  {ea_i:>+18.2f}  "
                    f"[{mn_i:+.2f}, {mx_i:+.2f}]"
                )
            print(
                f"  {'mean':>8}  {float(np.nanmean([np.nanmin(p) for p in self.block_pmfs])):>+18.2f}"
            )
            print(f"  {'σ':>8}  {float(np.nanstd([np.nanmin(p) for p in self.block_pmfs], ddof=min(1, self.n_blocks - 1))):>18.3f}")
            print(f"  max σ(PMF) = {np.nanmax(self.pmf_block_std):.3f} kJ/mol")

        print(sep)

    # ------------------------------------------------------------------
    # Smoke test
    # ------------------------------------------------------------------

    @staticmethod
    def _smoke_test():
        """
        Self-contained smoke test with synthetic umbrella data.

        Generates pullx files in a temporary directory and runs the full
        pipeline (full WHAM → cumulative → block-split → save → load) to
        verify that all code paths execute without error.
        """
        import tempfile

        rng      = np.random.default_rng(42)
        n_win    = 15
        n_frames = 600         # frames per window (≈ 1.2 ns at 2 ps/frame)
        k        = 1000.0
        T        = 298.0
        dt       = 2.0         # ps per frame
        centers  = np.linspace(0.25, 1.65, n_win)   # nm

        print("=== ClayConvergence smoke test ===")
        print(f"  {n_win} windows, {n_frames} frames/window, "
              f"k = {k} kJ mol⁻¹ nm⁻²,  T = {T} K")

        with tempfile.TemporaryDirectory() as tmpdir:
            times = np.arange(n_frames) * dt
            sigma = np.sqrt(8.314e-3 * T / k)   # thermal width ~0.050 nm

            for i, c in enumerate(centers, start=1):
                z1 = rng.normal( c, sigma, n_frames)
                z2 = rng.normal(-c, sigma, n_frames)
                header = ['# synthetic pullx\n',
                          '@ title "pullx"\n',
                          '@ xaxis label "Time (ps)"\n',
                          '@ yaxis label "Coord (nm)"\n']
                rows = [f"  {t:.3f}  {a:.6f}  {b:.6f}\n"
                        for t, a, b in zip(times, z1, z2)]
                with open(os.path.join(tmpdir, f'pullx{i}.xvg'), 'w') as fh:
                    fh.writelines(header + rows)

            cc = ClayConvergence(
                umbrella_dir=tmpdir,
                n_windows=n_win,
                k=k,
                T=T,
                equil_skip_ps=0.0,      # no equil discard in smoke test
                n_bins=80,
                bulk_fraction=0.2,
                tolerance=1e-4,         # looser for speed
                max_iter=5000,
            )

            # --- full WHAM ---
            cc.run_full_wham(verbose=False)
            assert cc.pmf_full is not None, "pmf_full not set after run_full_wham"
            assert len(cc.r_bins) == 80,    "r_bins length mismatch"

            # --- cumulative ---
            fracs, pmfs = cc.run_cumulative(n_checkpoints=3, verbose=False)
            assert fracs.shape == (3,),      "wrong fracs shape"
            assert pmfs.shape  == (3, 80),   "wrong cumulative_pmfs shape"

            # --- block-split ---
            bpmfs = cc.run_block_split(n_blocks=3, verbose=False)
            assert bpmfs.shape == (3, 80),   "wrong block_pmfs shape"
            assert cc.pmf_block_std is not None

            # --- adsorption energies ---
            ae = cc.adsorption_energies()
            assert ae['full'] is not None
            assert len(ae['cumulative']) == 3
            assert len(ae['blocks'])     == 3

            # --- drift metric ---
            drift = cc.drift_metric()
            assert drift is not None and len(drift) == 80

            # --- save / load round-trip ---
            save_path = os.path.join(tmpdir, 'convergence_smoke.npz')
            cc.save(save_path)
            cc2 = ClayConvergence.load(save_path)
            assert np.allclose(cc2.pmf_full, cc.pmf_full), "save/load mismatch"

        print(f"  Full PMF range : "
              f"{np.nanmin(cc.pmf_full):.2f} … {np.nanmax(cc.pmf_full):.2f} kJ/mol")
        print(f"  ΔG_ads (full)  : {np.nanmin(cc.pmf_full):.2f} kJ/mol")
        print(f"  Cumulative ΔG  : "
              f"{[f'{v:.2f}' for v in ae['cumulative']]}")
        print(f"  Block σ(ΔG)    : {ae['blocks_std']:.3f} kJ/mol")
        print(f"  max σ(PMF)     : {np.nanmax(cc.pmf_block_std):.3f} kJ/mol")
        print("Smoke test: PASS ✓")


# ---------------------------------------------------------------------------
# Entry point for standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    ClayConvergence._smoke_test()
