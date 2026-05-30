#!/usr/bin/env python3
"""
ClayMBAR.py
===========
Multistate Bennett Acceptance Ratio (MBAR) PMF estimator for clay umbrella
sampling trajectories.

Unlike WHAM, MBAR does not require binning during the optimisation: it is an
asymptotically optimal maximum-likelihood estimator of the free energies f_k
for each sampled state.  The 1-D W(r) PMF is obtained in a post-processing
step via pymbar.FES, which histograms the reweighted samples onto the output
r-grid.

Key differences from ClayPMF (WHAM)
--------------------------------------
  - No binning artefacts in f_k estimation (MBAR is bin-free)
  - Analytical uncertainty estimate via MBAR covariance propagation
  - Slower than WHAM for large datasets (O(K²·N) per iteration)

Two-CIP pseudo-window design
------------------------------
Identical to ClayPMF._build_histograms: each GROMACS window contains two CIP
molecules on opposite sides of the clay.  Both are treated as independent
pseudo-windows with their own harmonic spring:

  State 2i   → CIP1, spring centre |c₁ᵢ|
  State 2i+1 → CIP2, spring centre |c₂ᵢ|

So K = 2 × n_windows pseudo-states.

Coordinate convention
----------------------
r = |z| is the GROMACS pull coordinate (distance from pore centre toward clay
surface).  Small r ≈ pore centre (bulk); large r ≈ clay surface.  Consistent
with ClayPMF.get_adsorption_energy() and ClayMeanForce.reference_to_bulk().

Dependencies
------------
  pymbar >= 4.0   (conda install -c conda-forge pymbar)
  numpy >= 1.21
  scipy

Usage
-----
    from ClayPMF  import ClayPMF
    from ClayMBAR import ClayMBAR

    pmf  = ClayPMF(umbrella_dir='...').load_data().run_wham()
    mbar = ClayMBAR(pmf)
    mbar.run_mbar(verbose=True)
    mbar.reference_to_bulk()
    mbar.print_summary()

    mbar.save('clay_mbar.npz')
    mbar2 = ClayMBAR.load('clay_mbar.npz')

References
----------
    Shirts & Chodera (2008) J. Chem. Phys. 129, 124105
    Chodera et al. (2007) J. Chem. Theory Comput. 3, 26–41
    pymbar: https://github.com/choderalab/pymbar
"""

import os
import warnings

import numpy as np

try:
    import pymbar
    _PYMBAR_AVAILABLE = True
except ImportError:
    _PYMBAR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Module-level constant
# ---------------------------------------------------------------------------
_K_B = 8.314462618e-3   # kJ mol⁻¹ K⁻¹


# ---------------------------------------------------------------------------
class ClayMBAR:
    """
    MBAR PMF estimator for clay umbrella sampling.

    Parameters
    ----------
    pmf : ClayPMF
        Instance with ``load_data()`` already called.  ``run_wham()`` is NOT
        required.  Must expose: ``z_data``, ``window_centers``, ``k``, ``T``,
        ``beta``, ``n_windows``.  Optionally ``n_bins`` for default grid size.
    n_bins : int or None
        Number of histogram bins for the W(r) output.  Default: inherit from
        ``pmf.n_bins`` (falls back to 200 if not set).
    xi_min : float or None
        Lower bound for the PMF r-grid [nm].  Defaults to 0.0.
    xi_max : float or None
        Upper bound for the PMF r-grid [nm].  Defaults to 1.05 × max |z|.
    bulk_fraction : float
        Fraction of the r range (high-r end) used as the bulk reference in
        ``reference_to_bulk()``.  Default 0.2.
    """

    K_B = _K_B

    def __init__(
        self,
        pmf,
        n_bins=None,
        xi_min=None,
        xi_max=None,
        bulk_fraction=0.2,
        stride=1,
        max_memory_gb=2.0,
        use_tqdm=True,
    ):
        """
        Parameters
        ----------
        stride : int
            Sub-sampling stride applied to every window before building the
            reduced-potential matrix.  ``stride=1`` uses all frames (default).
            Use ``stride=10`` (or higher) to reduce memory when trajectories
            are long (e.g. > 10 ns at 1-ps save interval).
            The u_kn matrix has shape (K, N_total/stride) × 8 bytes; for
            K=60 and N_total=6 M a stride of 10 reduces it from ~2.9 GB to
            ~290 MB.
        max_memory_gb : float
            Abort u_kn construction if the estimated matrix size exceeds this
            limit (GB).  Default 2.0.  Raise ``MemoryError`` with a clear
            message suggesting a higher ``stride``.
        use_tqdm : bool
            Show a ``tqdm`` progress bar during the u_kn build loop if tqdm
            is installed.  Silently falls back to no bar if tqdm is absent.
            Default ``True``.
        """
        if not _PYMBAR_AVAILABLE:
            raise ImportError(
                "pymbar is required for ClayMBAR.\n"
                "Install with:\n"
                "  conda install -c conda-forge pymbar\n"
                "or:\n"
                "  pip install pymbar"
            )

        self._pmf = pmf

        # --- Simulation parameters from ClayPMF -------------------------
        self.k          = float(pmf.k)
        self.T          = float(pmf.T)
        self.beta       = float(pmf.beta)   # mol / kJ
        self.n_windows  = int(pmf.n_windows)
        self.umbrella_dir = getattr(pmf, 'umbrella_dir', None)

        # --- Grid parameters ---------------------------------------------
        self.n_bins       = int(n_bins if n_bins is not None
                                else getattr(pmf, 'n_bins', 200))
        self.xi_min       = xi_min
        self.xi_max       = xi_max
        self.bulk_fraction  = float(bulk_fraction)
        self.stride         = max(1, int(stride))
        self.max_memory_gb  = float(max_memory_gb)
        self.use_tqdm       = bool(use_tqdm)

        # --- Results (set by run_mbar) -----------------------------------
        self.bin_centers_abs  = None   # (n_bins,) r grid  [nm]
        self.bin_edges        = None   # (n_bins+1,)       [nm]
        self.bin_width        = None   # float             [nm]
        self.pmf_mbar_1d      = None   # (n_bins,)  W(r)   [kJ/mol]
        self.pmf_mbar_1d_err  = None   # (n_bins,)  σ(W)   [kJ/mol] or None

        # --- Results (set by run_mbar_2d) --------------------------------
        self.pmf_mbar_2d      = None   # (n_r_bins, n_theta_bins) [kJ/mol]
        self.count2d_raw      = None   # (n_r_bins, n_theta_bins) raw counts
        self.r_centers_2d     = None   # (n_r_bins,)   [nm]
        self.theta_centers_2d = None   # (n_theta_bins,) [deg]
        self.r_edges_2d       = None   # (n_r_bins+1,)
        self.theta_edges_2d   = None   # (n_theta_bins+1,)

        # --- Internal state ----------------------------------------------
        self._mbar       = None    # pymbar.MBAR or None
        self._u_kn       = None    # (K, N_total) reduced potential matrix
        self._N_k        = None    # (K,) sample counts
        self._r0         = None    # (K,) spring centres [nm]
        self._x_all      = None    # (N_total,) all |z| samples [nm]
        self._used_stride = None   # actual stride applied in run_mbar()
        self._converged  = False

    # -----------------------------------------------------------------------
    # Data preparation
    # -----------------------------------------------------------------------

    def _build_pseudo_windows(self):
        """
        Build K = 2 × n_windows pseudo-state arrays from ``pmf.z_data``.

        Returns samples as a list of arrays (not pre-concatenated) so that
        the caller can check memory before allocating the full ``x_all`` and
        ``u_kn`` arrays.

        Returns
        -------
        r0          : ndarray, shape (K,)           spring centres [nm]
        samples_list: list of ndarray, length K     per-state |z| arrays
        N_k         : ndarray, shape (K,) int       samples per state
        """
        if self._pmf.z_data is None:
            raise RuntimeError(
                "z_data not loaded.  Call pmf.load_data() before ClayMBAR."
            )

        K = 2 * self.n_windows
        r0_list, samples_list, N_list = [], [], []

        s = self.stride
        for (z1, z2), (c1, c2) in zip(
            self._pmf.z_data, self._pmf.window_centers
        ):
            r1 = np.abs(z1)[::s]
            r2 = np.abs(z2)[::s]
            r0_list.append(abs(c1))
            r0_list.append(abs(c2))
            samples_list.append(r1)
            samples_list.append(r2)
            N_list.append(len(r1))
            N_list.append(len(r2))

        r0  = np.array(r0_list, dtype=float)
        N_k = np.array(N_list,  dtype=int)

        if len(r0) != K:
            raise RuntimeError(
                f"Expected K={K} pseudo-states, got {len(r0)}"
            )
        if int(N_k.sum()) != sum(len(a) for a in samples_list):
            raise RuntimeError("N_total mismatch between samples_list and N_k")

        return r0, samples_list, N_k

    # -----------------------------------------------------------------------
    # Memory helpers
    # -----------------------------------------------------------------------

    def _estimate_memory(self, K, N_total):
        """Return estimated u_kn memory in GB (float64 elements)."""
        return K * N_total * 8 / 1e9

    def _suggest_stride(self, K, N_total_unstrided):
        """
        Return the smallest stride that keeps u_kn under ``max_memory_gb``.

        Parameters
        ----------
        K : int
            Number of pseudo-states.
        N_total_unstrided : int
            Total sample count at ``stride=1``.

        Returns
        -------
        int
            Suggested stride ≥ current ``self.stride``.
        """
        current_gb = self._estimate_memory(K, N_total_unstrided)
        if current_gb <= self.max_memory_gb:
            return self.stride
        needed = int(np.ceil(current_gb / self.max_memory_gb))
        return max(self.stride, needed)

    def _build_r_grid(self, x_all):
        """Build bin edges, centres, and width for the PMF output r-grid."""
        r_min = float(self.xi_min) if self.xi_min is not None else 0.0
        r_max = (float(self.xi_max) if self.xi_max is not None
                 else float(x_all.max()) * 1.05)
        edges   = np.linspace(r_min, r_max, self.n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        width   = float(edges[1] - edges[0])
        return edges, centers, width

    # -----------------------------------------------------------------------
    # Core MBAR run
    # -----------------------------------------------------------------------

    def _run_pymbar_fes(self, u_data, N_k, verbose=True, solver_protocol='robust'):
        """Run ``pymbar.FES(u_data, N_k)`` in a background thread.

        When *verbose* is True a single-line elapsed-time bar is shown while
        the solver runs (tqdm required; falls back to silent wait if absent).
        pymbar's own verbose output is suppressed so it does not interleave
        with the progress display.

        Parameters
        ----------
        solver_protocol : str or list or None
            Passed via ``mbar_options`` to ``pymbar.MBAR``.  Use ``'robust'``
            (default) for large datasets — it runs L-BFGS-B then adaptive
            iterations and avoids the infinite cycling seen when ``'hybr'``
            (the pymbar default) fails to converge on millions of samples.
            Pass ``None`` to restore pymbar's built-in default sequence.
        """
        import threading as _threading

        _result = [None]
        _exc    = [None]

        _mbar_opts = {} if solver_protocol is None else {'solver_protocol': solver_protocol}

        def _worker():
            try:
                # Always pass verbose=False — we own the progress display.
                _result[0] = pymbar.FES(u_data, N_k, verbose=False,
                                        mbar_options=_mbar_opts)
            except Exception as _e:
                _exc[0] = _e

        _t = _threading.Thread(target=_worker, daemon=True)
        _t.start()

        if verbose:
            try:
                from tqdm.auto import tqdm as _tqdm
                with _tqdm(
                    desc='  MBAR solver',
                    bar_format='{desc}: {elapsed} elapsed',
                    total=None,
                ) as _pbar:
                    while _t.is_alive():
                        _t.join(timeout=1)
                        _pbar.refresh()
            except ImportError:
                _t.join()
        else:
            _t.join()

        if _exc[0] is not None:
            raise _exc[0]
        return _result[0]

    def run_mbar(self, uncertainty_method='analytical', verbose=True,
                 cache_path=None, save_cache=True, force_rerun=False,
                 solver_protocol='robust'):
        """
        Run MBAR and compute the 1-D W(r) PMF.

        The reduced potential u_kn[k, n] = β · ½k · (|z_n| − r₀_k)² is built
        for K = 2 × n_windows pseudo-states and N_total samples, then passed
        to ``pymbar.FES`` for histogram PMF calculation.

        Parameters
        ----------
        uncertainty_method : str or None
            Passed to ``pymbar.FES.get_fes()``.
            ``'analytical'`` (default): MBAR-propagated analytical uncertainty.
            ``None``: skip uncertainty (faster).
        verbose : bool
        cache_path : str or None
            Path to a .npz file.  If the file exists and ``force_rerun=False``,
            results are loaded from cache and the expensive MBAR solve is
            skipped.  If ``save_cache=True`` and the solve was run, results are
            saved to this path afterwards.
        save_cache : bool
            Save results to ``cache_path`` after running (default True).
            Has no effect when loading from cache or when ``cache_path`` is None.
        force_rerun : bool
            If True, ignore an existing cache file and always re-run MBAR
            (default False).  The new results overwrite the cache when
            ``save_cache=True``.
        solver_protocol : str or list or None
            Solver strategy passed through to ``pymbar.MBAR`` via
            ``mbar_options``.  Default ``'robust'`` uses L-BFGS-B followed by
            adaptive iterations, which converges reliably for large datasets
            (> 1 M samples).  Pass ``None`` to restore pymbar's built-in
            default (``'hybr'`` → Newton), which may stall on large systems.

        Returns
        -------
        self
        """
        def _log(msg):
            if verbose:
                print(msg, flush=True)

        # --- Default cache path (mirrors load_trajectory_data convention) -
        import os as _os
        if cache_path is None and self.umbrella_dir is not None:
            cache_path = _os.path.join(self.umbrella_dir, f"clay_mbar_s{self.stride}.npz")

        # --- Cache load-or-skip ------------------------------------------
        if cache_path is not None and not force_rerun and _os.path.exists(cache_path):
            _log(f"ClayMBAR.run_mbar(): loading cached results from {cache_path}")
            _loaded = ClayMBAR.load(cache_path)
            # Copy all computed attributes into self so callers can keep
            # their existing reference to this object.
            for _attr in (
                'bin_edges', 'bin_centers_abs', 'bin_width',
                'pmf_mbar_1d', 'pmf_mbar_1d_err',
                '_r0', '_N_k', '_f_k', '_x_all', '_u_kn',
                '_mbar', '_used_stride', '_converged',
                'pmf_mbar_2d', 'count2d_raw',
                'r_centers_2d', 'theta_centers_2d',
                'r_edges_2d', 'theta_edges_2d',
            ):
                setattr(self, _attr, getattr(_loaded, _attr, None))
            _log("  Loaded from cache — run_mbar() not repeated.")
            return self
        # -----------------------------------------------------------------

        _log("ClayMBAR.run_mbar(): building pseudo-window arrays ...")
        r0, samples_list, N_k = self._build_pseudo_windows()
        K       = len(N_k)
        N_total = int(N_k.sum())

        _log(f"  K = {K} pseudo-states,  N_total = {N_total:,} samples  (stride={self.stride})")

        # --- Memory check BEFORE any large allocation --------------------
        import warnings
        mem_gb = self._estimate_memory(K, N_total)
        _log(f"  Estimated u_kn memory: {mem_gb:.2f} GB  (limit: {self.max_memory_gb:.1f} GB)")
        if mem_gb > self.max_memory_gb:
            # N_total_unstrided ≈ N_total * stride (best estimate without re-reading data)
            suggested = self._suggest_stride(K, N_total * self.stride)
            raise MemoryError(
                f"u_kn matrix would use ~{mem_gb:.1f} GB, which exceeds "
                f"max_memory_gb={self.max_memory_gb:.1f} GB.\n"
                f"Increase stride (current: {self.stride}) to reduce memory.  "
                f"Suggested: stride={suggested}"
            )
        if mem_gb > 0.5 * self.max_memory_gb:
            warnings.warn(
                f"u_kn matrix will use ~{mem_gb:.1f} GB "
                f"(>{50:.0f}% of max_memory_gb={self.max_memory_gb:.1f} GB).  "
                "Consider increasing stride= if the kernel is slow.",
                ResourceWarning,
                stacklevel=2,
            )

        # --- Concatenate x_all in-place (no extra copy) ------------------
        x_all = np.empty(N_total, dtype=np.float64)
        offset = 0
        for arr in samples_list:
            n = len(arr)
            x_all[offset:offset + n] = arr
            offset += n

        # --- r-grid for output -------------------------------------------
        edges, centers, width = self._build_r_grid(x_all)

        # --- Reduced potential matrix u_kn (K, N_total) ------------------
        # Built row-by-row to avoid the (K, N_total) broadcast temporary
        # that the vectorised form x_all[np.newaxis,:] - r0[:,np.newaxis]
        # would create before assignment.
        _log("  Building u_kn matrix ...")
        u_kn = np.empty((K, N_total), dtype=np.float64)
        _row_iter = enumerate(r0)
        if self.use_tqdm:
            try:
                from tqdm.auto import tqdm as _tqdm
                _row_iter = enumerate(_tqdm(r0, desc='u_kn rows', leave=False))
            except ImportError:
                pass
        _c = self.beta * 0.5 * self.k
        for k, r0_k in _row_iter:
            u_kn[k] = _c * (x_all - r0_k) ** 2
        # shape (K, N_total)

        # --- Initialise pymbar.FES ---------------------------------------
        _log("  Initialising pymbar.FES (MBAR iteration — may take several minutes) ...")
        try:
            fes = self._run_pymbar_fes(u_kn, N_k, verbose=verbose,
                                        solver_protocol=solver_protocol)
        except TypeError:
            # Older pymbar (< 4.0) expects 3-D u_kln (K, K, N_max).
            _log("  (falling back to 3-D u_kln layout for older pymbar) ...")
            N_max = int(N_k.max())
            u_kln = np.zeros((K, K, N_max), dtype=float)
            offset = 0
            _kln_iter = range(K)
            if self.use_tqdm:
                try:
                    from tqdm.auto import tqdm as _tqdm
                    _kln_iter = _tqdm(_kln_iter, desc='u_kln rows', leave=False)
                except ImportError:
                    pass
            for k in _kln_iter:
                nk = int(N_k[k])
                for ll in range(K):
                    u_kln[k, ll, :nk] = (
                        self.beta * 0.5 * self.k
                        * (x_all[offset:offset + nk] - r0[ll]) ** 2
                    )
                offset += nk
                if not self.use_tqdm and verbose and K > 20 and (k + 1) % max(1, K // 10) == 0:
                    _log(f"    u_kln: {k + 1}/{K} states")
            _log("  Initialising pymbar.FES from u_kln (MBAR iteration — may take several minutes) ...")
            fes = self._run_pymbar_fes(u_kln, N_k, verbose=verbose,
                                        solver_protocol=solver_protocol)

        # --- Generate histogram FES in the unbiased state ----------------
        # The unbiased reduced potential u_n = 0 everywhere.
        u_n_zero = np.zeros(N_total, dtype=float)
        _log(f"  Generating histogram FES ({self.n_bins} bins) ...")
        fes.generate_fes(
            u_n_zero,
            x_all,
            fes_type='histogram',
            histogram_parameters={'bin_edges': edges},
        )

        # --- Retrieve PMF values and uncertainty -------------------------
        # pymbar._get_fes_histogram stores bin_label only for occupied bins;
        # querying an empty bin raises KeyError.  Only pass centers whose
        # histogram bin contains at least one sample.
        hist_counts, _ = np.histogram(x_all, bins=edges)
        populated = hist_counts > 0                   # shape (n_bins,)
        query_centers = centers[populated]
        _log(f"  Querying FES at {populated.sum()}/{len(centers)} occupied bin centres ...")
        try:
            results_partial = fes.get_fes(
                query_centers,
                reference_point='from-lowest',
                uncertainty_method=uncertainty_method,
            )
        except Exception as exc:
            warnings.warn(
                f"FES.get_fes() failed with uncertainty_method="
                f"'{uncertainty_method}': {exc}\n"
                "Retrying without uncertainty estimate.",
                RuntimeWarning,
            )
            results_partial = fes.get_fes(
                query_centers,
                reference_point='from-lowest',
                uncertainty_method=None,
            )

        # Expand back to full n_bins arrays; NaN for unpopulated bins.
        f_i_partial  = np.asarray(results_partial['f_i'], dtype=float)
        f_i_full     = np.full(len(centers), np.nan)
        f_i_full[populated] = f_i_partial

        df_i_full = None
        if 'df_i' in results_partial and results_partial['df_i'] is not None:
            df_i_full = np.full(len(centers), np.nan)
            df_i_full[populated] = np.asarray(results_partial['df_i'], dtype=float)

        # --- Convert kT → kJ/mol -----------------------------------------
        kT = self.K_B * self.T
        pmf_kJmol = f_i_full * kT
        pmf_err   = df_i_full * kT if df_i_full is not None else None

        # NOTE: pmf_kJmol is stored as-is (from-lowest reference from pymbar).
        # Call reference_to_bulk() afterwards to set the bulk plateau to 0.

        # --- Store results -----------------------------------------------
        self.bin_edges       = edges
        self.bin_centers_abs = centers
        self.bin_width       = width
        self.pmf_mbar_1d     = pmf_kJmol
        self.pmf_mbar_1d_err = pmf_err
        self._u_kn           = u_kn
        self._N_k            = N_k
        self._r0             = r0
        self._x_all          = x_all
        self._used_stride    = self.stride   # record stride actually used

        # Expose underlying MBAR object if pymbar makes it available
        try:
            self._mbar = fes.mbar
        except AttributeError:
            self._mbar = None

        self._converged = True
        _log("ClayMBAR.run_mbar(): done.")

        # Save cache so a future call can skip the expensive solve
        if cache_path is not None and save_cache:
            self.save(cache_path)
            _log(f"  Results cached to: {cache_path}")

        return self

    # -----------------------------------------------------------------------
    # 2-D MBAR reweighting
    # -----------------------------------------------------------------------

    def run_mbar_2d(
        self,
        theta_per_window,
        n_r_bins=None,
        n_theta_bins=36,
        theta_range=(0.0, 90.0),
        bulk_fraction=None,
        chunk_size=100_000,
        verbose=True,
        cache_path=None,
        save_cache=True,
        force_rerun=False,
    ):
        """
        Compute the 2-D MBAR PMF W(r, θ) by post-hoc reweighting.

        After ``run_mbar()`` has optimised the MBAR free energies f_k, this
        method uses those weights to project all N_total samples onto a
        2-D (r, θ) histogram representing the unbiased ensemble.

        The weight for each sample n in the unbiased ensemble is

            w_n  ∝  1 / Σ_k  N_k · exp(f_k − u_kn)

        which is then used as a probability weight in ``np.histogram2d``.
        W(r, θ) = −kT ln ρ(r, θ) is zero-referenced at the high-r bulk region.

        Requires ``run_mbar()`` to have been called on the *same* object
        (i.e. ``self._x_all`` and ``self._u_kn`` must still be in memory).

        Typical usage::

            from ClayPMF2D import ClayPMF2D
            from ClayMBAR  import ClayMBAR

            pmf2d = ClayPMF2D(umbrella_dir='...').load_data()
            pmf2d.load_theta_data(traj_files=[...], topology='...')

            mbar  = ClayMBAR(pmf2d).run_mbar()
            mbar.run_mbar_2d(pmf2d.theta_data)
            mbar.save('results/clay_mbar_2d.npz')

        Parameters
        ----------
        theta_per_window : list of length n_windows
            Each element is a tuple ``(theta1_arr, theta2_arr)`` containing
            the tilt angle [degrees] for CIP1 and CIP2 in that window.
            Pass **raw full-length arrays** — the same ``stride`` that was
            applied to ``z_data`` in ``run_mbar()`` is applied automatically
            here, so the theta arrays are always aligned with ``_x_all``.
            Use ``ClayPMF2D.theta_data`` directly.
        n_r_bins : int or None
            Number of r bins.  Defaults to ``self.n_bins`` (same as 1-D
            MBAR) for consistency.
        n_theta_bins : int
            Number of θ bins.  Default 36 (2.5° resolution for 0–90°).
        theta_range : tuple of (float, float)
            (θ_min, θ_max) in degrees.  Default (0.0, 90.0).
        bulk_fraction : float or None
            Fraction of the r range (high-r) averaged for the zero
            reference.  Defaults to ``self.bulk_fraction``.
        chunk_size : int
            Number of samples processed per iteration when building the
            weighted 2-D histogram.  Reducing this lowers peak memory at the
            cost of slightly more Python overhead.  Default 100 000.
        verbose : bool
        cache_path : str or None
            Path to a .npz file.  If the file exists and ``force_rerun=False``,
            the 2-D results are loaded from it and the computation is skipped.
            Defaults to ``<umbrella_dir>/clay_mbar_2d_s<stride>.npz`` when
            ``umbrella_dir`` is set.
        save_cache : bool
            Save results to ``cache_path`` after running (default True).
        force_rerun : bool
            Ignore an existing cache and always recompute (default False).

        Sets
        ----
        self.pmf_mbar_2d       : ndarray (n_r_bins, n_theta_bins)   [kJ/mol]
        self.r_centers_2d      : ndarray (n_r_bins,)                [nm]
        self.theta_centers_2d  : ndarray (n_theta_bins,)            [deg]
        self.r_edges_2d        : ndarray (n_r_bins+1,)
        self.theta_edges_2d    : ndarray (n_theta_bins+1,)

        Returns
        -------
        self
        """
        import os as _os

        def _log(msg):
            if verbose:
                print(msg, flush=True)

        # --- Default cache path -------------------------------------------
        if cache_path is None and self.umbrella_dir is not None:
            cache_path = _os.path.join(
                self.umbrella_dir,
                f"clay_mbar_2d_s{self._used_stride if self._used_stride else self.stride}.npz",
            )

        # --- Cache load-or-skip -------------------------------------------
        if cache_path is not None and not force_rerun and _os.path.exists(cache_path):
            _log(f"ClayMBAR.run_mbar_2d(): loading cached 2-D results from {cache_path}")
            _loaded = ClayMBAR.load(cache_path)
            for _attr in (
                'pmf_mbar_2d', 'count2d_raw',
                'r_centers_2d', 'theta_centers_2d',
                'r_edges_2d', 'theta_edges_2d',
            ):
                setattr(self, _attr, getattr(_loaded, _attr, None))
            _log("  Loaded from cache — run_mbar_2d() not repeated.")
            return self

        from scipy.special import logsumexp

        def _log(msg):
            if verbose:
                print(msg, flush=True)

        # --- Validate prerequisites ----------------------------------------
        # _u_kn and _x_all are NOT saved to cache (too large).  When the
        # object was loaded from cache (or run_mbar() read from cache), these
        # are None.  If the live ClayPMF/_pmf object is still attached we can
        # rebuild them cheaply without re-solving MBAR (f_k is already known).
        if self._u_kn is None or self._N_k is None or self._x_all is None:
            _pmf_obj = getattr(self, '_pmf', None)
            if _pmf_obj is not None and getattr(_pmf_obj, 'z_data', None) is not None:
                _log(
                    "  _u_kn / _x_all not in memory (loaded from cache). "
                    "Rebuilding from trajectory data (no MBAR re-solve needed) ..."
                )
                _r0_rb, _slist, _N_k_rb = self._build_pseudo_windows()
                _K_rb      = len(_N_k_rb)
                _N_tot_rb  = int(_N_k_rb.sum())
                _x_all_rb  = np.empty(_N_tot_rb, dtype=np.float64)
                _off       = 0
                for _arr in _slist:
                    _n = len(_arr)
                    _x_all_rb[_off:_off + _n] = _arr
                    _off += _n
                _c_rb   = self.beta * 0.5 * self.k
                _u_kn_rb = np.empty((_K_rb, _N_tot_rb), dtype=np.float64)
                for _ki, _r0k in enumerate(_r0_rb):
                    _u_kn_rb[_ki] = _c_rb * (_x_all_rb - _r0k) ** 2
                self._x_all = _x_all_rb
                self._u_kn  = _u_kn_rb
                self._N_k   = _N_k_rb
                self._r0    = _r0_rb
                _log(f"  Rebuilt: K={_K_rb}, N_total={_N_tot_rb:,}")
            else:
                raise RuntimeError(
                    "run_mbar_2d() requires in-memory sample arrays "
                    "(_u_kn / _x_all), but they are None and no live ClayPMF "
                    "object is attached.  Either:\n"
                    "  (a) call run_mbar() on a live ClayPMF/ClayPMF2D object "
                    "before run_mbar_2d(), or\n"
                    "  (b) recreate ClayMBAR(pmf=...) with the ClayPMF2D "
                    "object and call run_mbar() again."
                )
        if len(theta_per_window) != self.n_windows:
            raise ValueError(
                f"theta_per_window has {len(theta_per_window)} entries; "
                f"expected {self.n_windows} (= n_windows)."
            )

        # --- Build theta_all and align with _x_all / _u_kn ---------------
        # theta may have been computed at a different output frequency than
        # z_data.  We handle three cases per state:
        #
        #   len(theta_raw) == N_k_state  → already aligned, use as-is
        #   len(theta_raw) >  N_k_state  → theta is finer; subsample theta
        #   len(theta_raw) <  N_k_state  → theta is coarser; subsample z
        #
        # In the coarser-theta case we select every (N_k_state // n_theta)th
        # column from _u_kn and the corresponding rows from _x_all, so the
        # MBAR weights remain consistent with the samples being histogrammed.
        # f_k from the full MBAR solve is still valid — it does not depend on
        # which subset of samples we evaluate weights at.
        _log("ClayMBAR.run_mbar_2d(): aligning theta arrays with z samples ...")
        theta_list   = []
        z_col_idx_list = []   # selected column indices into _u_kn / _x_all
        N_k_2d       = []
        z_offset     = 0

        for i, (theta1, theta2) in enumerate(theta_per_window):
            for j, theta_raw_arr in enumerate([theta1, theta2]):
                t_raw  = np.asarray(theta_raw_arr, dtype=float)
                n_z    = int(self._N_k[2 * i + j])   # z-samples for this state
                n_th   = len(t_raw)
                state_z_cols = np.arange(z_offset, z_offset + n_z)

                if n_th == n_z:
                    # Perfect frame alignment
                    t_aligned = t_raw
                    sel_cols  = state_z_cols
                elif n_th > n_z:
                    # theta finer than z: subsample theta to match z count
                    th_stride = n_th // n_z
                    t_aligned = t_raw[::th_stride][:n_z]
                    sel_cols  = state_z_cols
                else:
                    # theta coarser than z: subsample z to match theta count
                    # Take evenly-spaced z indices that align with theta frames
                    z_stride  = n_z // n_th
                    sel_cols  = state_z_cols[::z_stride][:n_th]
                    t_aligned = t_raw

                theta_list.append(t_aligned)
                z_col_idx_list.append(sel_cols)
                N_k_2d.append(len(t_aligned))
                z_offset += n_z

        theta_all  = np.concatenate(theta_list)
        all_z_cols = np.concatenate(z_col_idx_list)
        x_all_2d   = self._x_all[all_z_cols]
        u_kn_2d    = self._u_kn[:, all_z_cols]
        N_k_2d     = np.array(N_k_2d, dtype=int)

        _log(
            f"  theta frames used: {len(theta_all):,}  "
            f"(z-frames in full run: {int(self._N_k.sum()):,})"
        )

        # Clip angles to the declared range
        theta_min, theta_max = float(theta_range[0]), float(theta_range[1])
        theta_all = np.clip(theta_all, theta_min, theta_max)
        _log(
            f"  theta range : [{theta_min:.1f}, {theta_max:.1f}]° "
            f"  n_theta_bins = {n_theta_bins}"
        )

        # --- Obtain MBAR free energies f_k --------------------------------
        if self._mbar is not None and hasattr(self._mbar, 'f_k'):
            f_k = np.asarray(self._mbar.f_k, dtype=float)
            _log(f"  Using f_k from stored MBAR object (K={len(f_k)}).")
        elif getattr(self, '_f_k', None) is not None:
            # Loaded from file via ClayMBAR.load() — f_k was saved directly
            f_k = np.asarray(self._f_k, dtype=float)
            _log(f"  Using f_k loaded from archive (K={len(f_k)}).")
        else:
            if self._u_kn is None:
                raise RuntimeError(
                    "f_k is unavailable and _u_kn is None — cannot re-run "
                    "pymbar.MBAR.  Either call run_mbar() first, or load a "
                    "ClayMBAR archive that was saved after run_mbar()."
                )
            _log(
                "  self._mbar not available — re-running pymbar.MBAR to "
                "obtain f_k (this may take a moment) ..."
            )
            _tmp_mbar = pymbar.MBAR(self._u_kn, self._N_k, verbose=False)
            f_k = np.asarray(_tmp_mbar.f_k, dtype=float)

        # --- Build 2-D grid -----------------------------------------------
        _n_r_bins      = int(n_r_bins if n_r_bins is not None else self.n_bins)
        r_edges_2d     = np.linspace(
            float(self.bin_edges[0]), float(self.bin_edges[-1]),
            _n_r_bins + 1,
        )
        theta_edges_2d = np.linspace(theta_min, theta_max, n_theta_bins + 1)

        # --- Chunked MBAR-weighted 2-D histogram --------------------------
        # Processing in chunks keeps peak memory at O(K × chunk_size) rather
        # than O(K × N_total) for the intermediate weight vector.
        # We slice self._u_kn[:, start:end] directly — no recomputation.
        # Per-chunk shift:  w_c = exp(log_w_c - shift_c),  shift_c = max(log_w_c)
        # Since log_w_c = -log_denom_c ≤ 0 for typical MBAR, shift_c ≤ 0 so
        # exp(shift_c) ∈ (0, 1] — no overflow risk.  Each chunk's histogram
        # is re-scaled by exp(shift_c) before accumulation, then the final
        # histogram is divided by its sum (absolute scale cancels).
        N_total  = int(N_k_2d.sum())
        n_chunks = (N_total + chunk_size - 1) // chunk_size
        log_N_k  = np.log(N_k_2d.astype(float))   # shape (K,)
        hist2d   = np.zeros((_n_r_bins, n_theta_bins), dtype=np.float64)

        _log(
            f"  2-D histogram: {_n_r_bins} r-bins × {n_theta_bins} θ-bins  "
            f"(N_total={N_total:,}, {n_chunks} chunks of {chunk_size:,}) ..."
        )

        _chunk_iter = range(n_chunks)
        if self.use_tqdm:
            try:
                from tqdm.auto import tqdm as _tqdm
                _chunk_iter = _tqdm(
                    _chunk_iter, desc='2D histogram chunks', leave=False
                )
            except ImportError:
                pass

        for chunk_idx in _chunk_iter:
            start = chunk_idx * chunk_size
            end   = min(start + chunk_size, N_total)

            u_chunk = u_kn_2d[:, start:end]          # (K, chunk_len)

            # log unnormalised weight: log w_n = -logsumexp_k(f_k + log_N_k - u_kn)
            log_denom_c = logsumexp(
                f_k[:, np.newaxis] + log_N_k[:, np.newaxis] - u_chunk,
                axis=0,
            )
            log_w_c = -log_denom_c

            # Shift for numerical stability; shift_c ≤ 0 so exp never overflows
            shift_c = float(log_w_c.max())
            w_c = np.exp(log_w_c - shift_c)             # max(w_c) = 1.0

            chunk_hist, _, _ = np.histogram2d(
                x_all_2d[start:end],
                theta_all[start:end],
                bins=[r_edges_2d, theta_edges_2d],
                weights=w_c,
            )
            # Restore absolute scale before accumulation
            hist2d += chunk_hist * np.exp(shift_c)

        # Normalise once (per-chunk absolute scale cancels in the ratio)
        total_w = hist2d.sum()
        if total_w > 0.0:
            hist2d /= total_w

        r_centers_2d     = 0.5 * (r_edges_2d[:-1]     + r_edges_2d[1:])
        theta_centers_2d = 0.5 * (theta_edges_2d[:-1] + theta_edges_2d[1:])

        # --- Raw count histogram (no weights) — marks which bins were visited ---
        # MBAR reweighting assigns non-zero probability to essentially every
        # bin, so hist2d > 0 almost everywhere.  The raw count histogram
        # identifies bins that were never directly sampled — those should be
        # NaN in the PMF so the plotter can leave them blank.
        count2d_raw, _, _ = np.histogram2d(
            x_all_2d, theta_all,
            bins=[r_edges_2d, theta_edges_2d],
        )

        # --- Convert weighted density → PMF  W = -kT ln ρ ---------------
        kT = self.K_B * self.T
        with np.errstate(divide='ignore', invalid='ignore'):
            pmf_2d = np.where(hist2d > 0.0, -kT * np.log(hist2d), np.nan)
        # Mask bins with zero raw counts (MBAR extrapolates but these were never sampled)
        pmf_2d = np.where(count2d_raw > 0, pmf_2d, np.nan)

        # --- Reference to bulk (LOW-r = pore centre = bulk) ----------------
        # IMPORTANT: In this clay system low-r is the pore CENTRE (bulk-like)
        # and high-r is the clay SURFACE (adsorption site).
        # Bulk reference is therefore the LOWEST frac of the r axis.
        # Do NOT change this to high-r — that references to the clay surface.
        # This matches reference_to_bulk(), reference_to_bulk_2d(), and
        # adsorption_energy() which all use  r.min() + frac*(r.max()-r.min()).
        frac = float(bulk_fraction if bulk_fraction is not None
                     else self.bulk_fraction)
        r_bulk_thresh = (
            r_centers_2d.min()
            + frac * (r_centers_2d.max() - r_centers_2d.min())
        )
        bulk_r_mask = r_centers_2d <= r_bulk_thresh  # low-r = bulk
        if bulk_r_mask.any():
            shift = float(np.nanmean(pmf_2d[bulk_r_mask, :]))
            if not np.isnan(shift):
                pmf_2d -= shift
        else:
            warnings.warn(
                f"bulk_fraction={frac:.2f} left no r-bins for 2-D bulk "
                "reference; PMF zero-point unchanged.",
                RuntimeWarning,
            )

        # --- Store results ------------------------------------------------
        self.pmf_mbar_2d      = pmf_2d
        self.count2d_raw      = count2d_raw
        self.r_centers_2d     = r_centers_2d
        self.theta_centers_2d = theta_centers_2d
        self.r_edges_2d       = r_edges_2d
        self.theta_edges_2d   = theta_edges_2d

        _n_valid = int(np.sum(~np.isnan(pmf_2d)))
        _n_total = pmf_2d.size
        _log(
            f"ClayMBAR.run_mbar_2d(): done.  "
            f"{_n_valid}/{_n_total} bins populated."
        )

        # Save cache
        if cache_path is not None and save_cache:
            self.save(cache_path)
            _log(f"  2-D results cached to: {cache_path}")

        return self

    def reference_to_bulk_2d(self, bulk_fraction=None):
        """
        Re-zero the 2-D MBAR PMF at the high-r bulk plateau.

        The reference is the mean W(r, θ) over all θ values in the
        ``bulk_fraction`` high-r slice of the r grid, consistent with
        ``reference_to_bulk()`` for the 1-D PMF.

        Parameters
        ----------
        bulk_fraction : float or None
            Fraction of r range (LOW-r pore-centre end) to use as bulk reference.
            Defaults to ``self.bulk_fraction``.

        Returns
        -------
        self
        """
        if self.pmf_mbar_2d is None:
            raise RuntimeError("Call run_mbar_2d() first.")

        frac = float(bulk_fraction if bulk_fraction is not None
                     else self.bulk_fraction)
        r = self.r_centers_2d
        r_bulk_thresh = r.min() + frac * (r.max() - r.min())
        bulk_r_mask = r <= r_bulk_thresh

        if not bulk_r_mask.any():
            warnings.warn(
                f"bulk_fraction={frac:.2f} leaves no 2-D bulk r-bins; "
                "PMF reference unchanged.",
                RuntimeWarning,
            )
            return self

        shift = float(np.nanmean(self.pmf_mbar_2d[bulk_r_mask, :]))
        if not np.isnan(shift):
            self.pmf_mbar_2d -= shift
        return self

    # -----------------------------------------------------------------------
    # Post-processing
    # -----------------------------------------------------------------------

    def reference_to_bulk(self, bulk_fraction=None):
        """
        Zero-reference the MBAR PMF at the bulk plateau.

        The bulk region is the HIGH-r fraction of the r grid
        (large r = far from clay = pore interior), consistent with
        ``ClayPMF.get_adsorption_energy()`` and
        ``ClayMeanForce.reference_to_bulk()``.

        Parameters
        ----------
        bulk_fraction : float or None
            Fraction of the r range (high-r end) to use as bulk.
            Defaults to ``self.bulk_fraction``.

        Returns
        -------
        self
        """
        if self.pmf_mbar_1d is None:
            raise RuntimeError("Call run_mbar() first.")

        frac = float(bulk_fraction if bulk_fraction is not None
                     else self.bulk_fraction)
        r = self.bin_centers_abs
        r_bulk_thresh = r.min() + frac * (r.max() - r.min())
        bulk_mask = r <= r_bulk_thresh

        if not bulk_mask.any():
            warnings.warn(
                f"bulk_fraction={frac:.2f} leaves no bulk points; "
                "PMF reference unchanged.",
                RuntimeWarning,
            )
            return self

        shift = float(np.nanmean(self.pmf_mbar_1d[bulk_mask]))
        self.pmf_mbar_1d = self.pmf_mbar_1d - shift
        return self

    def adsorption_energy(self, r_surface=0.5, r_bulk_start=None):
        """
        Compute adsorption free energy ΔG_ads = min(W_surface) − ⟨W_bulk⟩.

        A negative value indicates favourable adsorption.  Consistent with
        ``ClayPMF.get_adsorption_energy()``.

        Parameters
        ----------
        r_surface : float
            Lower boundary [nm] of the clay-surface region (r ≥ r_surface).
            r ≈ 0 is bulk (pore centre); large r is near clay. Default 0.5 nm.
        r_bulk_start : float or None
            Upper boundary [nm] of the bulk region (r ≤ r_bulk_start).
            If None, uses the low-r ``bulk_fraction`` threshold.

        Returns
        -------
        dG_ads : float  (kJ/mol)
        r_min  : float  (nm) — r position of the PMF minimum
        """
        if self.pmf_mbar_1d is None:
            raise RuntimeError("Call run_mbar() first.")

        r   = self.bin_centers_abs
        pmf = self.pmf_mbar_1d

        if r_bulk_start is None:
            r_bulk_start = r.min() + self.bulk_fraction * (r.max() - r.min())

        mask_surf = r >= r_surface
        mask_bulk = r <= r_bulk_start

        if not mask_surf.any():
            raise ValueError(
                f"No MBAR bins in surface region (r ≥ {r_surface} nm). "
                f"r range: [{r.min():.3f}, {r.max():.3f}] nm."
            )
        if not mask_bulk.any():
            raise ValueError(
                f"No MBAR bins in bulk region (r ≤ {r_bulk_start:.3f} nm). "
                f"r range: [{r.min():.3f}, {r.max():.3f}] nm."
            )

        pmf_surf_arr = pmf[mask_surf]
        r_surf_arr   = r[mask_surf]
        pmf_bulk_val = float(np.nanmean(pmf[mask_bulk]))

        idx_min = np.nanargmin(pmf_surf_arr)
        dG_ads  = float(pmf_surf_arr[idx_min]) - pmf_bulk_val
        r_min   = float(r_surf_arr[idx_min])
        return dG_ads, r_min

    def overlap_matrix(self):
        """
        Return the K × K MBAR state-overlap matrix.

        Diagonal entries are 1.  Off-diagonal values close to 0 signal
        poor sampling connectivity between neighbouring umbrella windows.

        Returns
        -------
        overlap : ndarray, shape (K, K)

        Raises
        ------
        RuntimeError
            If the underlying pymbar MBAR object is not available.
        """
        if self._mbar is None:
            raise RuntimeError(
                "MBAR object not available.  Run run_mbar() with a pymbar "
                "version that exposes fes.mbar."
            )
        try:
            ov_dict = self._mbar.compute_overlap()
            return np.asarray(ov_dict['matrix'])
        except Exception as exc:
            raise RuntimeError(
                f"Could not compute overlap matrix: {exc}"
            ) from exc

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    def print_summary(self):
        """Print a human-readable summary of the MBAR results."""
        if self.pmf_mbar_1d is None:
            print("ClayMBAR: run_mbar() not yet called.")
            return

        pmf   = self.pmf_mbar_1d
        r     = self.bin_centers_abs
        valid = ~np.isnan(pmf)

        print("=" * 60)
        print("ClayMBAR Summary")
        print("=" * 60)
        print(f"  K  (pseudo-states)  : {2 * self.n_windows}")
        print(f"  n_windows           : {self.n_windows}")
        if self._N_k is not None:
            print(f"  N_total samples     : {int(self._N_k.sum()):,}")
            print(f"  Samples / state     : {int(self._N_k.mean()):,} (mean)")
        print(f"  T                   : {self.T} K")
        print(f"  k                   : {self.k:.0f} kJ mol⁻¹ nm⁻²")
        print(f"  n_bins              : {self.n_bins}")
        print(f"  r range             : [{r.min():.3f}, {r.max():.3f}] nm")
        print(f"  Valid PMF bins      : {int(valid.sum())}/{self.n_bins}")
        if valid.any():
            print(f"  PMF min             : {float(np.nanmin(pmf)):.2f} kJ/mol")
            print(f"  PMF max             : {float(np.nanmax(pmf)):.2f} kJ/mol")
        if self.pmf_mbar_1d_err is not None:
            valid_e = ~np.isnan(self.pmf_mbar_1d_err)
            if valid_e.any():
                print(f"  max σ(W)            : "
                      f"{float(np.nanmax(self.pmf_mbar_1d_err)):.3f} kJ/mol")
        try:
            dG, r_min = self.adsorption_energy()
            print(f"  ΔG_ads              : {dG:.2f} kJ/mol"
                  f"  (min at r = {r_min:.3f} nm)")
        except Exception:
            pass
        print("=" * 60)

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, filepath):
        """
        Save MBAR results to a compressed NumPy .npz archive.

        Saves all data required to:
        - Plot 1-D and 2-D PMFs without recomputation.
        - Call ``run_mbar_2d()`` on a loaded object (requires ``f_k``).

        Parameters
        ----------
        filepath : str
            Output path (e.g. ``'results/clay_mbar.npz'``).
        """
        if self.pmf_mbar_1d is None:
            raise RuntimeError("Call run_mbar() before saving.")

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        arrays = dict(
            # --- 1-D grid & results ---
            bin_centers_abs = self.bin_centers_abs,
            bin_edges       = self.bin_edges,
            pmf_mbar_1d     = self.pmf_mbar_1d,
            # --- MBAR internal state (needed for run_mbar_2d on reload) ---
            r0  = self._r0,
            N_k = self._N_k,
        )

        # f_k: extract from live _mbar object, or fall back to _f_k attribute
        # (set when loaded from a previous archive)
        _f_k_to_save = None
        if self._mbar is not None and hasattr(self._mbar, 'f_k'):
            _f_k_to_save = np.asarray(self._mbar.f_k, dtype=float)
        elif getattr(self, '_f_k', None) is not None:
            _f_k_to_save = np.asarray(self._f_k, dtype=float)
        if _f_k_to_save is not None:
            arrays['f_k'] = _f_k_to_save

        if self.pmf_mbar_1d_err is not None:
            arrays['pmf_mbar_1d_err'] = self.pmf_mbar_1d_err

        # --- 2-D results (optional) ---
        if self.pmf_mbar_2d is not None:
            arrays['pmf_mbar_2d']      = self.pmf_mbar_2d
            arrays['r_centers_2d']     = self.r_centers_2d
            arrays['theta_centers_2d'] = self.theta_centers_2d
            arrays['r_edges_2d']       = self.r_edges_2d
            arrays['theta_edges_2d']   = self.theta_edges_2d
            if self.count2d_raw is not None:
                arrays['count2d_raw']  = self.count2d_raw

        # Metadata: [k, T, beta, n_windows, n_bins, bulk_fraction, stride,
        #            max_memory_gb, use_tqdm]  (extend as needed; load reads
        #            by index so old archives with 6 elements still work)
        arrays['_meta'] = np.array(
            [self.k, self.T, self.beta,
             float(self.n_windows), float(self.n_bins), self.bulk_fraction,
             float(self.stride), float(self.max_memory_gb),
             float(getattr(self, 'use_tqdm', True))]
        )

        np.savez_compressed(filepath, **arrays)
        _f_k_msg = f"  f_k: saved (K={len(_f_k_to_save)})" if _f_k_to_save is not None else "  f_k: NOT saved (run_mbar_2d unavailable on reload)"
        print(f"Saved MBAR results → {filepath}")
        print(_f_k_msg)

    @classmethod
    def load(cls, filepath):
        """
        Reload MBAR results from a .npz archive (no ClayPMF object required).

        Parameters
        ----------
        filepath : str

        Returns
        -------
        obj : ClayMBAR  (with run_mbar results already populated)
        """
        data = np.load(filepath, allow_pickle=False)
        meta = data['_meta']
        k, T, beta, n_windows, n_bins, bulk_fraction = (
            float(meta[0]), float(meta[1]), float(meta[2]),
            int(meta[3]),   int(meta[4]),   float(meta[5]),
        )
        # meta[6..8] added in v2; fall back to defaults for old archives
        stride        = int(meta[6])   if len(meta) > 6 else 1
        max_memory_gb = float(meta[7]) if len(meta) > 7 else 2.0
        use_tqdm      = bool(meta[8])  if len(meta) > 8 else True

        # Bypass __init__ (which requires pymbar import + ClayPMF object)
        obj = cls.__new__(cls)
        obj._pmf          = None
        obj.k             = k
        obj.T             = T
        obj.beta          = beta
        obj.n_windows     = n_windows
        obj.n_bins        = n_bins
        obj.bulk_fraction = bulk_fraction
        obj.stride        = stride
        obj.max_memory_gb = max_memory_gb
        obj.use_tqdm      = use_tqdm
        obj.xi_min        = None
        obj.xi_max        = None
        obj.umbrella_dir  = None

        obj.bin_centers_abs = data['bin_centers_abs']
        obj.bin_edges       = data['bin_edges']
        obj.bin_width       = float(obj.bin_edges[1] - obj.bin_edges[0])
        obj.pmf_mbar_1d     = data['pmf_mbar_1d']
        obj.pmf_mbar_1d_err = (data['pmf_mbar_1d_err']
                                if 'pmf_mbar_1d_err' in data else None)
        obj._r0             = data['r0']
        obj._N_k            = data['N_k']
        # f_k: stored directly so run_mbar_2d works without re-running pymbar
        obj._f_k            = data['f_k'] if 'f_k' in data else None
        obj._x_all          = None   # trajectory data not persisted
        obj._u_kn           = None   # u_kn not persisted (re-run if needed)
        obj._mbar           = None
        obj._used_stride    = stride  # same as self.stride for loaded objects
        obj._converged      = True

        # --- 2-D results (optional) ---
        if 'pmf_mbar_2d' in data:
            obj.pmf_mbar_2d      = data['pmf_mbar_2d']
            obj.count2d_raw      = data['count2d_raw'] if 'count2d_raw' in data else None
            obj.r_centers_2d     = data['r_centers_2d']
            obj.theta_centers_2d = data['theta_centers_2d']
            obj.r_edges_2d       = data['r_edges_2d']
            obj.theta_edges_2d   = data['theta_edges_2d']
        else:
            obj.pmf_mbar_2d      = None
            obj.count2d_raw      = None
            obj.r_centers_2d     = None
            obj.theta_centers_2d = None
            obj.r_edges_2d       = None
            obj.theta_edges_2d   = None

        _f_k_msg = f"  f_k: loaded (K={len(obj._f_k)})" if obj._f_k is not None else "  f_k: not in archive (run_mbar_2d unavailable)"
        print(f"Loaded MBAR results ← {filepath}")
        print(_f_k_msg)
        return obj

    # -----------------------------------------------------------------------
    # Repr
    # -----------------------------------------------------------------------

    def __repr__(self):
        status = 'not run'
        if self.pmf_mbar_1d is not None:
            n_valid = int(np.sum(~np.isnan(self.pmf_mbar_1d)))
            status = f'PMF ready ({n_valid}/{self.n_bins} valid bins)'
        return (
            f"ClayMBAR("
            f"n_windows={self.n_windows}, "
            f"K={2 * self.n_windows}, "
            f"T={self.T} K, "
            f"k={self.k:.0f} kJ/mol/nm², "
            f"n_bins={self.n_bins}, "
            f"status={status!r})"
        )
