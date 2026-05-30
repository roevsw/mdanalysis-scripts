#!/usr/bin/env python3
"""
ClayPMF.py
==========
1D WHAM analysis for CIP adsorption PMF on montmorillonite clay surface.

Computes the potential of mean force (PMF) as a function of z-distance from
the clay surface using GROMACS umbrella sampling pull-coordinate data (pullx*.xvg).

Designed for the 2-CIP-molecule setup where each simulation window restrains
two CIP molecules simultaneously on opposite sides of the clay surface in the
z-direction.  Each pullx file therefore contributes *two* independent z-time
series, which are treated as separate pseudo-windows (60 total for 30 pullx
files) to double the effective statistics.

Outputs
-------
- PMF(z)   : signed PMF on [-z_max, +z_max], in kJ/mol
- PMF(|z|) : symmetrised PMF on [0, z_max],  in kJ/mol (analogous to gmx wham -sym)

Usage
-----
    from ClayPMF import ClayPMF

    pmf = ClayPMF(
        umbrella_dir='/path/to/Umbrella/',
        n_windows=30,
        k=1000.0,           # kJ/(mol·nm²)
        T=298.15,           # K
        equil_skip_ps=1000.0,
    )
    pmf.load_data()
    pmf.run_wham()
    pmf.bootstrap_errors(n_bootstrap=200)
    pmf.load_gmx_profile('sym_profile.xvg', unit='kT')
    fig, axes = pmf.plot_pmf(plot_gmx=True)
    pmf.save_results(outdir='.', prefix='pmf_python')
    dG, dG_err = pmf.get_adsorption_energy()
"""

import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import MDAnalysis as mda
    _MDA_AVAILABLE = True
except ImportError:
    _MDA_AVAILABLE = False


class ClayPMF:
    """
    1D WHAM for CIP adsorption PMF on clay surface from GROMACS umbrella sampling.

    Parameters
    ----------
    umbrella_dir : str
        Directory containing pullx*.xvg files (and optionally umbrella*.mdp).
    n_windows : int
        Number of umbrella windows, i.e. number of pullx files. Default 30.
    k : float
        Harmonic spring constant in kJ/(mol·nm²). Default 1000.0.
    T : float
        Temperature in K. Default 298.15.
    equil_skip_ps : float
        Equilibration time to discard from each window in ps. Default 1000.0.
    n_bins : int
        Number of histogram bins for WHAM. Default 200.
    xi_min : float or None
        Lower edge of the RC grid in nm. Auto-detected from data when None.
    xi_max : float or None
        Upper edge of the RC grid in nm. Auto-detected from data when None.
    pullx_prefix : str
        Filename prefix for pull-position files (e.g. 'pullx' → pullx1.xvg).
    tolerance : float
        WHAM self-consistency convergence criterion. Default 1e-6
        (consistent with gmx wham -tol 1e-06).
    max_iter : int
        Maximum WHAM iterations. Default 50000.
    verbose : bool
        Print progress messages. Default True.
    """

    # Boltzmann constant, kJ/(mol·K)
    K_B = 8.314462618e-3

    def __init__(
        self,
        umbrella_dir,
        n_windows=30,
        k=1000.0,
        T=298.15,
        equil_skip_ps=1000.0,
        n_bins=200,
        xi_min=None,
        xi_max=None,
        pullx_prefix='pullx',
        tolerance=1e-6,
        max_iter=50000,
        verbose=True,
    ):
        self.umbrella_dir  = os.path.abspath(umbrella_dir)
        self.n_windows     = n_windows
        self.k             = float(k)
        self.T             = float(T)
        self.equil_skip_ps = float(equil_skip_ps)
        self.n_bins        = int(n_bins)
        self.xi_min        = xi_min
        self.xi_max        = xi_max
        self.pullx_prefix  = pullx_prefix
        self.tolerance     = tolerance
        self.max_iter      = int(max_iter)
        self.verbose       = verbose

        self.beta = 1.0 / (self.K_B * self.T)   # mol / kJ

        # --- set by load_data() ---
        self.z_data         = None   # list of (z1_arr, z2_arr) per window
        self.window_centers = None   # list of (c1, c2) per window

        # --- set by _build_histograms() ---
        self.bins          = None   # shape (n_bins+1,)
        self.bin_centers   = None   # shape (n_bins,)
        self.bin_width     = None
        self.histograms    = None   # shape (2*n_windows, n_bins)
        self.biases        = None   # shape (2*n_windows, n_bins)
        self.n_snapshots   = None   # shape (2*n_windows,)

        # --- set by run_wham() ---
        self.f             = None   # WHAM free energies, shape (2*n_windows,)
        self.P_unb         = None   # unbiased probability density, shape (n_bins,)
        self.pmf_signed    = None   # PMF(z), kJ/mol, shape (n_bins,)
        self.bin_centers_abs = None
        self.pmf_abs       = None   # PMF(|z|), kJ/mol

        # --- set by bootstrap_errors() ---
        self.pmf_signed_std  = None
        self.pmf_abs_std     = None

        # --- set by reference_to_bulk() ---
        self._bulk_shift             = 0.0
        self.bulk_correction_enabled = False

        # --- set by load_gmx_profile() ---
        self.gmx_z   = None
        self.gmx_pmf = None         # converted to kJ/mol
        self.gmx_pmf_std = None

        # --- set by detect_clay_surface() ---
        self.z_clay_surface = None   # z-coordinate of outermost clay layer (nm)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg):
        if self.verbose:
            print(msg)

    @staticmethod
    def _read_pullx(filepath):
        """
        Parse a GROMACS pullx xvg file.

        Skips all header lines beginning with '#' or '@'.

        Returns
        -------
        time : np.ndarray  (ps)
        z1   : np.ndarray  (nm) – pull coord 1 (CIP1, positive-z side)
        z2   : np.ndarray  (nm) – pull coord 2 (CIP2, negative-z side)
        """
        data = np.loadtxt(filepath, comments=['#', '@'])
        if data.ndim < 2 or data.shape[0] == 0:
            raise ValueError(
                f"pullx file appears empty or has no numeric data: {filepath}"
            )
        return data[:, 0], data[:, 1], data[:, 2]

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self):
        """
        Read all pullx*.xvg files, discard equilibration, and extract
        production z-trajectories and window centres.

        Window centres are estimated from the mean of the first 25 frames
        (~0.5 ps at 0.02 ps output interval) because ``pull_coord1_start=yes``
        fixes the restraint origin at the t=0 position.

        Returns
        -------
        self
        """
        self._log(
            f"Loading {self.n_windows} pullx files from:\n  {self.umbrella_dir}"
        )

        self.z_data         = []
        self.window_centers = []

        for i in range(1, self.n_windows + 1):
            fpath = os.path.join(
                self.umbrella_dir, f'{self.pullx_prefix}{i}.xvg'
            )
            if not os.path.isfile(fpath):
                raise FileNotFoundError(f"Pull-x file not found: {fpath}")

            time, z1, z2 = self._read_pullx(fpath)

            # Window centre from mean of first 25 frames (pull_coord_start=yes
            # → restraint origin = initial position)
            n_cf = min(25, len(z1))
            c1 = float(np.mean(z1[:n_cf]))
            c2 = float(np.mean(z2[:n_cf]))

            # Discard equilibration
            mask = time >= self.equil_skip_ps
            z1_prod = z1[mask]
            z2_prod = z2[mask]

            if len(z1_prod) == 0:
                import warnings
                warnings.warn(
                    f"Window {i}: no frames remain after "
                    f"equil_skip_ps={self.equil_skip_ps} ps "
                    f"(last time = {time[-1]:.1f} ps). Skipping window.",
                    UserWarning, stacklevel=2,
                )
                continue

            self.z_data.append((z1_prod, z2_prod))
            self.window_centers.append((c1, c2))

            if self.verbose and i % 5 == 0:
                self._log(
                    f"  win {i:2d}: c1={c1:+.3f} nm  c2={c2:+.3f} nm  "
                    f"n_frames={len(z1_prod):,}"
                )

        self._log(
            f"Done — {self.n_windows} windows × 2 CIPs = "
            f"{2 * self.n_windows} pseudo-windows total."
        )
        return self

    # ------------------------------------------------------------------
    # Clay surface detection (MDAnalysis)
    # ------------------------------------------------------------------

    def detect_clay_surface(
        self,
        structure_file,
        clay_selection='resname MMT',
        surface_pct=95,
    ):
        """
        Detect the outermost clay layer z-position using MDAnalysis.

        The pull coordinate in GROMACS measures displacement from the clay
        COM (z = 0).  Calling this method stores the clay surface z-position
        so that plot methods can label the x-axis as "distance from clay
        surface" rather than "distance from clay COM".

        Parameters
        ----------
        structure_file : str
            Path to a GROMACS structure file (.gro, .tpr, or any
            MDAnalysis-readable topology).
        clay_selection : str
            MDAnalysis selection string for clay atoms.
            Default ``'resname MMT'``.
        surface_pct : float
            Percentile of |z| clay-atom positions used to define the
            surface, e.g. 95 uses the 95th percentile to avoid outliers.
            Default 95.

        Returns
        -------
        z_surface : float
            z-coordinate of the clay surface (nm).

        Sets
        ----
        self.z_clay_surface : float  (nm)

        Notes
        -----
        MDAnalysis positions are in Angstroms; this method converts to nm.
        The clay slab straddles the z=0 ≡ z=Lz PBC boundary (standard MMT
        setup).  Atoms are folded into [-Lz/2, +Lz/2] via minimum-image
        convention before the percentile is taken, matching the GROMACS
        pull-coordinate frame where z=0 is the clay COM.
        """
        if not _MDA_AVAILABLE:
            raise ImportError(
                "MDAnalysis is required for detect_clay_surface(). "
                "Install with:  pip install MDAnalysis"
            )

        u    = mda.Universe(structure_file)
        clay = u.select_atoms(clay_selection)

        if len(clay) == 0:
            available = sorted(set(u.residues.resnames))
            raise ValueError(
                f"No atoms matched '{clay_selection}'. "
                f"Available residue names: {available}"
            )

        # MDAnalysis positions in Angstroms; box z-length in Angstroms.
        Lz_ang = float(u.dimensions[2])
        z_ang  = clay.positions[:, 2]

        # Centre the z-axis so that z=0 is the pore mid-plane (same convention
        # as GROMACS pull coordinate).  PBC is only in xy for this mesopore
        # system — no z-folding needed.
        z_centered = z_ang - Lz_ang / 2.0

        # Use Si atoms (primary structural reference, always split between the
        # two clay layers) to locate the surface — same as ZDirectionalAnalysis
        # calculate_clay_interface_boundaries().
        si_sel = u.select_atoms(
            f"({clay_selection}) and "
            "(name Si or name SI or name Sio or name SIO)"
        )

        if len(si_sel) > 0:
            si_z = si_sel.positions[:, 2] - Lz_ang / 2.0
            upper_si = si_z[si_z > 0]
            if len(upper_si) == 0:
                # Fallback: all Si (should not happen for a split clay)
                upper_si = np.abs(si_z)
            z_surface = float(np.mean(upper_si)) / 10.0   # Å → nm
            method = f"mean of {len(upper_si)} upper Si atoms (z>0)"
        else:
            # Fallback: all clay atoms, take mean of upper half
            upper = z_centered[z_centered > 0]
            if len(upper) == 0:
                upper = np.abs(z_centered)
            z_surface = float(np.mean(upper)) / 10.0
            method = f"mean of {len(upper)} upper clay atoms (no Si found)"

        self.z_clay_surface = z_surface

        self._log(
            f"Clay surface detected: z_surface = {z_surface:.3f} nm\n"
            f"  Method: {method}\n"
            f"  Lz = {Lz_ang/10.0:.3f} nm; selection='{clay_selection}'; "
            f"n_clay_atoms={len(clay):,}\n"
            f"  Plots will label x-axis as 'distance from clay surface'."
        )
        return z_surface

    # ------------------------------------------------------------------
    # Histogram / bias construction
    # ------------------------------------------------------------------

    def _build_histograms(self):
        """
        Construct histograms and harmonic bias arrays for all
        2 × n_windows pseudo-windows, using **reflected** |z| coordinates.

        Both CIP branches are folded onto the positive half-space before
        histogramming, exactly analogous to ``gmx wham -sym``.  This avoids
        the ill-conditioned WHAM problem that arises when the positive-z
        (CIP1) and negative-z (CIP2) branches are only connected at z ≈ 0.

        Pseudo-window indexing:
          2i   → CIP1:  r = |z1|,   centre = |c1|
          2i+1 → CIP2:  r = |z2|,   centre = |c2|

        The RC grid spans [0, r_max] (pure |z|).

        Sets
        ----
        self.bins, self.bin_centers, self.bin_width,
        self.histograms, self.biases, self.n_snapshots
        """
        if self.z_data is None:
            raise RuntimeError("Call load_data() before running WHAM.")

        # RC grid over |z| ∈ [0, r_max] -----------------------------------
        if self.xi_max is None:
            all_r = np.concatenate(
                [np.concatenate([np.abs(z1), np.abs(z2)])
                 for z1, z2 in self.z_data]
            )
            r_max = all_r.max() * 1.05
        else:
            r_max = float(self.xi_max)

        r_min = 0.0   # |z| ≥ 0 always

        self.bins        = np.linspace(r_min, r_max, self.n_bins + 1)
        self.bin_centers = 0.5 * (self.bins[:-1] + self.bins[1:])
        self.bin_width   = float(self.bins[1] - self.bins[0])

        # Allocate arrays --------------------------------------------------
        R    = 2 * self.n_windows
        hists = np.zeros((R, self.n_bins), dtype=float)
        bias  = np.zeros((R, self.n_bins), dtype=float)
        nsn   = np.zeros(R, dtype=float)

        for i, ((z1, z2), (c1, c2)) in enumerate(
            zip(self.z_data, self.window_centers)
        ):
            idx1 = 2 * i       # CIP1 pseudo-window
            idx2 = 2 * i + 1   # CIP2 pseudo-window

            # Reflect to positive half-space
            r1  = np.abs(z1)
            r2  = np.abs(z2)
            rc1 = abs(c1)
            rc2 = abs(c2)

            h1, _ = np.histogram(r1, bins=self.bins)
            h2, _ = np.histogram(r2, bins=self.bins)
            hists[idx1] = h1.astype(float)
            hists[idx2] = h2.astype(float)

            # Harmonic bias: V_i(r) = ½ k (r − |ξ₀|)²
            bias[idx1] = 0.5 * self.k * (self.bin_centers - rc1) ** 2
            bias[idx2] = 0.5 * self.k * (self.bin_centers - rc2) ** 2

            nsn[idx1] = len(z1)
            nsn[idx2] = len(z2)

        self.histograms  = hists
        self.biases      = bias
        self.n_snapshots = nsn

        self._log(
            f"Histogram grid: [0.000, {r_max:.3f}] nm (|z|), "
            f"{self.n_bins} bins, Δξ = {self.bin_width:.4f} nm"
        )
        return self

    # ------------------------------------------------------------------
    # Core WHAM solver
    # ------------------------------------------------------------------

    def run_wham(self, tolerance=None, max_iter=None):
        """
        Solve the WHAM self-consistency equations (Kumar et al. 1992).

        The unbiased probability density is::

            P(ξ) = Σᵢ Hᵢ(ξ) / Σᵢ [ Nᵢ · exp(fᵢ − β·Vᵢ(ξ)) ]

        with the self-consistency condition::

            exp(−fᵢ) = Σ_ξ P(ξ) · exp(−β·Vᵢ(ξ)) · Δξ

        The WHAM free energies ``fᵢ`` are normalised to zero mean each
        iteration to prevent drift.

        Parameters
        ----------
        tolerance : float or None
            Override instance tolerance.
        max_iter : int or None
            Override instance max_iter.

        Returns
        -------
        pmf_signed : np.ndarray
            PMF(z) in kJ/mol, NaN where unsampled. Zero-referenced at minimum.
        pmf_abs : np.ndarray
            PMF(|z|) in kJ/mol (symmetrised). Zero-referenced at minimum.
        """
        if tolerance is None:
            tolerance = self.tolerance
        if max_iter is None:
            max_iter = self.max_iter

        # Build histograms on first call (or after xi_min/xi_max change)
        if self.histograms is None:
            self._build_histograms()

        R       = 2 * self.n_windows
        H_total = np.sum(self.histograms, axis=0)   # (n_bins,) total counts

        # Warm start from previous solution (useful in bootstrap)
        f = self.f.copy() if self.f is not None else np.zeros(R)

        # Precompute exp(−β·V) once – shape (R, n_bins)
        exp_neg_bV = np.exp(-self.beta * self.biases)

        self._log(
            f"Running WHAM: {R} pseudo-windows, {self.n_bins} bins, "
            f"tol={tolerance:.1e}, max_iter={max_iter}"
        )

        diff = np.inf
        for iteration in range(max_iter):
            f_old = f.copy()

            # --- denominator -------------------------------------------
            # denom[j] = Σ_k N_k · exp(f_k) · exp(−β·V_k(ξ_j))
            # Vectorised: (R,) @ (R, n_bins) → (n_bins,)
            denom = (self.n_snapshots * np.exp(f)) @ exp_neg_bV
            denom = np.where(denom > 0.0, denom, np.inf)

            # --- unbiased probability (unnormalized) -------------------
            P_unb = H_total / denom                     # (n_bins,)
            norm  = np.sum(P_unb) * self.bin_width
            P_norm = P_unb / max(norm, 1e-300)

            # --- update free energies ----------------------------------
            # exp(−f_k) = Σ_j P_norm(ξ_j) · exp(−β·V_k(ξ_j)) · Δξ
            # Vectorised: (R, n_bins) @ (n_bins,) → (R,)
            integrals = (exp_neg_bV @ P_norm) * self.bin_width
            f = -np.log(np.where(integrals > 0, integrals, 1e-300))

            # Normalise to prevent drift
            f -= f.mean()

            diff = float(np.max(np.abs(f - f_old)))
            if diff < tolerance:
                self._log(
                    f"  Converged after {iteration + 1} iterations "
                    f"(Δf = {diff:.2e})"
                )
                break
        else:
            warnings.warn(
                f"WHAM did not converge after {max_iter} iterations "
                f"(last Δf = {diff:.2e}). Consider increasing max_iter.",
                RuntimeWarning,
            )

        self.f = f

        # --- Final unbiased probability --------------------------------
        denom = (self.n_snapshots * np.exp(f)) @ exp_neg_bV
        denom = np.where(denom > 0.0, denom, np.inf)
        P_unb = H_total / denom

        norm = np.sum(P_unb) * self.bin_width
        if norm > 0:
            P_unb /= norm
        self.P_unb = P_unb

        # --- PMF(z) ---------------------------------------------------
        with np.errstate(divide='ignore', invalid='ignore'):
            pmf = np.where(P_unb > 0, -np.log(P_unb) / self.beta, np.nan)

        pmf -= np.nanmin(pmf)

        # --- PMF(|z|): this IS the direct output (|z|-WHAM) -----------
        self.bin_centers_abs = self.bin_centers.copy()   # [0, r_max]
        self.pmf_abs         = pmf.copy()

        # --- PMF(z): symmetric signed profile by reflection ------------
        # PMF_signed(z) = PMF_abs(|z|).  Construct full [-r_max, r_max] grid.
        r_vals  = self.bin_centers                       # [0, r_max]
        # Mirror: z = -r_max ... -Δξ  (flip; skip z=0 to avoid duplicate)
        z_neg   = -r_vals[::-1][:-1]                     # [-r_max, ..., -Δξ]
        z_pos   = r_vals                                 # [0, r_max]
        pmf_neg = pmf[::-1][:-1]
        pmf_pos = pmf

        self._bin_centers_signed = np.concatenate([z_neg, z_pos])
        self.pmf_signed          = np.concatenate([pmf_neg, pmf_pos])

        return self.pmf_signed, self.pmf_abs

    # ------------------------------------------------------------------
    # Symmetrisation (kept for backward compatibility — now trivial)
    # ------------------------------------------------------------------

    def _symmetrize_pmf(self):
        """Backward-compat stub — PMF(|z|) is computed directly by the |z| WHAM."""
        return self.pmf_abs, self.bin_centers_abs

    # ------------------------------------------------------------------
    # Aliases so ClayPMF can be used as the WHAM source in plot_meanforce_vs_wham
    # (which expects the same interface as ClayPMF3D: r_centers, pmf_r, pmf_r_std)
    # ------------------------------------------------------------------
    @property
    def r_centers(self):
        return self.bin_centers_abs

    @property
    def pmf_r(self):
        return self.pmf_abs

    @property
    def pmf_r_std(self):
        return self.pmf_abs_std

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def bootstrap_errors(self, n_bootstrap=200):
        """
        Estimate PMF uncertainty via Poisson bootstrap resampling.

        Each bootstrap sample resamples every histogram bin count as
        Poisson(Hᵢ(ξ)) and re-runs WHAM from the converged free energies
        as a warm start.

        Sets ``self.pmf_signed_std`` and ``self.pmf_abs_std``.

        Parameters
        ----------
        n_bootstrap : int
            Number of bootstrap iterations. Default 200.

        Returns
        -------
        pmf_signed_std : np.ndarray
        pmf_abs_std    : np.ndarray
        """
        if self.f is None:
            self.run_wham()

        self._log(f"Bootstrap: {n_bootstrap} samples (Poisson resampling)…")

        hist_orig = self.histograms.copy()
        f_init    = self.f.copy()

        pmf_signed_samples = []
        pmf_abs_samples    = []

        # Suppress per-iteration output during bootstrap
        _verbose_backup = self.verbose
        self.verbose = False

        try:
            for _ in tqdm(range(n_bootstrap), desc='Bootstrap', disable=not _verbose_backup):
                self.histograms = np.random.poisson(hist_orig).astype(float)
                self.f = f_init.copy()   # warm start

                pmf_s, pmf_a = self.run_wham()

                # Re-zero each sample at its minimum
                pmf_s = pmf_s - np.nanmin(pmf_s)
                pmf_a = pmf_a - np.nanmin(pmf_a)

                pmf_signed_samples.append(pmf_s.copy())
                pmf_abs_samples.append(pmf_a.copy())
        finally:
            # Always restore originals even if an exception occurs
            self.histograms = hist_orig
            self.f          = f_init
            self.verbose    = _verbose_backup

        # Re-run final WHAM to restore P_unb, pmf_signed, pmf_abs
        self.run_wham()

        # Signed PMF std
        pmf_signed_arr        = np.array(pmf_signed_samples)   # (B, n_bins)
        self.pmf_signed_std   = np.nanstd(pmf_signed_arr, axis=0)

        # Abs PMF std – all samples should have the same r_vals length
        # (same bins), but use the minimum in case of edge differences
        min_len = min(len(p) for p in pmf_abs_samples)
        pmf_abs_arr      = np.array([p[:min_len] for p in pmf_abs_samples])
        self.pmf_abs_std = np.nanstd(pmf_abs_arr, axis=0)

        self.verbose = _verbose_backup
        self._log(
            f"Bootstrap done.  "
            f"max σ(PMF_signed) = {np.nanmax(self.pmf_signed_std):.3f} kJ/mol  |  "
            f"max σ(PMF_|z|) = {np.nanmax(self.pmf_abs_std):.3f} kJ/mol"
        )
        return self.pmf_signed_std, self.pmf_abs_std

    # ------------------------------------------------------------------
    # Bulk referencing
    # ------------------------------------------------------------------

    def reference_to_bulk(self, bulk_fraction=0.2, enabled=True):
        """
        Re-zero all PMFs so that the bulk plateau equals zero.

        The bulk reference level is the **median** of ``pmf_abs`` over the
        *first* ``bulk_fraction`` of r-bins (small r ≈ 0 = bulk / pore centre).
        Calling again with new parameters replaces the previous correction.

        Parameters
        ----------
        bulk_fraction : float
            Fraction of r-axis bins used as bulk reference. Default 0.2.
        enabled : bool
            If False, undo any existing correction and leave PMFs at
            their min=0 baseline.
        """
        if self.pmf_abs is None:
            raise RuntimeError("run_wham() must be called first.")

        # Undo previous correction if any
        if self.bulk_correction_enabled and self._bulk_shift != 0.0:
            for arr in (self.pmf_abs, self.pmf_signed):
                if arr is not None:
                    arr += self._bulk_shift
            self._bulk_shift = 0.0
            self.bulk_correction_enabled = False

        self.bulk_fraction = bulk_fraction

        if not enabled:
            self._log("reference_to_bulk: disabled — PMFs left at min=0.")
            return

        n_bulk = max(1, int(bulk_fraction * len(self.pmf_abs)))
        # Small r (r ≈ 0) = pore centre = bulk in ClayPMF coordinate convention
        shift  = float(np.nanmedian(self.pmf_abs[:n_bulk]))

        self.pmf_abs    -= shift
        if self.pmf_signed is not None:
            self.pmf_signed -= shift

        self._bulk_shift             = shift
        self.bulk_correction_enabled = True

        self._log(
            f"reference_to_bulk: bulk_fraction={bulk_fraction:.2f} "
            f"({n_bulk} bins), median shift = {shift:+.3f} kJ/mol applied."
        )

    # ------------------------------------------------------------------
    # gmx wham comparison
    # ------------------------------------------------------------------

    def load_gmx_profile(self, filepath, unit='kT'):
        """
        Load a gmx wham profile xvg file for comparison.

        Parameters
        ----------
        filepath : str
            Path to the xvg file (e.g. ``sym_profile.xvg``).
        unit : {'kT', 'kJ/mol', 'kcal/mol'}
            Energy unit used in the gmx wham run. Values are converted
            to kJ/mol for consistent comparison. Default ``'kT'``.

        Returns
        -------
        gmx_z   : np.ndarray  (nm)
        gmx_pmf : np.ndarray  (kJ/mol)
        """
        data = np.loadtxt(filepath, comments=['#', '@'])
        self.gmx_z   = data[:, 0]
        self.gmx_pmf = data[:, 1].copy()

        if unit == 'kT':
            kT = self.K_B * self.T   # kJ/mol
            self.gmx_pmf *= kT
        elif unit == 'kcal/mol':
            self.gmx_pmf *= 4.184
        # 'kJ/mol' → no conversion needed

        # Optionally load std column (e.g. from gmx wham bootstrap output)
        self.gmx_pmf_std = None
        if data.shape[1] > 2:
            self.gmx_pmf_std = data[:, 2].copy()
            if unit == 'kT':
                self.gmx_pmf_std *= kT
            elif unit == 'kcal/mol':
                self.gmx_pmf_std *= 4.184

        self._log(f"Loaded gmx WHAM profile from {filepath} (unit={unit})")
        return self.gmx_z, self.gmx_pmf

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save_results(self, outdir='.', prefix='pmf_python'):
        """
        Save PMF results to plain-text files.

        Files written:
        - ``{prefix}_signed.dat`` : z(nm)  PMF(kJ/mol)  [std(kJ/mol)]
        - ``{prefix}_abs.dat``    : |z|(nm) PMF(kJ/mol) [std(kJ/mol)]

        Returns
        -------
        fname_signed, fname_abs : str
        """
        if self.pmf_signed is None:
            raise RuntimeError("Call run_wham() before saving.")

        os.makedirs(outdir, exist_ok=True)

        # Signed PMF
        fname_s = os.path.join(outdir, f'{prefix}_signed.dat')
        z_signed = getattr(self, '_bin_centers_signed', self.bin_centers)
        cols_s  = [z_signed, self.pmf_signed]
        hdr_s   = 'z_nm  PMF_kJmol'
        if self.pmf_signed_std is not None:
            cols_s.append(self.pmf_signed_std)
            hdr_s += '  PMF_std_kJmol'
        np.savetxt(fname_s, np.column_stack(cols_s), header=hdr_s, fmt='%.6f')

        # |z| PMF
        fname_a = os.path.join(outdir, f'{prefix}_abs.dat')
        n = len(self.bin_centers_abs)
        cols_a  = [self.bin_centers_abs, self.pmf_abs[:n]]
        hdr_a   = 'abs_z_nm  PMF_kJmol'
        if self.pmf_abs_std is not None:
            m = min(n, len(self.pmf_abs_std))
            cols_a = [
                self.bin_centers_abs[:m],
                self.pmf_abs[:m],
                self.pmf_abs_std[:m],
            ]
            hdr_a += '  PMF_std_kJmol'
        np.savetxt(fname_a, np.column_stack(cols_a), header=hdr_a, fmt='%.6f')

        self._log(f"Saved:\n  {fname_s}\n  {fname_a}")
        return fname_s, fname_a

    # ------------------------------------------------------------------
    # Thermodynamics
    # ------------------------------------------------------------------

    def get_adsorption_energy(self, r_surface=2.0, r_bulk_start=0.5):
        """
        Compute adsorption free energy from the symmetrised PMF(|z|).

        ΔG_ads = ⟨PMF⟩_surface − ⟨PMF⟩_bulk

        A negative value indicates favourable adsorption.

        Parameters
        ----------
        r_surface : float
            Lower boundary (nm) of the surface region (r ≥ r_surface).
            r ≈ 0 is bulk (pore centre); large r is near clay. Default 2.0 nm.
        r_bulk_start : float
            Upper boundary (nm) of the bulk region (r ≤ r_bulk_start).
            Default 0.5 nm.

        Returns
        -------
        dG_ads : float  (kJ/mol)
        dG_err : float or None  (kJ/mol, from bootstrap std if available)
        """
        if self.pmf_abs is None:
            raise RuntimeError("Call run_wham() first.")

        r   = self.bin_centers_abs
        pmf = self.pmf_abs

        mask_surf = r >= r_surface
        mask_bulk = r <= r_bulk_start

        if not mask_surf.any():
            raise ValueError(
                f"No PMF bins in the surface region (r ≥ {r_surface} nm). "
                f"r range is [{r.min():.3f}, {r.max():.3f}] nm."
            )
        if not mask_bulk.any():
            raise ValueError(
                f"No PMF bins in the bulk region (r ≤ {r_bulk_start} nm). "
                f"r range is [{r.min():.3f}, {r.max():.3f}] nm."
            )

        pmf_surf = float(np.nanmean(pmf[mask_surf]))
        pmf_bulk = float(np.nanmean(pmf[mask_bulk]))
        dG_ads   = pmf_surf - pmf_bulk

        dG_err = None
        if self.pmf_abs_std is not None:
            n   = min(len(r), len(self.pmf_abs_std))
            rn  = r[:n]
            sn  = self.pmf_abs_std[:n]
            e_s = float(np.nanmean(sn[rn >= r_surface])) if (rn >= r_surface).any() else 0.0
            e_b = float(np.nanmean(sn[rn <= r_bulk_start])) if (rn <= r_bulk_start).any() else 0.0
            dG_err = float(np.sqrt(e_s**2 + e_b**2))

        self._log(
            f"ΔG_ads = {dG_ads:.2f} kJ/mol"
            + (f"  ±{dG_err:.2f} kJ/mol" if dG_err is not None else "")
            + f"  (surface r≥{r_surface} nm, bulk r≤{r_bulk_start} nm)"
        )
        return dG_ads, dG_err

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self):
        status = 'not loaded'
        if self.pmf_signed is not None:
            status = 'PMF computed'
        elif self.z_data is not None:
            status = 'data loaded'
        return (
            f"ClayPMF("
            f"n_windows={self.n_windows}, "
            f"k={self.k:.0f} kJ/mol/nm², "
            f"T={self.T} K, "
            f"equil={self.equil_skip_ps:.0f} ps, "
            f"n_bins={self.n_bins}, "
            f"status={status!r})"
        )
