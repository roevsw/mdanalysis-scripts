#!/usr/bin/env python3
"""
ClayKd.py
=========
Compute dissociation constant (Kd) and binding free energy (ΔG°) from an MFEP
produced by ClayPath.

CONVENTION (FIXED)
------------------
  PATH START (index 0)    →  BULK (large r, e.g., 3.0 nm, W ≈ 0 kJ/mol)
  PATH END (index -1)     →  SURFACE (small r, e.g., 0.2 nm, W negative)
  r[0] > r[-1]            →  r decreases along the path

This convention is HARD-CODED. No automatic detection is performed.
If your data does not follow this convention, reverse it before passing.

Two methods are implemented:

  1. Endpoint method (quick estimate):
       ΔG°_ep  = W_surface - W_bulk          [kJ/mol]  (negative if favorable)
       Kd_ep   = exp(ΔG°_ep / RT) × 1 M      [M]

  2. Partition-function method (Roux 2004, Biophys J 86:1087):
       I_bound = ∫_{r_saddle}^{r_surface} exp(−W(r)/kT) dr   [nm]
       Kd_pf   = 1 / (c° × I_bound)                           [M]
       ΔG°_pf  = −RT ln(c° × I_bound)                         [kJ/mol]

where:
  - c° = 0.6022 nm⁻³ (standard concentration 1 M in molecular units)
  - r_saddle is the transition state (maximum PMF along the path)
  - Integration is from saddle TO surface (bound region only)

Parameters
----------
clay_path : ClayPath
    A ClayPath instance after run_string_method() has been called.
    Must contain r_path and pmf_path following the convention above.
T : float
    Temperature in Kelvin. Default 298.15 K.
c_standard : float or None
    Standard concentration in nm⁻³. If None, uses 0.6022 nm⁻³ (1 M).
    Set to 1.0 for units of nm³ (molecular units).

Usage
-----
    >>> kd = ClayKd(clay_path)
    >>> results = kd.compute()
    >>> kd.summary()
"""

import warnings
import numpy as np

class ClayKd:
    """Compute Kd and ΔG° from MFEP following bulk→surface convention."""

    # Physical constants
    _K_B = 8.314462618e-3   # kJ mol⁻¹ K⁻¹
    _C_STD = 0.6022          # nm⁻³ (1 M = 0.6022 molecules/nm³)
    _N_AVOGADRO = 6.02214076e23  # mol⁻¹

    # ------------------------------------------------------------------
    def __init__(self, clay_path, T=298.15, c_standard=None):
        """
        Initialize with a ClayPath object.

        Parameters
        ----------
        clay_path : ClayPath
            Must have r_path and pmf_path following the convention:
            r[0] > r[-1] (bulk at start, surface at end)
        T : float
            Temperature in Kelvin.
        c_standard : float or None
            Standard concentration in nm⁻³. None = 0.6022 nm⁻³ (1 M).
        """
        # Validate input
        if clay_path.pmf_path is None:
            raise RuntimeError(
                "clay_path.pmf_path is None. Call run_string_method() first."
            )

        # Store references
        self.cp = clay_path
        self.T = float(T)
        self.c_std = float(c_standard) if c_standard is not None else self._C_STD

        # Raw data (following your convention)
        self.r_raw = np.asarray(clay_path.r_path, dtype=float)
        self.w_raw = np.asarray(clay_path.pmf_path, dtype=float)

        # Validate convention
        self._validate_convention()

        # Results storage
        self._results = None

        # Pre-computed values (set during compute)
        self.r = None          # r array (after any processing)
        self.w = None          # w array (after zero referencing)
        self.bulk_mean = None  # Mean W in bulk region
        self.pmf_ref = None    # Zero-referenced PMF

    # ------------------------------------------------------------------
    def _validate_convention(self):
        """
        Validate that input follows the bulk→surface convention.
        Issues warnings but does not modify data.
        """
        r = self.r_raw
        w = self.w_raw

        # Check 1: r should decrease (bulk large → surface small)
        if len(r) >= 2:
            if r[0] <= r[-1]:
                warnings.warn(
                    f"r_path: r[0]={r[0]:.3f} nm, r[-1]={r[-1]:.3f} nm.\n"
                    "Convention expects r[0] > r[-1] (bulk at start, surface at end).\n"
                    "Consider reversing your path with: r = r[::-1], w = w[::-1]",
                    RuntimeWarning, stacklevel=2
                )

        # Check 2: First point should be near bulk (W ≈ 0 or plateau)
        # This is a soft check - just inform the user
        n_check = min(3, len(w))
        first_vals = w[:n_check]
        print(f"  Bulk reference check: first {n_check} W values = {first_vals}")
        print(f"    (expect ≈ 0 kJ/mol for bulk region)")

        # Check 3: Last point should be negative (adsorption well)
        if len(w) > 0 and w[-1] > 0:
            warnings.warn(
                f"Last point W[-1]={w[-1]:.2f} kJ/mol is positive.\n"
                "Convention expects negative (favorable adsorption).\n"
                "Check that PMF is properly zero-referenced.",
                RuntimeWarning, stacklevel=2
            )

    # ------------------------------------------------------------------
    def _zero_reference_to_bulk(self, n_bulk_points=5):
        """
        Zero-reference PMF to bulk region (first n_bulk_points).

        Parameters
        ----------
        n_bulk_points : int
            Number of images from the START of the path (bulk region)
            to average for zero reference. Default 5.

        Returns
        -------
        pmf_ref : ndarray
            Zero-referenced PMF (bulk = 0)
        bulk_mean : float
            Mean value subtracted
        """
        n_points = min(n_bulk_points, len(self.w_raw))
        if n_points < 1:
            raise ValueError(f"n_bulk_points={n_bulk_points} but only {len(self.w_raw)} images")

        # Bulk is at the START of the path (index 0)
        self.bulk_mean = np.mean(self.w_raw[:n_points])
        self.pmf_ref = self.w_raw - self.bulk_mean
        self.r = self.r_raw.copy()  # Keep original r (decreasing)

        print(f"  Bulk reference: averaged {n_points} images, mean = {self.bulk_mean:.3f} kJ/mol")
        print(f"  PMF range after referencing: {self.pmf_ref.min():.2f} to {self.pmf_ref.max():.2f} kJ/mol")

        return self.pmf_ref, self.bulk_mean

    # ------------------------------------------------------------------
    def _find_saddle(self, r_saddle=None):
        """
        Find the saddle point (transition state) along the path.

        Parameters
        ----------
        r_saddle : float or None
            If None, saddle is the maximum PMF along the path.
            If float, finds the closest image.

        Returns
        -------
        idx : int
            Index of saddle in the array
        r_saddle : float
            r-coordinate at saddle (nm)
        w_saddle : float
            PMF at saddle (kJ/mol)
        """
        if self.pmf_ref is None:
            raise RuntimeError("Call _zero_reference_to_bulk() first.")

        if r_saddle is None:
            # Saddle = maximum PMF (highest energy point between bulk and surface)
            # Because bulk=0 and surface is negative, the maximum is the barrier
            idx = int(np.argmax(self.pmf_ref))
            r_saddle_val = float(self.r[idx])
            w_saddle_val = float(self.pmf_ref[idx])
            print(f"  Auto-detected saddle: image {idx}, r={r_saddle_val:.3f} nm, W={w_saddle_val:.2f} kJ/mol")
        else:
            # Find closest image to user-provided r
            r_saddle_val = float(r_saddle)
            idx = int(np.argmin(np.abs(self.r - r_saddle_val)))
            w_saddle_val = float(self.pmf_ref[idx])
            print(f"  User-provided saddle: r={r_saddle_val:.3f} nm -> image {idx}, W={w_saddle_val:.2f} kJ/mol")

        # Validate saddle is not at endpoints (poor sampling)
        if idx == 0:
            warnings.warn(
                f"Saddle at first image (r={r_saddle_val:.3f} nm). "
                "This suggests the barrier is not captured; extend path further into bulk.",
                RuntimeWarning
            )
        if idx == len(self.r) - 1:
            warnings.warn(
                f"Saddle at last image (r={r_saddle_val:.3f} nm). "
                "This suggests desorption barrier is not captured.",
                RuntimeWarning
            )

        return idx, r_saddle_val, w_saddle_val

    # ------------------------------------------------------------------
    def _split_path_at_saddle(self, saddle_idx):
        """
        Split path into bound region and bulk region at the saddle.

        With convention (bulk at start, surface at end, r decreasing):
          - Bound region: from saddle TO surface (indices saddle_idx to end)
          - Bulk region: from bulk TO saddle (indices 0 to saddle_idx)

        For integration, we need r INCREASING (from surface to saddle, or
        saddle to bulk). So we reverse the bound region.

        Parameters
        ----------
        saddle_idx : int
            Index of saddle point.

        Returns
        -------
        r_bound_inc : ndarray
            r in bound region, increasing from surface to saddle (nm)
        w_bound_inc : ndarray
            PMF in bound region, aligned with r_bound_inc (kJ/mol)
        r_bulk_inc : ndarray
            r in bulk region, increasing from saddle to bulk (nm)
        w_bulk_inc : ndarray
            PMF in bulk region, aligned with r_bulk_inc (kJ/mol)
        """
        # Bound region: from saddle to surface (end of path)
        # r is decreasing here (saddle → surface has smaller r)
        r_bound_dec = self.r[saddle_idx:]      # decreasing
        w_bound_dec = self.pmf_ref[saddle_idx:]

        # Bulk region: from bulk to saddle (start of path to saddle)
        # r is decreasing here too (bulk → saddle has smaller r)
        r_bulk_dec = self.r[:saddle_idx+1]     # decreasing
        w_bulk_dec = self.pmf_ref[:saddle_idx+1]

        # Reverse bound region so r increases from surface to saddle
        r_bound_inc = r_bound_dec[::-1]
        w_bound_inc = w_bound_dec[::-1]

        # Reverse bulk region so r increases from saddle to bulk
        r_bulk_inc = r_bulk_dec[::-1]
        w_bulk_inc = w_bulk_dec[::-1]

        print(f"  Path split at saddle (image {saddle_idx}):")
        print(f"    Bound region: {len(r_bound_inc)} images, r ∈ [{r_bound_inc.min():.3f}, {r_bound_inc.max():.3f}] nm")
        print(f"    Bulk region:  {len(r_bulk_inc)} images, r ∈ [{r_bulk_inc.min():.3f}, {r_bulk_inc.max():.3f}] nm")

        return r_bound_inc, w_bound_inc, r_bulk_inc, w_bulk_inc

    # ------------------------------------------------------------------
    def _log_integral(self, r, w, RT):
        """
        Compute ∫ exp(-w(r)/RT) dr using log-domain for numerical stability.

        This avoids underflow when w has large positive values (high barriers).

        Parameters
        ----------
        r : ndarray
            r-coordinates (nm), must be strictly increasing
        w : ndarray
            PMF values (kJ/mol)
        RT : float
            Thermal energy (kJ/mol)

        Returns
        -------
        integral : float
            ∫ exp(-w/RT) dr (nm)
        """
        if len(r) < 2:
            return 0.0

        # Ensure r is strictly increasing (remove duplicates)
        unique_mask = np.concatenate([[True], np.diff(r) > 1e-8])
        if not np.all(unique_mask):
            r = r[unique_mask]
            w = w[unique_mask]
            if len(r) < 2:
                return 0.0

        # Log-domain integration
        log_f = -w / RT
        max_log_f = np.max(log_f)

        # Scale to avoid underflow: f = exp(log_f - max_log_f)
        f_scaled = np.exp(log_f - max_log_f)

        # Trapezoidal rule
        dr = np.diff(r)
        integral_scaled = np.sum(0.5 * (f_scaled[:-1] + f_scaled[1:]) * dr)

        # Scale back
        integral = integral_scaled * np.exp(max_log_f)

        return float(integral)

    # ------------------------------------------------------------------
    def compute(self, r_saddle=None, n_bulk_points=5, verbose=True):
        """
        Compute ΔG° and Kd from the MFEP.

        Parameters
        ----------
        r_saddle : float or None
            r-coordinate (nm) of the dividing surface between bound and bulk.
            If None, the saddle point (maximum PMF) is used.
        n_bulk_points : int
            Number of images from the START of the path (bulk region)
            to average for zero reference. Default 5.
        verbose : bool
            Print progress information.

        Returns
        -------
        dict
            Contains all computed quantities (see keys below).
        """
        def _log(msg):
            if verbose:
                print(msg)

        _log("\n" + "="*56)
        _log("ClayKd: Computing binding free energy and Kd")
        _log("Convention: bulk at start (r large), surface at end (r small)")
        _log("="*56)

        # Physical constants
        RT = self._K_B * self.T
        _log(f"  Temperature: {self.T:.2f} K  (kT = {RT:.4f} kJ/mol)")
        _log(f"  c° standard: {self.c_std:.4f} nm⁻³  ({self.c_std/self._C_STD:.2f} × 1 M)")

        # Step 1: Zero reference to bulk
        _log("\n[Step 1] Zero referencing to bulk region...")
        self._zero_reference_to_bulk(n_bulk_points=n_bulk_points)

        # Step 2: Find saddle point
        _log("\n[Step 2] Locating saddle point (transition state)...")
        saddle_idx, r_saddle_val, w_saddle_val = self._find_saddle(r_saddle)

        # Step 3: Split path at saddle
        _log("\n[Step 3] Splitting path at saddle...")
        r_bound, w_bound, r_bulk, w_bulk = self._split_path_at_saddle(saddle_idx)

        # Step 4: Compute bound partition integral I_bound
        _log("\n[Step 4] Computing bound partition integral I_bound...")
        if len(r_bound) >= 2:
            I_bound = self._log_integral(r_bound, w_bound, RT)
            _log(f"  I_bound = {I_bound:.6f} nm")
        else:
            raise RuntimeError(
                f"Bound region has only {len(r_bound)} image(s). "
                "Need at least 2 images for integration. "
                "Check that saddle point is not at the path end."
            )

        # Step 5: Compute bulk partition integral I_bulk (optional, for reference)
        _log("\n[Step 5] Computing bulk partition integral I_bulk (reference)...")
        if len(r_bulk) >= 2:
            I_bulk = self._log_integral(r_bulk, w_bulk, RT)
            _log(f"  I_bulk = {I_bulk:.6f} nm")
        else:
            I_bulk = 0.0
            _log(f"  I_bulk = {I_bulk:.6f} nm (insufficient points)")

        # Step 6: Compute Kd and ΔG° using partition-function method
        _log("\n[Step 6] Computing Kd and ΔG° (partition-function method)...")
        if I_bound <= 0.0:
            raise RuntimeError(
                f"I_bound = {I_bound:.6f} nm ≤ 0. "
                "Check that PMF in bound region is negative (favorable)."
            )

        # Kd = 1 / (c° × I_bound)  [M]
        Kd_pf = 1.0 / (self.c_std * I_bound)

        # ΔG° = -RT ln(c° × I_bound)  [kJ/mol]
        dG_pf = -RT * np.log(self.c_std * I_bound)

        _log(f"  c° × I_bound = {self.c_std * I_bound:.6f}")
        _log(f"  ΔG°_bind = {dG_pf:+.2f} kJ/mol")
        _log(f"  Kd = {self._fmt_kd(Kd_pf)}")

        # Step 7: Compute endpoint estimate
        _log("\n[Step 7] Computing endpoint estimate...")
        w_bulk = self.pmf_ref[0]      # Should be ≈ 0 after referencing
        w_surface = self.pmf_ref[-1]  # Should be negative

        dG_ep = w_surface - w_bulk    # ΔG = W(surface) - W(bulk)
        Kd_ep = np.exp(dG_ep / RT)

        _log(f"  W(bulk)    = {w_bulk:+.2f} kJ/mol")
        _log(f"  W(surface) = {w_surface:+.2f} kJ/mol")
        _log(f"  ΔG°_ep     = {dG_ep:+.2f} kJ/mol")
        _log(f"  Kd_ep      = {self._fmt_kd(Kd_ep)}")

        # Step 8: Compute activation barriers
        _log("\n[Step 8] Computing activation barriers...")
        dG_fwd = w_saddle_val - self.pmf_ref[0]   # Bulk → TS
        dG_rev = w_saddle_val - self.pmf_ref[-1]  # Surface → TS

        _log(f"  Forward barrier (bulk → TS):  {dG_fwd:+.2f} kJ/mol")
        _log(f"  Reverse barrier (well → TS): {dG_rev:+.2f} kJ/mol")

        # Store all results
        self._results = {
            # Partition-function method
            'dG_pf_kJ': dG_pf,
            'Kd_pf_M': Kd_pf,
            'I_bound_nm': I_bound,
            'I_bulk_nm': I_bulk,

            # Endpoint method
            'dG_ep_kJ': dG_ep,
            'Kd_ep_M': Kd_ep,

            # Saddle / barrier info
            'r_saddle_nm': r_saddle_val,
            'w_saddle_kJ': w_saddle_val,
            'dG_fwd_kJ': dG_fwd,
            'dG_rev_kJ': dG_rev,

            # Path info
            'T_K': self.T,
            'c_std_nm3': self.c_std,
            'n_bound': len(r_bound),
            'n_bulk': len(r_bulk),
            'n_total': len(self.r),
            'w_bulk_kJ': w_bulk,
            'w_surface_kJ': w_surface,
            'bulk_shift_kJ': self.bulk_mean,
        }

        _log("\n" + "="*56)
        _log("Computation complete.")
        _log("="*56)

        return self._results

    # ------------------------------------------------------------------
    @staticmethod
    def _fmt_kd(Kd_M):
        """Format Kd in the most readable sub-unit."""
        if Kd_M >= 1e-3:
            return f"{Kd_M * 1e3:.2f} mM"
        elif Kd_M >= 1e-6:
            return f"{Kd_M * 1e6:.2f} μM"
        elif Kd_M >= 1e-9:
            return f"{Kd_M * 1e9:.2f} nM"
        elif Kd_M >= 1e-12:
            return f"{Kd_M * 1e12:.2f} pM"
        else:
            return f"{Kd_M:.3e} M"

    # ------------------------------------------------------------------
    def summary(self):
        """Print a formatted summary of all results."""
        if self._results is None:
            self.compute()

        r = self._results
        sep = "━" * 58

        print("\n" + sep)
        print("  📊 ClayKd — Binding Free Energy & Dissociation Constant")
        print(f"  Convention: bulk at start (r={self.r[0]:.2f} nm) → surface at end (r={self.r[-1]:.2f} nm)")
        print(sep)

        # Temperature and path info
        print(f"\n  ⚙️  Simulation conditions:")
        print(f"     Temperature        : {r['T_K']:.2f} K")
        print(f"     kT                 : {self._K_B * r['T_K']:.4f} kJ/mol")
        print(f"     c° (standard conc.) : {r['c_std_nm3']:.4f} nm⁻³  (1 M = {self._C_STD:.4f} nm⁻³)")
        print(f"     Path images        : {r['n_total']} total")
        print(f"       • Bound region   : {r['n_bound']} images (r ≤ {r['r_saddle_nm']:.3f} nm)")
        print(f"       • Bulk region    : {r['n_bulk']} images (r > {r['r_saddle_nm']:.3f} nm)")

        # Saddle / barrier
        print(f"\n  🏔️  Transition state (saddle):")
        print(f"     r_saddle           : {r['r_saddle_nm']:.3f} nm")
        print(f"     W_saddle           : {r['w_saddle_kJ']:+.2f} kJ/mol")
        print(f"     Forward barrier    : {r['dG_fwd_kJ']:+.2f} kJ/mol  (bulk → TS)")
        print(f"     Reverse barrier    : {r['dG_rev_kJ']:+.2f} kJ/mol  (well → TS)")

        # Partition-function method
        print(f"\n  📐 Partition-function method (Roux 2004):")
        print(f"     I_bound            : {r['I_bound_nm']:.6f} nm")
        print(f"     c° × I_bound       : {r['c_std_nm3'] * r['I_bound_nm']:.6f}")
        print(f"     ΔG°_bind           : {r['dG_pf_kJ']:+.2f} kJ/mol")
        print(f"     Kd                 : {self._fmt_kd(r['Kd_pf_M'])}  ({r['Kd_pf_M']:.3e} M)")

        # Endpoint method
        print(f"\n  📐 Endpoint method (W_surface − W_bulk):")
        print(f"     W_bulk (start)     : {r['w_bulk_kJ']:+.2f} kJ/mol")
        print(f"     W_surface (end)    : {r['w_surface_kJ']:+.2f} kJ/mol")
        print(f"     ΔG°_bind           : {r['dG_ep_kJ']:+.2f} kJ/mol")
        print(f"     Kd                 : {self._fmt_kd(r['Kd_ep_M'])}  ({r['Kd_ep_M']:.3e} M)")

        # Check consistency
        print(f"\n  ✅ Consistency check:")
        ratio = r['Kd_pf_M'] / r['Kd_ep_M'] if r['Kd_ep_M'] != 0 else float('inf')
        if 0.5 < ratio < 2.0:
            print(f"     PF / Endpoint ratio: {ratio:.2f}  →  Good agreement ✓")
        else:
            print(f"     PF / Endpoint ratio: {ratio:.2f}  →  Large discrepancy! Check integration.")

        print(sep + "\n")

    # ------------------------------------------------------------------
    def compare_with_experiment(self, Kd_exp_M, temperature=None, verbose=True):
        """
        Compare computed Kd with experimental value.

        Parameters
        ----------
        Kd_exp_M : float
            Experimental dissociation constant in Molar.
        temperature : float or None
            Temperature of experiment (K). If None, uses simulation T.
        verbose : bool
            Print comparison table.

        Returns
        -------
        dict
            Comparison metrics.
        """
        if self._results is None:
            self.compute()

        T_exp = temperature if temperature is not None else self.T
        RT_exp = self._K_B * T_exp

        # Convert experimental Kd to ΔG°
        dG_exp = -RT_exp * np.log(Kd_exp_M)

        # Differences
        dG_diff_pf = self._results['dG_pf_kJ'] - dG_exp
        dG_diff_ep = self._results['dG_ep_kJ'] - dG_exp

        # Ratios (always >= 1)
        ratio_pf = max(self._results['Kd_pf_M'], Kd_exp_M) / min(self._results['Kd_pf_M'], Kd_exp_M)
        ratio_ep = max(self._results['Kd_ep_M'], Kd_exp_M) / min(self._results['Kd_ep_M'], Kd_exp_M)

        results = {
            'Kd_exp_M': Kd_exp_M,
            'dG_exp_kJ': dG_exp,
            'T_exp_K': T_exp,
            'ratio_pf': ratio_pf,
            'ratio_ep': ratio_ep,
            'dG_diff_pf_kJ': dG_diff_pf,
            'dG_diff_ep_kJ': dG_diff_ep,
            'within_factor_10_pf': ratio_pf < 10.0,
            'within_factor_10_ep': ratio_ep < 10.0,
            'within_factor_2_pf': ratio_pf < 2.0,
            'within_factor_2_ep': ratio_ep < 2.0,
        }

        if verbose:
            print("\n" + "━" * 58)
            print("  🔬 Comparison with experimental data")
            print("━" * 58)
            print(f"\n  Experimental:")
            print(f"     Temperature     : {T_exp:.2f} K")
            print(f"     Kd              : {self._fmt_kd(Kd_exp_M)}")
            print(f"     ΔG°             : {dG_exp:+.2f} kJ/mol")
            print(f"\n  Partition-function method:")
            print(f"     ΔΔG°            : {dG_diff_pf:+.2f} kJ/mol")
            print(f"     Kd ratio        : {ratio_pf:.1f}x  ({'✓ within 10×' if ratio_pf < 10 else '✗ outside 10×'})")
            print(f"\n  Endpoint method:")
            print(f"     ΔΔG°            : {dG_diff_ep:+.2f} kJ/mol")
            print(f"     Kd ratio        : {ratio_ep:.1f}x  ({'✓ within 10×' if ratio_ep < 10 else '✗ outside 10×'})")
            print("━" * 58 + "\n")

        return results

    # ------------------------------------------------------------------
    def bootstrap_uncertainty(self, n_bootstrap=100, r_saddle=None, n_bulk_points=5, verbose=True):
        """
        Estimate uncertainty via bootstrap resampling of MFEP images.

        Parameters
        ----------
        n_bootstrap : int
            Number of bootstrap resamples.
        r_saddle : float or None
            Saddle position (if None, auto-detected per bootstrap sample).
        n_bulk_points : int
            Number of bulk points for zero reference.
        verbose : bool
            Print progress.

        Returns
        -------
        dict
            Mean and standard deviation for key quantities.
        """
        if verbose:
            print(f"\n  Bootstrapping with {n_bootstrap} resamples...")

        # Storage for bootstrap results
        boot_results = {
            'dG_pf_kJ': [],
            'Kd_pf_M': [],
            'I_bound_nm': [],
            'dG_ep_kJ': [],
            'Kd_ep_M': [],
            'r_saddle_nm': [],
            'dG_fwd_kJ': [],
            'dG_rev_kJ': [],
        }

        n_images = len(self.r_raw)

        for i in range(n_bootstrap):
            if verbose and (i+1) % 20 == 0:
                print(f"    Bootstrap {i+1}/{n_bootstrap}")

            # Resample images with replacement
            indices = np.random.choice(n_images, n_images, replace=True)

            # Create temporary arrays with resampled order
            r_boot = self.r_raw[indices]
            w_boot = self.w_raw[indices]

            # Sort by r to maintain path order? No - bootstrap resamples randomly.
            # Need to sort to restore physical order (bulk to surface)
            sort_idx = np.argsort(r_boot)[::-1]  # Descending: bulk (large r) to surface (small r)
            r_boot = r_boot[sort_idx]
            w_boot = w_boot[sort_idx]

            # Store original data temporarily
            r_orig, w_orig = self.r_raw, self.w_raw
            self.r_raw, self.w_raw = r_boot, w_boot

            # Recompute
            try:
                # Zero reference
                self._zero_reference_to_bulk(n_bulk_points=n_bulk_points)

                # Find saddle
                saddle_idx, r_saddle_val, w_saddle_val = self._find_saddle(r_saddle)

                # Split path
                r_bound, w_bound, r_bulk, w_bulk = self._split_path_at_saddle(saddle_idx)

                # Integrate
                RT = self._K_B * self.T
                if len(r_bound) >= 2:
                    I_bound = self._log_integral(r_bound, w_bound, RT)
                else:
                    I_bound = 0.0

                # Compute values
                if I_bound > 0:
                    Kd_pf = 1.0 / (self.c_std * I_bound)
                    dG_pf = -RT * np.log(self.c_std * I_bound)
                else:
                    Kd_pf = np.nan
                    dG_pf = np.nan

                dG_ep = self.pmf_ref[-1] - self.pmf_ref[0]
                Kd_ep = np.exp(dG_ep / RT)

                dG_fwd = w_saddle_val - self.pmf_ref[0]
                dG_rev = w_saddle_val - self.pmf_ref[-1]

                # Store
                boot_results['dG_pf_kJ'].append(dG_pf)
                boot_results['Kd_pf_M'].append(Kd_pf)
                boot_results['I_bound_nm'].append(I_bound)
                boot_results['dG_ep_kJ'].append(dG_ep)
                boot_results['Kd_ep_M'].append(Kd_ep)
                boot_results['r_saddle_nm'].append(r_saddle_val)
                boot_results['dG_fwd_kJ'].append(dG_fwd)
                boot_results['dG_rev_kJ'].append(dG_rev)

            except Exception as e:
                if verbose:
                    print(f"    Warning: bootstrap {i+1} failed: {e}")
                continue

            # Restore original data
            self.r_raw, self.w_raw = r_orig, w_orig

        # Recompute with original data to restore state
        self.compute(r_saddle=r_saddle, n_bulk_points=n_bulk_points, verbose=False)

        # Calculate statistics
        uncertainty = {}
        for key, values in boot_results.items():
            if values:
                uncertainty[f'{key}_mean'] = np.nanmean(values)
                uncertainty[f'{key}_std'] = np.nanstd(values)
                uncertainty[f'{key}_ci95_lower'] = np.nanpercentile(values, 2.5)
                uncertainty[f'{key}_ci95_upper'] = np.nanpercentile(values, 97.5)
            else:
                uncertainty[f'{key}_mean'] = np.nan
                uncertainty[f'{key}_std'] = np.nan

        if verbose:
            print(f"\n  Bootstrap uncertainty (n={n_bootstrap}):")
            print(f"    ΔG°_pf = {uncertainty['dG_pf_kJ_mean']:.2f} ± {uncertainty['dG_pf_kJ_std']:.2f} kJ/mol")
            print(f"    Kd_pf  = {self._fmt_kd(uncertainty['Kd_pf_M_mean'])} (±{uncertainty['Kd_pf_M_std']:.3e})")
            print(f"    r_saddle = {uncertainty['r_saddle_nm_mean']:.3f} ± {uncertainty['r_saddle_nm_std']:.3f} nm")

        return uncertainty


# --------------------------------------------------------------------------
# Quick smoke test
# --------------------------------------------------------------------------

def _smoke_test():
    """Run a self-contained smoke test with synthetic data."""
    print("\n" + "="*58)
    print("  ClayKd Smoke Test")
    print("="*58)

    # Create synthetic data following convention: bulk at start, surface at end
    r = np.linspace(3.0, 0.2, 50