#!/usr/bin/env python3
"""
ClayMeanForce.py
================
Umbrella Integration (UI) PMF from mean forces recorded in pullf*.xvg files.

Three independent 1D PMF estimators — pure Python, no PLUMED, no WHAM:

  RFD   Reverse Finite Differences
        Trapezoid integration of mean forces at window centres.
        Fastest; no hyperparameters. Equivalent to PMFLib's RFD integrator.

  RBF   Radial Basis Function smoothing of mean forces → integration.
        Reduces stochastic noise before integration via
        scipy.interpolate.RBFInterpolator.

  GPR   Gaussian Process Regression on mean forces → integration.
        Clay-aware composite kernel (short length scale near interface,
        long length scale in bulk). Gives posterior uncertainty bands for
        the PMF. Requires scikit-learn.

All three produce W(r) in kJ/mol, directly comparable to
ClayPMF3D.pmf_r (WHAM 1D marginal). The comparison is a publishable
cross-validation: if the simulation is well-converged, all methods
should agree within ~1–2 kJ/mol in the adsorption well.

Sign convention
---------------
GROMACS pullf*.xvg records the force that the harmonic spring exerts on
the pulled molecule (the restraint force):

    f_pull(t) = −k · (ξ(t) − ξ₀)

where ξ is the pull coordinate (distance from clay surface, nm) and ξ₀ is
the window reference position.

The umbrella integration mean force equals the PMF gradient at ξ₀:

    dW/dξ|_{ξ₀} = k · (ξ₀ − ⟨ξ⟩) = ⟨f_pull⟩     [kJ/(mol·nm)]

Physical check (diffuse layer, ξ₀ = 1.0 nm, ξ_ads = 0.5 nm):
  − The drug is attracted toward the surface → ⟨ξ⟩ < ξ₀
  − f_pull = −k(ξ − ξ₀) > 0 (pushes drug away from surface to balance)
  − mean(f_pull) = k(ξ₀ − ⟨ξ⟩) > 0
  − dW/dξ > 0 in diffuse layer (W increases from well toward bulk) ✓

Clay-system specifics
---------------------
The clay interface creates two regions with very different force statistics:

  Interface  (r ≤ r_stern ≈ z_clay_surface + 0.5 nm):
    steep PMF gradient, high force variance, short correlation length

  Bulk  (r ≥ r_bulk ≈ z_clay_surface + 2.0 nm):
    near-zero mean force, small variance, long correlation length

The GPR kernel accounts for this: it is a SUM of a short-length-scale RBF
(captures the interface region) and a long-length-scale RBF (captures bulk),
with hyperparameters optimised via log-marginal-likelihood.

GROMACS double-molecule setup
------------------------------
Each ClayPMF3D window contains TWO CIP molecules (columns f1, f2 in the
pullf file). Both are treated as independent sub-windows, doubling the
mean-force sample density along r. This is legitimate because each molecule
has its own independent spring; their forces are independent given the
window geometry.

Usage
-----
    from ClayMeanForce import ClayMeanForce

    mf = ClayMeanForce(pmf3d)
    mf.load_forces()         # reads pullf*.xvg (or falls back to pullx)
    mf.run_rfd()             # trapezoid integration
    mf.run_rbf()             # RBF-smoothed integration
    mf.run_gpr()             # GPR integration (requires sklearn)
    mf.reference_to_bulk()   # zero at bulk (consistent with ClayPMF3D)
    mf.print_summary()

    mf.save('meanforce.npz')
    mf2 = ClayMeanForce.load('meanforce.npz')   # reload without pmf3d

References
----------
    Kumar et al. (1992) J. Comput. Chem. 13, 1011–1021
    Kästner & Thiel (2005) J. Chem. Phys. 123, 144104
    Kästner (2011) WIREs Comput. Mol. Sci. 1, 932–942
"""

import os
import io as _io
import warnings

import numpy as np
from scipy.interpolate import RBFInterpolator

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        RBF, Matern, WhiteKernel, ConstantKernel as C,
    )
    _sklearn_available = True
except ImportError:
    _sklearn_available = False


# ---------------------------------------------------------------------------
# Helper — cumulative trapezoid (avoids scipy version fragility)
# ---------------------------------------------------------------------------

def _cumtrapz(y, x):
    """Return cumulative trapezoid integral starting at 0."""
    dx = np.diff(x)
    increments = 0.5 * (y[:-1] + y[1:]) * dx
    return np.concatenate([[0.0], np.cumsum(increments)])


def _propagate_error_cumtrapz(errors, x):
    """
    Propagate per-point errors through a cumulative trapezoid integral.

    For each step i the trapezoid uses the average of two adjacent
    points, so both contribute:

        σ²(W_i) = σ²(W_{i-1}) + (0.5·Δx_{i-1})² · (σ²(y_i) + σ²(y_{i-1}))

    This is the correct formula for the trapezoid rule, as opposed to
    the simpler (Δx·σ_i)² which ignores the left-point contribution
    and double-counts when steps are unequal.

    Parameters
    ----------
    errors : array-like, shape (N,)  — per-point 1-σ uncertainties
    x      : array-like, shape (N,)  — coordinate values (must be sorted)

    Returns
    -------
    sigma_W : ndarray, shape (N,)  — cumulative σ of the integral, σ(0)=0
    """
    errors = np.asarray(errors, dtype=float)
    x      = np.asarray(x,      dtype=float)
    N  = len(x)
    dx = np.diff(x)                          # shape (N-1,)
    integral_var = np.zeros(N)
    for i in range(1, N):
        contrib = (0.5 * dx[i - 1])**2 * (errors[i]**2 + errors[i - 1]**2)
        integral_var[i] = integral_var[i - 1] + contrib
    return np.sqrt(integral_var)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ClayMeanForce:
    """
    Umbrella Integration PMF from pullf*.xvg force files.

    Parameters
    ----------
    pmf3d : ClayPMF3D or None
        Source of simulation parameters (k, T, equil_skip_ps,
        umbrella_dir, window_centers, z_data).
        Pass None when reloading from file via ClayMeanForce.load().
    pullf_prefix : str
        Filename prefix for pull-force files. Default ``'pullf'``.
    umbrella_dir : str or None
        Directory containing pullf files. If None, uses pmf3d.umbrella_dir.
    n_blocks : int
        Number of time blocks for block-averaging mean force uncertainty.
        Typical: 5–10. Block averaging accounts for MD time-correlation.
    bulk_fraction : float
        Fraction of the r range (high-r end) used as the bulk reference
        when calling reference_to_bulk(). Default 0.2.
    interface_boundary : float or None
        r value (nm) separating interface from bulk for the GPR kernel.
        If None, auto-detected as pmf3d.z_clay_surface + 0.5 nm (or
        25th percentile of r0 if z_clay_surface is not set).
    rbf_smoothing : float
        Smoothing parameter for RBFInterpolator (0 = exact interpolation).
        Larger values → smoother force profile. Default 0.1.
    gpr_n_restarts : int
        Number of optimiser restarts for GPR hyperparameter tuning. Default 5.
    """

    K_B = 8.314462618e-3  # kJ mol⁻¹ K⁻¹

    def __init__(
        self,
        pmf3d,
        pullf_prefix='pullf',
        umbrella_dir=None,
        n_blocks=5,
        bulk_fraction=0.2,
        interface_boundary=None,
        rbf_smoothing=0.1,
        gpr_n_restarts=5,
    ):
        self._pmf3d = pmf3d

        if pmf3d is not None:
            self.k             = float(pmf3d.k)
            self.T             = float(pmf3d.T)
            self.kT            = self.K_B * self.T
            self.equil_skip_ps = float(pmf3d.equil_skip_ps)
            self.umbrella_dir  = os.path.abspath(umbrella_dir or pmf3d.umbrella_dir)
        else:
            self.k = self.T = self.kT = self.equil_skip_ps = None
            self.umbrella_dir = os.path.abspath(umbrella_dir) if umbrella_dir else None

        self.pullf_prefix        = pullf_prefix
        self.n_blocks            = int(n_blocks)
        self.bulk_fraction       = float(bulk_fraction)
        self._interface_boundary = interface_boundary
        self.rbf_smoothing       = float(rbf_smoothing)
        self.gpr_n_restarts      = int(gpr_n_restarts)

        # --- set by load_forces() ---
        # r0:         (M,) window reference positions [nm], sorted ascending
        # r_eval:     (M,) actual mean position = where MF is evaluated
        # mean_force: (M,) dW/dr at r0  [kJ/(mol·nm)]
        # force_err:  (M,) block-average SEM of mean force
        # force_std:  (M,) block-average std dev
        # n_frames:   (M,) production frames contributing to each point
        self.r0          = None
        self.r_eval      = None
        self.mean_force  = None
        self.force_err   = None
        self.force_std   = None
        self.n_frames    = None

        # --- set by run_*() ---
        # r_grid:      (N,) uniform output r grid
        # pmf_rfd/rbf/gpr: (N,) PMF in kJ/mol
        # pmf_rfd_err / pmf_rbf_err / pmf_gpr_std: (N,) uncertainty
        self.r_grid      = None
        self.pmf_rfd     = None
        self.pmf_rfd_err = None
        self.pmf_rbf     = None
        self.pmf_rbf_err = None
        self.pmf_gpr     = None
        self.pmf_gpr_std = None

        self._bulk_shift        = 0.0
        self._gpr_kernel_params = None

    # -----------------------------------------------------------------------
    # File I/O helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _read_pullf(filepath):
        """
        Read a GROMACS pullf*.xvg file.

        Returns
        -------
        time : ndarray, shape (N,)  [ps]
        f1   : ndarray, shape (N,)  force on mol 1 [kJ/(mol·nm)]
        f2   : ndarray, shape (N,)  force on mol 2 [kJ/(mol·nm)]
        """
        good_lines = []
        with open(filepath, 'r') as fh:
            for line in fh:
                s = line.strip()
                if not s or s[0] in ('#', '@', '&'):
                    continue
                if len(s.split()) >= 3:
                    good_lines.append(s)
        if not good_lines:
            empty = np.empty(0, dtype=float)
            return empty, empty, empty
        data = np.loadtxt(_io.StringIO('\n'.join(good_lines)), ndmin=2)
        if data.shape[0] == 0 or data.shape[1] < 3:
            empty = np.empty(0, dtype=float)
            return empty, empty, empty
        return data[:, 0], data[:, 1], data[:, 2]

    # -----------------------------------------------------------------------
    # Block averaging
    # -----------------------------------------------------------------------

    @staticmethod
    def _block_stats(arr, n_blocks):
        """
        Block-average a 1D time series.

        Splits arr into n_blocks equal-length blocks, computes the mean of
        each block, then returns statistics over those block means. This
        accounts for temporal autocorrelation in MD data.

        Returns
        -------
        dict with keys: mean, sem, std, n (total frames).
        """
        n = len(arr)
        if n == 0:
            return {'mean': np.nan, 'sem': np.nan, 'std': np.nan, 'n': 0}
        if n_blocks <= 1 or n < 2 * n_blocks:
            mean = float(np.mean(arr))
            std  = float(np.std(arr, ddof=1)) if n > 1 else 0.0
            sem  = std / np.sqrt(n) if n > 1 else 0.0
            return {'mean': mean, 'sem': sem, 'std': std, 'n': n}
        bs = n // n_blocks
        block_means = np.array([
            np.mean(arr[b * bs: (b + 1) * bs])
            for b in range(n_blocks)
        ])
        mean = float(np.mean(block_means))
        std  = float(np.std(block_means, ddof=1))
        sem  = std / np.sqrt(n_blocks)
        return {'mean': mean, 'sem': sem, 'std': std, 'n': n}

    # -----------------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------------

    def load_forces(self, use_pullf_files=True):
        """
        Load mean forces from pullf*.xvg (preferred) or from pullx data.

        Each GROMACS window contains two CIP molecules (columns f1, f2).
        Both are treated as independent sub-windows, giving M = 2*n_loaded
        mean-force samples along the r axis.

        Sign convention (see module docstring):
            MF(r₀) = dW/dr = ⟨f_pull⟩ = −k·(⟨r⟩ − r₀) = k·(r₀ − ⟨r⟩)

        Parameters
        ----------
        use_pullf_files : bool
            True  → read pullf*.xvg directly (recommended).
            False → compute from pullx data: MF = k*(r0 − mean_z).

        Returns
        -------
        self
        """
        if self._pmf3d is None:
            raise RuntimeError(
                "pmf3d is required for load_forces(); "
                "use ClayMeanForce.load() to reload saved results."
            )
        pmf3d    = self._pmf3d
        n_loaded = len(pmf3d.window_centers)

        # Check pullf file availability once (before per-window loop)
        first_pullf = os.path.join(
            self.umbrella_dir, f'{self.pullf_prefix}1.xvg'
        )
        have_pullf = use_pullf_files and os.path.isfile(first_pullf)
        if use_pullf_files and not have_pullf:
            warnings.warn(
                f"pullf files not found in {self.umbrella_dir} "
                f"(tried {self.pullf_prefix}1.xvg); "
                f"falling back to pullx-based mean force."
            )

        # Access original pullx z-data (before trajectory resampling)
        z_source = (
            pmf3d._pullx_z_data
            if hasattr(pmf3d, '_pullx_z_data')
            else pmf3d.z_data
        )
        t_source = (
            pmf3d._pullx_t_data
            if hasattr(pmf3d, '_pullx_t_data')
            else getattr(pmf3d, 't_data', None)
        )

        r0_list   = []
        reval_list = []
        mf_list   = []
        err_list  = []
        std_list  = []
        nf_list   = []

        for i in range(n_loaded):
            c1, c2 = pmf3d.window_centers[i]   # nominal references [nm]
            z1, z2 = z_source[i]                # production z-arrays [nm]
            t1     = t_source[i] if t_source is not None else None  # production time [ps]

            if have_pullf:
                fpath = os.path.join(
                    self.umbrella_dir, f'{self.pullf_prefix}{i+1}.xvg'
                )
                if not os.path.isfile(fpath):
                    warnings.warn(
                        f"pullf file missing for window {i+1}: {fpath}; "
                        f"skipping this window."
                    )
                    continue
                time_f, f1, f2 = self._read_pullf(fpath)
                if len(time_f) == 0:
                    continue
                mask_f = time_f >= self.equil_skip_ps
                f1p, f2p = f1[mask_f], f2[mask_f]
                if len(f1p) == 0:
                    continue

                # dW/dr = ⟨f_pull⟩ (GROMACS sign: f_pull = −k(ξ−ξ₀) = +k(ξ₀−ξ))
                stats1 = self._block_stats(f1p, self.n_blocks)
                stats2 = self._block_stats(f2p, self.n_blocks)

                # r_eval: actual mean position during production (from pullx)
                # Use pullx because it has a cleaner time baseline than pullf
                r_eval_1 = float(np.mean(z1)) if len(z1) > 0 else c1
                r_eval_2 = float(np.mean(z2)) if len(z2) > 0 else c2

            else:
                # Fallback: MF = k*(r0 − mean_z)
                if len(z1) == 0:
                    continue
                stats1 = self._block_stats(self.k * (c1 - z1), self.n_blocks)
                stats2 = self._block_stats(self.k * (c2 - z2), self.n_blocks)
                r_eval_1 = float(np.mean(z1))
                r_eval_2 = float(np.mean(z2))

            for r0_val, r_eval_val, stats in [
                (c1, r_eval_1, stats1),
                (c2, r_eval_2, stats2),
            ]:
                if np.isnan(stats['mean']):
                    continue
                r0_list.append(r0_val)
                reval_list.append(r_eval_val)
                mf_list.append(stats['mean'])
                err_list.append(stats['sem'])
                std_list.append(stats['std'])
                nf_list.append(stats['n'])

        if len(r0_list) == 0:
            raise RuntimeError("No mean-force data could be loaded.")

        r0   = np.array(r0_list)
        reval = np.array(reval_list)
        mf   = np.array(mf_list)
        err  = np.array(err_list)
        std  = np.array(std_list)
        nf   = np.array(nf_list)

        # Sort by r_eval ascending (surface → bulk)
        idx = np.argsort(reval)
        self.r0         = r0[idx]
        self.r_eval     = reval[idx]
        self.mean_force = mf[idx]
        self.force_err  = err[idx]
        self.force_std  = std[idx]
        self.n_frames   = nf[idx]

        src = "pullf" if have_pullf else "pullx"
        print(
            f"ClayMeanForce: loaded {len(self.r_eval)} mean-force points "
            f"from {src}  "
            f"(r = {self.r_eval.min():.3f} … {self.r_eval.max():.3f} nm)"
        )
        return self

    # -----------------------------------------------------------------------
    # PMF methods
    # -----------------------------------------------------------------------

    def run_rfd(self, n_grid=300):
        """
        Reverse Finite Differences: trapezoid integration of mean forces.

        W(r) = ∫_{r_inner}^{r} dW/dr' dr'   (r_inner = innermost window)

        After reference_to_bulk(), W(bulk) = 0 and W(well) < 0 for
        favourable adsorption.

        Parameters
        ----------
        n_grid : int
            Number of points in the output uniform r grid.

        Returns
        -------
        self  (self.pmf_rfd and self.pmf_rfd_err are set)
        """
        self._check_loaded()
        r    = self.r_eval
        mf   = self.mean_force
        err  = self.force_err

        # Cumulative trapezoid from innermost (index 0) to bulk (last index)
        pmf_at_r = _cumtrapz(mf, r)

        # Error propagation: trapezoid rule — each step uses both endpoints
        err_at_r = _propagate_error_cumtrapz(err, r)

        r_grid = np.linspace(r.min(), r.max(), n_grid)
        self.r_grid      = r_grid
        self.pmf_rfd     = np.interp(r_grid, r, pmf_at_r)
        self.pmf_rfd_err = np.interp(r_grid, r, err_at_r)
        print(
            f"  RFD done.  W range: "
            f"{np.nanmin(self.pmf_rfd):.2f} … {np.nanmax(self.pmf_rfd):.2f} kJ/mol"
        )
        return self

    def run_rbf(self, n_grid=300):
        """
        RBF-smoothed mean forces → trapezoid integration.

        Uses scipy.interpolate.RBFInterpolator with thin-plate-spline kernel
        to smooth stochastic fluctuations in the force profile before
        integration.

        Parameters
        ----------
        n_grid : int

        Returns
        -------
        self  (self.pmf_rbf and self.pmf_rbf_err are set)
        """
        self._check_loaded()
        r  = self.r_eval
        mf = self.mean_force
        err = self.force_err

        r_grid = (self.r_grid if self.r_grid is not None
                  else np.linspace(r.min(), r.max(), n_grid))
        n_grid = len(r_grid)

        # Fit RBF to mean forces and to their errors (for uncertainty propagation)
        rbf_mf = RBFInterpolator(
            r[:, None], mf,
            smoothing=self.rbf_smoothing,
            kernel='thin_plate_spline',
        )
        rbf_err = RBFInterpolator(
            r[:, None], np.maximum(err, 1e-8),
            smoothing=self.rbf_smoothing,
            kernel='thin_plate_spline',
        )
        mf_smooth  = rbf_mf(r_grid[:, None])
        err_smooth = np.abs(rbf_err(r_grid[:, None]))

        # Integrate smoothed force
        pmf_rbf = _cumtrapz(mf_smooth, r_grid)

        # Propagate smoothed error: trapezoid rule — each step uses both endpoints
        err_rbf = _propagate_error_cumtrapz(err_smooth, r_grid)

        self.r_grid      = r_grid
        self.pmf_rbf     = pmf_rbf
        self.pmf_rbf_err = err_rbf
        print(
            f"  RBF done.  W range: "
            f"{np.nanmin(self.pmf_rbf):.2f} … {np.nanmax(self.pmf_rbf):.2f} kJ/mol"
        )
        return self

    def run_gpr(self, n_grid=300):
        """
        Gaussian Process Regression on mean forces → integration.

        Composite kernel
        ----------------
        kernel_short  = C · Matern(ν=2.5, l≈1.5·Δr)
            Short length-scale component.  Matern(2.5) is C²-continuous and
            captures the steep force gradients at the clay surface without
            over-smoothing the sharp repulsive spike.

        kernel_long   = C · RBF(l≈1.0 nm)
            Long length-scale component.  Smooth RBF represents the slow
            variation of the mean force through the bulk plateau.

        WhiteKernel   — learns any residual homoscedastic noise not already
            accounted for by the per-point SEM (passed via ``alpha``).

        Both length scales and amplitudes are optimised by maximising the
        log-marginal-likelihood (sklearn default) from multiple random
        restarts to avoid false optima.  Using a sum of two components
        avoids the single-kernel failure mode where LML collapses to one
        large length scale that smears the interface.

        Per-point heteroscedastic noise (block-SEM²) is passed as ``alpha``
        in normalised units (divide by y_var because normalize_y=True scales
        y but not alpha).

        Requires scikit-learn.

        Returns
        -------
        self  (self.pmf_gpr and self.pmf_gpr_std are set)
        """
        if not _sklearn_available:
            warnings.warn(
                "scikit-learn is not installed; run_gpr() skipped. "
                "Install with: conda install scikit-learn"
            )
            return self
        self._check_loaded()

        r  = self.r_eval
        mf = self.mean_force
        err = self.force_err

        r_grid = (self.r_grid if self.r_grid is not None
                  else np.linspace(r.min(), r.max(), n_grid))
        n_grid = len(r_grid)

        # ── Window spacing (positive-r side) ─────────────────────────────
        r_pos = np.sort(r[r > 1e-9])
        dr    = float(np.median(np.diff(r_pos))) if len(r_pos) > 1 else 0.1

        # ── Composite kernel ──────────────────────────────────────────────
        # Short component (interface): Matern(2.5) adapts to steep force
        # gradients at the clay surface.  Initialised at 1.5×Δr; LML is
        # free to tighten toward Δr/2 or loosen to 5×Δr.
        l_short = max(0.10, dr * 1.5)
        kernel_short = C(1.0, (1e-3, 1e3)) * Matern(
            length_scale=l_short,
            length_scale_bounds=(max(0.05, dr * 0.5), min(0.5, dr * 5.0)),
            nu=2.5,
        )
        # Long component (bulk): smooth RBF captures the slow plateau.
        # Initialised at 1.0 nm; LML free over [0.3, 5.0] nm.
        kernel_long = C(1.0, (1e-3, 1e3)) * RBF(
            length_scale=1.0,
            length_scale_bounds=(0.3, 5.0),
        )
        # Residual noise kernel: absorbs homoscedastic noise not captured
        # by the per-point SEM (alpha).  Together they give heteroscedastic
        # + homoscedastic noise — a more honest noise model.
        kernel = kernel_short + kernel_long + WhiteKernel(
            noise_level=0.1,
            noise_level_bounds=(1e-4, 10.0),
        )

        # ── Manual normalisation ─────────────────────────────────────────
        # sklearn's normalize_y=True divides y by y_std internally but does
        # NOT touch alpha — so alpha ends up in original units while the GP
        # works in normalised units.  The noise variance is then wrong by a
        # factor of y_std².  We normalise manually so that alpha, the kernel
        # amplitudes, and the y values are all in the same dimensionless space.
        force_std = max(float(np.std(mf)), 1e-6)
        mf_norm   = mf  / force_std
        err_norm  = np.where(
            np.isnan(err) | (err <= 0),
            0.1,                   # fallback: 10 % of unit variance
            err / force_std,
        )
        # alpha = per-point variance in normalised units (dimensionless)
        alpha = err_norm ** 2

        gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            n_restarts_optimizer=5,
            normalize_y=False,     # we normalised manually — no double-scaling
        )
        gpr.fit(r[:, None], mf_norm)
        self._gpr_kernel_params = str(gpr.kernel_)

        # Posterior mean + std in normalised space → back-transform
        mf_gpr_norm, mf_gpr_std_norm = gpr.predict(r_grid[:, None], return_std=True)
        mf_gpr     = mf_gpr_norm     * force_std
        mf_gpr_std = mf_gpr_std_norm * force_std

        # Integrate GPR posterior mean → PMF (same formula as RFD)
        pmf_gpr = _cumtrapz(mf_gpr, r_grid)

        # Propagate posterior std through the cumulative integral
        # trapezoid rule — each step uses both endpoints
        pmf_gpr_std = _propagate_error_cumtrapz(mf_gpr_std, r_grid)

        self.r_grid      = r_grid
        self.pmf_gpr     = pmf_gpr
        self.pmf_gpr_std = pmf_gpr_std
        print(
            f"  GPR done.  W range: "
            f"{np.nanmin(self.pmf_gpr):.2f} … {np.nanmax(self.pmf_gpr):.2f} kJ/mol"
        )
        print(f"  GPR kernel (optimised): {gpr.kernel_}")
        return self

    # -----------------------------------------------------------------------
    # Reference to bulk
    # -----------------------------------------------------------------------

    def reference_to_bulk(self, bulk_fraction=None):
        """
        Shift all PMF curves so that the bulk plateau equals 0 kJ/mol.

        The bulk is defined as the high-r fraction of the r_grid
        (bulk_fraction × total r range). Consistent with
        ClayPMF3D.reference_to_bulk().

        Parameters
        ----------
        bulk_fraction : float or None
            Overrides self.bulk_fraction if given.

        Returns
        -------
        self
        """
        frac = bulk_fraction if bulk_fraction is not None else self.bulk_fraction
        if self.r_grid is None:
            warnings.warn("No PMF computed yet; call run_rfd/rbf/gpr first.")
            return self

        r     = self.r_grid
        r_max = r.max()
        r_min = r.min()
        r_span = r_max - r_min

        # Detect symmetric two-surface system (r spans both signs).
        # In that case the bulk is the CENTRAL region (|r| small), not the
        # low-r or high-r tail.  For a single-surface system (r ≥ 0) the
        # bulk is the high-r tail.
        if r_min < -0.1 and r_max > 0.1:
            # Symmetric: use the central frac of the total span around r=0
            r_bulk_thresh = frac * r_span / 2.0
            bulk_mask = np.abs(r) <= r_bulk_thresh
        else:
            # Single-surface: high-r tail is the bulk
            r_bulk_thresh = r_max - frac * r_span
            bulk_mask = r >= r_bulk_thresh

        if not bulk_mask.any():
            warnings.warn("bulk_fraction too small; no bulk points found.")
            return self

        for attr in ('pmf_rfd', 'pmf_rbf', 'pmf_gpr'):
            arr = getattr(self, attr)
            if arr is not None:
                shift = float(np.nanmean(arr[bulk_mask]))
                setattr(self, attr, arr - shift)
                if attr == 'pmf_rfd':
                    self._bulk_shift = shift

        return self

    # -----------------------------------------------------------------------
    # Accessors / comparisons
    # -----------------------------------------------------------------------

    def adsorption_energy(self, method='rfd', r_surface=None,
                          r_surface_pos=None, r_surface_neg=None):
        """
        Return the adsorption free energy ΔG_ads = min(W) − W(bulk).

        After reference_to_bulk(), W(bulk) = 0 so ΔG_ads = min(W).

        Parameters
        ----------
        method : str  'rfd', 'rbf', or 'gpr'
        r_surface : float or None
            Legacy: restrict search to r ≤ r_surface (positive side only).
        r_surface_pos : float or None
            Restrict search to r ≤ r_surface_pos (positive clay surface).
        r_surface_neg : float or None
            Restrict search to r ≥ r_surface_neg (negative clay surface).
            Combined with r_surface_pos to bracket the inter-surface region.

        Returns
        -------
        delta_G : float  [kJ/mol]
        r_min   : float  [nm]
        """
        pmf = getattr(self, f'pmf_{method}')
        if pmf is None:
            raise RuntimeError(f"Run run_{method}() first.")
        r = self.r_grid
        if r_surface_pos is not None or r_surface_neg is not None:
            mask = np.ones(len(r), dtype=bool)
            if r_surface_pos is not None:
                mask &= r <= r_surface_pos
            if r_surface_neg is not None:
                mask &= r >= r_surface_neg
            pmf = pmf[mask]
            r   = r[mask]
        elif r_surface is not None:
            mask = r <= r_surface
            pmf = pmf[mask]
            r   = r[mask]
        idx_min = np.nanargmin(pmf)
        return float(pmf[idx_min]), float(r[idx_min])

    def compare_with_wham(self, pmf3d=None, bulk_fraction=None):
        """
        Compute RMSE and max deviation between UI PMFs and the WHAM 1D marginal.

        Parameters
        ----------
        pmf3d : ClayPMF3D or None
            If None, uses self._pmf3d.

        Returns
        -------
        dict  with keys: 'rmse_rfd', 'rmse_rbf', 'rmse_gpr',
                         'max_rfd', 'max_rbf', 'max_gpr'
        """
        src = pmf3d or self._pmf3d
        if src is None or src.pmf_r is None:
            raise RuntimeError("Provide a ClayPMF3D with a computed pmf_r.")
        if self.r_grid is None:
            raise RuntimeError("Run at least one PMF method first.")

        wham_on_grid = np.interp(self.r_grid, src.r_centers, src.pmf_r)
        results = {}
        for tag in ('rfd', 'rbf', 'gpr'):
            pmf = getattr(self, f'pmf_{tag}')
            if pmf is not None:
                diff = pmf - wham_on_grid
                results[f'rmse_{tag}'] = float(np.sqrt(np.nanmean(diff**2)))
                results[f'max_{tag}']  = float(np.nanmax(np.abs(diff)))
        return results

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _check_loaded(self):
        if self.mean_force is None:
            raise RuntimeError("Call load_forces() first.")

    def _get_interface_boundaries(self, percentile=80):
        """Detect both clay-water interfaces independently from force data.

        Parameters
        ----------
        percentile : float, optional
            Percentile of |MF| used as the repulsive-wall threshold.
            Windows whose |mean_force| exceeds this percentile are
            considered to be in the repulsive wall region.  Default 80.
            Lower values (e.g. 60) include more windows as 'wall';
            higher values (e.g. 90) restrict to only the most repulsive.

        Returns
        -------
        (r_neg, r_pos) : (float or None, float or None)
            r_neg : negative-surface boundary (r < 0)
            r_pos : positive-surface boundary (r > 0)

        Detection
        ---------
        The clay surface is the onset of the strongly repulsive wall.  Near
        the positive surface, the spring force is strongly positive
        (k·(r₀ − ⟨r⟩) > 0 because the wall pushes the molecule inward).
        Near the negative surface, the force is strongly negative for the
        same reason.

        Threshold: ``percentile``-th percentile of |MF| across all windows
        separates the large repulsive-wall forces from moderate
        adsorption-well gradients.  The innermost repulsive-wall point on
        each side is returned — this is the outermost clay-surface contact.
        Each side is detected independently so asymmetric slabs are handled
        correctly.
        """
        if self.mean_force is None:
            return None, None

        r  = self.r_eval
        mf = self.mean_force

        # Threshold separating repulsive-wall forces from bulk/well forces
        mf_thresh = float(np.percentile(np.abs(mf), float(percentile)))

        # --- Positive surface ---
        # clay pushes molecule toward centre → MF > 0
        r_pos = None
        pos_wall = (r > 1e-9) & (mf > mf_thresh)
        if pos_wall.sum() > 0:
            r_pos = float(np.min(r[pos_wall]))   # innermost (closest to bulk) repulsive point
        elif np.any(r > 1e-9):
            r_pos = float(np.percentile(r[r > 1e-9], 90))

        # --- Negative surface ---
        # clay pushes molecule toward centre → MF < 0
        r_neg = None
        neg_wall = (r < -1e-9) & (mf < -mf_thresh)
        if neg_wall.sum() > 0:
            r_neg = float(np.max(r[neg_wall]))   # innermost (least-negative) repulsive point
        elif np.any(r < -1e-9):
            r_neg = float(np.percentile(r[r < -1e-9], 10))

        return r_neg, r_pos

    def _get_interface_boundary(self):
        """Return r (nm) of the positive-side clay-water interface.

        Delegates to _get_interface_boundaries(); kept for backward compat.
        """
        if self._interface_boundary is not None:
            return float(self._interface_boundary)
        _, r_pos = self._get_interface_boundaries()
        if r_pos is not None:
            return r_pos
        return float(np.percentile(np.abs(self.r_eval), 90))

    # -----------------------------------------------------------------------
    # Save / Load
    # -----------------------------------------------------------------------

    def save(self, filepath):
        """
        Save all results to a .npz file (independent of pmf3d).

        Parameters
        ----------
        filepath : str  path to .npz file (extension added if absent)

        Returns
        -------
        filepath : str
        """
        if not filepath.endswith('.npz'):
            filepath += '.npz'
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        kw = dict(
            r0         = self.r0,
            r_eval     = self.r_eval,
            mean_force = self.mean_force,
            force_err  = self.force_err,
            force_std  = self.force_std,
            n_frames   = self.n_frames,
            k          = np.array(self.k),
            T          = np.array(self.T),
        )
        if self.r_grid is not None:
            kw['r_grid'] = self.r_grid
        for name in (
            'pmf_rfd', 'pmf_rfd_err',
            'pmf_rbf', 'pmf_rbf_err',
            'pmf_gpr', 'pmf_gpr_std',
        ):
            arr = getattr(self, name)
            if arr is not None:
                kw[name] = arr

        np.savez(filepath, **kw)
        print(f"ClayMeanForce: saved → {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath):
        """
        Load a previously saved ClayMeanForce result from .npz.

        Parameters
        ----------
        filepath : str

        Returns
        -------
        ClayMeanForce  (pmf3d = None; ready for plotting / comparison)
        """
        data = np.load(filepath)
        obj = cls(pmf3d=None)

        obj.k   = float(data['k'])
        obj.T   = float(data['T'])
        obj.kT  = obj.K_B * obj.T

        obj.r0         = data['r0']
        obj.r_eval     = data['r_eval']
        obj.mean_force = data['mean_force']
        obj.force_err  = data['force_err']
        obj.force_std  = data['force_std']
        obj.n_frames   = data['n_frames']

        if 'r_grid' in data:
            obj.r_grid = data['r_grid']
        for name in (
            'pmf_rfd', 'pmf_rfd_err',
            'pmf_rbf', 'pmf_rbf_err',
            'pmf_gpr', 'pmf_gpr_std',
        ):
            if name in data:
                setattr(obj, name, data[name])

        print(f"ClayMeanForce: loaded from {filepath}")
        return obj

    # -----------------------------------------------------------------------
    # Quick text summary
    # -----------------------------------------------------------------------

    def print_summary(self):
        """Print a concise summary of loaded forces and computed PMFs."""
        print("\n=== ClayMeanForce Summary ===")
        print(
            f"  k  = {self.k} kJ/(mol·nm²)   "
            f"T  = {self.T} K   "
            f"kT = {self.kT:.4f} kJ/mol"
        )
        if self.r_eval is not None:
            print(
                f"  {len(self.r_eval)} mean-force points: "
                f"r = {self.r_eval.min():.3f} … {self.r_eval.max():.3f} nm"
            )
            valid = ~np.isnan(self.mean_force)
            print(
                f"  Force range: "
                f"{self.mean_force[valid].min():.2f} … "
                f"{self.mean_force[valid].max():.2f} kJ/(mol·nm)"
            )
        for tag, label in (
            ('rfd', 'RFD (trapezoid)  '),
            ('rbf', 'RBF (smoothed)   '),
            ('gpr', 'GPR (Bayesian)   '),
        ):
            pmf = getattr(self, f'pmf_{tag}')
            if pmf is not None:
                print(
                    f"  PMF {label}: "
                    f"{np.nanmin(pmf):.2f} … {np.nanmax(pmf):.2f} kJ/mol"
                )
        if self._gpr_kernel_params:
            print(f"  GPR kernel: {self._gpr_kernel_params}")
        print()

    # -----------------------------------------------------------------------
    # Smoke-test / demo
    # -----------------------------------------------------------------------

    @classmethod
    def _smoke_test(cls):
        """
        Run a self-contained smoke test with synthetic data.
        No real trajectory or pullf files required.

        Synthetic PMF: double-Gaussian (adsorption well + Stern-layer hump).
        Analytic mean forces with Gaussian noise (~10 % of peak amplitude).
        All three methods are checked; each must reach RMSE < 2 kJ/mol.

        WHY NOT LJ: The LJ 1/r¹² core has ~10⁵ kJ/(mol·nm) force at r=0.3 nm.
        With 26 windows (Δr ≈ 0.1 nm), the trapezoid rule over-integrates the
        first step by thousands of kJ/mol, making RMSE >> 2 kJ/mol even with
        perfect forces.  That is a test-design artefact, not an algorithm bug:
        real clay US windows never sample inside the LJ core.

        Usage:
            python -c "from ClayMeanForce import ClayMeanForce; ClayMeanForce._smoke_test()"
        """
        import numpy.random as rng_mod

        rng = rng_mod.default_rng(42)

        # ── Synthetic PMF: double-Gaussian ────────────────────────────────
        # W(r) = −A₁·exp[−(r−r₁)²/(2σ₁²)]   ← adsorption well at 0.55 nm
        #      + A₂·exp[−(r−r₂)²/(2σ₂²)]    ← small Stern hump at 0.40 nm
        # Bulk (r → ∞) → 0.
        # Analytic dW/dr avoids np.gradient finite-difference errors.
        r_true = np.linspace(0.25, 3.0, 500)

        A1, r1, s1 = 10.0, 0.55, 0.18   # well:  depth 10 kJ/mol at 0.55 nm
        A2, r2, s2 =  3.0, 0.40, 0.08   # hump:  barrier 3 kJ/mol at 0.40 nm

        def _gauss(r, A, r0, s):
            return A * np.exp(-0.5 * ((r - r0) / s)**2)

        def _dgauss(r, A, r0, s):   # analytic dW/dr
            return A * (r - r0) / s**2 * np.exp(-0.5 * ((r - r0) / s)**2)

        W_true = -_gauss(r_true, A1, r1, s1) + _gauss(r_true, A2, r2, s2)
        W_true -= W_true[-1]   # set bulk = 0

        # Analytic mean force dW/dr (no finite-difference errors)
        mf_true = _dgauss(r_true, A1, r1, s1) + _dgauss(r_true, A2, r2, s2)

        # ── 26 umbrella windows, 0.30 → 2.80 nm ─────────────────────────
        r0_arr   = np.linspace(0.30, 2.80, 26)
        mf_samp  = np.interp(r0_arr, r_true, mf_true)
        peak_f   = float(np.max(np.abs(mf_true)))          # ≈ 56 kJ/(mol·nm)
        noise_sd = 0.10 * peak_f                            # 10 % noise
        noise    = rng.normal(0, noise_sd, len(r0_arr))

        # ── Build minimal ClayMeanForce object ────────────────────────────
        obj = cls(pmf3d=None)
        obj.k          = 1000.0
        obj.T          = 298.15
        obj.kT         = cls.K_B * 298.15
        obj.n_blocks   = 5
        obj.bulk_fraction      = 0.25
        obj._interface_boundary = 0.80   # nm (past the hump)
        obj.rbf_smoothing      = 0.30
        obj.gpr_n_restarts     = 3

        obj.r0         = r0_arr.copy()
        obj.r_eval     = r0_arr.copy()
        obj.mean_force = mf_samp + noise
        obj.force_err  = np.full(len(r0_arr), noise_sd)
        obj.force_std  = np.full(len(r0_arr), noise_sd)
        obj.n_frames   = np.full(len(r0_arr), 1000, dtype=int)

        print("=== ClayMeanForce smoke test ===")
        print(
            f"  True PMF: double-Gaussian  "
            f"well={-A1:.1f} kJ/mol @ {r1:.2f} nm, "
            f"hump=+{A2:.1f} kJ/mol @ {r2:.2f} nm"
        )
        print(f"  Peak force ≈ {peak_f:.1f} kJ/(mol·nm)  "
              f"noise ±{noise_sd:.1f} kJ/(mol·nm) (10 %)")

        obj.run_rfd(n_grid=200)
        obj.run_rbf(n_grid=200)
        obj.run_gpr(n_grid=200)
        obj.reference_to_bulk()
        obj.print_summary()

        # ── Accuracy check on the sampled r range ─────────────────────────
        W_ref = np.interp(obj.r_grid, r_true, W_true)
        all_pass = True
        for tag in ('rfd', 'rbf', 'gpr'):
            pmf = getattr(obj, f'pmf_{tag}')
            if pmf is None:
                print(f"  {tag.upper()} skipped (scikit-learn not installed)")
                continue
            rmse = float(np.sqrt(np.nanmean((pmf - W_ref)**2)))
            status = "PASS ✓" if rmse < 2.0 else "FAIL ✗"
            if rmse >= 2.0:
                all_pass = False
            print(f"  {tag.upper()} RMSE vs true PMF: {rmse:.3f} kJ/mol  [{status}]")

        print(f"\nSmoke test: {'ALL PASS ✓' if all_pass else 'SOME FAILURES ✗'}\n")
        return obj
