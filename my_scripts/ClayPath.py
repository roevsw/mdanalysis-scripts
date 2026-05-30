"""
ClayPath.py
===========
Minimum free energy path (MFEP) on a 2-D free energy surface using the
simplified string method (E et al., J. Chem. Phys. 126, 164103, 2007).

The string is a discretised curve of *n_images* images connecting two
specified endpoints.  At each iteration:

  1. Every *interior* image is moved one step down the normalised free
     energy gradient (gradient-descent step, normalised to unit length).
  2. The whole string is re-parameterised so images are equally spaced in
     normalised arc-length  (u = r/r_range,  v = θ/θ_range ∈ [0,1]).

Steps 1–2 are repeated until convergence (max interior-image displacement
< tol, measured in normalised units) or max_iter is reached.

Usage
-----
>>> from ClayPath import ClayPath
>>> cp = ClayPath(pmf2d_obj, n_images=60)
>>> cp.set_endpoints(r_start=1.5, theta_start=60.0, r_end=0.35, theta_end=15.0)
>>> cp.run_string_method(step_size=0.02, max_iter=3000, tol=1e-5)
>>> cp.print_summary()

or using automatic endpoint detection::

>>> cp.auto_endpoints(mode='adsorption')
>>> cp.run_string_method()

Requirements
------------
numpy, scipy (RectBivariateSpline, minimum_filter)
ClayPMF2D  (for type-hint and _smoke_test mock)
"""

import os
import sys
import warnings
import tempfile

import numpy as np

try:
    from scipy.interpolate import RectBivariateSpline
    from scipy.ndimage import minimum_filter
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

# ── ClayPMF2D import (type-hinting + smoke test) ────────────────────────────
try:
    from my_scripts.ClayPMF2D import ClayPMF2D as _ClayPMF2D
except ImportError:
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from ClayPMF2D import ClayPMF2D as _ClayPMF2D
    except ImportError:
        _ClayPMF2D = None


# ============================================================================
# ClayPath
# ============================================================================

class ClayPath:
    """
    String-method MFEP on a 2-D W(r, θ) free energy surface.

    Parameters
    ----------
    pmf2d : ClayPMF2D
        A ``ClayPMF2D`` instance that has been through ``run_wham_2d()``.
        Must have ``pmf_2d``, ``r_centers``, and ``theta_centers`` set.
    n_images : int
        Number of string images (including the two fixed endpoints).
        Default 60.
    nan_penalty : float
        kJ/mol value added to the local maximum for filling NaN (unsampled)
        bins before building the spline interpolator, so the string
        naturally avoids poorly-sampled regions.  Default 50.0.

    Attributes (set after run_string_method)
    ----------------------------------------
    r_path : ndarray  (n_images,) nm
    theta_path : ndarray  (n_images,) degrees
    pmf_path : ndarray  (n_images,) kJ/mol
    arc_length : ndarray  (n_images,) normalised cumulative arc-length
    converged : bool
    n_iter : int
    final_disp : float   max image displacement on last iteration (normalised)
    """

    K_B = 8.314462618e-3  # kJ mol⁻¹ K⁻¹

    def __init__(self, pmf2d, n_images=60, nan_penalty=50.0):
        if not _SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is required for ClayPath.  "
                "Install it with:  pip install scipy"
            )
        if pmf2d.pmf_2d is None:
            raise RuntimeError(
                "pmf2d.pmf_2d is None — call run_wham_2d() on the "
                "ClayPMF2D object first."
            )

        self.pmf2d       = pmf2d
        self.n_images    = int(n_images)
        self.nan_penalty = float(nan_penalty)

        # Axis arrays
        self.r_centers     = np.asarray(pmf2d.r_centers,     dtype=float)
        self.theta_centers = np.asarray(pmf2d.theta_centers, dtype=float)

        # Normalisation bounds (used to map both axes to [0, 1])
        self._r_lo    = float(self.r_centers[0])
        self._r_hi    = float(self.r_centers[-1])
        self._t_lo    = float(self.theta_centers[0])
        self._t_hi    = float(self.theta_centers[-1])
        self._r_range = max(self._r_hi - self._r_lo, 1e-15)
        self._t_range = max(self._t_hi - self._t_lo, 1e-15)

        # Build spline interpolator
        self._spline = self._build_spline(pmf2d.pmf_2d)

        # Endpoint storage
        self.r_start     = None
        self.theta_start = None
        self.r_end       = None
        self.theta_end   = None

        # Results (set by run_string_method)
        self.r_path     = None     # (n_images,) nm
        self.theta_path = None     # (n_images,) degrees
        self.pmf_path   = None     # (n_images,) kJ/mol
        self.arc_length = None     # (n_images,) normalised cumulative
        self.converged  = False
        self.n_iter     = 0
        self.final_disp = np.nan

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _build_spline(self, pmf_2d):
        """Construct the 2-D spline; NaN cells filled with large penalty."""
        pmf = np.array(pmf_2d, dtype=float)
        nan_mask = ~np.isfinite(pmf)
        if nan_mask.any():
            fill = float(np.nanmax(pmf)) + self.nan_penalty
            pmf[nan_mask] = fill
        return RectBivariateSpline(
            self.r_centers, self.theta_centers, pmf, kx=3, ky=3
        )

    def _to_norm(self, r, theta):
        """Convert physical (r, θ) to normalised (u, v) ∈ [0, 1]."""
        u = (np.asarray(r, dtype=float) - self._r_lo) / self._r_range
        v = (np.asarray(theta, dtype=float) - self._t_lo) / self._t_range
        return u, v

    def _from_norm(self, u, v):
        """Convert normalised (u, v) back to physical (r, θ)."""
        r     = np.asarray(u, dtype=float) * self._r_range + self._r_lo
        theta = np.asarray(v, dtype=float) * self._t_range + self._t_lo
        return r, theta

    def _reparameterise(self, u, v):
        """
        Re-interpolate (u, v) so images are equally spaced in normalised
        arc-length.  Endpoints are preserved exactly.
        """
        du = np.diff(u)
        dv = np.diff(v)
        ds = np.sqrt(du ** 2 + dv ** 2)
        s  = np.concatenate([[0.0], np.cumsum(ds)])
        s_total = s[-1]
        if s_total < 1e-15:
            return u.copy(), v.copy()
        s_unif = np.linspace(0.0, s_total, len(u))
        u_new  = np.interp(s_unif, s, u)
        v_new  = np.interp(s_unif, s, v)
        # Lock endpoints exactly to avoid drift from np.interp
        u_new[0],  v_new[0]  = u[0],  v[0]
        u_new[-1], v_new[-1] = u[-1], v[-1]
        return u_new, v_new

    def _norm_arc_length(self, u, v):
        """Cumulative normalised arc-length array (first entry = 0)."""
        du = np.diff(u)
        dv = np.diff(v)
        ds = np.sqrt(du ** 2 + dv ** 2)
        return np.concatenate([[0.0], np.cumsum(ds)])

    # ------------------------------------------------------------------ #
    # Minimum detection                                                    #
    # ------------------------------------------------------------------ #

    def find_minima(self, n_candidates=10):
        """
        Locate local minima in the 2-D PMF.

        Uses a 3 × 3 minimum filter to identify grid points that are lower
        than all 8 neighbours.

        Parameters
        ----------
        n_candidates : int
            Maximum number of minima to return.  Default 10.

        Returns
        -------
        list of dict
            Each entry has keys ``'r'``, ``'theta'``, ``'pmf'``,
            ``'ir'`` (r-bin index), ``'it'`` (θ-bin index).
            Sorted by PMF value (deepest first).
        """
        pmf_raw  = self.pmf2d.pmf_2d.copy()
        pmf_fill = np.where(np.isfinite(pmf_raw),
                            pmf_raw,
                            float(np.nanmax(pmf_raw)) + 100.0)

        local_min = (pmf_fill == minimum_filter(pmf_fill, size=3,
                                                mode='nearest'))
        local_min &= np.isfinite(pmf_raw)

        idx = np.argwhere(local_min)
        if len(idx) == 0:
            return []

        vals  = pmf_raw[idx[:, 0], idx[:, 1]]
        order = np.argsort(vals)
        idx   = idx[order]
        vals  = vals[order]

        out = []
        for k in range(min(n_candidates, len(idx))):
            ir, it = int(idx[k, 0]), int(idx[k, 1])
            out.append({
                'r':     float(self.r_centers[ir]),
                'theta': float(self.theta_centers[it]),
                'pmf':   float(vals[k]),
                'ir':    ir,
                'it':    it,
            })
        return out

    # ------------------------------------------------------------------ #
    # Endpoint setup                                                       #
    # ------------------------------------------------------------------ #

    def set_endpoints(self, r_start, theta_start, r_end, theta_end):
        """
        Set the two fixed endpoints of the MFEP search and initialise the
        string as a straight line between them.

        Parameters are clamped to the PMF grid boundaries automatically.

        Parameters
        ----------
        r_start, theta_start : float
            Start point (nm, degrees).  Typically the bulk position.
        r_end, theta_end : float
            End point (nm, degrees).  Typically the adsorption minimum.
        """
        r_start     = float(np.clip(r_start,     self._r_lo, self._r_hi))
        theta_start = float(np.clip(theta_start, self._t_lo, self._t_hi))
        r_end       = float(np.clip(r_end,       self._r_lo, self._r_hi))
        theta_end   = float(np.clip(theta_end,   self._t_lo, self._t_hi))

        self.r_start     = r_start
        self.theta_start = theta_start
        self.r_end       = r_end
        self.theta_end   = theta_end

        # Straight-line initial guess
        self.r_path     = np.linspace(r_start,     r_end,       self.n_images)
        self.theta_path = np.linspace(theta_start, theta_end,   self.n_images)
        self.pmf_path   = None
        self.arc_length = None
        self.converged  = False
        self.n_iter     = 0
        self.final_disp = np.nan

    def auto_endpoints(self, mode='adsorption'):
        """
        Automatically determine MFEP endpoints from the 2-D PMF topology.

        Parameters
        ----------
        mode : str
            ``'adsorption'`` (default) — start = outermost r-row minimum,
            end = global PMF minimum (deepest adsorption site).

            ``'minima'`` — start = global minimum, end = second deepest
            local minimum.  Useful for barrier-crossing paths between two
            adsorption states.

        Returns
        -------
        (r_start, theta_start, r_end, theta_end) : tuple of float

        Also calls ``set_endpoints()`` internally.
        """
        minima = self.find_minima(n_candidates=5)
        if len(minima) == 0:
            raise RuntimeError(
                "No local minima found in the 2-D PMF.  "
                "Check that run_wham_2d() produced a valid PMF."
            )

        if mode == 'minima':
            if len(minima) < 2:
                raise RuntimeError(
                    "Fewer than 2 local minima found — "
                    "use mode='adsorption' instead."
                )
            r_start     = minima[0]['r']
            theta_start = minima[0]['theta']
            r_end       = minima[1]['r']
            theta_end   = minima[1]['theta']

        else:  # 'adsorption'
            # End: deepest 2-D minimum (adsorbed state)
            r_end     = minima[0]['r']
            theta_end = minima[0]['theta']

            # Start: row with the smallest r value = bulk.
            # Bulk is at small r (r_centers[0] when ascending, r_centers[-1]
            # when descending).  Scan outward from the minimum-r end to find
            # the first row with at least one finite PMF value.
            _r_arr  = self.pmf2d.r_centers
            _n_rows = self.pmf2d.pmf_2d.shape[0]
            if float(_r_arr[0]) <= float(_r_arr[-1]):
                # ascending r: bulk = index 0, scan forward
                _scan = range(0, _n_rows)
            else:
                # descending r: bulk = index -1, scan backward
                _scan = range(-1, -_n_rows - 1, -1)

            bulk_row = None
            for _row_idx in _scan:
                _candidate = self.pmf2d.pmf_2d[_row_idx, :]
                if not np.all(np.isnan(_candidate)):
                    bulk_row = _candidate
                    _r_bulk  = float(self.pmf2d.r_centers[_row_idx])
                    break
            if bulk_row is None:
                raise RuntimeError(
                    "auto_endpoints: entire pmf_2d grid is NaN — "
                    "run run_wham_2d() before calling auto_endpoints()."
                )
            it_bulk     = int(np.nanargmin(bulk_row))
            r_start     = _r_bulk
            theta_start = float(self.theta_centers[it_bulk])

        self.set_endpoints(r_start, theta_start, r_end, theta_end)
        return r_start, theta_start, r_end, theta_end

    # ------------------------------------------------------------------ #
    # String method                                                        #
    # ------------------------------------------------------------------ #

    def run_string_method(self,
                          step_size=0.02,
                          max_iter=3000,
                          tol=1e-5,
                          verbose=True):
        """
        Run the simplified string method to find the MFEP.

        The string is evolved in normalised coordinate space
        ``(u, v) = (r/r_range, θ/θ_range) ∈ [0,1]²`` so that both axes
        contribute equally to arc-length.  At each iteration:

          1. Compute ∇W at every image via the cubic spline.
          2. Convert ∇W to normalised-space units and take a step of size
             *step_size* along the **unit** gradient vector (interior images
             only; endpoints are fixed).
          3. Re-parameterise the string to equal arc-length spacing.

        Parameters
        ----------
        step_size : float
            Step size in normalised units.  Both axes span [0, 1], so a
            step of 0.02 corresponds to 2 % of each axis's range per
            iteration.  Default 0.02.
        max_iter : int
            Maximum number of iterations.  Default 3000.
        tol : float
            Convergence threshold: maximum displacement of any interior
            image (in normalised units) on the last iteration.
            Default 1e-5.
        verbose : bool
            Print progress messages.  Default True.

        Returns
        -------
        r_path : ndarray  (n_images,) nm
        theta_path : ndarray  (n_images,) degrees

        Raises
        ------
        RuntimeError
            If ``set_endpoints()`` or ``auto_endpoints()`` has not been
            called first.
        """
        if self.r_path is None:
            raise RuntimeError(
                "Call set_endpoints() or auto_endpoints() before running "
                "the string method."
            )

        # Normalise initial path
        u, v = self._to_norm(self.r_path, self.theta_path)

        if verbose:
            print(
                f"String method: {self.n_images} images, "
                f"step_size={step_size}, tol={tol}, max_iter={max_iter}\n"
                f"  Start: r={self.r_start:.3f} nm, θ={self.theta_start:.1f}°\n"
                f"  End:   r={self.r_end:.3f} nm, θ={self.theta_end:.1f}°"
            )

        disp = np.nan
        for it in range(max_iter):
            r_now, theta_now = self._from_norm(u, v)

            # ── Gradient in physical space ──────────────────────────
            # dW/dr  [kJ mol⁻¹ nm⁻¹]   and   dW/dθ  [kJ mol⁻¹ deg⁻¹]
            g_r = self._spline.ev(r_now, theta_now, dx=1, dy=0)
            g_t = self._spline.ev(r_now, theta_now, dx=0, dy=1)

            # Convert to normalised-space gradient
            # u = r / r_range  →  dW/du = (dW/dr) * r_range
            # v = θ / θ_range  →  dW/dv = (dW/dθ) * θ_range
            g_u = g_r * self._r_range   # kJ/mol per normalised-r unit
            g_v = g_t * self._t_range   # kJ/mol per normalised-θ unit

            # Unit gradient in (u, v) space
            mag    = np.sqrt(g_u ** 2 + g_v ** 2)
            mag    = np.maximum(mag, 1e-10)
            g_u_hat = g_u / mag
            g_v_hat = g_v / mag

            # ── Gradient-descent step (interior images only) ─────────
            u_new = u.copy()
            v_new = v.copy()
            u_new[1:-1] = u[1:-1] - step_size * g_u_hat[1:-1]
            v_new[1:-1] = v[1:-1] - step_size * g_v_hat[1:-1]

            # Clamp interior images to valid grid range [0, 1]
            u_new[1:-1] = np.clip(u_new[1:-1], 0.0, 1.0)
            v_new[1:-1] = np.clip(v_new[1:-1], 0.0, 1.0)

            # ── Re-parameterise to equal arc-length ──────────────────
            u_new, v_new = self._reparameterise(u_new, v_new)

            # ── Convergence check ────────────────────────────────────
            disp = float(np.max(np.sqrt(
                (u_new[1:-1] - u[1:-1]) ** 2
                + (v_new[1:-1] - v[1:-1]) ** 2
            )))

            u, v = u_new, v_new

            if verbose and (it + 1) % 500 == 0:
                print(f"  iter {it + 1:5d}: max disp = {disp:.2e}")

            if disp < tol:
                self.converged  = True
                self.n_iter     = it + 1
                self.final_disp = disp
                if verbose:
                    print(
                        f"  Converged after {it + 1} iterations "
                        f"(max disp = {disp:.2e})"
                    )
                break
        else:
            self.n_iter     = max_iter
            self.final_disp = disp
            warnings.warn(
                f"String method did not converge after {max_iter} iterations "
                f"(final max disp = {disp:.2e}).  "
                "The path may not be the true MFEP — try increasing max_iter "
                "or reducing step_size.",
                RuntimeWarning,
                stacklevel=2,
            )
            if verbose:
                print(
                    f"  Did not converge after {max_iter} iterations "
                    f"(max disp = {disp:.2e})"
                )

        # ── Store results in physical coordinates ────────────────────
        self.r_path, self.theta_path = self._from_norm(u, v)
        self.pmf_path   = self._spline.ev(self.r_path, self.theta_path)
        self.arc_length = self._norm_arc_length(u, v)

        return self.r_path, self.theta_path

    # ------------------------------------------------------------------ #
    # Analysis                                                             #
    # ------------------------------------------------------------------ #

    def saddle_point(self):
        """
        Return the highest-energy point along the converged MFEP.

        Returns
        -------
        dict  with keys ``'r'`` (nm), ``'theta'`` (degrees),
              ``'pmf'`` (kJ/mol), ``'image_index'`` (int),
              ``'arc_length'`` (normalised).
        """
        self._require_path()
        idx = int(np.argmax(self.pmf_path))
        return {
            'r':           float(self.r_path[idx]),
            'theta':       float(self.theta_path[idx]),
            'pmf':         float(self.pmf_path[idx]),
            'image_index': idx,
            'arc_length':  float(self.arc_length[idx]),
        }

    def activation_energy(self):
        """
        Forward and reverse activation barriers along the MFEP.

        Returns
        -------
        dict
            ``'forward'``   : ΔG‡ from start to saddle (kJ/mol)
            ``'reverse'``   : ΔG‡ from end to saddle   (kJ/mol)
            ``'pmf_start'`` : PMF at start image        (kJ/mol)
            ``'pmf_end'``   : PMF at end image          (kJ/mol)
            ``'pmf_saddle'``: PMF at saddle             (kJ/mol)
        """
        self._require_path()
        pmf_start  = float(self.pmf_path[0])
        pmf_end    = float(self.pmf_path[-1])
        pmf_saddle = float(np.max(self.pmf_path))
        return {
            'forward':    pmf_saddle - pmf_start,
            'reverse':    pmf_saddle - pmf_end,
            'pmf_start':  pmf_start,
            'pmf_end':    pmf_end,
            'pmf_saddle': pmf_saddle,
        }

    def adsorption_energy(self):
        """
        Free energy change from start to end along the MFEP
        (PMF_end − PMF_start, in kJ/mol).  Negative = favourable adsorption.
        """
        self._require_path()
        return float(self.pmf_path[-1]) - float(self.pmf_path[0])

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, filepath):
        """
        Save the MFEP path and metadata to a compressed NumPy archive.

        Parameters
        ----------
        filepath : str
            Output ``.npz`` path (extension added if absent).
        """
        self._require_path()
        np.savez_compressed(
            filepath,
            r_path      = self.r_path,
            theta_path  = self.theta_path,
            pmf_path    = self.pmf_path,
            arc_length  = self.arc_length,
            r_centers   = self.r_centers,
            theta_centers = self.theta_centers,
            # 0-d metadata arrays
            meta_n_images    = np.array(self.n_images),
            meta_converged   = np.array(self.converged),
            meta_n_iter      = np.array(self.n_iter),
            meta_final_disp  = np.array(self.final_disp),
            meta_r_start     = np.array(
                self.r_start if self.r_start is not None else np.nan),
            meta_theta_start = np.array(
                self.theta_start if self.theta_start is not None else np.nan),
            meta_r_end       = np.array(
                self.r_end if self.r_end is not None else np.nan),
            meta_theta_end   = np.array(
                self.theta_end if self.theta_end is not None else np.nan),
        )
        print(f"MFEP saved: {filepath}")

    @classmethod
    def load(cls, filepath):
        """
        Load a saved ClayPath from a ``.npz`` file.

        The original ``pmf2d`` object is **not** restored (set to None).
        The spline is also not rebuilt; call ``_rebuild_spline_from_grid()``
        if you need to re-run the string method from a loaded file.

        Parameters
        ----------
        filepath : str

        Returns
        -------
        ClayPath instance
        """
        data = np.load(filepath, allow_pickle=True)

        def _s(key):
            v = data[key]
            return v.item() if v.ndim == 0 else v

        inst = object.__new__(cls)
        inst.n_images     = int(_s('meta_n_images'))
        inst.nan_penalty  = 50.0
        inst.converged    = bool(_s('meta_converged'))
        inst.n_iter       = int(_s('meta_n_iter'))
        inst.final_disp   = float(_s('meta_final_disp'))
        inst.r_start      = float(_s('meta_r_start'))
        inst.theta_start  = float(_s('meta_theta_start'))
        inst.r_end        = float(_s('meta_r_end'))
        inst.theta_end    = float(_s('meta_theta_end'))

        inst.r_path       = data['r_path'].copy()
        inst.theta_path   = data['theta_path'].copy()
        inst.pmf_path     = data['pmf_path'].copy()
        inst.arc_length   = data['arc_length'].copy()
        inst.r_centers    = data['r_centers'].copy()
        inst.theta_centers = data['theta_centers'].copy()

        inst._r_lo    = float(inst.r_centers[0])
        inst._r_hi    = float(inst.r_centers[-1])
        inst._t_lo    = float(inst.theta_centers[0])
        inst._t_hi    = float(inst.theta_centers[-1])
        inst._r_range = max(inst._r_hi - inst._r_lo, 1e-15)
        inst._t_range = max(inst._t_hi - inst._t_lo, 1e-15)

        # pmf2d / spline not available from file alone
        inst.pmf2d   = None
        inst._spline = None
        return inst

    # ------------------------------------------------------------------ #
    # Reporting                                                            #
    # ------------------------------------------------------------------ #

    def print_summary(self):
        """Print a concise summary of the MFEP results."""
        self._require_path()
        ae  = self.activation_energy()
        sad = self.saddle_point()
        ads = self.adsorption_energy()

        print("=" * 56)
        print("  ClayPath  –  MFEP Summary")
        print("=" * 56)
        print(f"  Images           : {self.n_images}")
        print(f"  Converged        : {self.converged}  "
              f"(iter={self.n_iter}, disp={self.final_disp:.2e})")
        print(f"  Start            : r={self.r_start:.3f} nm, "
              f"θ={self.theta_start:.1f}°, "
              f"W={ae['pmf_start']:+.2f} kJ/mol")
        print(f"  End              : r={self.r_end:.3f} nm, "
              f"θ={self.theta_end:.1f}°, "
              f"W={ae['pmf_end']:+.2f} kJ/mol")
        print(f"  Saddle point     : r={sad['r']:.3f} nm, "
              f"θ={sad['theta']:.1f}°, "
              f"W={ae['pmf_saddle']:+.2f} kJ/mol")
        print(f"  ΔG‡ forward      : {ae['forward']:+.2f} kJ/mol")
        print(f"  ΔG‡ reverse      : {ae['reverse']:+.2f} kJ/mol")
        print(f"  ΔG_ads (end-start): {ads:+.2f} kJ/mol")
        print("=" * 56)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _require_path(self):
        if self.pmf_path is None:
            raise RuntimeError(
                "No MFEP path available.  "
                "Run run_string_method() first."
            )

    # ------------------------------------------------------------------ #
    # Smoke test                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _smoke_test():
        """
        Self-contained smoke test on a synthetic 2-D double-well PMF.

        Potential: W(r, θ) = 4(r-1.0)² + 0.01(θ-45)² − 10·exp(−20(r-0.4)² − 0.003(θ-20)²)
        Grid: r ∈ [0, 1.8] nm (50 bins), θ ∈ [0°, 90°] (36 bins)

        Expected: string method should converge and produce a path from
        the bulk region to the adsorption minimum near (0.4 nm, 20°).
        """
        from types import SimpleNamespace

        n_r, n_theta = 50, 36
        r_c = np.linspace(0.05, 1.75, n_r)
        t_c = np.linspace(1.25, 88.75, n_theta)

        R, T = np.meshgrid(r_c, t_c, indexing='ij')
        pmf_2d = (
            4.0 * (R - 1.0) ** 2
            + 0.01 * (T - 45.0) ** 2
            - 10.0 * np.exp(-20.0 * (R - 0.4) ** 2 - 0.003 * (T - 20.0) ** 2)
        )

        mock = SimpleNamespace(
            pmf_2d        = pmf_2d,
            r_centers     = r_c,
            theta_centers = t_c,
        )

        cp = ClayPath(mock, n_images=40, nan_penalty=20.0)

        # Test auto-endpoints
        r_s, t_s, r_e, t_e = cp.auto_endpoints(mode='adsorption')
        assert r_s > r_e, "Start should be further from clay than end"

        # Run string method (relaxed tol for speed)
        r_path, theta_path = cp.run_string_method(
            step_size=0.03, max_iter=1000, tol=1e-4, verbose=False
        )

        assert r_path.shape == (40,),  "r_path shape mismatch"
        assert theta_path.shape == (40,), "theta_path shape mismatch"
        assert cp.pmf_path is not None, "pmf_path not set"
        assert cp.arc_length.shape == (40,), "arc_length shape mismatch"

        # Saddle / activation
        sad = cp.saddle_point()
        ae  = cp.activation_energy()
        assert ae['forward'] >= 0.0, "Forward barrier should be non-negative"

        # save / load round-trip
        with tempfile.TemporaryDirectory() as td:
            fpath = os.path.join(td, 'test_path')
            cp.save(fpath)
            cp2 = ClayPath.load(fpath + '.npz')
            assert np.allclose(cp.r_path,   cp2.r_path),   "r_path mismatch after load"
            assert np.allclose(cp.pmf_path, cp2.pmf_path), "pmf_path mismatch after load"

        print("Smoke test: PASS ✓")


# ============================================================================
# ClayKd — Dissociation constant from MFEP
# ============================================================================

class ClayKd:
    """
    Compute the standard binding free energy (ΔG°) and dissociation constant
    (Kd) for a molecule binding to a clay surface, from a 1-D MFEP produced
    by ClayPath.

    PATH CONVENTION (hard-coded):
        index 0   →  BULK   (large r, W ≈ 0 kJ/mol)
        index -1  →  SURFACE (small r, W negative)
        r[0] > r[-1]  (r decreases along the path)

    Two methods are implemented:

    **Endpoint method** (quick estimate)::

        ΔG°_ep  = W_surface - W_bulk          [kJ/mol]
        Kd_ep   = exp(ΔG°_ep / RT) × 1 M     [M]

    **Partition-function method** (Roux 2004, Biophys J 86:1087)::

        I_bound = ∫_{r_surface}^{r_saddle} exp(−W(r)/kT) dr   [nm]
        Kd_pf   = 1 / (c° × I_bound)                           [M]
        ΔG°_pf  = −RT ln(c° × I_bound)                         [kJ/mol]

    where r_saddle is the transition state (maximum PMF), and
    c° = 0.6022 nm⁻³ is the standard concentration (1 M in molecular units).

    Parameters
    ----------
    clay_path : ClayPath
        A ``ClayPath`` instance after ``run_string_method()`` has been called.
        Must satisfy r_path[0] > r_path[-1] (bulk at start, surface at end).
    T : float
        Temperature in Kelvin.  Default 298.15 K.
    c_standard : float or None
        Standard concentration in nm⁻³.  None → 0.6022 nm⁻³ (1 M).

    Usage
    -----
    >>> kd = ClayKd(clay_path)
    >>> results = kd.compute()
    >>> kd.summary()
    >>> kd.compare_with_experiment(Kd_exp_M=1e-5)
    >>> kd.bootstrap_uncertainty(n_bootstrap=200)
    """

    _K_B   = 8.314462618e-3   # kJ mol⁻¹ K⁻¹
    _C_STD = 0.6022           # nm⁻³  (1 M = N_A × 10⁻³ mol/nm³)

    # ------------------------------------------------------------------
    def __init__(self, clay_path, T=298.15, c_standard=None):
        if clay_path.pmf_path is None:
            raise RuntimeError(
                "clay_path.pmf_path is None — call run_string_method() first."
            )
        self.cp    = clay_path
        self.T     = float(T)
        self.c_std = float(c_standard) if c_standard is not None else self._C_STD

        self.r_raw = np.asarray(clay_path.r_path,   dtype=float)
        self.w_raw = np.asarray(clay_path.pmf_path, dtype=float)

        self._validate_convention()

        self._results  = None
        self.r         = None     # working copy (set during compute)
        self.pmf_ref   = None     # zero-referenced PMF
        self.bulk_mean = None     # mean W subtracted during referencing

    # ------------------------------------------------------------------
    def _validate_convention(self):
        """Warn if the path does not follow bulk→surface convention."""
        r = self.r_raw
        w = self.w_raw
        if len(r) >= 2 and r[0] <= r[-1]:
            warnings.warn(
                f"r_path: r[0]={r[0]:.3f} nm, r[-1]={r[-1]:.3f} nm. "
                "Convention expects r[0] > r[-1] (bulk at start, surface at end). "
                "Consider reversing: r = r[::-1], w = w[::-1]",
                RuntimeWarning, stacklevel=3,
            )
        if len(w) > 0 and w[-1] > 0:
            warnings.warn(
                f"PMF at last image W[-1]={w[-1]:.2f} kJ/mol is positive. "
                "Convention expects the surface end to be the adsorption well "
                "(negative). Check zero-referencing.",
                RuntimeWarning, stacklevel=3,
            )

    # ------------------------------------------------------------------
    def _zero_reference_to_bulk(self, n_bulk_points=5):
        """
        Zero-reference the PMF by subtracting the mean over the first
        ``n_bulk_points`` images (the bulk plateau).

        Sets ``self.pmf_ref``, ``self.r``, and ``self.bulk_mean``.
        """
        n_pts = max(1, min(int(n_bulk_points), len(self.w_raw)))
        self.bulk_mean = float(np.mean(self.w_raw[:n_pts]))
        self.pmf_ref   = self.w_raw - self.bulk_mean
        self.r         = self.r_raw.copy()

    # ------------------------------------------------------------------
    def _zero_reference_regression(self, n_bulk_points=10, verbose=False):
        """
        Zero-reference the PMF using a linear regression fit to the bulk
        plateau instead of a simple mean.

        Fits  W ≈ slope·r + intercept  over the first ``n_bulk_points``
        images (bulk region, large r) and evaluates the fitted line at the
        outermost bulk point (``r_raw[0]``) to obtain a noise-averaged
        estimate of the asymptotic PMF value.  That estimate is then
        subtracted as a constant shift — leaving the shape of the PMF
        intact while correcting for single-point noise at the boundary.

        Sets ``self.pmf_ref``, ``self.r``, ``self.bulk_mean``, and
        ``self._bulk_regression`` (diagnostic dict).

        Diagnostic keys
        ---------------
        slope_kJ_per_nm, intercept_kJ, r_ref_nm, w_ref_kJ,
        w_mean_kJ (raw mean for comparison), n_points, r_squared, rmse_kJ
        """
        n_pts = max(2, min(int(n_bulk_points), len(self.w_raw)))
        r_bulk = self.r_raw[:n_pts]
        w_bulk = self.w_raw[:n_pts]

        # Ordinary least-squares: [r, 1] · [slope, intercept]ᵀ = w
        A = np.column_stack([r_bulk, np.ones(n_pts)])
        coeff, _, _, _ = np.linalg.lstsq(A, w_bulk, rcond=None)
        slope = float(coeff[0])
        intercept = float(coeff[1])

        # Reference: evaluate fitted line at outermost bulk point
        r_ref = float(self.r_raw[0])
        w_ref = slope * r_ref + intercept

        # Diagnostics
        w_fit  = slope * r_bulk + intercept
        ss_res = float(np.sum((w_bulk - w_fit) ** 2))
        ss_tot = float(np.sum((w_bulk - float(np.mean(w_bulk))) ** 2))
        r_sq   = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 1.0
        rmse   = float(np.sqrt(ss_res / n_pts))

        self.bulk_mean = float(w_ref)
        self.pmf_ref   = self.w_raw - self.bulk_mean
        self.r         = self.r_raw.copy()
        self._bulk_regression = dict(
            slope_kJ_per_nm=slope,
            intercept_kJ=intercept,
            r_ref_nm=r_ref,
            w_ref_kJ=float(w_ref),
            w_mean_kJ=float(np.mean(w_bulk)),
            n_points=n_pts,
            r_squared=r_sq,
            rmse_kJ=rmse,
        )

        if verbose:
            _delta = float(w_ref) - float(np.mean(w_bulk))
            print(f"  [Bulk regression]  slope = {slope:+.4f} kJ/mol/nm, "
                  f"intercept = {intercept:+.3f} kJ/mol")
            print(f"  R\u00b2 = {r_sq:.4f},  RMSE = {rmse:.4f} kJ/mol,  "
                  f"n = {n_pts} points")
            print(f"  ref: W_fit(r={r_ref:.3f} nm) = {float(w_ref):+.3f} kJ/mol  "
                  f"(raw mean = {float(np.mean(w_bulk)):+.3f} kJ/mol, "
                  f"delta = {_delta:+.4f} kJ/mol)")

        return self._bulk_regression

    # ------------------------------------------------------------------
    def _find_saddle(self, r_saddle=None):
        """
        Locate the saddle (maximum PMF) along the path.

        Returns (saddle_idx, r_saddle_val, w_saddle_val).
        Warns if the saddle lands at a path endpoint (poor sampling).
        """
        if self.pmf_ref is None:
            raise RuntimeError("Call _zero_reference_to_bulk() first.")

        if r_saddle is None:
            idx = int(np.argmax(self.pmf_ref))
        else:
            idx = int(np.argmin(np.abs(self.r - float(r_saddle))))

        r_val = float(self.r[idx])
        w_val = float(self.pmf_ref[idx])

        if idx == 0:
            warnings.warn(
                f"Saddle is at the first image (r={r_val:.3f} nm). "
                "The barrier may not be sampled — extend the path further into bulk.",
                RuntimeWarning, stacklevel=3,
            )
        if idx == len(self.r) - 1:
            warnings.warn(
                f"Saddle is at the last image (r={r_val:.3f} nm). "
                "The desorption barrier may not be sampled.",
                RuntimeWarning, stacklevel=3,
            )
        return idx, r_val, w_val

    # ------------------------------------------------------------------
    def _split_path_at_saddle(self, saddle_idx):
        """
        Split the path at the saddle into bound and bulk segments.

        With path convention (bulk=start, surface=end, r decreasing):
            bound region = images[saddle_idx :]  (saddle → surface)
            bulk  region = images[: saddle_idx+1] (bulk → saddle)

        Each segment is reversed so that r increases before integration.

        Returns
        -------
        r_bound, w_bound, r_bulk, w_bulk : ndarray
            Each array has r strictly increasing (surface→saddle or
            saddle→bulk), ready for trapezoidal integration.
        """
        # Bound: saddle to surface (r decreasing → reverse for integration)
        r_bound = self.r[saddle_idx:][::-1]
        w_bound = self.pmf_ref[saddle_idx:][::-1]

        # Bulk: bulk to saddle (r decreasing → reverse for integration)
        r_bulk = self.r[:saddle_idx + 1][::-1]
        w_bulk = self.pmf_ref[:saddle_idx + 1][::-1]

        if len(r_bound) < 2:
            raise RuntimeError(
                f"Bound region contains only {len(r_bound)} image(s) "
                f"(saddle at image {saddle_idx}, path length {len(self.r)}). "
                "Verify that the MFEP has converged and the saddle lies "
                "well within the path."
            )
        return r_bound, w_bound, r_bulk, w_bulk

    # ------------------------------------------------------------------
    @staticmethod
    def _log_integral(r, w, RT):
        """
        Compute ∫ exp(−w(r)/RT) dr using log-domain arithmetic for
        numerical stability (avoids underflow at large positive barriers).

        Parameters
        ----------
        r : ndarray  — coordinates (nm), must be strictly increasing
        w : ndarray  — PMF values (kJ/mol)
        RT : float   — thermal energy (kJ/mol)

        Returns
        -------
        float  — ∫ exp(−w/RT) dr  (nm)
        """
        if len(r) < 2:
            return 0.0

        # Remove duplicate r values
        unique_mask = np.concatenate([[True], np.diff(r) > 1e-8])
        if not np.all(unique_mask):
            r = r[unique_mask]
            w = w[unique_mask]
            if len(r) < 2:
                return 0.0

        log_f     = -w / RT
        max_log_f = np.max(log_f)
        f_scaled  = np.exp(log_f - max_log_f)         # scale to O(1)
        dr        = np.diff(r)
        integral  = (np.sum(0.5 * (f_scaled[:-1] + f_scaled[1:]) * dr)
                     * np.exp(max_log_f))
        return float(integral)

    # ------------------------------------------------------------------
    def compute(self, r_saddle=None, n_bulk_points=5, verbose=True,
                use_regression=False):
        """
        Compute ΔG° and Kd from the MFEP.

        Parameters
        ----------
        r_saddle : float or None
            r-coordinate (nm) of the dividing surface.  None → auto-detect
            as the maximum-PMF image.
        n_bulk_points : int
            Number of leading images used for the zero reference.  Default 5.
            When ``use_regression=True`` this is also the regression window;
            using at least 8–10 points is recommended.
        verbose : bool
            Print step-by-step diagnostics.
        use_regression : bool
            If True, use a linear regression fit to the bulk plateau for the
            zero reference (``_zero_reference_regression``) instead of the
            simple mean (``_zero_reference_to_bulk``).
            The regression removes bias from a linearly drifting plateau.

        Returns
        -------
        dict with keys:
            dG_pf_kJ, Kd_pf_M, I_bound_nm, I_bulk_nm,
            dG_ep_kJ, Kd_ep_M,
            r_saddle_nm, w_saddle_kJ, dG_fwd_kJ, dG_rev_kJ,
            T_K, c_std_nm3, n_bound, n_bulk, n_total,
            w_bulk_kJ, w_surface_kJ, bulk_shift_kJ
        """
        def _log(msg):
            if verbose:
                print(msg)

        RT = self._K_B * self.T

        _log(f"\n{'='*56}")
        _log("  ClayKd: Computing binding free energy and Kd")
        _log(f"  T = {self.T:.2f} K   kT = {RT:.4f} kJ/mol   "
             f"c° = {self.c_std:.4f} nm⁻³")
        _log(f"{'='*56}")

        # Step 1 – zero-reference
        if use_regression:
            n_reg = max(n_bulk_points, 5)
            reg = self._zero_reference_regression(
                n_bulk_points=n_reg, verbose=False)
            _delta = reg['w_ref_kJ'] - reg['w_mean_kJ']
            _rsq = reg['r_squared']
            _log(f"  [1] Bulk ref (regression, n={n_reg}): "
                 f"slope = {reg['slope_kJ_per_nm']:+.4f} kJ/mol/nm, "
                 f"R\u00b2 = {_rsq:.4f}, "
                 f"shift = {self.bulk_mean:.3f} kJ/mol "
                 f"(\u0394 vs mean = {_delta:+.4f} kJ/mol)  "
                 f"(PMF range: {self.pmf_ref.min():.2f} \u2026 "
                 f"{self.pmf_ref.max():.2f} kJ/mol)")
        else:
            self._zero_reference_to_bulk(n_bulk_points=n_bulk_points)
            _log(f"  [1] Bulk ref: averaged {min(n_bulk_points, len(self.w_raw))} images, "
                 f"shift = {self.bulk_mean:.3f} kJ/mol  "
                 f"(PMF range: {self.pmf_ref.min():.2f} \u2026 "
                 f"{self.pmf_ref.max():.2f} kJ/mol)")

        # Step 2 – saddle
        saddle_idx, r_saddle_val, w_saddle_val = self._find_saddle(r_saddle)
        _log(f"  [2] Saddle: image {saddle_idx}, r = {r_saddle_val:.3f} nm, "
             f"W = {w_saddle_val:+.2f} kJ/mol")

        # Step 3 – split
        r_bound, w_bound, r_bulk, w_bulk_arr = self._split_path_at_saddle(saddle_idx)
        _log(f"  [3] Bound region: {len(r_bound)} images, "
             f"r ∈ [{r_bound.min():.3f}, {r_bound.max():.3f}] nm")
        _log(f"      Bulk  region: {len(r_bulk)} images, "
             f"r ∈ [{r_bulk.min():.3f}, {r_bulk.max():.3f}] nm")

        # Step 4 – integrals (log-domain)
        I_bound = self._log_integral(r_bound, w_bound, RT)
        I_bulk  = self._log_integral(r_bulk, w_bulk_arr, RT) if len(r_bulk) >= 2 else 0.0
        _log(f"  [4] I_bound = {I_bound:.6f} nm   I_bulk = {I_bulk:.6f} nm")

        if I_bound <= 0.0:
            raise RuntimeError(
                f"I_bound = {I_bound:.6f} nm ≤ 0. "
                "Verify that the PMF in the bound region is negative "
                "(favorable adsorption) after zero-referencing."
            )

        # Step 5 – Kd / ΔG° from partition function
        Kd_pf = 1.0 / (self.c_std * I_bound)
        dG_pf = -RT * np.log(self.c_std * I_bound)
        _log(f"  [5] PF:  ΔG° = {dG_pf:+.2f} kJ/mol   "
             f"Kd = {self._fmt_kd(Kd_pf)}")

        # Step 6 – endpoint estimate
        w_bulk_end    = float(self.pmf_ref[0])
        w_surface_end = float(self.pmf_ref[-1])
        dG_ep = w_surface_end - w_bulk_end
        Kd_ep = float(np.exp(dG_ep / RT))
        _log(f"  [6] EP:  W_bulk={w_bulk_end:+.2f}, W_surf={w_surface_end:+.2f}  "
             f"→ ΔG° = {dG_ep:+.2f} kJ/mol   Kd = {self._fmt_kd(Kd_ep)}")

        # Step 7 – activation barriers
        dG_fwd = w_saddle_val - w_bulk_end
        dG_rev = w_saddle_val - w_surface_end
        _log(f"  [7] Barriers: fwd = {dG_fwd:+.2f} kJ/mol   "
             f"rev = {dG_rev:+.2f} kJ/mol")

        self._results = dict(
            dG_pf_kJ=float(dG_pf),
            Kd_pf_M=float(Kd_pf),
            I_bound_nm=float(I_bound),
            I_bulk_nm=float(I_bulk),
            dG_ep_kJ=float(dG_ep),
            Kd_ep_M=float(Kd_ep),
            r_saddle_nm=float(r_saddle_val),
            w_saddle_kJ=float(w_saddle_val),
            dG_fwd_kJ=float(dG_fwd),
            dG_rev_kJ=float(dG_rev),
            T_K=self.T,
            c_std_nm3=self.c_std,
            n_bound=len(r_bound),
            n_bulk=len(r_bulk),
            n_total=len(self.r),
            w_bulk_kJ=w_bulk_end,
            w_surface_kJ=w_surface_end,
            bulk_shift_kJ=float(self.bulk_mean),
        )
        return self._results

    # ------------------------------------------------------------------
    @staticmethod
    def _fmt_kd(Kd_M):
        """Return Kd formatted in the most readable sub-unit."""
        if   Kd_M >= 1e-3:  return f"{Kd_M * 1e3:.2f} mM"
        elif Kd_M >= 1e-6:  return f"{Kd_M * 1e6:.2f} \u03bcM"
        elif Kd_M >= 1e-9:  return f"{Kd_M * 1e9:.2f} nM"
        elif Kd_M >= 1e-12: return f"{Kd_M * 1e12:.2f} pM"
        else:                return f"{Kd_M:.3e} M"

    # ------------------------------------------------------------------
    def summary(self):
        """Print a formatted summary of all computed quantities."""
        if self._results is None:
            self.compute()
        res = self._results
        RT  = self._K_B * res['T_K']
        sep = "\u2500" * 58

        print(sep)
        print("  ClayKd \u2014 Binding Free Energy & Dissociation Constant")
        print(f"  Convention: bulk start (r={self.r[0]:.2f} nm) "
              f"\u2192 surface end (r={self.r[-1]:.2f} nm)")
        print(sep)

        print(f"  Simulation conditions")
        print(f"    Temperature          : {res['T_K']:.2f} K  "
              f"(kT = {RT:.4f} kJ/mol)")
        print(f"    c\u00b0                   : {res['c_std_nm3']:.4f} nm\u207b\u00b3  "
              f"({res['c_std_nm3'] / self._C_STD:.2f} \u00d7 1 M)")
        print(f"    Path images          : {res['n_total']} total  "
              f"({res['n_bound']} bound / {res['n_bulk']} bulk)")
        print(f"    Bulk shift applied   : {res['bulk_shift_kJ']:+.3f} kJ/mol")

        print(sep)
        print(f"  Transition state (saddle)")
        print(f"    r_saddle             : {res['r_saddle_nm']:.3f} nm")
        print(f"    W_saddle             : {res['w_saddle_kJ']:+.2f} kJ/mol")
        print(f"    \u0394G\u2021 forward            : {res['dG_fwd_kJ']:+.2f} kJ/mol  (bulk \u2192 TS)")
        print(f"    \u0394G\u2021 reverse            : {res['dG_rev_kJ']:+.2f} kJ/mol  (well \u2192 TS)")

        print(sep)
        print("  Partition-function method  (Roux 2004, Biophys J 86:1087)")
        print(f"    I_bound              : {res['I_bound_nm']:.6f} nm")
        print(f"    c\u00b0 \u00d7 I_bound          : {res['c_std_nm3'] * res['I_bound_nm']:.6f}")
        print(f"    \u0394G\u00b0_bind             : {res['dG_pf_kJ']:+.2f} kJ/mol")
        print(f"    Kd                   : {self._fmt_kd(res['Kd_pf_M'])}"
              f"   ({res['Kd_pf_M']:.3e} M)")

        print(sep)
        print("  Endpoint method  (\u0394G = W_surface \u2212 W_bulk)")
        print(f"    W_bulk               : {res['w_bulk_kJ']:+.2f} kJ/mol")
        print(f"    W_surface            : {res['w_surface_kJ']:+.2f} kJ/mol")
        print(f"    \u0394G\u00b0_bind             : {res['dG_ep_kJ']:+.2f} kJ/mol")
        print(f"    Kd                   : {self._fmt_kd(res['Kd_ep_M'])}"
              f"   ({res['Kd_ep_M']:.3e} M)")

        print(sep)
        ratio = (res['Kd_pf_M'] / res['Kd_ep_M']
                 if res['Kd_ep_M'] != 0 else float('inf'))
        flag  = "\u2713 good" if 0.1 < ratio < 10.0 else "\u26a0 large discrepancy"
        print(f"  Consistency  PF/EP Kd ratio: {ratio:.2f}  \u2192  {flag}")
        print(sep)

    # ------------------------------------------------------------------
    def compare_with_experiment(self, Kd_exp_M, temperature=None,
                                verbose=True):
        """
        Compare computed Kd with an experimental value.

        Parameters
        ----------
        Kd_exp_M : float
            Experimental dissociation constant (Molar).
        temperature : float or None
            Temperature of the experiment (K).  None → uses simulation T.
        verbose : bool
            Print the comparison table.

        Returns
        -------
        dict
            Comparison metrics (ΔΔG°, Kd ratios, within-factor flags).
        """
        if self._results is None:
            self.compute(verbose=False)

        T_exp  = float(temperature) if temperature is not None else self.T
        RT_exp = self._K_B * T_exp
        dG_exp = -RT_exp * np.log(float(Kd_exp_M))

        res = self._results
        dG_diff_pf = res['dG_pf_kJ'] - dG_exp
        dG_diff_ep = res['dG_ep_kJ'] - dG_exp

        ratio_pf = (max(res['Kd_pf_M'], Kd_exp_M)
                    / min(res['Kd_pf_M'], Kd_exp_M))
        ratio_ep = (max(res['Kd_ep_M'], Kd_exp_M)
                    / min(res['Kd_ep_M'], Kd_exp_M))

        out = dict(
            Kd_exp_M=Kd_exp_M,
            dG_exp_kJ=dG_exp,
            T_exp_K=T_exp,
            dG_diff_pf_kJ=dG_diff_pf,
            dG_diff_ep_kJ=dG_diff_ep,
            ratio_pf=ratio_pf,
            ratio_ep=ratio_ep,
            within_factor_2_pf=ratio_pf  < 2.0,
            within_factor_10_pf=ratio_pf < 10.0,
            within_factor_2_ep=ratio_ep  < 2.0,
            within_factor_10_ep=ratio_ep < 10.0,
        )

        if verbose:
            sep = "\u2500" * 58
            print(sep)
            print("  Comparison with experimental data")
            print(sep)
            print(f"  Experiment")
            print(f"    T                    : {T_exp:.2f} K")
            print(f"    Kd_exp               : {self._fmt_kd(Kd_exp_M)}")
            print(f"    \u0394G\u00b0_exp             : {dG_exp:+.2f} kJ/mol")
            print(sep)
            for label, dG_sim, ratio in (
                ("Partition-function", res['dG_pf_kJ'], ratio_pf),
                ("Endpoint",          res['dG_ep_kJ'], ratio_ep),
            ):
                flag = ("\u2713 within 2\u00d7" if ratio < 2.0 else
                        "\u2713 within 10\u00d7" if ratio < 10.0 else
                        "\u26a0 outside 10\u00d7")
                print(f"  {label}")
                print(f"    \u0394\u0394G\u00b0                 : {dG_sim - dG_exp:+.2f} kJ/mol")
                print(f"    Kd ratio             : {ratio:.1f}\u00d7  {flag}")
            print(sep)

        return out

    # ------------------------------------------------------------------
    def bootstrap_uncertainty(self, n_bootstrap=200, r_saddle=None,
                              n_bulk_points=5, verbose=True):
        """
        Estimate Kd / ΔG° uncertainty by bootstrap resampling of MFEP images.

        Images are resampled with replacement and sorted by descending r
        (preserving bulk→surface ordering) before each calculation.
        This gives a measure of sensitivity to individual path images.

        Parameters
        ----------
        n_bootstrap : int
            Number of bootstrap resamples.  Default 200.
        r_saddle : float or None
            Saddle r (nm); None → auto-detect per resample.
        n_bulk_points : int
            Bulk averaging points for zero-reference.
        verbose : bool
            Print progress and summary.

        Returns
        -------
        dict
            Keys: ``{quantity}_mean``, ``{quantity}_std``,
            ``{quantity}_ci95_lower``, ``{quantity}_ci95_upper``
            for each of dG_pf_kJ, Kd_pf_M, I_bound_nm,
            dG_ep_kJ, Kd_ep_M, r_saddle_nm, dG_fwd_kJ, dG_rev_kJ.
        """
        if verbose:
            print(f"\n  Bootstrap uncertainty  (n = {n_bootstrap})...")

        keys = ('dG_pf_kJ', 'Kd_pf_M', 'I_bound_nm',
                'dG_ep_kJ', 'Kd_ep_M', 'r_saddle_nm',
                'dG_fwd_kJ', 'dG_rev_kJ')
        boot = {k: [] for k in keys}

        r_orig = self.r_raw.copy()
        w_orig = self.w_raw.copy()

        for i in range(n_bootstrap):
            if verbose and (i + 1) % 50 == 0:
                print(f"    {i+1}/{n_bootstrap}")

            # Resample images, then sort descending r (bulk→surface order)
            idx_boot  = np.random.choice(len(r_orig), len(r_orig), replace=True)
            r_b       = r_orig[idx_boot]
            w_b       = w_orig[idx_boot]
            sort_desc = np.argsort(r_b)[::-1]
            self.r_raw = r_b[sort_desc]
            self.w_raw = w_b[sort_desc]

            try:
                res_b = self.compute(r_saddle=r_saddle,
                                     n_bulk_points=n_bulk_points,
                                     verbose=False)
                for k in keys:
                    boot[k].append(res_b[k])
            except Exception as exc:
                if verbose:
                    print(f"    Warning: bootstrap {i+1} failed: {exc}")
            finally:
                # Always restore original data, even on failure
                self.r_raw = r_orig
                self.w_raw = w_orig

        # Restore clean state with original data
        self.compute(r_saddle=r_saddle, n_bulk_points=n_bulk_points,
                     verbose=False)

        # Summarise
        uncertainty = {}
        for k in keys:
            vals = [v for v in boot[k] if np.isfinite(v)]
            if vals:
                uncertainty[f'{k}_mean']       = float(np.mean(vals))
                uncertainty[f'{k}_std']        = float(np.std(vals))
                uncertainty[f'{k}_ci95_lower'] = float(np.percentile(vals,  2.5))
                uncertainty[f'{k}_ci95_upper'] = float(np.percentile(vals, 97.5))
                uncertainty[f'{k}_values']     = np.array(vals)
            else:
                for suf in ('_mean', '_std', '_ci95_lower', '_ci95_upper'):
                    uncertainty[f'{k}{suf}'] = float('nan')
                uncertainty[f'{k}_values'] = np.empty(0)

        if verbose:
            def _ci(k):
                return (f"{uncertainty[f'{k}_mean']:.2f} "
                        f"\u00b1 {uncertainty[f'{k}_std']:.2f}  "
                        f"[{uncertainty[f'{k}_ci95_lower']:.2f}, "
                        f"{uncertainty[f'{k}_ci95_upper']:.2f}]")
            sep = "\u2500" * 58
            print(sep)
            print(f"  Bootstrap results  (n = {n_bootstrap})")
            print(sep)
            print(f"  \u0394G\u00b0_pf  (kJ/mol) : {_ci('dG_pf_kJ')}")
            print(f"  Kd_pf    (M)     : "
                  f"{uncertainty['Kd_pf_M_mean']:.3e} "
                  f"\u00b1 {uncertainty['Kd_pf_M_std']:.3e}")
            print(f"  r_saddle (nm)    : {_ci('r_saddle_nm')}")
            print(f"  \u0394G\u2021_fwd (kJ/mol) : {_ci('dG_fwd_kJ')}")
            print(sep)

        self._bootstrap_results = uncertainty
        return uncertainty

    # ------------------------------------------------------------------
    def _check_pmf_convergence(self, n_bulk_points=10, tol_slope=0.5,
                               tol_rms=0.5, verbose=True):
        """
        Check whether the PMF has converged to a flat bulk plateau.

        Fits a linear trend to the first ``n_bulk_points`` images (the
        bulk region) and reports:

        1. **Slope** (kJ mol⁻¹ nm⁻¹) — a non-zero slope means the PMF
           is still drifting; the zero-reference is biased.
        2. **RMS deviation** (kJ/mol) — large RMS indicates a noisy
           bulk plateau and an uncertain zero-reference.

        Parameters
        ----------
        n_bulk_points : int
            Number of leading path images to analyse.  Default 10.
        tol_slope : float
            Maximum acceptable |slope| in kJ mol⁻¹ nm⁻¹.  Default 0.5.
        tol_rms : float
            Maximum acceptable RMS deviation in kJ/mol.  Default 0.5.
        verbose : bool
            Print the convergence report.

        Returns
        -------
        dict
            slope, rms, w_range, converged_slope, converged_rms, converged
        """
        n = max(2, min(int(n_bulk_points), len(self.w_raw)))
        r_b = self.r_raw[:n]
        w_b = self.w_raw[:n]

        # Linear trend: w ≈ a·r + b
        coeffs = np.polyfit(r_b, w_b, 1)
        slope  = float(coeffs[0])                              # kJ/mol per nm
        w_fit  = np.polyval(coeffs, r_b)
        rms    = float(np.sqrt(np.mean((w_b - w_fit) ** 2)))  # kJ/mol
        w_rng  = float(w_b.max() - w_b.min())                 # kJ/mol

        conv_slope = abs(slope) <= tol_slope
        conv_rms   = rms       <= tol_rms
        converged  = conv_slope and conv_rms

        if verbose:
            _ok   = "\u2713"
            _warn = "\u26a0"
            sep = "\u2500" * 58
            print(sep)
            print(f"  PMF Bulk Convergence Check  (first {n} images)")
            print(sep)
            print(f"  Slope    : {slope:+.3f} kJ mol\u207b\u00b9 nm\u207b\u00b9  "
                  f"  {_ok if conv_slope else _warn} "
                  f"(tol = \u00b1{tol_slope:.1f})")
            print(f"  RMS dev  : {rms:.3f} kJ/mol  "
                  f"  {_ok if conv_rms else _warn} "
                  f"(tol = {tol_rms:.1f})")
            print(f"  Range    : {w_rng:.3f} kJ/mol  "
                  f"over r \u2208 [{r_b.min():.3f}, {r_b.max():.3f}] nm")
            if not converged:
                print(f"  \u26a0 Bulk plateau has NOT converged.")
                if not conv_slope:
                    drift_bias = abs(slope) * float(r_b.max() - r_b.min())
                    print(f"    Slope drift  \u2192 zero-reference bias "
                          f"\u2248 {drift_bias:.2f} kJ/mol over bulk region.")
                    print(f"    Recommendation: extend sampling to larger r "
                          f"(currently r_max = {r_b.max():.3f} nm).")
                if not conv_rms:
                    print(f"    RMS too large \u2192 zero-reference uncertainty "
                          f"\u2248 \u00b1{rms:.2f} kJ/mol.")
            else:
                print(f"  \u2713 Bulk plateau appears converged.")
            print(sep)

        return dict(slope=slope, rms=rms, w_range=w_rng,
                    converged_slope=conv_slope, converged_rms=conv_rms,
                    converged=converged)

    # ------------------------------------------------------------------
    def compute_surface(self, binding_site_area_nm2=1.0, r_saddle=None,
                        n_bulk_points=5, verbose=True):
        """
        Surface-corrected Kd for planar mineral binding.

        **Why this matters:**
        The standard ``compute()`` (Roux 2004 1D formula) evaluates:

            Kd_1D = 1 / (c° × I_bound)    [I_bound in nm]

        This is correct for a 3D confined pocket (e.g. protein).  For a
        *planar* mineral surface the ligand binds over a lateral footprint
        A_site on the surface.  The physical binding *volume* is:

            V_bind = A_site × I_bound      [nm³ = nm² × nm]

        and the corrected dissociation constant is:

            Kd_surf = 1 / (c° × A_site × I_bound)              [M]
            ΔG°_surf = −RT ln(c° × A_site × I_bound)    [kJ/mol]

        Relative to the 1D formula, the area correction shifts ΔG° by:

            ΔΔG°_area = −RT ln(A_site)    [kJ/mol]

        The standard ``compute()`` implicitly sets A_site = 1 nm².

        Parameters
        ----------
        binding_site_area_nm2 : float
            Footprint of one binding site on the mineral surface (nm²).
            Typical values for clay minerals:

            ============  ====================
            Species       A_site (nm²)
            ============  ====================
            H₂O, Na⁺      0.05 – 0.15
            Acetate        0.2  – 0.4
            CIP, drug      0.5  – 2.0
            Polymer        5    – 20
            ============  ====================

            Use ``sensitivity_to_area()`` to assess the impact of this
            choice on Kd before committing to a single value.
        r_saddle : float or None
            Saddle r (nm); None → auto-detect.
        n_bulk_points : int
            Bulk images for zero-reference.
        verbose : bool

        Returns
        -------
        dict
            All keys from ``compute()``, plus:
            ``Kd_surf_M``, ``dG_surf_kJ``, ``V_bind_nm3``,
            ``binding_site_area_nm2``, ``dG_area_correction_kJ``.
        """
        # Ensure base compute is fresh
        res = self.compute(r_saddle=r_saddle, n_bulk_points=n_bulk_points,
                           verbose=False)
        RT  = self._K_B * self.T
        A   = float(binding_site_area_nm2)
        I   = res['I_bound_nm']

        V_bind  = A * I                               # nm³
        Kd_surf = 1.0 / (self.c_std * V_bind)        # M
        dG_surf = -RT * np.log(self.c_std * V_bind)  # kJ/mol
        dG_corr = -RT * np.log(A)                    # kJ/mol vs 1D formula

        surf = dict(res)
        surf.update(
            Kd_surf_M=float(Kd_surf),
            dG_surf_kJ=float(dG_surf),
            V_bind_nm3=float(V_bind),
            binding_site_area_nm2=A,
            dG_area_correction_kJ=float(dG_corr),
        )
        self._surf_results = surf

        if verbose:
            sep = "\u2500" * 58
            print(sep)
            print("  ClayKd \u2014 Surface-Corrected Formula")
            print(f"  (Planar surface: V_bind = A_site \u00d7 I_bound)")
            print(sep)
            print(f"  Binding site area A_site : {A:.3f} nm\u00b2")
            print(f"  I_bound                  : {I:.4f} nm")
            print(f"  V_bind = A \u00d7 I           : {V_bind:.4f} nm\u00b3")
            print(sep)
            print(f"  1D formula (A=1 nm\u00b2) :")
            print(f"    \u0394G\u00b0_1D  = {res['dG_pf_kJ']:+.2f} kJ/mol   "
                  f"Kd_1D  = {self._fmt_kd(res['Kd_pf_M'])}")
            print(f"  Surface formula :")
            print(f"    \u0394G\u00b0_surf = {dG_surf:+.2f} kJ/mol   "
                  f"Kd_surf = {self._fmt_kd(Kd_surf)}")
            print(f"  Area correction \u0394\u0394G\u00b0 = \u2212RT\u00b7ln({A:.3f} nm\u00b2) "
                  f"= {dG_corr:+.2f} kJ/mol")
            print(sep)

        return surf

    # ------------------------------------------------------------------
    def sensitivity_to_area(self, area_range=(0.05, 20.0), n_points=10,
                             r_saddle=None, n_bulk_points=5, verbose=True):
        """
        Scan Kd and ΔG° over a range of binding site areas.

        Since A_site is often uncertain (it depends on the adsorption
        geometry and is not directly computed from the 1D MFEP), this
        method quantifies how sensitive Kd is to the assumed value.
        The spread across the table represents the irreducible geometric
        uncertainty of the 1D→3D projection.

        Parameters
        ----------
        area_range : tuple
            (min, max) binding site area in nm².  Default (0.05, 20.0).
        n_points : int
            Number of area values to evaluate (log-spaced).
        r_saddle, n_bulk_points : see compute()
        verbose : bool
            Print a formatted sensitivity table.

        Returns
        -------
        dict
            ``areas_nm2``, ``Kd_M``, ``dG_kJ`` (ndarrays, length n_points).
        """
        if self._results is None:
            self.compute(r_saddle=r_saddle, n_bulk_points=n_bulk_points,
                         verbose=False)

        RT    = self._K_B * self.T
        I     = self._results['I_bound_nm']
        areas = np.logspace(np.log10(float(area_range[0])),
                            np.log10(float(area_range[1])),
                            int(n_points))
        Kds   = np.array([1.0 / (self.c_std * A * I) for A in areas])
        dGs   = np.array([-RT * np.log(self.c_std * A * I) for A in areas])

        # Labelled marker sites for context
        _markers = [
            (0.10, "H\u2082O / Na\u207a"),
            (0.30, "acetate"),
            (0.70, "small drug"),
            (1.50, "CIP-size"),
            (5.00, "polymer"),
        ]

        if verbose:
            sep   = "\u2500" * 58
            _rule = "\u2500"
            _ha   = "A_site (nm\u00b2)"
            _hdg  = "\u0394G\u00b0 (kJ/mol)"
            print(sep)
            print("  Kd Sensitivity to Binding Site Area")
            print(f"  (I_bound = {I:.4f} nm,  T = {self.T:.1f} K)")
            print(sep)
            print(f"  {_ha:>14}  {_hdg:>14}  {'Kd':>12}  Note")
            print(f"  {_rule*14}  {_rule*14}  {_rule*12}  {_rule*16}")
            for A, dG, Kd in zip(areas, dGs, Kds):
                note = next((lbl for a_m, lbl in _markers
                             if abs(A - a_m) / a_m < 0.2), "")
                print(f"  {A:14.3f}  {dG:+14.2f}  "
                      f"{self._fmt_kd(Kd):>12}  {note}")
            print(sep)
            # ΔG range from A_site uncertainty alone
            dG_span = abs(float(dGs[0]) - float(dGs[-1]))
            print(f"  ΔG\u00b0 span from A_site uncertainty: "
                  f"{dG_span:.1f} kJ/mol  "
                  f"({area_range[0]}\u2013{area_range[1]} nm\u00b2)")
            print(f"  Kd range: "
                  f"{self._fmt_kd(float(Kds[-1]))} \u2013 "
                  f"{self._fmt_kd(float(Kds[0]))}")
            print(sep)

        return dict(areas_nm2=areas, Kd_M=Kds, dG_kJ=dGs)


# ============================================================================
# ClayKdImproved — surface-specific corrections on top of ClayKd
# ============================================================================

class ClayKdImproved(ClayKd):
    """
    ClayKd with two additional mineral-surface-specific corrections:

    1. **Translational entropy correction** (``compute_entropic_correction``,
       ``compute_with_entropy_correction``)

       When a ligand moves from 3-D bulk to a 2-D surface, it loses
       translational freedom in the direction perpendicular to the surface.
       The standard Roux PMF already captures the *enthalpic* binding well, but
       the reference-state volume comparison is sometimes adjusted explicitly:

           ΔG_conf = RT × ln(V_free / V_bound)   [kJ/mol, always ≥ 0]

       where  V_free  = 1/c°  ≈ 1.66 nm³  (volume per molecule at 1 M)
              V_bound = A_site × δ           (bound-layer volume)

       **Caveat**: the PMF integral already encodes the change in accessible
       volume along r; this correction is only meaningful when the PMF was
       computed in a geometry that did *not* fully sample the lateral (2-D)
       degrees of freedom and a separate bookkeeping term is required.
       Use it as a diagnostic, not a blind correction.

    2. **Orientation-averaged binding** (``compute_orientation_averaged_kd``)

       Correct Kd from a full 2-D PMF W(r, θ) by integrating over both r and
       the tilt angle θ with the appropriate sin(θ) Jacobian factor:

           I_bound_2D = ∫_{r_surf}^{r*} ∫_{0}^{π/2} exp(−βW(r,θ)) sin(θ) dθ dr
           Kd         = 1 / (c° × A_site × I_bound_2D)
    """

    # ------------------------------------------------------------------
    # 0. Quadrature helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _simpson_integral(r, w, RT):
        """
        Compute ∫ exp(−w(r)/RT) dr using composite Simpson's rule.

        More accurate than the trapezoidal ``_log_integral`` for PMFs with
        sharp features near the surface (4th-order vs 2nd-order accurate).
        ``scipy.integrate.simpson`` is used internally — it handles both
        uniform and non-uniform grids and any number of points (odd or even)
        without special-casing.  Log-domain scaling is retained for
        numerical stability.

        Falls back to ``_log_integral`` (trapezoidal) for fewer than 3 points.

        Parameters
        ----------
        r  : ndarray  — coordinates (nm), must be strictly increasing
        w  : ndarray  — PMF values (kJ/mol)
        RT : float    — thermal energy (kJ/mol)

        Returns
        -------
        float  — ∫ exp(−w/RT) dr  (nm)
        """
        if len(r) < 2:
            return 0.0

        # --- deduplicate (same guard as _log_integral) ---
        unique_mask = np.concatenate([[True], np.diff(r) > 1e-8])
        if not np.all(unique_mask):
            r = r[unique_mask]
            w = w[unique_mask]
            if len(r) < 2:
                return 0.0

        # --- trapezoidal fallback for very short arrays ---
        if len(r) < 3:
            return ClayKd._log_integral(r, w, RT)

        # --- log-domain scaling to avoid underflow ---
        log_f     = -w / RT
        max_log_f = float(np.max(log_f))
        f_scaled  = np.exp(log_f - max_log_f)

        # scipy.integrate.simpson handles non-uniform grids and odd/even n
        try:
            from scipy.integrate import simpson as _scipy_simpson
        except ImportError:                         # scipy < 1.6 fallback
            from scipy.integrate import simps as _scipy_simpson  # type: ignore
        integral_scaled = float(_scipy_simpson(f_scaled, x=r))
        return float(integral_scaled * np.exp(max_log_f))

    # ------------------------------------------------------------------
    # 1. Translational entropy correction
    # ------------------------------------------------------------------

    def compute_entropic_correction(
            self,
            A_site_nm2: float = 1.0,
            delta_nm: float   = 0.4,
            temperature=None) -> float:
        """
        Estimate the translational-entropy free-energy penalty for 3D→2D binding.

        The bound state confines the ligand to a volume
        ``V_bound = A_site × δ`` [nm³], whereas in the bulk the available
        volume per molecule is ``V_free = 1/c°`` ≈ 1.66 nm³.  The
        configurational free-energy *penalty* is:

            ΔG_conf = RT × ln(V_free / V_bound)   ≥ 0  [kJ/mol]

        A positive value means binding is *less* favourable after the
        correction (entropy is lost upon binding).

        Parameters
        ----------
        A_site_nm2 : float
            Lateral footprint of the binding site (nm²).  Default 1.0 nm².
        delta_nm : float
            Thickness of the bound layer (nm).  Typical clay value ≈ 0.3–0.5 nm.
            Default 0.4 nm.
        temperature : float or None
            Temperature (K).  None → uses ``self.T``.

        Returns
        -------
        dG_conf_kJ : float
            Configurational entropy penalty (kJ/mol, ≥ 0 when V_bound < V_free).
        """
        T   = self.T if temperature is None else float(temperature)
        RT  = self._K_B * T          # kJ/mol

        V_free  = 1.0 / self.c_std   # nm³  per molecule at c°
        V_bound = A_site_nm2 * delta_nm        # nm³

        # POSITIVE when V_bound < V_free (entropy loss upon binding)
        dG_conf = RT * np.log(V_free / V_bound)
        return float(dG_conf)

    def compute_with_entropy_correction(
            self,
            A_site_nm2: float = 1.0,
            delta_nm: float   = 0.4,
            r_saddle=None,
            n_bulk_points: int = 5,
            verbose: bool = True) -> dict:
        """
        Compute Kd and ΔG° with the translational entropy correction applied.

        Steps
        -----
        1. Run the standard Roux partition-function calculation (``compute``).
        2. Evaluate the configurational entropy penalty ``ΔG_conf``.
        3. Apply the surface area factor (same as ``compute_surface``).
        4. Add the entropy penalty to ΔG (makes binding *less* favourable):

               ΔG_corr = ΔG_surf + ΔG_conf

        Parameters
        ----------
        A_site_nm2 : float
            Binding site area (nm²).
        delta_nm : float
            Bound-layer thickness (nm).
        r_saddle, n_bulk_points, verbose : forwarded to ``compute``.

        Returns
        -------
        dict
            All keys from ``compute_surface()`` plus:
            ``dG_conf_kJ``, ``delta_nm``,
            ``dG_corrected_kJ``, ``Kd_corrected_M``.
        """
        RT = self._K_B * self.T

        # 1. Surface-corrected result
        surf = self.compute_surface(
            binding_site_area_nm2=A_site_nm2,
            r_saddle=r_saddle,
            n_bulk_points=n_bulk_points,
            verbose=False)

        # 2. Entropy penalty (always ≥ 0, opposes binding)
        dG_conf = self.compute_entropic_correction(
            A_site_nm2=A_site_nm2,
            delta_nm=delta_nm)

        # 3. Corrected quantities
        dG_corr = surf['dG_surf_kJ'] + dG_conf    # penalty → less negative
        Kd_corr = float(np.exp(dG_corr / RT))     # Kd_M = exp(ΔG/RT)

        result = dict(surf)
        result.update(
            dG_conf_kJ     = dG_conf,
            delta_nm       = delta_nm,
            dG_corrected_kJ = dG_corr,
            Kd_corrected_M  = Kd_corr,
        )

        if verbose:
            _line = "\u2500" * 58
            _ok   = "\u2713"
            print(_line)
            print("  ClayKdImproved \u2014 Surface + Entropy Correction")
            print(_line)
            print(f"  Binding site area    : {A_site_nm2:.3f} nm\u00b2")
            print(f"  Bound-layer \u03b4        : {delta_nm:.2f} nm  "
                  f"(V_bound = {A_site_nm2*delta_nm:.3f} nm\u00b3)")
            V_free = 1.0 / self.c_std
            print(f"  V_free  (1/c\u00b0)       : {V_free:.3f} nm\u00b3")
            print(_line)
            print(f"  \u0394G\u00b0_surf (no ent.)  : {surf['dG_surf_kJ']:+.2f} kJ/mol"
                  f"   Kd = {self._fmt_kd(surf['Kd_surf_M'])}")
            print(f"  \u0394G_conf (ent. pen.) : {dG_conf:+.2f} kJ/mol")
            print(f"  \u0394G\u00b0_corrected       : {dG_corr:+.2f} kJ/mol"
                  f"   Kd = {self._fmt_kd(Kd_corr)}")
            print(_line)
            if dG_conf > 3.0:
                print("  \u26a0  Large entropy penalty: V_bound << V_free.")
                print(f"     Check A_site and \u03b4; consider increasing A_site.")

        return result

    # ------------------------------------------------------------------
    # 2. Orientation-averaged Kd from 2D PMF
    # ------------------------------------------------------------------

    def compute_orientation_averaged_kd(
            self,
            pmf_2d,
            r_centers,
            theta_centers,
            r_saddle_nm=None,
            A_site_nm2: float = 1.0,
            verbose: bool     = True) -> dict:
        """
        Compute Kd from a full 2-D PMF W(r, θ) with orientational averaging.

        The partition function integral over the bound well is:

            I_bound_2D = ∫_{r_surf}^{r*} ∫_{0}^{θ_max}
                             exp(−βW(r,θ)) sin(θ) dθ dr

        and the dissociation constant is:

            Kd = 1 / (c° × A_site × I_bound_2D)

        Parameters
        ----------
        pmf_2d : ndarray, shape (n_r, n_theta)
            2-D PMF W(r, θ) [kJ/mol].  Must be zero-referenced to bulk.
        r_centers : array-like, shape (n_r,)
            r coordinates [nm], **increasing** from surface to bulk.
            (Opposite to ClayPath convention; pass np.flip if needed.)
        theta_centers : array-like, shape (n_theta,)
            θ coordinates [degrees], typically 0–90°.
        r_saddle_nm : float or None
            r cutoff for the bound region.  None → use ``self._results``
            saddle from the last ``compute()`` call (if available), otherwise
            midpoint of r_centers.
        A_site_nm2 : float
            Lateral area of the binding site [nm²].  Default 1.0 nm².
        verbose : bool

        Returns
        -------
        dict with keys: ``I_bound_2D_nm_rad``, ``dG_2D_kJ``, ``Kd_2D_M``,
        ``r_saddle_nm``, ``n_bound_r``, ``n_theta``.
        """
        pmf_2d      = np.asarray(pmf_2d,      dtype=float)
        r_centers   = np.asarray(r_centers,   dtype=float)
        theta_deg   = np.asarray(theta_centers, dtype=float)

        if pmf_2d.shape != (len(r_centers), len(theta_deg)):
            raise ValueError(
                f"pmf_2d shape {pmf_2d.shape} must be "
                f"(n_r={len(r_centers)}, n_theta={len(theta_deg)})")

        RT         = self._K_B * self.T
        theta_rad  = np.deg2rad(theta_deg)

        # --- determine saddle cutoff ---
        if r_saddle_nm is None:
            if self._results is not None:
                r_saddle_nm = self._results['r_saddle_nm']
            else:
                r_saddle_nm = float(np.median(r_centers))

        # bound region: r < r_saddle (surface side)
        bound_mask = r_centers <= r_saddle_nm
        r_bound    = r_centers[bound_mask]
        pmf_bound  = pmf_2d[bound_mask]              # (n_r_bound, n_theta)

        if len(r_bound) < 2:
            raise ValueError(
                f"r_saddle_nm={r_saddle_nm:.3f} leaves fewer than 2 bound "
                f"r points. Adjust r_saddle_nm.")

        # --- theta integration with sin(θ) Jacobian ---
        # theta_integrand[i_r] = ∫ exp(-W(r_i,θ)) sin(θ) dθ
        boltz          = np.exp(-pmf_bound / RT)          # (n_r_bound, n_theta)
        sin_w          = np.sin(theta_rad)                 # (n_theta,)
        theta_integrand = np.trapz(boltz * sin_w[np.newaxis, :],
                                   theta_rad, axis=1)      # (n_r_bound,)

        # --- r integration ---
        I_bound_2D = float(np.trapz(theta_integrand, r_bound))  # nm·rad

        dG_2D = float(-RT * np.log(self.c_std * A_site_nm2 * I_bound_2D))
        Kd_2D = float(np.exp(dG_2D / RT))   # Kd_M = exp(ΔG/RT)

        result = dict(
            I_bound_2D_nm_rad = I_bound_2D,
            dG_2D_kJ          = dG_2D,
            Kd_2D_M           = Kd_2D,
            r_saddle_nm       = r_saddle_nm,
            n_bound_r         = int(bound_mask.sum()),
            n_theta           = len(theta_deg),
        )

        if verbose:
            _line = "\u2500" * 58
            print(_line)
            print("  ClayKdImproved \u2014 2D Orientation-Averaged Kd")
            print(_line)
            print(f"  r range (bound)   : [{r_bound[0]:.3f}, {r_bound[-1]:.3f}] nm"
                  f"  ({result['n_bound_r']} points)")
            print(f"  \u03b8 range           : [{theta_deg[0]:.1f}, {theta_deg[-1]:.1f}]\u00b0"
                  f"  ({result['n_theta']} points)")
            print(f"  r_saddle          : {r_saddle_nm:.3f} nm")
            print(f"  A_site            : {A_site_nm2:.3f} nm\u00b2")
            print(_line)
            print(f"  I_bound_2D        : {I_bound_2D:.6f} nm\u00b7rad")
            print(f"  \u0394G\u00b0_2D            : {dG_2D:+.2f} kJ/mol")
            print(f"  Kd_2D             : {self._fmt_kd(Kd_2D)}")
            print(_line)

        return result

    # ------------------------------------------------------------------
    # 3. Cation-mediated binding correction
    # ------------------------------------------------------------------

    #: Clay-site binding constants K_c (M⁻¹) for montmorillonite.
    #: Monovalent cations bind outer-sphere (weak); divalent inner-sphere (strong).
    #: Sources: Sposito 1984; Nir et al. 1994; Tournassat et al. 2009.
    _CLAY_K_DEFAULT = {
        'Na': 1.0,    # M⁻¹  outer-sphere, weakly hydrated
        'K':  3.0,    # M⁻¹  less hydrated than Na⁺, slightly stronger
        'Mg': 30.0,   # M⁻¹  divalent, inner-sphere possible on smectites
        'Ca': 50.0,   # M⁻¹  divalent, strongest among common cations
    }

    #: Default ΔΔG corrections (kJ/mol) for the drug–cation–clay bridge
    #: relative to bare-clay binding.  Negative → bridge *stabilises* binding.
    #: Divalent cations form stronger bridges (higher charge density).
    #: Typical fluoroquinolone–clay estimates (Aristilde & Sposito 2010;
    #: Zhao et al. 2015).
    _CLAY_DDG_DEFAULT = {
        'Na':  -3.0,   # kJ/mol  monovalent bridge
        'K':   -4.0,   # kJ/mol  monovalent bridge (CIP K-complexes slightly stronger)
        'Mg': -10.0,   # kJ/mol  divalent bridge
        'Ca':  -8.0,   # kJ/mol  divalent bridge (Ca less charge-dense than Mg)
    }

    #: Readable labels for verbose output.
    _CATION_LABELS = {
        'Na': 'Na\u207a',
        'K':  'K\u207a',
        'Mg': 'Mg\u00b2\u207a',
        'Ca': 'Ca\u00b2\u207a',
    }

    def compute_cation_binding_contribution(
            self,
            cation_concentrations_M: dict,
            K_clay: dict       = None,
            dG_bridge_kJ: dict = None,
            verbose: bool      = True) -> dict:
        """
        Apparent drug Kd corrected for competing cation–clay occupancy.

        **Physical model** (montmorillonite)
        -------------------------------------
        Clay interlayer/surface sites are occupied by Na⁺, K⁺, Mg²⁺ or Ca²⁺
        in a competitive Langmuir equilibrium::

            Drug + Cation-Clay ⇌ Drug–Cation–Clay

        Site occupancies (competitive Langmuir):

            θ₀  = 1 / (1 + Σ_c K_c [c])            bare clay
            θ_c = K_c [c] / (1 + Σ_c K_c [c])      cation c occupies site

        Each cation state contributes a bridge ΔΔG_c (kJ/mol, ≤ 0).  The
        apparent partition function is:

            Z = θ₀ exp(−G₀/RT) + Σ_c θ_c exp(−(G₀ + ΔΔG_c)/RT)

        giving::

            ΔG_app  = −RT ln(Z)
            Kd_app  = exp(ΔG_app / RT)

        Parameters
        ----------
        cation_concentrations_M : dict
            Salt concentrations [M] for any subset of ``{'Na', 'K', 'Mg', 'Ca'}``.
            Example (montmorillonite + 0.1 M NaCl + 5 mM CaCl₂)::

                {'Na': 0.1, 'Ca': 0.005}

            Keys not listed default to 0 M.  Na⁺ is always present as the
            clay counter-ion (include it explicitly if relevant).
        K_clay : dict or None
            Override clay binding constants (M⁻¹).  Missing keys use
            ``_CLAY_K_DEFAULT``.
        dG_bridge_kJ : dict or None
            Override per-cation ΔΔG_bridge values (kJ/mol, should be ≤ 0).
            Missing keys use ``_CLAY_DDG_DEFAULT``.
        verbose : bool

        Returns
        -------
        dict
            ``dG_base_kJ``        — PMF-derived ΔG° (uncorrected)
            ``dG_app_kJ``         — apparent ΔG° at given cation concentrations
            ``Kd_base_M``         — uncorrected Kd
            ``Kd_app_M``          — apparent Kd
            ``theta``             — dict of site occupancies (θ₀ + Σθ_c = 1)
            ``K_clay``            — clay binding constants used
            ``dG_bridge_kJ``      — bridge ΔΔG values used
            ``cation_conc_M``     — cation concentrations used
        """
        _ALL_CATIONS = ('Na', 'K', 'Mg', 'Ca')

        # --- ensure base Kd is available ---
        if self._results is None:
            self.compute(verbose=False)
        dG_base = float(self._results['dG_pf_kJ'])
        Kd_base = float(self._results['Kd_pf_M'])
        RT = self._K_B * self.T

        # --- merge defaults with user overrides ---
        K_c   = dict(self._CLAY_K_DEFAULT)
        ddG_c = dict(self._CLAY_DDG_DEFAULT)
        if K_clay is not None:
            K_c.update(K_clay)
        if dG_bridge_kJ is not None:
            ddG_c.update(dG_bridge_kJ)

        # --- concentrations (M) for each cation ---
        conc = {c: float(cation_concentrations_M.get(c, 0.0))
                for c in _ALL_CATIONS}

        # --- competitive Langmuir site occupancy ---
        denom = 1.0 + sum(K_c[c] * conc[c] for c in _ALL_CATIONS)
        theta_0 = 1.0 / denom
        theta   = {c: K_c[c] * conc[c] / denom for c in _ALL_CATIONS}

        # --- apparent partition function (dimensionless) ---
        # each term: θ × exp(−G_state/RT) = θ × exp(−(G_base + ΔΔG_c)/RT)
        Z = theta_0 * np.exp(-dG_base / RT)
        for c in _ALL_CATIONS:
            if theta[c] > 0.0:
                Z += theta[c] * np.exp(-(dG_base + ddG_c[c]) / RT)

        dG_app = float(-RT * np.log(Z))
        Kd_app = float(np.exp(dG_app / RT))

        # --- pack result ---
        all_theta = {'0': theta_0}
        all_theta.update(theta)

        result = dict(
            dG_base_kJ   = dG_base,
            dG_app_kJ    = dG_app,
            Kd_base_M    = Kd_base,
            Kd_app_M     = Kd_app,
            theta        = all_theta,
            K_clay       = dict(K_c),
            dG_bridge_kJ = dict(ddG_c),
            cation_conc_M = dict(conc),
        )

        if verbose:
            _line = "\u2500" * 62
            _ok   = "\u2713"
            _lbl  = self._CATION_LABELS
            print(_line)
            print("  ClayKdImproved \u2014 Cation-Mediated Binding")
            print("  System: montmorillonite clay")
            print(_line)

            # --- cation occupancy table ---
            _h1   = "K_clay (M\u207b\u00b9)"
            _h2   = "\u0394\u0394G (kJ/mol)"
            _hth  = "\u03b8"
            _dash = "-"
            _zero = "0.00"
            _bare = "(bare)"
            _cat  = "Cation"
            _conc = "[c] (mM)"
            print(f"  {_cat:<8}  {_conc:>10}  "
                  f"{_h1:>14}  "
                  f"{_hth:>6}  "
                  f"{_h2:>14}")
            print(f"  {_dash*8}  {_dash*10}  {_dash*14}  {_dash*6}  {_dash*14}")
            # bare clay row
            _empty = ""
            print(f"  {_bare:<8}  {_empty:>10}  {_empty:>14}  "
                  f"{theta_0:>6.3f}  {_zero:>14}")
            for c in _ALL_CATIONS:
                label = _lbl.get(c, c)
                th    = theta[c]
                flag  = f"  {_ok}" if conc[c] > 0 else ""
                print(f"  {label:<8}  {conc[c]*1e3:>10.2f}  "
                      f"{K_c[c]:>14.1f}  "
                      f"{th:>6.3f}  "
                      f"{ddG_c[c]:>+14.2f}{flag}")
            print(_line)

            # --- free energy summary ---
            ddG_eff = dG_app - dG_base
            print(f"  ΔG°_base (PMF)       : {dG_base:+.2f} kJ/mol"
                  f"   Kd = {self._fmt_kd(Kd_base)}")
            print(f"  Effective ΔΔG_cation : {ddG_eff:+.2f} kJ/mol")
            print(f"  ΔG°_apparent         : {dG_app:+.2f} kJ/mol"
                  f"   Kd = {self._fmt_kd(Kd_app)}")
            print(_line)
            # warn if no cations given
            if all(conc[c] == 0 for c in _ALL_CATIONS):
                print("  \u26a0  All cation concentrations are 0. "
                      "Kd_app equals Kd_base.")
            # highlight dominant cation
            dominant = max(theta, key=lambda c: theta[c])
            if theta[dominant] > 0.5:
                print(f"  Dominant site state: {_lbl.get(dominant, dominant)}-clay"
                      f"  (\u03b8 = {theta[dominant]:.2f})")

        return result

    # ------------------------------------------------------------------
    # 4. Long-range electrostatic tail correction
    # ------------------------------------------------------------------

    def correct_long_range_tail(
            self,
            r_bulk_start=None,
            bulk_pmf_plateau=None,
            ionic_strength_M=0.1,
            verbose=True):
        """
        Correct the PMF tail when long-range clay electrostatics prevent
        full decay to zero in the bulk region.

        Convention (same as ClayKd): ``r[0]`` is bulk (large r, W≈0),
        ``r[-1]`` is surface (small r, W < 0).  ``r`` is the distance
        of the solute from the clay surface.

        Due to the finite simulation box, the PMF at large r (bulk) may
        carry a residual non-zero plateau rather than converging to 0.
        This method:

        1. Estimates the plateau from the bulk region (r ≥ r_bulk_start).
        2. Subtracts a tapered correction profile:

           .. code-block:: text

               correction(r) = W_plateau                          for r ≥ r_bulk_start
               correction(r) = W_plateau × exp(−(r_bulk_start−r)/λ_D)  for r < r_bulk_start

           This applies the full correction in the true bulk (large r) and
           smoothly decays toward zero near the surface (small r), where
           specific clay–solute interactions dominate and no correction is
           needed.

        3. Updates ``self.pmf_ref`` in-place.

        Parameters
        ----------
        r_bulk_start : float or None
            Distance (nm) above which the bulk region begins.  Defaults
            to the top 20 % of the r range (auto-detected).
        bulk_pmf_plateau : float or None
            Expected asymptotic PMF value in bulk.  If ``None``, computed
            as the mean of ``self.pmf_ref`` in the bulk region.
        ionic_strength_M : float
            Ionic strength (M) used for the Debye length calculation
            λ_D = 0.304 / √I  nm  (T ≈ 298 K, 1:1 electrolyte scaling).
        verbose : bool
            Print a short diagnostic table.

        Returns
        -------
        w_corrected : ndarray
            Corrected PMF (same shape as ``self.pmf_ref``).
        plateau_kJ_mol : float
            The plateau value that was subtracted in the bulk region.

        Notes
        -----
        For multi-valent salts (MgCl₂, CaCl₂) the effective ionic
        strength should include the contribution of divalent ions:
        ``I = 0.5 × (c_Na + c_K + 4×c_Mg + 4×c_Ca + c_Cl)``.
        """
        if self.pmf_ref is None:
            raise RuntimeError(
                "self.pmf_ref is None — call compute() first."
            )

        r = self.r
        w = self.pmf_ref.copy()

        # --- auto-detect bulk start ---
        if r_bulk_start is None:
            r_min, r_max = float(r.min()), float(r.max())
            r_bulk_start = r_min + 0.80 * (r_max - r_min)   # top 20 % of range

        # --- bulk mask: large r (far from surface) ---
        bulk_mask = r >= r_bulk_start

        if not np.any(bulk_mask):
            warnings.warn(
                f"No path points found with r ≥ {r_bulk_start:.3f} nm. "
                f"r range is [{r.min():.3f}, {r.max():.3f}] nm. "
                "Returning uncorrected PMF.",
                RuntimeWarning, stacklevel=2,
            )
            return w, 0.0

        # --- estimate plateau ---
        if bulk_pmf_plateau is None:
            bulk_pmf_plateau = float(np.mean(w[bulk_mask]))

        if bulk_pmf_plateau == 0.0:
            if verbose:
                print("  correct_long_range_tail: plateau = 0.00 kJ/mol — "
                      "no correction needed.")
            return w, 0.0

        # --- Debye length (nm) at T ≈ 298 K ---
        # λ_D = 0.304 / √(I/M)  nm  (1:1 electrolyte, Debye-Hückel)
        lambda_D = 0.304 / np.sqrt(max(ionic_strength_M, 1e-6))

        # --- tapered correction profile ---
        #   • Bulk region (r ≥ r_bulk_start): full plateau subtraction
        #   • Surface region (r < r_bulk_start): exponential taper to 0
        #     so the strong specific binding near the surface is untouched
        surface_decay = bulk_pmf_plateau * np.exp(
            -(r_bulk_start - r) / lambda_D
        )
        tail_correction = np.where(r >= r_bulk_start,
                                   bulk_pmf_plateau,
                                   surface_decay)

        w_corrected = w - tail_correction

        # --- store correction metadata ---
        self._tail_corr_meta = {
            'plateau_kJ_mol':    float(bulk_pmf_plateau),
            'r_bulk_start_nm':   float(r_bulk_start),
            'lambda_D_nm':       float(lambda_D),
            'ionic_strength_M':  float(ionic_strength_M),
            'n_bulk_points':     int(np.sum(bulk_mask)),
        }

        # --- update self.pmf_ref ---
        self.pmf_ref = w_corrected

        if verbose:
            _sep = "\u2500" * 62
            _lam = "λ_D"
            _pm  = "\u00b1"
            _delta = "\u0394"
            _w   = "W"
            print(_sep)
            print("  Debye-Hückel Tail Correction")
            print(_sep)
            _ion_str = f"  Ionic strength          : {ionic_strength_M:.4f} M"
            _lam_str = f"  {_lam}  (Debye length)    : {lambda_D:.3f} nm"
            _rbs_str = f"  Bulk region r ≥         : {r_bulk_start:.3f} nm"
            _nbs_str = f"  Bulk points used        : {int(np.sum(bulk_mask))}"
            _plt_str = f"  Plateau {_w}(bulk)         : {bulk_pmf_plateau:+.3f} kJ/mol"
            print(_ion_str)
            print(_lam_str)
            print(_rbs_str)
            print(_nbs_str)
            print(_plt_str)
            _bulk_new = float(np.mean(w_corrected[bulk_mask]))
            _surf_new = float(w_corrected[np.argmin(r)])
            _bulk_str = f"  {_w}(bulk) after correction : {_bulk_new:+.3f} kJ/mol"
            _surf_str = f"  {_w}(surface, r_min)        : {_surf_new:+.3f} kJ/mol"
            print(_bulk_str)
            print(_surf_str)
            _dg_old = float(self._results['dG_kJ_mol']) if self._results else float('nan')
            print(f"  Note: call compute() again to refresh Kd with corrected PMF.")
            print(_sep)

        return w_corrected, float(bulk_pmf_plateau)

    # ------------------------------------------------------------------
    def report_kd_with_confidence(self, confidence_level=0.95, verbose=True):
        """
        Report Kd with confidence intervals from bootstrap resampling.

        Kd is log-normally distributed (it is the exponential of ΔG/kT),
        so this method reports the **geometric mean** and **empirical
        percentile CI** — which correctly capture the asymmetric
        uncertainty instead of assuming a symmetric Gaussian.

        Calls ``bootstrap_uncertainty(n_bootstrap=200)`` automatically if
        it has not been run yet for this instance.

        Parameters
        ----------
        confidence_level : float
            Confidence level for the interval.  0.95 → 95% CI, 0.90 → 90%.
        verbose : bool
            Print the formatted report.

        Returns
        -------
        dict
            geometric_mean_M, ci_lower_M, ci_upper_M,
            multiplicative_factor, n_bootstrap, log_std.
        """
        if not hasattr(self, '_bootstrap_results'):
            if verbose:
                print("  Running bootstrap for uncertainty estimation...")
            self.bootstrap_uncertainty(n_bootstrap=200, verbose=verbose)

        br      = self._bootstrap_results
        kd_vals = br.get('Kd_pf_M_values')
        if kd_vals is None or len(kd_vals) == 0:
            raise RuntimeError(
                "No bootstrap Kd values available. "
                "Re-run bootstrap_uncertainty() to refresh.")

        # guard: Kd must be positive
        kd_vals = kd_vals[kd_vals > 0]
        if len(kd_vals) == 0:
            raise RuntimeError("All bootstrap Kd values are non-positive.")

        # --- log-space statistics (Kd ~ log-normal) ---
        log_kd   = np.log(kd_vals)
        log_mean = float(np.mean(log_kd))
        log_std  = float(np.std(log_kd, ddof=1))

        geometric_mean = float(np.exp(log_mean))

        # --- empirical percentile CI (no parametric assumption needed) ---
        alpha    = 1.0 - confidence_level
        pct_lo   = 100.0 * (alpha / 2.0)
        pct_hi   = 100.0 * (1.0 - alpha / 2.0)
        ci_lower = float(np.percentile(kd_vals, pct_lo))
        ci_upper = float(np.percentile(kd_vals, pct_hi))
        mult     = (ci_upper / geometric_mean
                    if geometric_mean > 0 else float('nan'))

        result = {
            'geometric_mean_M':      geometric_mean,
            'ci_lower_M':            ci_lower,
            'ci_upper_M':            ci_upper,
            'multiplicative_factor': mult,
            'n_bootstrap':           int(len(kd_vals)),
            'log_std':               log_std,
        }

        if verbose:
            _sep  = "\u2500" * 62
            _cl   = f"{confidence_level * 100:.0f}"
            _kd_g = self._fmt_kd(geometric_mean)
            _kd_l = self._fmt_kd(ci_lower)
            _kd_u = self._fmt_kd(ci_upper)
            _pct  = "%"
            print(_sep)
            print("  Kd Confidence Report  (bootstrap percentile CI)")
            print(_sep)
            print(f"  Bootstrap resamples     : {len(kd_vals)}")
            print(f"  Confidence level        : {_cl}{_pct}")
            print(f"  Geometric mean Kd       : {_kd_g}")
            print(f"  {_cl}{_pct} CI                  : [{_kd_l},  {_kd_u}]")
            print(f"  Multiplicative factor   : {mult:.2f}x")
            print(f"  log(Kd) std dev         : {log_std:.3f}")
            print(_sep)

        return result

    # ------------------------------------------------------------------
    # 6. Ensemble Kd from multiple independent paths
    # ------------------------------------------------------------------
    @classmethod
    def compute_ensemble_kd(
        cls,
        clay_kd_list,
        r_saddle=None,
        n_bulk_points=5,
        use_regression=False,
        confidence_level=0.95,
        verbose=True,
    ):
        """
        Compute an ensemble-averaged Kd from a list of independent
        ``ClayKd`` (or ``ClayKdImproved``) objects, each representing a
        separate MFEP run (different random seeds, initial conditions, or
        replicas).

        Unlike ``bootstrap_uncertainty`` — which resamples a *single* path
        to estimate statistical noise — this method captures *inter-replica*
        variability: how much Kd changes between independent simulations.

        Parameters
        ----------
        clay_kd_list : list of ClayKd
            Independent ``ClayKd``/``ClayKdImproved`` instances.  Each must
            have been constructed with ``ClayKd(clay_path, T=...)`` but need
            not have had ``compute()`` called yet — this method calls it.
        r_saddle : float or None
            Passed to each object's ``compute()``.  None = auto-detect.
        n_bulk_points : int
            Passed to each object's ``compute()``.  Default 5.
        use_regression : bool
            Use regression bulk reference for each path.  Default False.
        confidence_level : float
            Confidence level for the CI.  For small ensembles (n < 30) a
            Student-t interval on log(Kd) is used; for n ≥ 30 it falls back
            to a normal approximation.
        verbose : bool
            Print a summary table.

        Returns
        -------
        dict with keys:
            n_paths, Kd_geom_mean_M, Kd_geom_std_factor,
            Kd_ci_lower_M, Kd_ci_upper_M, Kd_values_M,
            dG_values_kJ, dG_mean_kJ, dG_std_kJ,
            confidence_level, log_kd_mean, log_kd_std
        """
        from scipy import stats as _stats

        if len(clay_kd_list) < 2:
            raise ValueError(
                "compute_ensemble_kd requires at least 2 independent paths; "
                f"got {len(clay_kd_list)}."
            )

        kd_vals  = []
        dg_vals  = []
        for i, kd_obj in enumerate(clay_kd_list):
            res = kd_obj.compute(
                r_saddle=r_saddle,
                n_bulk_points=n_bulk_points,
                verbose=False,
                use_regression=use_regression,
            )
            kd_vals.append(float(res["Kd_pf_M"]))
            dg_vals.append(float(res["dG_pf_kJ"]))

        kd_arr  = np.array(kd_vals)
        dg_arr  = np.array(dg_vals)
        log_kd  = np.log(kd_arr)
        n       = len(kd_arr)

        log_mean = float(np.mean(log_kd))
        log_std  = float(np.std(log_kd, ddof=1))
        geom_mean = float(np.exp(log_mean))
        geom_std  = float(np.exp(log_std))   # multiplicative factor

        # CI on log(Kd): t-distribution for small n, normal for n>=30
        alpha   = 1.0 - confidence_level
        log_sem = log_std / float(np.sqrt(n))
        if n < 30:
            t_crit = float(_stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
        else:
            t_crit = float(_stats.norm.ppf(1.0 - alpha / 2.0))
        ci_lo = float(np.exp(log_mean - t_crit * log_sem))
        ci_hi = float(np.exp(log_mean + t_crit * log_sem))

        dg_mean = float(np.mean(dg_arr))
        dg_std  = float(np.std(dg_arr, ddof=1))

        result = dict(
            n_paths=n,
            Kd_geom_mean_M=geom_mean,
            Kd_geom_std_factor=geom_std,
            Kd_ci_lower_M=ci_lo,
            Kd_ci_upper_M=ci_hi,
            Kd_values_M=kd_arr.tolist(),
            dG_values_kJ=dg_arr.tolist(),
            dG_mean_kJ=dg_mean,
            dG_std_kJ=dg_std,
            confidence_level=confidence_level,
            log_kd_mean=log_mean,
            log_kd_std=log_std,
        )

        if verbose:
            _cl      = int(round(confidence_level * 100))
            _pct     = "%"
            _sep     = "-" * 60
            _hdr_dg  = "\u0394G\u00b0 (kJ/mol)"   # pre-assign to avoid backslash in f-expr
            _lbl_dg  = "\u0394G\u00b0 mean \u00b1 std"
            _lbl_pm  = "\u00b1"
            print(_sep)
            print(f"  Ensemble Kd  ({n} independent paths)")
            print(_sep)
            _hdr_kd  = "Kd"
            _hdr_p   = "Path"
            print(f"  {_hdr_p:>6}  {_hdr_kd:>14}  {_hdr_dg:>16}")
            for idx, (kd_i, dg_i) in enumerate(zip(kd_vals, dg_vals)):
                _kd_str = ClayKd._fmt_kd(kd_i)
                print(f"  {idx+1:>6}  {_kd_str:>14}  {dg_i:>+16.2f}")
            print(_sep)
            _gm = ClayKd._fmt_kd(geom_mean)
            _lo = ClayKd._fmt_kd(ci_lo)
            _hi = ClayKd._fmt_kd(ci_hi)
            _ci_type = "t" if n < 30 else "normal"
            print(f"  Geometric mean Kd        : {_gm}")
            print(f"  Geometric std factor     : {geom_std:.2f}x")
            print(f"  {_cl}{_pct} CI ({_ci_type}-dist)        : [{_lo},  {_hi}]")
            print(f"  log(Kd) std dev          : {log_std:.3f}")
            print(f"  {_lbl_dg:24s} : "
                  f"{dg_mean:+.2f} {_lbl_pm} {dg_std:.2f} kJ/mol")
            print(_sep)

        return result

if __name__ == '__main__':
    ClayPath._smoke_test()
