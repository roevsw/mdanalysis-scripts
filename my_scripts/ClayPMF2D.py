#!/usr/bin/env python3
"""
ClayPMF2D.py
============
2D WHAM analysis: distance × orientation PMF for CIP on montmorillonite clay.

Coordinates
-----------
r = |z|    : distance from clay COM (nm), same biased coordinate as 1D WHAM
θ          : tilt angle of CIP aromatic ring vs. clay surface normal (degrees)
             θ = 0°  → plane normal ∥ z → ring lying flat (parallel to surface)
             θ = 90° → plane normal ⊥ z → ring edge-on (perpendicular to surface)

             How θ is computed
             ~~~~~~~~~~~~~~~~~~
             θ is the angle between the best-fit plane normal of the ring
             atoms and the z-axis (clay surface normal).  The normal is found
             by singular value decomposition (SVD) of the centred ring-atom
             coordinates; the last right-singular vector minimises residuals
             and is therefore the plane normal.

                 z (clay normal)
                 ↑
                 |  ← ring normal (tilted)
                 | ╱
                 |╱ θ
                 ────────────────  clay surface

             When θ is small the ring normal is nearly parallel to z, meaning
             the ring lies flat (face-parallel to the clay).  When θ → 90° the
             normal is nearly perpendicular to z and the ring stands edge-on.

             Plane-normal geometry (θ = 0° case, ring lying flat):

                        normal ↑
                           |
                    ───────┼──────   ← ring plane
                           |

             Note: θ is folded into [0°, 90°] by taking abs(cos θ) before
             the arccos.  This means θ cannot distinguish which face of the
             ring faces the clay — a molecule-fixed orientation vector
             (e.g. N1 → piperazine-N) is needed for that.

Key mathematical insight
------------------------
The umbrella bias acts *only* on r = |z|, so the WHAM free energies {f_i}
satisfy the same self-consistency equations as in 1D.  They are determined
entirely by the marginal r-distribution, and the full 2D unbiased probability
follows analytically without any additional outer iteration:

    D(r)    = Σᵢ Nᵢ · exp(fᵢ − β·Vᵢ(r))      Vᵢ(r) = ½k(r − r₀ᵢ)²
    P(r, θ) ∝ [ Σᵢ Hᵢ(r, θ) ] / D(r)
    W(r, θ) = −kT ln P(r, θ)   [shifted so that min = 0]

The marginal PMF(r) from 2D WHAM is identical (within sampling noise) to the
1D ClayPMF result.

Grid resolution
---------------
n_r_bins     : bins along the r-axis.  Default 50; 50–100 recommended for 2D.
n_theta_bins : bins along the θ-axis.  Default 36 (= 2.5° per bin over 0–90°).
               Increasing to 90 gives 1°/bin but requires denser sampling.

These are *constructor parameters* — not hardcoded.  In the notebook they are
exposed as ``N_R_BINS`` / ``N_THETA_BINS`` in the §13 parameter cell and
forwarded to the constructor.  Changing either value invalidates the θ cache
only if ``theta_range`` changes; the cache stores raw per-frame θ values, so
re-binning is automatic.  Set ``THETA_FORCE_RERUN=True`` only when the
underlying trajectory data needs to be re-read.

Typical usage
-------------
    from ClayPMF2D import ClayPMF2D

    pmf2d = ClayPMF2D(
        umbrella_dir='Umbrella/',
        n_windows=30,
        k=1000.0,            # kJ/(mol·nm²)
        T=298.15,
        equil_skip_ps=1000.0,
        n_r_bins=50,
        n_theta_bins=36,
    )
    pmf2d.load_data()
    pmf2d.detect_clay_surface('md.gro')
    pmf2d.define_selections({
        'CIP_ring': {
            'cip_ring': 'resname api and (name C4 C4a C8a C1 C2 C3 C5 N1)',
        },
    })
    pmf2d.load_theta_data(
        traj_files=[f'Umbrella/traj{i}.xtc' for i in range(1, 31)],
        topology='md.tpr',
    )
    pmf2d.run_wham_2d()
    pmf2d.bootstrap_errors_2d(n_bootstrap=100)
    fig, ax   = pmf2d.plot_2d_pmf(unit='kJ/mol', zero_at='bulk')
    fig2, axs = pmf2d.plot_marginals(unit='kJ/mol')
    pmf2d.save_results()

Compatibility with ClayPMFPlotter
----------------------------------
ClayPMF2D sets the same attributes as ClayPMF
(pmf_abs, bin_centers_abs, pmf_signed, _bin_centers_signed, z_clay_surface,
pmf_abs_std, pmf_signed_std) so that ClayPMFPlotter can consume the
marginal PMF(r) transparently.
"""

import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from tqdm import tqdm

try:
    import MDAnalysis as mda
    _MDA_AVAILABLE = True
except ImportError:
    _MDA_AVAILABLE = False


class ClayPMF2D:
    """
    2D WHAM for CIP adsorption: distance |z| × tilt-angle θ.

    Parameters
    ----------
    umbrella_dir : str
        Directory containing pullx*.xvg files.
    n_windows : int
        Number of umbrella windows (= number of pullx files). Default 30.
    k : float
        Harmonic spring constant in kJ/(mol·nm²). Default 1000.0.
    T : float
        Temperature in K. Default 298.15.
    equil_skip_ps : float
        Equilibration time to discard from each window (ps). Default 1000.0.
    n_r_bins : int
        Number of bins for r = |z|. Default 50.
    n_theta_bins : int
        Number of bins for θ. Default 36.
    theta_range : tuple
        (min_deg, max_deg) for θ. Default (0.0, 90.0).
    xi_min : float or None
        Lower r-grid edge (nm). Auto-detected from data if None.
    xi_max : float or None
        Upper r-grid edge (nm). Auto-detected from data if None.
    pullx_prefix : str
        Filename prefix for pull-position files. Default 'pullx'.
    tolerance : float
        WHAM convergence criterion. Default 1e-6.
    max_iter : int
        Maximum WHAM iterations. Default 50000.
    verbose : bool
        Print progress messages. Default True.
    """

    K_B = 8.314462618e-3   # kJ / (mol·K)

    def __init__(
        self,
        umbrella_dir,
        n_windows=30,
        k=1000.0,
        T=298.15,
        equil_skip_ps=1000.0,
        n_r_bins=50,
        n_theta_bins=36,
        theta_range=(0.0, 90.0),
        xi_min=None,
        xi_max=None,
        pullx_prefix='pullx',
        tolerance=1e-6,
        max_iter=50000,
        verbose=True,
    ):
        self.umbrella_dir  = os.path.abspath(umbrella_dir)
        self.n_windows     = int(n_windows)
        self.k             = float(k)
        self.T             = float(T)
        self.equil_skip_ps = float(equil_skip_ps)
        self.n_r_bins      = int(n_r_bins)
        self.n_theta_bins  = int(n_theta_bins)
        self.theta_range   = tuple(theta_range)
        self.xi_min        = xi_min
        self.xi_max        = xi_max
        self.pullx_prefix  = pullx_prefix
        self.tolerance     = tolerance
        self.max_iter      = int(max_iter)
        self.verbose       = verbose

        self.beta = 1.0 / (self.K_B * self.T)   # mol / kJ

        # --- custom atom selections (set by define_selections / add_selection) ---
        self.custom_selections = {}   # {category: {name: selection_string}}

        # --- set by load_data() ---
        self.z_data         = None   # list of (z1_prod, z2_prod) per window
        self.t_data         = None   # list of (t_prod, t_prod) per window [ps]
        self.window_centers = None   # list of (c1, c2) per window [nm]

        # --- set by detect_clay_surface() ---
        self.z_clay_surface = None   # nm

        # --- set by load_theta_data() or set_theta_data_direct() ---
        self.theta_data = None   # list of (theta1, theta2) per window [deg]

        # --- set by _build_histograms_2d() ---
        self.r_bins         = None   # (n_r_bins+1,)
        self.r_centers      = None   # (n_r_bins,)
        self.r_width        = None
        self.theta_bins     = None   # (n_theta_bins+1,)
        self.theta_centers  = None   # (n_theta_bins,)
        self.theta_width    = None
        self.histograms_2d  = None   # (2*n_windows, n_r_bins, n_theta_bins)
        self.biases_2d      = None   # (2*n_windows, n_r_bins)  kJ/mol
        self.n_snapshots    = None   # (2*n_windows,)

        # --- set by run_wham_2d() ---
        self.f              = None   # WHAM free energies (2*n_windows,)
        self.P_2d           = None   # unbiased P(r,θ) (n_r_bins, n_theta_bins)
        self.pmf_2d         = None   # W(r,θ) kJ/mol
        self.pmf_abs        = None   # W(r), marginal over θ, kJ/mol
        self.pmf_theta      = None   # W(θ), marginal over r, kJ/mol
        # ClayPMFPlotter compatibility:
        self.bin_centers_abs     = None   # alias for r_centers
        self.pmf_signed          = None   # mirror of pmf_abs on [-r_max, +r_max]
        self._bin_centers_signed = None
        self.gmx_z               = None   # not applicable for 2D (set to None)
        self.gmx_pmf             = None

        # --- set by bootstrap_errors_2d() ---
        self.pmf_2d_std  = None
        self.pmf_abs_std = None
        self.pmf_signed_std = None   # ClayPMFPlotter compat

        # --- set by reference_to_bulk() ---
        self.bulk_fraction            = None   # fraction of r-axis treated as bulk
        self.bulk_correction_enabled  = False
        self._bulk_shift              = 0.0    # kJ/mol subtracted from all PMFs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg):
        if self.verbose:
            print(msg)

    @staticmethod
    def _read_pullx(filepath):
        """Parse GROMACS pullx xvg. Returns (time [ps], z1 [nm], z2 [nm])."""
        data = np.loadtxt(filepath, comments=['#', '@'], ndmin=2)
        return data[:, 0], data[:, 1], data[:, 2]

    @staticmethod
    def _tilt_from_coords(coords):
        """
        Compute aromatic-ring tilt angle from atomic coordinates.

        θ = angle between the plane normal and the z-axis:
            θ = 0°  : normal ∥ z → ring lying flat (parallel to clay surface)
            θ = 90° : normal ⊥ z → ring edge-on

        Uses SVD: the last right-singular vector spans the minimum-variance
        direction = normal to the best-fit plane.

        Parameters
        ----------
        coords : np.ndarray  shape (N, 3), N ≥ 3
            Atomic positions in any unit (only directions matter).

        Returns
        -------
        theta : float  [degrees, 0–90]
        """
        centred = coords - coords.mean(axis=0)
        # Last row of Vt from SVD = direction of minimum variance = plane normal
        _, _, Vt = np.linalg.svd(centred, full_matrices=False)
        normal = Vt[-1]
        cos_theta = float(np.clip(abs(normal[2]) / np.linalg.norm(normal), 0.0, 1.0))
        return float(np.degrees(np.arccos(cos_theta)))

    # ------------------------------------------------------------------
    # Selection management  (mirrors MolecularAnalysis.define_selections)
    # ------------------------------------------------------------------

    def define_selections(self, selections_dict):
        """
        Register named MDAnalysis selection strings, organised by category.

        Mirrors the ``MolecularAnalysis.define_selections`` API so atom groups
        are defined once and referenced by name throughout the analysis.
        Pass a registered name directly to ``load_theta_data`` instead of a
        full selection string.

        Parameters
        ----------
        selections_dict : dict
            Nested  ``{category: {name: selection_string}}``
            or flat ``{name: selection_string}``.

        Returns
        -------
        self

        Examples
        --------
        >>> pmf2d.define_selections({
        ...     'CIP_ring': {
        ...         # Single selection for both CIP molecules — split by residue ID
        ...         'cip_ring': 'resname api and (name C4 C4a C8a N1 C2 C3)',
        ...     },
        ...     'CIP_parts': {
        ...         'quinolone':      'resname api and (name N1 C C2 C3 C7 C8)',
        ...         'piperazine':     'resname api and (name N N2 C10 C11 C12 C13)',
        ...         'carboxylic_acid':'resname api and (name O1 O2 C1)',
        ...         'cyclopropyl':    'resname api and (name C4 C5 C6)',
        ...     },
        ...     'MMT_surface': {
        ...         'surface_oxygen':  'resname MMT and name Ob',
        ...         'surface_silicon': 'resname MMT and name Si',
        ...     },
        ... })
        >>> # Then reference by name in load_theta_data:
        >>> pmf2d.load_theta_data(
        ...     traj_files=[...], topology='md.tpr',
        ...     # resid split into CIP1/CIP2 is automatic
        ... )
        """
        first_value = next(iter(selections_dict.values()))
        is_nested = isinstance(first_value, dict)

        if is_nested:
            for category, selections in selections_dict.items():
                if category not in self.custom_selections:
                    self.custom_selections[category] = {}
                self.custom_selections[category].update(selections)
        else:
            if 'default' not in self.custom_selections:
                self.custom_selections['default'] = {}
            self.custom_selections['default'].update(selections_dict)

        if self.verbose:
            print("Selections registered:")
            for cat, sels in self.custom_selections.items():
                for name in sels:
                    print(f"  [{cat}] {name}")
        return self

    def sel(self, name):
        """
        Retrieve a registered selection string by name.

        Searches all categories.  Raises ``KeyError`` if *name* is not
        registered.

        Parameters
        ----------
        name : str

        Returns
        -------
        str
            MDAnalysis selection string.
        """
        for cat, sels in self.custom_selections.items():
            if name in sels:
                return sels[name]
        available = [n for sels in self.custom_selections.values() for n in sels]
        raise KeyError(
            f"Selection '{name}' not found. "
            f"Available: {', '.join(available) if available else '(none registered)'}"
        )

    def _resolve_sel(self, sel_or_name):
        """Return the full selection string for a registered name, or the string as-is."""
        for sels in self.custom_selections.values():
            if sel_or_name in sels:
                return sels[sel_or_name]
        return sel_or_name

    # ------------------------------------------------------------------

    def _to_unit(self, arr, unit):
        """Convert PMF array from kJ/mol to the requested unit."""
        if unit == 'kJ/mol':
            return arr.copy()
        elif unit == 'kcal/mol':
            return arr.copy() / 4.184
        elif unit in ('kT', 'kBT'):
            return arr.copy() * self.beta
        else:
            raise ValueError(f"Unknown unit '{unit}'. Choose 'kJ/mol', 'kcal/mol', or 'kT'.")

    # ------------------------------------------------------------------
    # Data loading: z from pullx (identical to 1D ClayPMF)
    # ------------------------------------------------------------------

    def load_data(self):
        """
        Read pullx*.xvg files, discard equilibration, store z-trajectories.

        Also stores the time array for each window (needed for
        frame-matching with theta data computed from trajectories).

        Returns
        -------
        self
        """
        self._log(
            f"Loading {self.n_windows} pullx files from:\n  {self.umbrella_dir}"
        )

        self.z_data         = []
        self.t_data         = []
        self.window_centers = []

        for i in range(1, self.n_windows + 1):
            fpath = os.path.join(
                self.umbrella_dir, f'{self.pullx_prefix}{i}.xvg'
            )
            if not os.path.isfile(fpath):
                raise FileNotFoundError(f"Pull-x file not found: {fpath}")

            time, z1, z2 = self._read_pullx(fpath)

            # Window centres: mean of first 25 frames (pull_coord_start=yes)
            n_cf = min(25, len(z1))
            c1   = float(np.mean(z1[:n_cf]))
            c2   = float(np.mean(z2[:n_cf]))

            mask    = time >= self.equil_skip_ps
            t_prod  = time[mask]
            z1_prod = z1[mask]
            z2_prod = z2[mask]

            if len(z1_prod) == 0:
                raise ValueError(
                    f"Window {i}: no frames after equil_skip_ps="
                    f"{self.equil_skip_ps} ps  (last t={time[-1]:.1f} ps)."
                )

            self.z_data.append((z1_prod, z2_prod))
            self.t_data.append(t_prod)          # same time array for both CIPs
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
        # Preserve original pullx data so load_theta_data() always has access
        # to the dense pullx arrays for interpolation, even after z_data /
        # t_data have been overwritten by a previous load_theta_data() call.
        self._pullx_z_data = list(self.z_data)
        self._pullx_t_data = list(self.t_data)
        return self

    # ------------------------------------------------------------------
    # Clay surface detection (same logic as 1D ClayPMF)
    # ------------------------------------------------------------------

    def detect_clay_surface(
        self,
        structure_file,
        clay_selection='resname MMT',
        surface_pct=95,
    ):
        """
        Detect the outermost clay layer z-position using MDAnalysis.

        Stores ``self.z_clay_surface`` (nm) for axis labelling.

        Parameters
        ----------
        structure_file : str
            GROMACS structure file (.gro, .tpr, or any MDA-readable topology).
        clay_selection : str
            MDAnalysis selection for clay atoms. Default ``'resname MMT'``.
        surface_pct : float
            Kept for API compatibility; not used (mean of upper Si atoms is used).

        Returns
        -------
        z_surface : float  [nm]
        """
        if not _MDA_AVAILABLE:
            raise ImportError(
                "MDAnalysis is required for detect_clay_surface(). "
                "Install with:  pip install MDAnalysis"
            )

        u      = mda.Universe(structure_file)
        Lz_ang = float(u.dimensions[2])

        si_sel = u.select_atoms(
            f"({clay_selection}) and "
            "(name Si or name SI or name Sio or name SIO)"
        )
        if len(si_sel) > 0:
            si_z     = si_sel.positions[:, 2] - Lz_ang / 2.0
            upper_si = si_z[si_z > 0]
            if len(upper_si) == 0:
                upper_si = np.abs(si_z)
            z_surface = float(np.mean(upper_si)) / 10.0
        else:
            clay  = u.select_atoms(clay_selection)
            z_c   = clay.positions[:, 2] - Lz_ang / 2.0
            upper = z_c[z_c > 0]
            if len(upper) == 0:
                upper = np.abs(z_c)
            z_surface = float(np.mean(upper)) / 10.0

        self.z_clay_surface = z_surface
        self._log(f"Clay surface: z_surface = {z_surface:.3f} nm")
        return z_surface

    # ------------------------------------------------------------------
    # Theta loading from GROMACS trajectories
    # ------------------------------------------------------------------

    def load_theta_data(
        self,
        traj_files,
        topology,
        cip_selection=None,
        stride=1,
        cache_file=None,
        save_cache=True,
        force_recompute=False,
    ):
        """
        Compute CIP orientation angles from GROMACS trajectories.

        For each window, reads the trajectory, computes tilt angle θ of
        CIP1 and CIP2 at every production frame, then linearly interpolates
        z-values from the pullx file onto those frame times.  This gives
        matched (r, θ) pairs at the trajectory output frequency.

        θ is the angle between the aromatic ring plane-normal and the z-axis
        (0° = flat; 90° = edge-on).

        The two CIP molecules are identified automatically from the selection:
        all matching atoms are split by residue ID (ascending order).  The
        two molecules share the same atom names — no per-residue selection
        string is needed.

        Parameters
        ----------
        traj_files : list of str
            One trajectory (.xtc or .trr) per window (len == n_windows).
        topology : str or list of str
            Topology file(s) (.tpr or .gro).  A single string is used for
            every window.
        cip_selection : str or None
            MDAnalysis selection string **or** a name registered via
            :meth:`define_selections` that selects the aromatic ring atoms
            for **both** CIP molecules (≥ 3 atoms per residue).  The
            selection is split into two groups by residue ID (ascending).
            If ``None``, looks up ``'cip_ring'`` from registered selections.
            Example literal: ``'resname api and (name C4 C4a C8a C1 C2 C3 C5 N1)'``
        stride : int
            Use every *stride*-th trajectory frame. Default 1.
        cache_file : str or None
            Path to the ``.npz`` cache file.  If ``None`` (default), uses
            ``<umbrella_dir>/theta_cache_s<stride>.npz``.
        save_cache : bool
            Save computed angles to *cache_file* for future runs. Default ``True``.
        force_recompute : bool
            Ignore an existing cache and recompute from trajectories. Default ``False``.

        Sets
        ----
        self.theta_data : list of (theta1, theta2)  [degrees]
        self.z_data     : resampled to trajectory frame times
        self.t_data     : updated to trajectory frame times [ps]

        Returns
        -------
        self
        """
        if self.z_data is None:
            raise RuntimeError("Call load_data() before load_theta_data().")
        if len(traj_files) != self.n_windows:
            raise ValueError(
                f"Expected {self.n_windows} traj_files, got {len(traj_files)}."
            )

        # --- default cache path ---
        if cache_file is None:
            cache_file = os.path.join(
                self.umbrella_dir, f"theta_cache_s{stride}.npz"
            )

        # --- try loading from cache ---
        if save_cache and not force_recompute and os.path.isfile(cache_file):
            self._log(f"Loading θ from cache: {cache_file}")
            try:
                data  = np.load(cache_file, allow_pickle=False)
                n_win = int(data['n_windows'])
                if n_win != self.n_windows:
                    raise ValueError(
                        f"Cache has {n_win} windows but n_windows={self.n_windows}"
                    )
                self.theta_data = []
                new_z_data = []
                new_t_data = []
                for i in range(n_win):
                    self.theta_data.append((data[f't1_{i}'], data[f't2_{i}']))
                    new_z_data.append((data[f'z1_{i}'], data[f'z2_{i}']))
                    new_t_data.append(data[f't_{i}'])
                self.z_data = new_z_data
                self.t_data = new_t_data
                self._log(f"θ loaded from cache ({n_win} windows).")
                return self
            except Exception as _e:
                self._log(f"  Cache load failed ({_e}); recomputing from trajectories…")

        if not _MDA_AVAILABLE:
            raise ImportError("MDAnalysis is required for load_theta_data().")

        # Resolve selection string: explicit arg (name or literal) > 'cip_ring'
        if cip_selection is None:
            cip_selection = self.sel('cip_ring')
        else:
            cip_selection = self._resolve_sel(cip_selection)

        if isinstance(topology, str):
            topology = [topology] * self.n_windows

        # Sort by the trailing integer in the filename so that e.g.
        # umbrella10.xtc comes *after* umbrella9.xtc (not before, as
        # alphabetical/glob order would give).
        import re as _re
        def _num_key(p):
            m = _re.search(r'(\d+)[^/\\]*$', os.path.basename(p))
            return int(m.group(1)) if m else 0
        _paired = sorted(zip(traj_files, topology), key=lambda x: _num_key(x[0]))
        traj_files = [p[0] for p in _paired]
        topology   = [p[1] for p in _paired]

        self._log(f"Computing θ from {self.n_windows} trajectories (stride={stride})…")
        self._log(f"  CIP sel : {cip_selection}")
        self._log("  (splitting into CIP1/CIP2 by residue ID)")

        self.theta_data = []
        new_z_data      = []
        new_t_data      = []

        for i, (traj_file, top_file) in enumerate(zip(traj_files, topology)):
            if not os.path.isfile(traj_file):
                raise FileNotFoundError(f"Trajectory not found: {traj_file}")
            if not os.path.isfile(top_file):
                raise FileNotFoundError(f"Topology not found: {top_file}")

            u        = mda.Universe(top_file, traj_file)

            # Apply NoJump transformation so that molecular COMs never
            # teleport by a box vector between frames.  This keeps the
            # per-frame unwrap() call consistent: without it, a ring sitting
            # right on the PBC boundary could be reassembled differently in
            # consecutive frames, causing spurious θ discontinuities.
            # NoJump operates on the full Universe trajectory in-place.
            try:
                from MDAnalysis.transformations import NoJump
                u.trajectory.add_transformations(NoJump())
            except ImportError:
                self._log(
                    "  WARNING: MDAnalysis.transformations.NoJump not available "
                    "(upgrade MDAnalysis ≥ 2.1 for full PBC safety). "
                    "Proceeding without inter-frame unwrapping."
                )

            all_cip  = u.select_atoms(cip_selection)

            # Split into two groups by sorted residue ID — same atom names,
            # different residues
            resids   = sorted(set(all_cip.resids))
            if len(resids) != 2:
                raise ValueError(
                    f"cip_selection '{cip_selection}' matched atoms from "
                    f"{len(resids)} residue(s) {resids}; expected exactly 2 "
                    "(one per CIP molecule).  Refine the selection so it spans "
                    "exactly 2 residues."
                )
            # Ring-atom subsets (for SVD plane fitting)
            atoms1 = all_cip.select_atoms(f'resid {resids[0]}')
            atoms2 = all_cip.select_atoms(f'resid {resids[1]}')

            if len(atoms1) < 3:
                raise ValueError(
                    f"resid {resids[0]} matched {len(atoms1)} ring atom(s) "
                    "(need ≥ 3 for plane fitting).  Check cip_selection."
                )
            if len(atoms2) < 3:
                raise ValueError(
                    f"resid {resids[1]} matched {len(atoms2)} ring atom(s) "
                    "(need ≥ 3 for plane fitting).  Check cip_selection."
                )

            # Whole-residue AtomGroups used for unwrap().
            # unwrap() requires a bond-contiguous AtomGroup; selecting only
            # ring atoms breaks the bond graph (e.g. C–C bonds to non-ring
            # atoms are missing) and raises ValueError.  We unwrap the full
            # residue instead — MDAnalysis updates atom positions in-place in
            # the Universe, so atoms1/atoms2 (subsets of the same residues)
            # automatically reflect the corrected coordinates.
            mol1 = u.select_atoms(f'resid {resids[0]}')
            mol2 = u.select_atoms(f'resid {resids[1]}')

            if i == 0:
                self._log(
                    f"  resid {resids[0]} → CIP1 ({len(atoms1)} ring atoms, "
                    f"{len(mol1)} total), "
                    f"resid {resids[1]} → CIP2 ({len(atoms2)} ring atoms, "
                    f"{len(mol2)} total)"
                )

            # Detect whether the topology defines bonds (required for unwrap()).
            # .tpr files include the full bond graph; .gro/.pdb do not.
            # If bonds are absent, rely on the NoJump trajectory transformation
            # applied above for inter-frame PBC continuity and skip per-frame
            # unwrap() — avoids NoDataError with GRO topologies.
            try:
                _has_bonds = len(u.bonds) > 0
            except Exception:
                _has_bonds = False
            if not _has_bonds and i == 0:
                self._log(
                    "  WARNING: topology has no bonds defined (e.g. GRO file). "
                    "Per-frame unwrap() skipped; NoJump handles PBC continuity. "
                    "Use a .tpr topology for fully rigorous unwrapping."
                )

            # Single pass over trajectory — compute theta for both molecules.
            # mol.unwrap() reassembles the whole molecule across PBC using the
            # full bond graph (requires .tpr topology).  After unwrap(), the
            # ring-atom positions in atoms1/atoms2 are already corrected
            # because they share the same underlying Universe coordinate array.
            t_frames, th1_list, th2_list = [], [], []
            for ts in u.trajectory[::stride]:
                if ts.time < self.equil_skip_ps:
                    continue
                t_frames.append(ts.time)
                if _has_bonds:
                    mol1.unwrap()
                    mol2.unwrap()
                th1_list.append(self._tilt_from_coords(atoms1.positions))
                th2_list.append(self._tilt_from_coords(atoms2.positions))

            if len(t_frames) == 0:
                raise ValueError(
                    f"Window {i+1}: no trajectory frames after "
                    f"equil_skip_ps={self.equil_skip_ps} ps."
                )

            t_traj  = np.array(t_frames)
            theta1  = np.array(th1_list)
            theta2  = np.array(th2_list)

            # Interpolate z1, z2 from pullx onto trajectory frame times.
            # Normalize both time arrays to start at 0 before interpolating:
            # if the trajectory was output with a reset start time (e.g. GROMACS
            # tinit=0 on a continuation run) while the pullx file kept cumulative
            # time, a direct np.interp would extrapolate everything to the
            # left-boundary value and every frame would get the same z.
            # Use _pullx_t/z_data (set by load_data()) rather than self.t/z_data
            # which may have been overwritten by a previous load_theta_data()
            # cache-load, turning the interpolation into a no-op on corrupt values.
            t_pullx   = self._pullx_t_data[i] if hasattr(self, '_pullx_t_data') else self.t_data[i]
            z1_pullx, z2_pullx = (self._pullx_z_data[i] if hasattr(self, '_pullx_z_data') else self.z_data[i])

            t_pullx_rel = t_pullx - t_pullx[0]
            t_traj_rel  = t_traj  - t_traj[0]

            if i == 0:
                self._log(
                    f"  Time alignment: pullx[0]={t_pullx[0]:.1f} ps, "
                    f"traj[0]={t_traj[0]:.1f} ps  "
                    f"(offset={t_pullx[0]-t_traj[0]:.1f} ps → normalized to 0)"
                )

            z1_matched = np.interp(t_traj_rel, t_pullx_rel, z1_pullx)
            z2_matched = np.interp(t_traj_rel, t_pullx_rel, z2_pullx)

            self.theta_data.append((theta1, theta2))
            new_z_data.append((z1_matched, z2_matched))
            new_t_data.append(t_traj)

            if self.verbose and (i + 1) % 5 == 0:
                self._log(
                    f"  win {i+1:2d}: n_frames={len(t_traj):,}  "
                    f"θ₁=[{theta1.min():.1f}°,{theta1.max():.1f}°]  "
                    f"θ₂=[{theta2.min():.1f}°,{theta2.max():.1f}°]"
                )

        # Replace z_data and t_data with trajectory-frame-aligned versions
        self.z_data = new_z_data
        self.t_data = new_t_data
        self._log("θ loaded; z-data resampled to trajectory frame times.")

        # --- save cache ---
        if save_cache:
            self._log(f"Saving θ cache → {cache_file}")
            save_dict = {'n_windows': np.array(self.n_windows)}
            for i, ((th1, th2), (z1, z2), t) in enumerate(
                zip(self.theta_data, self.z_data, self.t_data)
            ):
                save_dict[f't1_{i}'] = th1
                save_dict[f't2_{i}'] = th2
                save_dict[f'z1_{i}'] = z1
                save_dict[f'z2_{i}'] = z2
                save_dict[f't_{i}']  = t
            np.savez(cache_file, **save_dict)
            self._log(f"  saved {len(self.theta_data)} windows → {cache_file}")

        return self

    def set_theta_data_direct(self, theta_data):
        """
        Supply pre-computed theta arrays directly (no trajectory needed).

        Parameters
        ----------
        theta_data : list of tuple of np.ndarray
            ``theta_data[i] = (theta1_arr, theta2_arr)`` in degrees.
            Lengths must match ``self.z_data[i][0]`` and ``self.z_data[i][1]``.

        Returns
        -------
        self
        """
        if self.z_data is None:
            raise RuntimeError("Call load_data() first.")
        if len(theta_data) != self.n_windows:
            raise ValueError(
                f"Expected {self.n_windows} elements, got {len(theta_data)}."
            )
        for i, ((z1, z2), (th1, th2)) in enumerate(
            zip(self.z_data, theta_data)
        ):
            if len(th1) != len(z1) or len(th2) != len(z2):
                raise ValueError(
                    f"Window {i+1}: length mismatch — "
                    f"z1={len(z1)}, θ1={len(th1)}; z2={len(z2)}, θ2={len(th2)}."
                )
        self.theta_data = [(np.asarray(th1), np.asarray(th2))
                           for th1, th2 in theta_data]
        return self

    # ------------------------------------------------------------------
    # 2D histogram construction
    # ------------------------------------------------------------------

    def _build_histograms_2d(self):
        """
        Build 2D histograms Hᵢ(r, θ) for all 2 × n_windows pseudo-windows.

        r = |z| (reflected to positive half-space, same as 1D WHAM).
        θ ∈ [theta_range[0], theta_range[1]] degrees.

        Sets
        ----
        r_bins, r_centers, r_width
        theta_bins, theta_centers, theta_width
        bin_centers_abs   (alias for r_centers — ClayPMFPlotter compat)
        histograms_2d     (2*n_windows, n_r_bins, n_theta_bins)
        biases_2d         (2*n_windows, n_r_bins)  kJ/mol  [only depends on r]
        n_snapshots       (2*n_windows,)
        """
        if self.z_data is None:
            raise RuntimeError("Call load_data() first.")
        if self.theta_data is None:
            raise RuntimeError(
                "Call load_theta_data() or set_theta_data_direct() first."
            )

        # --- r grid ---------------------------------------------------
        if self.xi_max is None:
            all_r = np.concatenate([
                np.concatenate([np.abs(z1), np.abs(z2)])
                for z1, z2 in self.z_data
            ])
            r_max = all_r.max() * 1.05
        else:
            r_max = float(self.xi_max)

        self.r_bins    = np.linspace(0.0, r_max, self.n_r_bins + 1)
        self.r_centers = 0.5 * (self.r_bins[:-1] + self.r_bins[1:])
        self.r_width   = float(self.r_bins[1] - self.r_bins[0])

        # --- theta grid -----------------------------------------------
        self.theta_bins    = np.linspace(
            self.theta_range[0], self.theta_range[1], self.n_theta_bins + 1
        )
        self.theta_centers = 0.5 * (self.theta_bins[:-1] + self.theta_bins[1:])
        self.theta_width   = float(self.theta_bins[1] - self.theta_bins[0])

        # ClayPMFPlotter compatibility alias
        self.bin_centers_abs = self.r_centers

        # --- Allocate arrays ------------------------------------------
        R      = 2 * self.n_windows
        hists  = np.zeros((R, self.n_r_bins, self.n_theta_bins), dtype=float)
        bias   = np.zeros((R, self.n_r_bins), dtype=float)
        nsn    = np.zeros(R, dtype=float)

        for i, ((z1, z2), (c1, c2), (th1, th2)) in enumerate(
            zip(self.z_data, self.window_centers, self.theta_data)
        ):
            idx1, idx2 = 2 * i, 2 * i + 1
            r1, r2   = np.abs(z1), np.abs(z2)
            rc1, rc2 = abs(c1), abs(c2)

            h1, _, _ = np.histogram2d(r1, th1, bins=[self.r_bins, self.theta_bins])
            h2, _, _ = np.histogram2d(r2, th2, bins=[self.r_bins, self.theta_bins])
            hists[idx1] = h1.astype(float)
            hists[idx2] = h2.astype(float)

            # Harmonic bias Vᵢ(r) = ½k(r − r₀ᵢ)²  [kJ/mol]
            bias[idx1] = 0.5 * self.k * (self.r_centers - rc1) ** 2
            bias[idx2] = 0.5 * self.k * (self.r_centers - rc2) ** 2

            nsn[idx1] = len(z1)
            nsn[idx2] = len(z2)

        self.histograms_2d = hists
        self.biases_2d     = bias
        self.n_snapshots   = nsn

        self._log(
            f"2D histogram grid: "
            f"r ∈ [0, {r_max:.3f}] nm ({self.n_r_bins} bins, Δr={self.r_width:.4f} nm); "
            f"θ ∈ [{self.theta_range[0]:.0f}°, {self.theta_range[1]:.0f}°] "
            f"({self.n_theta_bins} bins, Δθ={self.theta_width:.2f}°). "
            f"Total counts: {int(np.sum(hists)):,}"
        )
        return self

    # ------------------------------------------------------------------
    # WHAM solver
    # ------------------------------------------------------------------

    def run_wham_2d(self, tolerance=None, max_iter=None, fix_f=None):
        """
        Solve the 2D WHAM equations.

        **Step 1 — 1D WHAM for {fᵢ}**
        Since the bias Vᵢ depends only on r, the WHAM self-consistency
        condition reduces to a 1D problem on the marginal r-histograms:

            H̄ᵢ(r) = Σ_θ Hᵢ(r, θ)
            {fᵢ} ← standard 1D WHAM on {H̄ᵢ(r), Vᵢ(r)}

        **Step 2 — 2D unbiased probability**

            D(r)    = Σᵢ Nᵢ · exp(fᵢ − β·Vᵢ(r))
            P(r, θ) ∝ [ Σᵢ Hᵢ(r, θ) ] / D(r)

        **Step 3 — PMF**

            W(r, θ) = −kT ln P(r, θ)  [shifted so global min = 0]

        Also computes marginal PMF(r) and PMF(θ), and the signed profile
        PMF(z) (mirror reflection on [-r_max, +r_max]) for ClayPMFPlotter.

        Parameters
        ----------
        tolerance : float or None   Override instance tolerance.
        max_iter : int or None      Override instance max_iter.
        fix_f : array-like or None
            If provided, skip the 1D WHAM iteration entirely and use these
            values as the window free energies {fᵢ}.  Shape must be
            (2*n_windows,) — same layout as ``ClayPMF.f`` (CIP1/CIP2
            pseudo-windows interleaved).  Use this when the trajectory data
            used for θ is too sparse to give reliable WHAM convergence:
            pass the ``f`` array from a separately-run 1D ClayPMF on the
            full-density pullx data::

                pmf2d.run_wham_2d(fix_f=pmf.f)

            The injected values are mean-centred internally.  Stored as
            ``self._fix_f`` so that :meth:`bootstrap_errors_2d` can reuse
            them automatically.

        Returns
        -------
        pmf_2d : np.ndarray  shape (n_r_bins, n_theta_bins), kJ/mol
        """
        if tolerance is None:
            tolerance = self.tolerance
        if max_iter is None:
            max_iter = self.max_iter

        if self.histograms_2d is None:
            self._build_histograms_2d()

        R = 2 * self.n_windows

        # --- Step 1: marginal r-histograms & 1D WHAM for {fᵢ} ---
        H_1d_per_win = np.sum(self.histograms_2d, axis=2)  # (R, n_r_bins)
        H_1d_total   = np.sum(H_1d_per_win, axis=0)        # (n_r_bins,)

        # Precompute exp(−β·Vᵢ(r))  shape (R, n_r_bins)
        exp_neg_bV = np.exp(-self.beta * self.biases_2d)

        # --- Resolve f source ---
        if fix_f is not None:
            f = np.asarray(fix_f, dtype=float).copy()
            if f.shape != (R,):
                raise ValueError(
                    f"fix_f has shape {f.shape}; expected ({R},) = "
                    f"(2 × {self.n_windows} pseudo-windows)."
                )
            f -= f.mean()
            self._fix_f = f.copy()   # store for bootstrap
            self._log(
                f"  Using externally supplied f values (fix_f; skipping 1D WHAM). "
                f"Range: [{f.min():.3f}, {f.max():.3f}] kJ/mol"
            )
        else:
            # Remember any stored fix_f from a prior call is superseded
            self._fix_f = None

            self._log(
                f"1D WHAM for fᵢ: R={R}, n_r_bins={self.n_r_bins}, "
                f"tol={tolerance:.1e}, max_iter={max_iter}"
            )

            f    = self.f.copy() if self.f is not None else np.zeros(R)
            diff = np.inf

            for iteration in range(max_iter):
                f_old = f.copy()

                # denom[j] = Σᵢ Nᵢ exp(fᵢ) exp(−β·Vᵢ(rⱼ))
                denom = (self.n_snapshots * np.exp(f)) @ exp_neg_bV  # (n_r_bins,)
                denom = np.where(denom > 0.0, denom, np.inf)

                P_r  = H_1d_total / denom
                norm = np.sum(P_r) * self.r_width
                P_r /= max(norm, 1e-300)

                # exp(−fᵢ) = Σⱼ P(rⱼ) exp(−β·Vᵢ(rⱼ)) Δr
                integrals = (exp_neg_bV @ P_r) * self.r_width  # (R,)
                f = -np.log(np.where(integrals > 0, integrals, 1e-300))
                f -= f.mean()

                diff = float(np.max(np.abs(f - f_old)))
                if diff < tolerance:
                    self._log(
                        f"  Converged after {iteration+1} iterations (Δf={diff:.2e})"
                    )
                    break
            else:
                warnings.warn(
                    f"1D WHAM for fᵢ did not converge after {max_iter} iterations "
                    f"(Δf={diff:.2e}). Increase max_iter or tolerance.",
                    RuntimeWarning,
                )

        self.f = f

        # --- Step 2: 2D unbiased probability --------------------------
        # D(r) = Σᵢ Nᵢ exp(fᵢ − β·Vᵢ(r))  shape (n_r_bins,)
        D_r = (self.n_snapshots * np.exp(f)) @ exp_neg_bV  # (n_r_bins,)
        D_r = np.where(D_r > 0.0, D_r, np.inf)

        H_2d_total = np.sum(self.histograms_2d, axis=0)  # (n_r_bins, n_theta_bins)

        # P(r, θ) = H_2d_total(r, θ) / D(r)   [broadcast D_r over θ]
        P_2d = H_2d_total / D_r[:, np.newaxis]

        # Normalise: ∫∫ P dr dθ = 1
        norm_2d = np.sum(P_2d) * self.r_width * self.theta_width
        if norm_2d > 0:
            P_2d /= norm_2d
        self.P_2d = P_2d

        # --- Step 3: 2D PMF -------------------------------------------
        with np.errstate(divide='ignore', invalid='ignore'):
            pmf_2d = np.where(P_2d > 0, -np.log(P_2d) / self.beta, np.nan)
        pmf_2d -= np.nanmin(pmf_2d)
        self.pmf_2d = pmf_2d

        # --- Marginal PMF(r) ------------------------------------------
        # P(r) = Σ_θ P(r,θ) Δθ
        P_r_marg = np.sum(P_2d, axis=1) * self.theta_width
        with np.errstate(divide='ignore', invalid='ignore'):
            pmf_r = np.where(P_r_marg > 0, -np.log(P_r_marg) / self.beta, np.nan)
        pmf_r -= np.nanmin(pmf_r)
        self.pmf_abs = pmf_r
        _argmax_Pr = int(np.nanargmax(P_r_marg))
        self._log(
            f"  P_r_marg peak at r_centers[{_argmax_Pr}] = "
            f"{self.r_centers[_argmax_Pr]:.3f} nm"
            + (f"  (dist from clay = "
               f"{self.z_clay_surface - self.r_centers[_argmax_Pr]:.3f} nm)"
               if self.z_clay_surface else "")
        )

        # --- Marginal PMF(θ) ------------------------------------------
        # P(θ) = Σ_r P(r,θ) Δr
        P_th_marg = np.sum(P_2d, axis=0) * self.r_width
        with np.errstate(divide='ignore', invalid='ignore'):
            pmf_theta = np.where(P_th_marg > 0, -np.log(P_th_marg) / self.beta, np.nan)
        pmf_theta -= np.nanmin(pmf_theta)
        self.pmf_theta = pmf_theta

        # --- Signed PMF for ClayPMFPlotter compatibility --------------
        r_vals  = self.r_centers
        z_neg   = -r_vals[::-1][:-1]              # [−r_max, …, −Δr]
        z_pos   = r_vals                           # [0, r_max]
        pmf_neg = pmf_r[::-1][:-1]
        pmf_pos = pmf_r
        self._bin_centers_signed = np.concatenate([z_neg, z_pos])
        self.pmf_signed          = np.concatenate([pmf_neg, pmf_pos])

        self._log(
            f"2D PMF: [{np.nanmin(pmf_2d):.2f}, {np.nanmax(pmf_2d):.2f}] kJ/mol | "
            f"W(r): [{np.nanmin(pmf_r):.2f}, {np.nanmax(pmf_r):.2f}] kJ/mol | "
            f"W(θ): [{np.nanmin(pmf_theta):.2f}, {np.nanmax(pmf_theta):.2f}] kJ/mol"
        )
        return pmf_2d

    # ------------------------------------------------------------------
    # Bulk reference correction
    # ------------------------------------------------------------------

    def reference_to_bulk(self, bulk_fraction=0.2, enabled=True):
        """
        Set the bulk free energy to zero by shifting all stored PMFs.

        The reference value is the **median** of ``pmf_abs`` over the
        first ``bulk_fraction`` of bins (small r = far from clay = bulk).
        Using the median rather than the mean makes the correction robust
        to sparse edge bins that may have large noise.

        The shift is applied in-place to:
          * ``self.pmf_abs``    — marginal W(r)
          * ``self.pmf_2d``     — full 2D W(r, θ)
          * ``self.pmf_theta``  — marginal W(θ)
          * ``self.pmf_signed`` — signed mirror profile

        The stored ``_bulk_shift`` can be inspected or undone manually.
        Calling this method again with new parameters replaces the previous
        correction (the old shift is reversed first).

        Parameters
        ----------
        bulk_fraction : float
            Fraction of the r-axis (from the low-r / far-from-clay end)
            used to compute the reference level.  Default 0.2 (20 %).
        enabled : bool
            If False, undo any existing correction and return.  Use this
            to switch the correction off without re-running WHAM.
        """
        if self.pmf_abs is None:
            raise RuntimeError(
                "run_wham_2d() must be called before reference_to_bulk()."
            )

        # Undo previous correction first (idempotent)
        if self.bulk_correction_enabled and self._bulk_shift != 0.0:
            self.pmf_abs    += self._bulk_shift
            self.pmf_2d     += self._bulk_shift
            if self.pmf_theta  is not None: self.pmf_theta  += self._bulk_shift
            if self.pmf_signed is not None: self.pmf_signed += self._bulk_shift
            self._bulk_shift = 0.0
            self.bulk_correction_enabled = False

        self.bulk_fraction = bulk_fraction

        if not enabled:
            self._log("reference_to_bulk: correction disabled — PMFs left at min=0.")
            return

        n_bulk = max(1, int(bulk_fraction * self.n_r_bins))
        bulk_vals = self.pmf_abs[:n_bulk]
        shift = float(np.nanmedian(bulk_vals))

        self.pmf_abs    -= shift
        self.pmf_2d     -= shift
        if self.pmf_theta  is not None: self.pmf_theta  -= shift
        if self.pmf_signed is not None: self.pmf_signed -= shift

        self._bulk_shift             = shift
        self.bulk_correction_enabled = True

        self._log(
            f"reference_to_bulk: bulk_fraction={bulk_fraction:.2f} "
            f"({n_bulk} bins), median shift = {shift:+.3f} kJ/mol applied."
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _run_1d_wham_internal(self):
        """
        Run a 1D WHAM on the *current* self.z_data (trajectory-rate data,
        after load_theta_data) using the same r-grid and biases as the 2D WHAM.

        Returns
        -------
        r_centers : np.ndarray  (n_r_bins,)
        pmf_abs   : np.ndarray  (n_r_bins,)  kJ/mol, min-zeroed
        """
        if self.r_bins is None:
            self._build_histograms_2d()

        R = 2 * self.n_windows
        hists_1d = np.zeros((R, self.n_r_bins), dtype=float)
        nsn      = np.zeros(R, dtype=float)

        for i, ((z1, z2), (c1, c2)) in enumerate(
            zip(self.z_data, self.window_centers)
        ):
            idx1, idx2 = 2 * i, 2 * i + 1
            r1, r2 = np.abs(z1), np.abs(z2)
            h1, _ = np.histogram(r1, bins=self.r_bins)
            h2, _ = np.histogram(r2, bins=self.r_bins)
            hists_1d[idx1] = h1.astype(float)
            hists_1d[idx2] = h2.astype(float)
            nsn[idx1] = len(z1)
            nsn[idx2] = len(z2)

        H_total    = np.sum(hists_1d, axis=0)
        exp_neg_bV = np.exp(-self.beta * self.biases_2d)

        f    = np.zeros(R)
        diff = np.inf
        for iteration in range(self.max_iter):
            f_old = f.copy()
            denom = (nsn * np.exp(f)) @ exp_neg_bV
            denom = np.where(denom > 0, denom, np.inf)
            P_r   = H_total / denom
            norm  = np.sum(P_r) * self.r_width
            P_r  /= max(norm, 1e-300)
            integrals = (exp_neg_bV @ P_r) * self.r_width
            f = -np.log(np.where(integrals > 0, integrals, 1e-300))
            f -= f.mean()
            diff = float(np.max(np.abs(f - f_old)))
            if diff < self.tolerance:
                break

        with np.errstate(divide='ignore', invalid='ignore'):
            pmf_r = np.where(P_r > 0, -np.log(P_r) / self.beta, np.nan)
        pmf_r -= np.nanmin(pmf_r)
        return self.r_centers.copy(), pmf_r

    def wham_diagnostics(self, compare_pullx_1d=None, unit='kJ/mol', figsize=(15, 5)):
        """
        Print diagnostic statistics and plot a 3-panel figure to help
        diagnose issues with the 2D WHAM marginal W(r).

        **Panel 1** — frames per pseudo-window (checks for uneven trajectories).
        **Panel 2** — raw H_1d_total as a function of r (total biased counts).
        **Panel 3** — PMF comparison:
          * Blue solid   : 2D WHAM marginal W(r)
          * Green dashed : 1D WHAM on the **same** trajectory-rate z_data
          * Red dotted   : reference 1D WHAM (pullx-rate), if provided

        Key diagnostic logic
        --------------------
        * If green ≈ red  → trajectory data is fine; the bug is in the 2D code.
        * If green ≠ red  → trajectory data is insufficient/wrong; fix the data.
        * If green ≈ blue → 2D marginal is consistent with 1D; no bug in 2D code.

        Parameters
        ----------
        compare_pullx_1d : tuple or None
            ``(r_centers, pmf_abs)`` from a full pullx-rate 1D WHAM
            (e.g. ``(pmf.bin_centers_abs, pmf.pmf_abs)``).
        unit : str
        figsize : tuple

        Returns
        -------
        fig, axes
        """
        if self.histograms_2d is None:
            self._build_histograms_2d()
        if self.pmf_abs is None:
            self.run_wham_2d()

        # ---------- text output ----------
        print("=" * 60)
        print("WHAM Diagnostics")
        print("=" * 60)
        print(f"n_windows : {self.n_windows}  (R = {2*self.n_windows} pseudo-windows)")
        print(f"n_r_bins  : {self.n_r_bins}   n_theta_bins: {self.n_theta_bins}")
        print(f"k = {self.k} kJ/mol/nm²   T = {self.T} K")
        print()

        print("Frames per pseudo-window (CIP1 | CIP2):")
        for i in range(self.n_windows):
            n1 = int(self.n_snapshots[2 * i])
            n2 = int(self.n_snapshots[2 * i + 1])
            c1, c2 = self.window_centers[i]
            print(
                f"  win {i+1:2d}  rc={abs(c1):.3f}/{abs(c2):.3f} nm  "
                f"CIP1={n1:5d}  CIP2={n2:5d}"
            )
        print(f"Total frames: {int(self.n_snapshots.sum()):,}")
        print()

        H_1d_per_win = np.sum(self.histograms_2d, axis=2)
        H_1d_total   = np.sum(H_1d_per_win, axis=0)
        hist_total_counts = int(H_1d_total.sum())
        argmax_r = int(np.argmax(H_1d_total))
        print(f"H_1d_total: {hist_total_counts:,} counts total  "
              f"(sum n_snapshots={int(self.n_snapshots.sum()):,})")
        if hist_total_counts < self.n_snapshots.sum() * 0.9:
            print(
                "  WARNING: H_1d_total significantly less than n_snapshots — "
                "some frames may fall outside the r-grid."
            )
        print(
            f"  Peak at r_centers[{argmax_r}] = {self.r_centers[argmax_r]:.3f} nm"
        )
        print()

        if self.f is not None:
            print(f"WHAM f values (f_i, mean-centred, kJ/mol):")
            print(f"  range [{self.f.min():.3f}, {self.f.max():.3f}]  "
                  f"std={self.f.std():.3f}")
            for i in range(0, len(self.f), 2):
                print(f"  win {i//2+1:2d}: CIP1={self.f[i]:+.3f}  CIP2={self.f[i+1]:+.3f}")
            print()

        P_r_marg = np.sum(self.P_2d, axis=1) * self.theta_width
        argmax_Pr = int(np.nanargmax(P_r_marg))
        print(
            f"P_r_marg peak at r_centers[{argmax_Pr}] = "
            f"{self.r_centers[argmax_Pr]:.3f} nm  "
            f"(= distance {(self.z_clay_surface or 0.0) - self.r_centers[argmax_Pr]:.3f} nm "
            f"from clay surface)" if self.z_clay_surface else
            f"P_r_marg peak at r_centers[{argmax_Pr}] = {self.r_centers[argmax_Pr]:.3f} nm"
        )
        print()

        # ---------- run 1D WHAM on trajectory data ----------
        print("Running 1D WHAM on trajectory-rate z_data …")
        r_traj, pmf_traj = self._run_1d_wham_internal()
        argmax_1d_traj = int(np.nanargmax(-pmf_traj + np.nanmax(pmf_traj)))
        print(
            f"  → W(r) minimum at r_centers[{argmax_1d_traj}] = "
            f"{r_traj[argmax_1d_traj]:.3f} nm"
        )
        print()

        # ---------- plot ----------
        x_shift = self.z_clay_surface or 0.0

        def _to_plot_x(r_arr):
            return (x_shift - r_arr) if x_shift > 0 else r_arr

        def _shift_zero_at_bulk(pmf_arr):
            # If reference_to_bulk() has already been applied, the stored
            # arrays are corrected — just convert units.  Otherwise fall
            # back to a local median shift for the diagnostic only.
            pmf = self._to_unit(pmf_arr.copy(), unit)
            if not self.bulk_correction_enabled:
                frac = self.bulk_fraction if self.bulk_fraction is not None else 0.2
                n_b  = max(1, int(frac * len(pmf)))
                pmf -= float(np.nanmedian(pmf[:n_b]))
            return pmf

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # --- panel 1: n_snapshots ---
        ax = axes[0]
        ax.bar(range(len(self.n_snapshots)), self.n_snapshots,
               color=['steelblue' if i % 2 == 0 else 'coral'
                      for i in range(len(self.n_snapshots))])
        ax.set_xlabel('Pseudo-window index (blue=CIP1, red=CIP2)', fontsize=9)
        ax.set_ylabel('n_snapshots')
        ax.set_title('Frames per pseudo-window')
        ax.grid(True, alpha=0.3, axis='y')

        # --- panel 2: H_1d_total ---
        ax = axes[1]
        ax.plot(_to_plot_x(self.r_centers), H_1d_total, 'b-', lw=1.5)
        ax.set_xlabel('Distance from clay (nm)' if x_shift > 0 else 'r (nm)')
        ax.set_ylabel('Count')
        ax.set_title('H_1d_total (raw biased counts, traj-rate)')
        ax.grid(True, alpha=0.3)
        if x_shift > 0:
            ax.axvline(0.0, ls='--', c='grey', lw=1, alpha=0.6)

        # --- panel 3: PMF comparison ---
        ax = axes[2]
        # 2D marginal
        pmf_2d_plot = _shift_zero_at_bulk(self.pmf_abs)
        ax.plot(_to_plot_x(self.r_centers), pmf_2d_plot, 'b-', lw=2,
                label='2D WHAM marginal')
        # 1D WHAM on traj data
        pmf_traj_plot = _shift_zero_at_bulk(pmf_traj)
        ax.plot(_to_plot_x(r_traj), pmf_traj_plot, 'g--', lw=2,
                label='1D WHAM (traj-rate data)')
        # reference 1D pullx WHAM
        if compare_pullx_1d is not None:
            r1d, pmf1d_raw = compare_pullx_1d
            pmf1d_plot = _shift_zero_at_bulk(np.asarray(pmf1d_raw))
            ax.plot(_to_plot_x(np.asarray(r1d)), pmf1d_plot, 'r:', lw=2,
                    label='1D WHAM (pullx-rate ref.)')
        ax.set_xlabel('Distance from clay (nm)' if x_shift > 0 else 'r (nm)')
        ax.set_ylabel(f'W ({unit})')
        ax.set_title('PMF: 2D marginal vs 1D WHAM checks')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if x_shift > 0:
            ax.axvline(0.0, ls='--', c='grey', lw=1, alpha=0.6)

        fig.tight_layout()
        print("Diagnostic plot ready.")
        return fig, axes

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def marginalize_to_r(self):
        """Return ``(r_centers, pmf_r)`` in kJ/mol."""
        if self.pmf_abs is None:
            self.run_wham_2d()
        return self.r_centers.copy(), self.pmf_abs.copy()

    def marginalize_to_theta(self):
        """Return ``(theta_centers, pmf_theta)`` in kJ/mol."""
        if self.pmf_theta is None:
            self.run_wham_2d()
        return self.theta_centers.copy(), self.pmf_theta.copy()

    def conditional_pmf(self, r_index=None, theta_index=None):
        """
        Conditional PMF at fixed r or fixed θ.

        Parameters
        ----------
        r_index : int or None
            Index into r_centers.  Returns W(θ | r = r_centers[r_index]).
        theta_index : int or None
            Index into theta_centers.  Returns W(r | θ = theta_centers[theta_index]).

        Returns
        -------
        centers : np.ndarray   bin centres for the free variable
        cond    : np.ndarray   conditional PMF in kJ/mol (min-shifted to 0)
        """
        if self.P_2d is None:
            self.run_wham_2d()

        if r_index is not None:
            P_slice = self.P_2d[r_index, :]
            norm    = np.sum(P_slice) * self.theta_width
            P_norm  = P_slice / max(norm, 1e-300)
            with np.errstate(divide='ignore', invalid='ignore'):
                cond = np.where(P_norm > 0, -np.log(P_norm) / self.beta, np.nan)
            cond -= np.nanmin(cond)
            return self.theta_centers.copy(), cond

        elif theta_index is not None:
            P_slice = self.P_2d[:, theta_index]
            norm    = np.sum(P_slice) * self.r_width
            P_norm  = P_slice / max(norm, 1e-300)
            with np.errstate(divide='ignore', invalid='ignore'):
                cond = np.where(P_norm > 0, -np.log(P_norm) / self.beta, np.nan)
            cond -= np.nanmin(cond)
            return self.r_centers.copy(), cond

        else:
            raise ValueError("Provide r_index or theta_index.")

    def coupling_free_energy(self):
        """
        Compute ΔΔW(r, θ) = W(r, θ) − W(r) − W(θ).

        Negative values indicate that r and θ are positively coupled
        (e.g. flat orientation preferred near surface).
        Positive values indicate anticorrelation.

        Returns
        -------
        ddW : np.ndarray  shape (n_r_bins, n_theta_bins), kJ/mol
        """
        if self.pmf_2d is None:
            self.run_wham_2d()
        return (
            self.pmf_2d
            - self.pmf_abs[:, np.newaxis]
            - self.pmf_theta[np.newaxis, :]
        )

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def bootstrap_errors_2d(self, n_bootstrap=100):
        """
        Estimate 2D and marginal PMF uncertainties via Poisson bootstrap.

        Resamples every histogram bin count as Poisson(Hᵢ(r, θ)) and
        re-runs the WHAM using the converged {fᵢ} as a warm start.

        Sets
        ----
        self.pmf_2d_std      : np.ndarray  (n_r_bins, n_theta_bins)
        self.pmf_abs_std     : np.ndarray  (n_r_bins,)
        self.pmf_signed_std  : np.ndarray  (2*n_r_bins − 1,)

        Returns
        -------
        pmf_2d_std : np.ndarray
        pmf_abs_std : np.ndarray
        """
        if self.pmf_2d is None:
            self.run_wham_2d()

        self._log(f"Bootstrap 2D: {n_bootstrap} samples…")

        hists_orig   = self.histograms_2d.copy()
        f_init       = self.f.copy()
        verbose_orig = self.verbose        # silence inner run_wham_2d calls

        pmf_2d_samples  = np.zeros((n_bootstrap, self.n_r_bins, self.n_theta_bins))
        pmf_abs_samples = np.zeros((n_bootstrap, self.n_r_bins))

        # If fix_f was used in run_wham_2d(), reuse it for every bootstrap
        # sample so that the sparse-traj WHAM is never invoked.
        _fix_f = getattr(self, '_fix_f', None)

        self.verbose = False
        try:
            for b in tqdm(
                range(n_bootstrap), desc='Bootstrap 2D', disable=not verbose_orig
            ):
                self.histograms_2d = np.random.poisson(hists_orig).astype(float)
                self.f = f_init.copy()    # warm start (ignored when fix_f is set)
                self.run_wham_2d(fix_f=_fix_f)
                pmf_2d_samples[b]  = self.pmf_2d
                pmf_abs_samples[b] = self.pmf_abs
        finally:
            # Always restore original state even if an error occurs
            self.verbose       = verbose_orig
            self.histograms_2d = hists_orig
            self.f             = f_init
        self.run_wham_2d(fix_f=_fix_f)

        self.pmf_2d_std  = np.std(pmf_2d_samples, axis=0)
        self.pmf_abs_std = np.std(pmf_abs_samples, axis=0)

        # Signed std for ClayPMFPlotter compatibility
        std_neg = self.pmf_abs_std[::-1][:-1]
        self.pmf_signed_std = np.concatenate([std_neg, self.pmf_abs_std])

        self._log(
            f"Bootstrap done. "
            f"Max 2D σ = {np.nanmax(self.pmf_2d_std):.3f} kJ/mol; "
            f"max W(r) σ = {np.nanmax(self.pmf_abs_std):.3f} kJ/mol."
        )
        return self.pmf_2d_std, self.pmf_abs_std

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_results(self, outdir='.', prefix='pmf2d'):
        """
        Save all PMF results to ``<outdir>/<prefix>.npz``.

        Arrays saved
        ------------
        r_centers, theta_centers, pmf_2d, pmf_abs, pmf_theta, P_2d, f
        (+ pmf_2d_std, pmf_abs_std if bootstrap was run)

        Returns
        -------
        path : str   Full path to the .npz file.
        """
        if self.pmf_2d is None:
            raise RuntimeError("Run run_wham_2d() first.")

        os.makedirs(outdir, exist_ok=True)
        out = os.path.join(outdir, f'{prefix}.npz')

        save_kw = dict(
            r_centers     = self.r_centers,
            theta_centers = self.theta_centers,
            pmf_2d        = self.pmf_2d,
            pmf_abs       = self.pmf_abs,
            pmf_theta     = self.pmf_theta,
            P_2d          = self.P_2d,
            f             = self.f,
        )
        if self.pmf_2d_std is not None:
            save_kw['pmf_2d_std']  = self.pmf_2d_std
            save_kw['pmf_abs_std'] = self.pmf_abs_std

        np.savez(out, **save_kw)
        self._log(f"Results saved → {out}")
        return out

