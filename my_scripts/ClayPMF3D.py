#!/usr/bin/env python3
"""
ClayPMF3D.py
============
3D WHAM analysis: distance × tilt-angle × cation-coordination PMF for
CIP on montmorillonite clay.

Coordinates
-----------
r = |z|    : distance from clay COM (nm) — same biased coordinate as 1D/2D WHAM.
             r = 0 corresponds to the clay centre-of-mass.  The clay surface
             is at r ≈ z_clay_surface (detected via detect_clay_surface()).
θ          : tilt angle of CIP aromatic ring vs. clay surface normal (degrees)
             θ = 0°  → ring lying flat (parallel to surface)
             θ = 90° → ring edge-on (perpendicular to surface)
n_cat      : number of cations (Na⁺, K⁺, Mg²⁺, Ca²⁺ or mixture) within
             cation_cutoff nm of the CIP centre-of-mass.  Integer-valued;
             binned with unit-width bins centred on 0, 1, 2, …

Mathematical basis
------------------
The umbrella bias acts only on r, so the WHAM free-energy offsets {f_i}
satisfy the standard 1D equations on the marginal r-histograms:

    H̄_i(r) = Σ_{θ,n} H_i(r, θ, n)
    D(r)    = Σ_i N_i exp(f_i − β V_i(r))   where V_i(r) = ½k(r − r₀ᵢ)²
    P(r,θ,n) ∝ H_total(r,θ,n) / D(r)
    W(r,θ,n) = −kT ln P(r,θ,n)   [shifted so bulk minimum = 0]

All energies are in kJ/mol.  Temperature default 298.15 K.

Key differences from WHAM_3D_Implementation.md design document
--------------------------------------------------------------
1. kJ/mol throughout (doc used kcal/mol).  K_B = 8.314462618e-3 kJ/(mol·K).
2. No spurious ×1000 factor anywhere.
3. fix_f: the WHAM iteration skips starvation NaNs (numerator ≤ 0 → keep
   f_i unchanged that iteration, identical to ClayPMF2D practice).
4. reference_to_bulk() zeros the bulk level instead of shifting the
   minimum to zero.
5. Fully vectorised 1D-WHAM inner loop (numpy broadcasting).
6. Cation is generalised via cation_name ('Na', 'K', 'Mg', 'Ca', or list).

Usage
-----
    from ClayPMF3D import ClayPMF3D

    pmf3d = ClayPMF3D(
        umbrella_dir='Umbrella/',
        n_windows=30,
        k=1000.0,          # kJ/(mol·nm²)
        T=298.15,
        n_r_bins=50,
        n_theta_bins=18,
        cation_range=(0, 5),
        cation_name='Na',
        cation_cutoff=0.5, # nm
    )
    pmf3d.load_data()
    pmf3d.detect_clay_surface('md.gro')
    pmf3d.define_selections({
        'CIP_ring': {'cip_ring': 'resname api and (name C4 C4a C8a C1 C2 C3 C5 N1)'},
    })
    pmf3d.load_trajectory_data(
        traj_files=[f'Umbrella/traj{i}.xtc' for i in range(1, 31)],
        topology='md.tpr',
    )
    pmf3d.run_wham_3d()
    pmf3d.reference_to_bulk()
    pmf3d.print_results()
    pmf3d.plot_marginals()
"""

import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

try:
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import distance_array as _mda_dist
    _MDA_AVAILABLE = True
except ImportError:
    _MDA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Common GROMACS residue names for each cation type.
#: Override via the cation_selection argument in load_trajectory_data().
_CATION_RESNAMES = {
    'Na': ['NA', 'SOD', 'NA+'],
    'K':  ['K', 'POT', 'K+'],
    'Mg': ['MG', 'MG2', 'MG+2'],
    'Ca': ['CA', 'CAL', 'CA+2'],
}

#: Volume per molecule at 1 mol/L standard concentration (nm³).
_V_1M_NM3 = 1.0e-3 / 6.022140857e23 * 1e27   # ≈ 1.6605 nm³


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ClayPMF3D:
    """
    3D WHAM: distance × tilt-angle × cation-coordination PMF.

    Parameters
    ----------
    umbrella_dir : str
        Directory containing pullx*.xvg files.
    n_windows : int
        Number of umbrella windows. Default 30.
    k : float
        Harmonic spring constant (kJ/mol/nm²). Default 1000.0.
    T : float
        Temperature (K). Default 298.15.
    equil_skip_ps : float
        Equilibration time to discard per window (ps). Default 1000.0.
    n_r_bins : int
        Bins for r = |z|. Default 50.
    n_theta_bins : int
        Bins for tilt angle θ. Default 18.
    theta_range : tuple
        (min_deg, max_deg). Default (0.0, 90.0).
    cation_range : tuple of int
        (min_n, max_n) for cation coordination number.  Integer-valued;
        one bin per integer value.  Default (0, 5) → 6 bins.
    cation_name : str or list of str
        Cation type(s): 'Na', 'K', 'Mg', 'Ca', or a list such as
        ['Na', 'Ca'] for a mixture.  Used for labels and MDAnalysis
        selection auto-build.
    cation_cutoff : float
        Distance cutoff (nm) from CIP centre for counting coordinated
        cations.  Default 0.5.
    xi_min, xi_max : float or None
        Override r-grid bounds (nm).
    pullx_prefix : str
        Filename prefix for pull-position files. Default 'pullx'.
    tolerance : float
        WHAM convergence tolerance. Default 1e-6.
    max_iter : int
        Maximum WHAM iterations. Default 50000.
    verbose : bool
        Print progress. Default True.
    """

    K_B = 8.314462618e-3   # kJ / (mol·K)
    V_1M = _V_1M_NM3       # nm³ at 1 mol/L

    def __init__(
        self,
        umbrella_dir,
        n_windows=30,
        k=1000.0,
        T=298.15,
        equil_skip_ps=1000.0,
        n_r_bins=50,
        n_theta_bins=18,
        theta_range=(0.0, 90.0),
        cation_range=(0, 5),
        cation_name='Na',
        cation_cutoff=0.5,
        cip_group=None,
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
        self.theta_range   = tuple(float(x) for x in theta_range)
        self.cation_range  = (int(cation_range[0]), int(cation_range[1]))
        self.cation_name   = cation_name
        self.cation_cutoff = float(cation_cutoff)
        self.cip_group  = cip_group  # None = total molecule; str key from CIP_parts
        self.xi_min        = xi_min
        self.xi_max        = xi_max
        self.pullx_prefix  = pullx_prefix
        self.tolerance     = tolerance
        self.max_iter      = int(max_iter)
        self.verbose       = verbose

        self.beta = 1.0 / (self.K_B * self.T)   # mol/kJ

        # Derived cation grid: one bin per integer in [cation_range[0], cation_range[1]]
        n_min, n_max = self.cation_range
        self.cation_bins    = np.arange(n_min - 0.5, n_max + 1.5, 1.0)
        self.n_cation_bins  = len(self.cation_bins) - 1   # = n_max - n_min + 1
        self.cation_centers = np.arange(n_min, n_max + 1, dtype=float)
        self.cation_width   = 1.0

        # Custom MDAnalysis selections (set by define_selections)
        self.custom_selections = {}

        # --- set by load_data() ---
        self.z_data         = None   # list of (z1_prod, z2_prod) per window
        self.t_data         = None   # list of t_prod per window [ps]
        self.window_centers = None   # list of (c1, c2) per window [nm]

        # --- set by detect_clay_surface() ---
        self.z_clay_surface = None

        # --- set by load_trajectory_data() or set_3d_data_direct() ---
        self.theta_data = None   # list of (theta1, theta2) per window [deg]
        self.ncat_data  = None   # list of (ncat1, ncat2) per window [int]

        # --- set by _build_histograms_3d() ---
        self.r_bins         = None   # (n_r_bins+1,)
        self.r_centers      = None   # (n_r_bins,)
        self.r_width        = None
        self.theta_bins     = None   # (n_theta_bins+1,)
        self.theta_centers  = None   # (n_theta_bins,)
        self.theta_width    = None
        self.histograms_3d  = None   # (2*n_windows, n_r_bins, n_theta_bins, n_cation_bins)
        self.biases_3d      = None   # (2*n_windows, n_r_bins) kJ/mol
        self.n_snapshots    = None   # (2*n_windows,)

        # --- set by run_wham_3d() ---
        self.f              = None   # WHAM free energies (2*n_windows,)
        self.P_3d           = None   # unbiased P(r,θ,n)
        self.pmf_3d         = None   # W(r,θ,n) kJ/mol
        # 1D marginals:
        self.pmf_r          = None   # W(r)
        self.pmf_theta      = None   # W(θ)
        self.pmf_cation     = None   # W(n_cat)
        # ClayPMFPlotter compatibility:
        self.pmf_abs             = None
        self.bin_centers_abs     = None
        self.pmf_signed          = None
        self._bin_centers_signed = None
        self.pmf_abs_std         = None
        self.pmf_signed_std      = None

        # --- set by bootstrap_errors_3d() ---
        self.pmf_3d_std    = None
        self.pmf_r_std     = None

        # --- set by reference_to_bulk() ---
        self.bulk_fraction           = None
        self.bulk_correction_enabled = False
        self._bulk_shift             = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _to_unit(self, arr, unit):
        """Convert PMF array from kJ/mol to requested unit."""
        if unit == 'kJ/mol':
            return arr.copy()
        elif unit == 'kcal/mol':
            return arr.copy() / 4.184
        elif unit in ('kT', 'kBT'):
            return arr.copy() * self.beta
        raise ValueError(f"Unknown unit '{unit}'. Use 'kJ/mol', 'kcal/mol', or 'kT'.")

    @staticmethod
    def _tilt_from_coords(coords):
        """
        Tilt angle between aromatic-ring plane normal and z-axis (degrees).
        θ = 0°: ring flat; θ = 90°: ring edge-on.  Uses SVD.
        """
        centred = coords - coords.mean(axis=0)
        _, _, Vt = np.linalg.svd(centred, full_matrices=False)
        normal = Vt[-1]
        cos_theta = float(np.clip(abs(normal[2]) / np.linalg.norm(normal), 0.0, 1.0))
        return float(np.degrees(np.arccos(cos_theta)))

    @staticmethod
    def _build_cation_selection(cation_name):
        """
        Build an MDAnalysis selection string from cation name(s).

        Parameters
        ----------
        cation_name : str or list of str
            'Na', 'K', 'Mg', 'Ca', or a list for mixtures.

        Returns
        -------
        str : MDAnalysis selection string
        """
        if isinstance(cation_name, str):
            names = [cation_name]
        else:
            names = list(cation_name)
        parts = []
        for n in names:
            resnames = _CATION_RESNAMES.get(n, [n])
            inner = ' or '.join(f'resname {r}' for r in resnames)
            parts.append(f'({inner})')
        return ' or '.join(parts)

    @property
    def _cation_label(self):
        """Human-readable cation label for plot titles.

        Returns e.g. ``'Na'`` for total coordination, or ``'Na (carboxylate)'``
        when *cip_group* is set.
        """
        base = '/'.join(self.cation_name) if isinstance(self.cation_name, (list, tuple)) else str(self.cation_name)
        if self.cip_group is not None:
            return f'{base} ({self.cip_group})'
        return base

    # ------------------------------------------------------------------
    # Selection management (mirrors ClayPMF2D)
    # ------------------------------------------------------------------

    def define_selections(self, selections_dict):
        """
        Register named MDAnalysis selection strings by category.

        Parameters
        ----------
        selections_dict : dict
            Nested ``{category: {name: sel_string}}`` or flat ``{name: sel_string}``.
        """
        first_value = next(iter(selections_dict.values()))
        is_nested = isinstance(first_value, dict)
        if is_nested:
            for cat, sels in selections_dict.items():
                self.custom_selections.setdefault(cat, {}).update(sels)
        else:
            self.custom_selections.setdefault('default', {}).update(selections_dict)
        if self.verbose:
            print("Selections registered:")
            for cat, sels in self.custom_selections.items():
                for name in sels:
                    print(f"  [{cat}] {name}")
        return self

    def sel(self, name):
        """Return registered selection string by name."""
        for sels in self.custom_selections.values():
            if name in sels:
                return sels[name]
        available = [n for sels in self.custom_selections.values() for n in sels]
        raise KeyError(
            f"Selection '{name}' not found. "
            f"Available: {', '.join(available) or '(none)'}"
        )

    def _resolve_sel(self, sel_or_name):
        for sels in self.custom_selections.values():
            if sel_or_name in sels:
                return sels[sel_or_name]
        return sel_or_name

    # ------------------------------------------------------------------
    # Data loading: z from pullx  (identical to ClayPMF2D)
    # ------------------------------------------------------------------

    @staticmethod
    def _read_pullx(filepath):
        # Pre-filter lines: skip XVG comment/separator lines and any row that
        # has fewer than 3 columns (e.g. a bare timestamp written by GROMACS
        # before a mid-row crash).  This is more robust than relying on
        # np.loadtxt's column-count consistency check.
        import io as _io
        good_lines = []
        with open(filepath, 'r') as _fh:
            for _line in _fh:
                _s = _line.strip()
                if not _s or _s[0] in ('#', '@', '&'):
                    continue
                if len(_s.split()) >= 3:
                    good_lines.append(_s)
        if not good_lines:
            # File exists but has no usable data rows (e.g. simulation crashed
            # before writing any pull-x output).  Return empty arrays so
            # load_data() can detect this as a trailing window.
            empty = np.empty(0, dtype=float)
            return empty, empty, empty
        data = np.loadtxt(_io.StringIO('\n'.join(good_lines)), ndmin=2)
        if data.shape[0] == 0 or data.shape[1] < 3:
            empty = np.empty(0, dtype=float)
            return empty, empty, empty
        return data[:, 0], data[:, 1], data[:, 2]

    def load_data(self):
        """
        Read pullx*.xvg files, discard equilibration, store z-trajectories.

        Returns
        -------
        self
        """
        self._log(f"Loading {self.n_windows} pullx files from:\n  {self.umbrella_dir}")
        _rep_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(self.umbrella_dir))))
        self.z_data         = []
        self.t_data         = []
        self.window_centers = []

        # First pass: collect every window's data (or None if empty after equil skip).
        import warnings
        _raw = []   # list of (i, c1, c2, z1_prod, z2_prod, t_prod) or None-entry
        for i in range(1, self.n_windows + 1):
            fpath = os.path.join(self.umbrella_dir, f'{self.pullx_prefix}{i}.xvg')
            if not os.path.isfile(fpath):
                raise FileNotFoundError(f"Pull-x file not found: {fpath}")
            time, z1, z2 = self._read_pullx(fpath)
            n_cf = min(25, len(z1))
            c1 = float(np.mean(z1[:n_cf])) if n_cf > 0 else float('nan')
            c2 = float(np.mean(z2[:n_cf])) if n_cf > 0 else float('nan')
            mask   = time >= self.equil_skip_ps
            t_prod = time[mask]
            if len(t_prod) == 0:
                _last_t = float(time[-1]) if len(time) > 0 else 0.0
                _raw.append((i, c1, c2, None, None, None, _last_t))
            else:
                _raw.append((i, c1, c2, z1[mask], z2[mask], t_prod, None))

        # Find the last window that has actual data; everything after it is
        # "trailing" (simulation likely crashed due to extreme energy at the
        # closest clay-CIP separation).  Those are safe to drop silently.
        last_good = max(
            (idx for idx, entry in enumerate(_raw) if entry[3] is not None),
            default=-1,
        )
        if last_good == -1:
            raise ValueError(
                "No windows have data after equil_skip_ps="
                f"{self.equil_skip_ps} ps."
            )

        for idx, entry in enumerate(_raw):
            i, c1, c2, z1_prod, z2_prod, t_prod, traj_len = entry
            if z1_prod is None:
                if idx > last_good:
                    # trailing crashed window — warn and skip
                    self._log(
                        f"  [{_rep_id}] SKIP Window {i}: trajectory too short "
                        f"({traj_len:.1f} ps ≤ equil_skip_ps={self.equil_skip_ps} ps). "
                        f"Likely simulation crash (CIP–clay energy diverged). "
                        f"Dropped as trailing window."
                    )
                else:
                    # gap in the middle — this is a real problem
                    raise ValueError(
                        f"[{_rep_id}] Window {i}: no frames after equil_skip_ps="
                        f"{self.equil_skip_ps} ps (trajectory length "
                        f"= {traj_len:.1f} ps), and it is not a trailing "
                        f"window (window {_raw[last_good][0]} has data). "
                        f"Cannot proceed with a gap in the umbrella sequence."
                    )
                continue
            self.z_data.append((z1_prod, z2_prod))
            self.t_data.append(t_prod)
            self.window_centers.append((c1, c2))
            if self.verbose and i % 5 == 0:
                self._log(
                    f"  win {i:2d}: c1={c1:+.3f} nm  c2={c2:+.3f} nm  "
                    f"n_frames={len(t_prod):,}"
                )

        # Preserve original pullx data for interpolation in load_trajectory_data()
        self._pullx_z_data = list(self.z_data)
        self._pullx_t_data = list(self.t_data)
        n_loaded  = len(self.z_data)
        n_skipped = self.n_windows - n_loaded
        if n_skipped:
            self._log(
                f"Done — {n_loaded}/{self.n_windows} windows loaded "
                f"({n_skipped} trailing window(s) dropped)."
            )
            # Update n_windows so downstream methods (load_trajectory_data,
            # run_wham_3d, cache checks) all use the actual window count.
            self.n_windows = n_loaded
        else:
            self._log(
                f"Done — {self.n_windows} windows × 2 CIPs = "
                f"{2 * self.n_windows} pseudo-windows total."
            )
        return self

    # ------------------------------------------------------------------
    # Clay surface detection (identical to ClayPMF2D)
    # ------------------------------------------------------------------

    def detect_clay_surface(
        self,
        structure_file,
        clay_selection='resname MMT',
        surface_pct=95,
    ):
        """
        Detect the outermost clay layer z-position.

        Sets ``self.z_clay_surface`` (nm).

        Returns
        -------
        z_surface : float  [nm]
        """
        if not _MDA_AVAILABLE:
            raise ImportError("MDAnalysis required for detect_clay_surface().")
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
        self._log(f"Clay surface: z_clay_surface = {z_surface:.3f} nm")
        return z_surface

    # ------------------------------------------------------------------
    # 3D trajectory data loading  (new: θ + n_cation)
    # ------------------------------------------------------------------

    def load_trajectory_data(
        self,
        traj_files,
        topology,
        cip_selection=None,
        cation_selection=None,
        cip_cation_center_selection=None,
        stride=1,
        cache_file=None,
        save_cache=True,
        force_recompute=False,
    ):
        """
        Compute CIP tilt-angle (θ) and cation coordination (n_cat) from
        GROMACS trajectories.

        For each window, reads the trajectory, computes θ (SVD of ring
        atoms) and n_cat (number of cations within ``cation_cutoff`` nm of
        the CIP centre) at every production frame, then linearly
        interpolates z-values from the pullx file onto those frame times.

        The two CIP molecules are identified by splitting the CIP selection
        by residue ID (ascending order), exactly as in ClayPMF2D.

        Parameters
        ----------
        traj_files : list of str
            One trajectory (.xtc or .trr) per window.
        topology : str or list of str
            Topology file(s) (.tpr or .gro).
        cip_selection : str or None
            MDAnalysis selection for CIP ring atoms (≥ 3 per residue).
            If None, uses registered 'cip_ring' selection.
        cation_selection : str or None
            MDAnalysis selection for cation atoms.  If None, auto-built
            from ``self.cation_name`` using the internal resname table.
            Override this when your force field uses non-standard resnames
            (e.g. 'resname NA and not resname MMT').
        cip_cation_center_selection : str or None
            Atoms of EACH CIP used as the reference centre for counting
            nearby cations.  If None, the full CIP molecule COM is used.
            Example: ``'resname api and (name O1 O2)'`` for carboxylate
            oxygens only.
        stride : int
            Use every stride-th trajectory frame. Default 1.
        cache_file : str or None
            Path to .npz cache.  Default: <umbrella_dir>/theta_ncat_cache_s<stride>.npz
        save_cache : bool
        force_recompute : bool

        Sets
        ----
        self.theta_data, self.ncat_data, self.z_data, self.t_data

        Returns
        -------
        self
        """
        if self.z_data is None:
            raise RuntimeError("Call load_data() before load_trajectory_data().")
        rep_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(self.umbrella_dir))))
        if len(traj_files) < self.n_windows:
            raise ValueError(
                f"[{rep_id}] Expected {self.n_windows} traj_files, "
                f"got {len(traj_files)}."
            )
        if len(traj_files) > self.n_windows:
            # Trailing windows were dropped by load_data; trim the file list.
            n_extra = len(traj_files) - self.n_windows
            self._log(
                f"  [{rep_id}] Trimming traj_files from {len(traj_files)} → "
                f"{self.n_windows} to match loaded pullx windows "
                f"({n_extra} trailing file(s) ignored)."
            )
            traj_files = list(traj_files)[: self.n_windows]

        if cache_file is None:
            _grp_tag = f"_{self.cip_group}" if self.cip_group else ""
            cache_file = os.path.join(
                self.umbrella_dir, f"theta_ncat_cache{_grp_tag}_s{stride}.npz"
            )

        # --- try loading from cache ---
        if save_cache and not force_recompute and os.path.isfile(cache_file):
            self._log(f"Loading θ/n_cat from cache: {cache_file}")
            try:
                data  = np.load(cache_file, allow_pickle=False)
                n_win = int(data['n_windows'])
                if n_win > self.n_windows:
                    raise ValueError(
                        f"Cache has {n_win} windows but n_windows={self.n_windows}; "
                        "cache appears corrupt or from a different system."
                    )
                if n_win < self.n_windows:
                    # Trailing windows were dropped during the original trajectory
                    # loading (e.g. a crashed simulation window had no frames).
                    # Accept the cache and align n_windows / window_centers to match.
                    self._log(
                        f"  Cache has {n_win} windows < n_windows={self.n_windows}; "
                        f"trimming to {n_win} (trailing-window drop preserved in cache)."
                    )
                    self.n_windows = n_win
                    if len(self.window_centers) > n_win:
                        self.window_centers = self.window_centers[:n_win]
                self.theta_data = []
                self.ncat_data  = []
                new_z = []
                new_t = []
                for i in range(n_win):
                    self.theta_data.append((data[f'th1_{i}'], data[f'th2_{i}']))
                    self.ncat_data.append((data[f'nc1_{i}'], data[f'nc2_{i}']))
                    new_z.append((data[f'z1_{i}'], data[f'z2_{i}']))
                    new_t.append(data[f't_{i}'])
                self.z_data = new_z
                self.t_data = new_t
                self._log(f"Loaded from cache ({n_win} windows).")
                return self
            except Exception as _e:
                self._log(f"  Cache load failed ({_e}); recomputing…")

        if not _MDA_AVAILABLE:
            raise ImportError("MDAnalysis is required for load_trajectory_data().")

        # Resolve CIP ring selection
        if cip_selection is None:
            cip_selection = self.sel('cip_ring')
        else:
            cip_selection = self._resolve_sel(cip_selection)

        # Resolve cation selection
        if cation_selection is None:
            cation_selection = self._build_cation_selection(self.cation_name)
        self._log(f"  Cation selection: {cation_selection}")

        # Auto-derive cip_cation_center_selection from self.cip_group if not given
        # _use_min_dist=True  → count cations within cutoff of ANY atom in the group
        # _use_min_dist=False → use centre-of-mass of ctr atoms (backward-compatible default)
        _use_min_dist = False
        if cip_cation_center_selection is None and self.cip_group is not None:
            try:
                cip_cation_center_selection = self.sel(self.cip_group)
                _use_min_dist = True
                self._log(
                    f"  CIP functional group '{self.cip_group}' → "
                    f"min-distance to group atoms: {cip_cation_center_selection}"
                )
            except KeyError:
                raise KeyError(
                    f"cip_group='{self.cip_group}' not found in registered selections. "
                    f"Available keys: {list(self.custom_selections.keys())}. "
                    "Register it via define_selections() before calling load_trajectory_data()."
                )

        if isinstance(topology, str):
            topology = [topology] * len(traj_files)

        # Sort traj_files (and topology) numerically by the trailing integer in
        # the filename so that e.g. umbrella31.xtc comes *after* umbrella9.xtc
        # rather than before it (which is what alphabetical/glob order gives).
        import re as _re
        def _num_key(path):
            m = _re.search(r'(\d+)[^/\\]*$', path)
            return int(m.group(1)) if m else 0
        _paired = sorted(zip(traj_files, topology), key=lambda x: _num_key(x[0]))
        traj_files, topology = [p[0] for p in _paired], [p[1] for p in _paired]

        # ── Sanity-check table: show exactly which pullx is paired with which traj ──
        _pullx_files = []
        if hasattr(self, '_pullx_t_data') and hasattr(self, '_pullx_z_data'):
            # Reconstruct pullx filenames from umbrella_dir using the same numerical sort
            _px_candidates = sorted(
                [
                    os.path.join(self.umbrella_dir, f)
                    for f in os.listdir(self.umbrella_dir)
                    if f.startswith(self.pullx_prefix) and f.endswith('.xvg')
                ],
                key=_num_key,
            )
            _pullx_files = _px_candidates[: self.n_windows]

        _col = 28  # column width for filenames
        print(f"\n{'Win':>4}  {'pullx file':<{_col}}  {'trajectory file'}")
        print(f"{'----':>4}  {'-'*_col}  {'-'*_col}")
        for _wi in range(self.n_windows):
            _px = os.path.basename(_pullx_files[_wi]) if _wi < len(_pullx_files) else '—'
            _tx = os.path.basename(traj_files[_wi]) if _wi < len(traj_files) else '—'
            print(f"  {_wi+1:>2}  {_px:<{_col}}  {_tx}")
        print()
        # ─────────────────────────────────────────────────────────────────────────

        cutoff_A = self.cation_cutoff * 10.0   # nm → Angstroms

        self._log(
            f"Computing θ and n_{self._cation_label} from "
            f"{self.n_windows} trajectories (stride={stride})…"
        )

        self.theta_data = []
        self.ncat_data  = []
        new_z_data      = []
        new_t_data      = []

        for i, (traj_file, top_file) in enumerate(zip(traj_files, topology)):
            if not os.path.isfile(traj_file):
                raise FileNotFoundError(f"Trajectory not found: {traj_file}")
            if not os.path.isfile(top_file):
                raise FileNotFoundError(f"Topology not found: {top_file}")

            u = mda.Universe(top_file, traj_file)

            # NoJump transformation to prevent PBC discontinuities
            try:
                from MDAnalysis.transformations import NoJump
                u.trajectory.add_transformations(NoJump())
            except ImportError:
                if i == 0:
                    self._log(
                        "  WARNING: MDAnalysis.transformations.NoJump not available "
                        "(upgrade MDAnalysis ≥ 2.1). Proceeding without it."
                    )

            all_cip = u.select_atoms(cip_selection)
            resids  = sorted(set(all_cip.resids))
            if len(resids) != 2:
                raise ValueError(
                    f"cip_selection matched atoms from {len(resids)} residue(s) "
                    f"{resids}; expected exactly 2."
                )

            atoms1 = all_cip.select_atoms(f'resid {resids[0]}')
            atoms2 = all_cip.select_atoms(f'resid {resids[1]}')
            mol1   = u.select_atoms(f'resid {resids[0]}')
            mol2   = u.select_atoms(f'resid {resids[1]}')

            for ag in (atoms1, atoms2):
                if len(ag) < 3:
                    raise ValueError(
                        f"resid {ag.resids[0]} matched {len(ag)} ring atom(s); "
                        "need ≥ 3 for plane fitting."
                    )

            # CIP cation-centre atom groups
            if cip_cation_center_selection is not None:
                cc_sel = self._resolve_sel(cip_cation_center_selection)
                ctr1 = u.select_atoms(f'({cc_sel}) and resid {resids[0]}')
                ctr2 = u.select_atoms(f'({cc_sel}) and resid {resids[1]}')
                if len(ctr1) == 0 or len(ctr2) == 0:
                    raise ValueError(
                        f"cip_cation_center_selection '{cc_sel}' matched "
                        f"{len(ctr1)}/{len(ctr2)} atoms for CIP1/CIP2."
                    )
            else:
                ctr1 = mol1
                ctr2 = mol2

            cations = u.select_atoms(cation_selection)
            if len(cations) == 0 and i == 0:
                warnings.warn(
                    f"Cation selection '{cation_selection}' matched 0 atoms. "
                    "n_cation will be 0 for all frames.",
                    RuntimeWarning,
                )

            if i == 0:
                self._log(
                    f"  resid {resids[0]} → CIP1 ({len(atoms1)} ring atoms, "
                    f"{len(mol1)} total), "
                    f"resid {resids[1]} → CIP2 ({len(atoms2)} ring atoms, "
                    f"{len(mol2)} total)"
                )
                self._log(f"  Cation atoms in topology: {len(cations)}")

            t_frames = []
            th1_list, th2_list = [], []
            nc1_list, nc2_list = [], []

            for ts in u.trajectory[::stride]:
                if ts.time < self.equil_skip_ps:
                        continue
                t_frames.append(ts.time)
                mol1.unwrap()
                mol2.unwrap()

                # Tilt angles
                th1_list.append(self._tilt_from_coords(atoms1.positions))
                th2_list.append(self._tilt_from_coords(atoms2.positions))

                # Cation coordination numbers
                if len(cations) > 0:
                    box = ts.dimensions
                    pos_cat = cations.positions
                    if _use_min_dist:
                        # Min distance from any atom in functional group to each cation
                        d1 = _mda_dist(ctr1.positions, pos_cat, box=box)  # (n_grp1, n_cat)
                        d2 = _mda_dist(ctr2.positions, pos_cat, box=box)
                        nc1_list.append(int(np.sum(d1.min(axis=0) <= cutoff_A)))
                        nc2_list.append(int(np.sum(d2.min(axis=0) <= cutoff_A)))
                    else:
                        # Centre-of-mass distance (default / backward-compatible)
                        pos_c1 = ctr1.center_of_mass().reshape(1, 3)
                        pos_c2 = ctr2.center_of_mass().reshape(1, 3)
                        dists1 = _mda_dist(pos_c1, pos_cat, box=box)[0]
                        dists2 = _mda_dist(pos_c2, pos_cat, box=box)[0]
                        nc1_list.append(int(np.sum(dists1 <= cutoff_A)))
                        nc2_list.append(int(np.sum(dists2 <= cutoff_A)))
                else:
                    nc1_list.append(0)
                    nc2_list.append(0)

            if len(t_frames) == 0:
                # ── Diagnose why: reset timestamps vs true empty window ────────
                _all_times = np.array([ts.time for ts in u.trajectory[::stride]])
                _max_t = float(_all_times[-1]) if len(_all_times) > 0 else 0.0
                _hint = (
                    f"  NOTE: All {len(_all_times)} frames have timestamps "
                    f"≤ {_max_t:.1f} ps (< equil_skip_ps={self.equil_skip_ps} ps). "
                    f"This likely means GROMACS reset timestamps to 0 in a "
                    f"continuation/restart run. "
                    f"To fix: re-generate the trajectory with continuous time stamps "
                    f"(gmx trjcat -settime or rerun with correct -t0), "
                    f"OR lower equil_skip_ps below {_max_t:.1f} ps for this replicate."
                    if _max_t < self.equil_skip_ps
                    else f"  The file has {len(_all_times)} frames but none pass "
                         f"the equil_skip_ps={self.equil_skip_ps} ps filter."
                )
                is_last_window = (i == self.n_windows - 1)
                if is_last_window:
                    # Last window is empty — treat as a trailing crashed window,
                    # silently drop and stop.
                    n_prev = self.n_windows
                    self.n_windows = i
                    if len(self.window_centers) > i:
                        self.window_centers = self.window_centers[:i]
                    self._log(
                        f"  [{rep_id}] Trailing window {i+1} has no frames after "
                        f"equil_skip_ps={self.equil_skip_ps} ps — dropped "
                        f"(n_windows {n_prev} → {i})."
                    )
                    break
                raise RuntimeError(
                    f"[{rep_id}] Window {i+1}/{self.n_windows} has no trajectory "
                    f"frames after equil_skip_ps={self.equil_skip_ps} ps "
                    f"(traj: {os.path.basename(traj_file)}). "
                    + _hint
                )

            t_traj = np.array(t_frames)
            theta1 = np.array(th1_list)
            theta2 = np.array(th2_list)
            ncat1  = np.array(nc1_list, dtype=int)
            ncat2  = np.array(nc2_list, dtype=int)

            # Interpolate z1, z2 from pullx onto trajectory frame times
            t_pullx = self._pullx_t_data[i] if hasattr(self, '_pullx_t_data') else self.t_data[i]
            z1_pullx, z2_pullx = (
                self._pullx_z_data[i] if hasattr(self, '_pullx_z_data') else self.z_data[i]
            )
            t_pullx_rel = t_pullx - t_pullx[0]
            t_traj_rel  = t_traj  - t_traj[0]

            z1_matched = np.interp(t_traj_rel, t_pullx_rel, z1_pullx)
            z2_matched = np.interp(t_traj_rel, t_pullx_rel, z2_pullx)

            self.theta_data.append((theta1, theta2))
            self.ncat_data.append((ncat1, ncat2))
            new_z_data.append((z1_matched, z2_matched))
            new_t_data.append(t_traj)

            if self.verbose and (i + 1) % 5 == 0:
                self._log(
                    f"  win {i+1:2d}: n_frames={len(t_traj):,}  "
                    f"θ₁=[{theta1.min():.1f}°,{theta1.max():.1f}°]  "
                    f"n_{self._cation_label}_avg=[{ncat1.mean():.2f},{ncat2.mean():.2f}]"
                )

        self.z_data = new_z_data
        self.t_data = new_t_data
        self._log("θ and n_cat loaded; z-data resampled to trajectory frame times.")

        # --- save cache ---
        if save_cache:
            self._log(f"Saving cache → {cache_file}")
            save_dict = {'n_windows': np.array(self.n_windows)}
            for i, ((th1, th2), (nc1, nc2), (z1, z2), t) in enumerate(
                zip(self.theta_data, self.ncat_data, self.z_data, self.t_data)
            ):
                save_dict[f'th1_{i}'] = th1
                save_dict[f'th2_{i}'] = th2
                save_dict[f'nc1_{i}'] = nc1
                save_dict[f'nc2_{i}'] = nc2
                save_dict[f'z1_{i}']  = z1
                save_dict[f'z2_{i}']  = z2
                save_dict[f't_{i}']   = t
            np.savez(cache_file, **save_dict)
            self._log(f"  Saved {len(self.theta_data)} windows → {cache_file}")

        return self

    def set_3d_data_direct(self, theta_data, ncat_data):
        """
        Supply pre-computed θ and n_cat arrays directly (no trajectory needed).

        Parameters
        ----------
        theta_data : list of (theta1, theta2) arrays  [degrees]
        ncat_data  : list of (ncat1, ncat2) arrays   [int]
            Lengths must match self.z_data[i][0] / [1].

        Returns
        -------
        self
        """
        if self.z_data is None:
            raise RuntimeError("Call load_data() first.")
        if len(theta_data) != self.n_windows or len(ncat_data) != self.n_windows:
            raise ValueError(
                f"Expected {self.n_windows} elements in theta_data and ncat_data."
            )
        for i, ((z1, z2), (th1, th2), (nc1, nc2)) in enumerate(
            zip(self.z_data, theta_data, ncat_data)
        ):
            for name, za, arr in [('θ1', z1, th1), ('θ2', z2, th2),
                                   ('nc1', z1, nc1), ('nc2', z2, nc2)]:
                if len(arr) != len(za):
                    raise ValueError(
                        f"Window {i+1}: length mismatch — {name}: "
                        f"len={len(arr)} vs z: len={len(za)}."
                    )
        self.theta_data = [
            (np.asarray(th1, float), np.asarray(th2, float))
            for th1, th2 in theta_data
        ]
        self.ncat_data = [
            (np.asarray(nc1, int), np.asarray(nc2, int))
            for nc1, nc2 in ncat_data
        ]
        return self

    # ------------------------------------------------------------------
    # 3D histogram construction
    # ------------------------------------------------------------------

    def _build_histograms_3d(self):
        """
        Build 3D histograms H_i(r, θ, n_cat) for all 2×n_windows pseudo-windows.

        Sets
        ----
        r_bins, r_centers, r_width, theta_bins, theta_centers, theta_width,
        cation_bins, cation_centers, cation_width,
        histograms_3d, biases_3d, n_snapshots
        """
        if self.z_data is None:
            raise RuntimeError("Call load_data() first.")
        if self.theta_data is None or self.ncat_data is None:
            raise RuntimeError(
                "Call load_trajectory_data() or set_3d_data_direct() first."
            )

        # --- r grid ---
        if self.xi_max is None:
            all_r = np.concatenate([
                np.concatenate([np.abs(z1), np.abs(z2)])
                for z1, z2 in self.z_data
            ])
            r_max = all_r.max() * 1.05
        else:
            r_max = float(self.xi_max)

        r_min = 0.0 if self.xi_min is None else float(self.xi_min)
        self.r_bins    = np.linspace(r_min, r_max, self.n_r_bins + 1)
        self.r_centers = 0.5 * (self.r_bins[:-1] + self.r_bins[1:])
        self.r_width   = float(self.r_bins[1] - self.r_bins[0])

        # --- theta grid ---
        self.theta_bins    = np.linspace(
            self.theta_range[0], self.theta_range[1], self.n_theta_bins + 1
        )
        self.theta_centers = 0.5 * (self.theta_bins[:-1] + self.theta_bins[1:])
        self.theta_width   = float(self.theta_bins[1] - self.theta_bins[0])

        # ClayPMFPlotter compatibility alias
        self.bin_centers_abs = self.r_centers

        # --- allocate ---
        R     = 2 * self.n_windows
        hists = np.zeros(
            (R, self.n_r_bins, self.n_theta_bins, self.n_cation_bins), dtype=float
        )
        bias  = np.zeros((R, self.n_r_bins), dtype=float)
        nsn   = np.zeros(R, dtype=float)

        bins_3d = [self.r_bins, self.theta_bins, self.cation_bins]

        for i, ((z1, z2), (c1, c2), (th1, th2), (nc1, nc2)) in enumerate(
            zip(self.z_data, self.window_centers, self.theta_data, self.ncat_data)
        ):
            idx1, idx2 = 2 * i, 2 * i + 1
            r1, r2 = np.abs(z1), np.abs(z2)
            rc1, rc2 = abs(c1), abs(c2)

            # Clip n_cat to declared range so no counts fall outside the grid
            nc1c = np.clip(nc1, self.cation_range[0], self.cation_range[1]).astype(float)
            nc2c = np.clip(nc2, self.cation_range[0], self.cation_range[1]).astype(float)

            data1 = np.column_stack([r1, th1, nc1c])
            data2 = np.column_stack([r2, th2, nc2c])

            h1, _ = np.histogramdd(data1, bins=bins_3d)
            h2, _ = np.histogramdd(data2, bins=bins_3d)

            hists[idx1] = h1.astype(float)
            hists[idx2] = h2.astype(float)

            # Harmonic bias V_i(r) = ½k(r − r₀ᵢ)²  [kJ/mol]
            bias[idx1] = 0.5 * self.k * (self.r_centers - rc1) ** 2
            bias[idx2] = 0.5 * self.k * (self.r_centers - rc2) ** 2

            nsn[idx1] = len(z1)
            nsn[idx2] = len(z2)

        self.histograms_3d = hists
        self.biases_3d     = bias
        self.n_snapshots   = nsn

        total_counts = int(np.sum(hists))
        self._log(
            f"3D histogram grid: "
            f"r ∈ [{r_min:.3f}, {r_max:.3f}] nm ({self.n_r_bins} bins, Δr={self.r_width:.4f} nm); "
            f"θ ∈ [{self.theta_range[0]:.0f}°, {self.theta_range[1]:.0f}°] "
            f"({self.n_theta_bins} bins, Δθ={self.theta_width:.2f}°); "
            f"n_cat ∈ [{self.cation_range[0]}, {self.cation_range[1]}] "
            f"({self.n_cation_bins} integer bins). "
            f"Total counts: {total_counts:,}"
        )
        return self

    # ------------------------------------------------------------------
    # WHAM solver
    # ------------------------------------------------------------------

    def run_wham_3d(self, tolerance=None, max_iter=None, fix_f=None):
        """
        Solve the 3D WHAM equations.

        **Step 1 — 1D WHAM for {f_i}** (vectorised, same as ClayPMF2D):
            H̄_i(r) = Σ_{θ,n} H_i(r, θ, n)
            {f_i}  ← standard 1D WHAM on {H̄_i(r), V_i(r)}

        **Step 2 — 3D unbiased probability**:
            D(r)      = Σ_i N_i exp(f_i) exp(−β V_i(r))
            P(r,θ,n) ∝ H_total(r,θ,n) / D(r)

        **Step 3 — 3D PMF**:
            W(r,θ,n) = −kT ln P(r,θ,n)   [global min shifted to 0]

        Also computes 1D marginals W(r), W(θ), W(n_cat) and signed
        profile W(z) for ClayPMFPlotter compatibility.

        Parameters
        ----------
        tolerance : float or None
        max_iter  : int or None
        fix_f : array-like or None
            If provided, skip 1D WHAM and use these values as {f_i}.
            Shape: (2*n_windows,).  Pass the ``f`` attribute from a
            ClayPMF1D/ClayPMF2D run on the full-density pullx data::

                pmf3d.run_wham_3d(fix_f=pmf.f)

        Returns
        -------
        pmf_3d : np.ndarray  shape (n_r_bins, n_theta_bins, n_cation_bins), kJ/mol
        """
        if tolerance is None:
            tolerance = self.tolerance
        if max_iter is None:
            max_iter = self.max_iter

        if self.histograms_3d is None:
            self._build_histograms_3d()

        R = 2 * self.n_windows

        # --- Step 1: marginal r-histograms & 1D WHAM for {f_i} ---
        # Collapse θ and n axes → (R, n_r_bins)
        H_1d_per_win = np.sum(self.histograms_3d, axis=(2, 3))
        H_1d_total   = np.sum(H_1d_per_win, axis=0)     # (n_r_bins,)

        # exp(−β V_i(r))  shape (R, n_r_bins)
        exp_neg_bV = np.exp(-self.beta * self.biases_3d)

        if fix_f is not None:
            f = np.asarray(fix_f, dtype=float).copy()
            if f.shape != (R,):
                raise ValueError(
                    f"fix_f has shape {f.shape}; expected ({R},)."
                )
            f -= f.mean()
            self._fix_f = f.copy()
            self._log(
                f"  Using externally supplied f values (fix_f; skipping 1D WHAM). "
                f"Range: [{f.min():.3f}, {f.max():.3f}] kJ/mol"
            )
        else:
            self._fix_f = None
            self._log(
                f"1D WHAM for f_i: R={R}, n_r_bins={self.n_r_bins}, "
                f"tol={tolerance:.1e}, max_iter={max_iter}"
            )

            f    = self.f.copy() if self.f is not None else np.zeros(R)
            diff = np.inf

            for iteration in range(max_iter):
                f_old = f.copy()

                # denom[j] = Σ_i N_i exp(f_i) exp(−β V_i(r_j))
                denom = (self.n_snapshots * np.exp(f)) @ exp_neg_bV  # (n_r_bins,)
                denom = np.where(denom > 0.0, denom, np.inf)

                P_r  = H_1d_total / denom
                norm = np.sum(P_r) * self.r_width
                P_r /= max(norm, 1e-300)

                # exp(−f_i) = Σ_j P(r_j) exp(−β V_i(r_j)) Δr
                integrals = (exp_neg_bV @ P_r) * self.r_width   # (R,)

                # starvation fix: skip windows with zero integral
                good = integrals > 0
                f_new = f.copy()
                f_new[good] = -np.log(integrals[good])
                f_new -= f_new.mean()

                diff = float(np.max(np.abs(f_new - f_old)))
                f = f_new

                if diff < tolerance:
                    self._log(
                        f"  Converged after {iteration + 1} iterations (Δf={diff:.2e})"
                    )
                    break
            else:
                warnings.warn(
                    f"1D WHAM did not converge after {max_iter} iterations "
                    f"(Δf={diff:.2e}). Increase max_iter or tolerance.",
                    RuntimeWarning,
                )

        self.f = f

        # --- Step 2: 3D unbiased probability ---
        # D(r) = Σ_i N_i exp(f_i) exp(−β V_i(r))  shape (n_r_bins,)
        D_r = (self.n_snapshots * np.exp(f)) @ exp_neg_bV
        D_r = np.where(D_r > 0.0, D_r, np.inf)

        # H_total_3d: (n_r_bins, n_theta_bins, n_cation_bins)
        H_total_3d = np.sum(self.histograms_3d, axis=0)

        # P(r,θ,n) = H_total(r,θ,n) / D(r)   [broadcast D_r]
        P_3d = H_total_3d / D_r[:, np.newaxis, np.newaxis]

        # Normalise: ∫∫∫ P dr dθ dn = 1
        norm_3d = np.sum(P_3d) * self.r_width * self.theta_width * self.cation_width
        if norm_3d > 0:
            P_3d /= norm_3d
        self.P_3d = P_3d

        # --- Step 3: 3D PMF ---
        with np.errstate(divide='ignore', invalid='ignore'):
            pmf_3d = np.where(P_3d > 0, -np.log(P_3d) / self.beta, np.nan)
        pmf_3d -= np.nanmin(pmf_3d)
        self.pmf_3d = pmf_3d

        # --- Marginal W(r) ---
        P_r_marg = np.sum(P_3d, axis=(1, 2)) * self.theta_width * self.cation_width
        with np.errstate(divide='ignore', invalid='ignore'):
            pmf_r = np.where(P_r_marg > 0, -np.log(P_r_marg) / self.beta, np.nan)
        pmf_r -= np.nanmin(pmf_r)
        self.pmf_r   = pmf_r
        self.pmf_abs = pmf_r   # ClayPMFPlotter compat

        # --- Marginal W(θ) ---
        P_th_marg = np.sum(P_3d, axis=(0, 2)) * self.r_width * self.cation_width
        with np.errstate(divide='ignore', invalid='ignore'):
            pmf_theta = np.where(P_th_marg > 0, -np.log(P_th_marg) / self.beta, np.nan)
        pmf_theta -= np.nanmin(pmf_theta)
        self.pmf_theta = pmf_theta

        # --- Marginal W(n_cat) ---
        P_nc_marg = np.sum(P_3d, axis=(0, 1)) * self.r_width * self.theta_width
        with np.errstate(divide='ignore', invalid='ignore'):
            pmf_cation = np.where(P_nc_marg > 0, -np.log(P_nc_marg) / self.beta, np.nan)
        pmf_cation -= np.nanmin(pmf_cation)
        self.pmf_cation = pmf_cation

        # --- Signed PMF (ClayPMFPlotter compat) ---
        r_vals  = self.r_centers
        z_neg   = -r_vals[::-1][:-1]
        pmf_neg = pmf_r[::-1][:-1]
        self._bin_centers_signed = np.concatenate([z_neg, r_vals])
        self.pmf_signed          = np.concatenate([pmf_neg, pmf_r])

        self._log(
            f"3D PMF: [{np.nanmin(pmf_3d):.2f}, {np.nanmax(pmf_3d):.2f}] kJ/mol | "
            f"W(r): [{np.nanmin(pmf_r):.2f}, {np.nanmax(pmf_r):.2f}] kJ/mol | "
            f"W(θ): [{np.nanmin(pmf_theta):.2f}, {np.nanmax(pmf_theta):.2f}] kJ/mol | "
            f"W(n): [{np.nanmin(pmf_cation):.2f}, {np.nanmax(pmf_cation):.2f}] kJ/mol"
        )
        return pmf_3d

    # ------------------------------------------------------------------
    # Bulk reference correction
    # ------------------------------------------------------------------

    def reference_to_bulk(self, bulk_fraction=0.2, enabled=True):
        """
        Set the bulk free energy to zero by shifting all stored PMFs.

        Uses the **median** of ``pmf_r`` over the first ``bulk_fraction``
        of r-bins as the bulk reference level (small r = far from clay in
        the coordinate system used by ClayPMF2D).

        Applies the same shift to pmf_3d, pmf_r, pmf_theta, pmf_cation,
        pmf_signed.  Calling again with new parameters replaces the old
        correction.

        Parameters
        ----------
        bulk_fraction : float
            Fraction of r-axis bins used as bulk reference. Default 0.2.
        enabled : bool
            If False, undo existing correction and return.
        """
        if self.pmf_r is None:
            raise RuntimeError("run_wham_3d() must be called first.")

        # Undo previous correction
        if self.bulk_correction_enabled and self._bulk_shift != 0.0:
            for arr in (self.pmf_3d, self.pmf_r, self.pmf_abs,
                        self.pmf_theta, self.pmf_cation, self.pmf_signed):
                if arr is not None:
                    arr += self._bulk_shift
            self._bulk_shift = 0.0
            self.bulk_correction_enabled = False

        self.bulk_fraction = bulk_fraction

        if not enabled:
            self._log("reference_to_bulk: disabled — PMFs left at min=0.")
            return

        n_bulk = max(1, int(bulk_fraction * self.n_r_bins))
        shift  = float(np.nanmedian(self.pmf_r[:n_bulk]))

        self.pmf_3d     -= shift
        self.pmf_r      -= shift
        self.pmf_abs    -= shift
        if self.pmf_theta  is not None: self.pmf_theta  -= shift
        if self.pmf_cation is not None: self.pmf_cation -= shift
        if self.pmf_signed is not None: self.pmf_signed -= shift

        self._bulk_shift             = shift
        self.bulk_correction_enabled = True

        self._log(
            f"reference_to_bulk: bulk_fraction={bulk_fraction:.2f} "
            f"({n_bulk} bins), median shift = {shift:+.3f} kJ/mol applied."
        )

    # ------------------------------------------------------------------
    # Marginalization
    # ------------------------------------------------------------------

    def marginalize_to_2d(self, keep_axes=(0, 1)):
        """
        Marginalise the 3D PMF to 2D by summing over the remaining axis.

        Axis ordering: 0 = r, 1 = θ, 2 = n_cation.

        Parameters
        ----------
        keep_axes : tuple of two ints
            Axes to retain.  Options: (0,1), (0,2), (1,2).

        Returns
        -------
        pmf_2d : np.ndarray  shape (keep_bins_axis0, keep_bins_axis1), kJ/mol
        ax0_centers, ax1_centers : np.ndarray
        """
        if self.P_3d is None:
            self.run_wham_3d()

        all_axes = {0, 1, 2}
        keep = tuple(sorted(keep_axes))
        if len(keep) != 2 or not set(keep).issubset(all_axes):
            raise ValueError("keep_axes must be two distinct values from {0, 1, 2}.")

        sum_axis = (all_axes - set(keep)).pop()
        widths   = [self.r_width, self.theta_width, self.cation_width]
        centers  = [self.r_centers, self.theta_centers, self.cation_centers]

        P_2d = np.sum(self.P_3d, axis=sum_axis) * widths[sum_axis]

        # Normalise
        w0, w1 = widths[keep[0]], widths[keep[1]]
        norm   = np.sum(P_2d) * w0 * w1
        if norm > 0:
            P_2d /= norm

        with np.errstate(divide='ignore', invalid='ignore'):
            pmf_2d = np.where(P_2d > 0, -np.log(P_2d) / self.beta, np.nan)
        pmf_2d -= np.nanmin(pmf_2d)

        return pmf_2d, centers[keep[0]], centers[keep[1]]

    def marginalize_to_1d(self, keep_axis=0):
        """
        Marginalise the 3D PMF to 1D.

        Parameters
        ----------
        keep_axis : int
            0 = r, 1 = θ, 2 = n_cation.

        Returns
        -------
        pmf_1d : np.ndarray, centers : np.ndarray
        """
        if self.P_3d is None:
            self.run_wham_3d()

        if keep_axis not in (0, 1, 2):
            raise ValueError("keep_axis must be 0, 1, or 2.")

        sum_axes = tuple(ax for ax in (0, 1, 2) if ax != keep_axis)
        widths   = [self.r_width, self.theta_width, self.cation_width]
        centers  = [self.r_centers, self.theta_centers, self.cation_centers]

        P_1d = np.sum(self.P_3d, axis=sum_axes) * np.prod([widths[a] for a in sum_axes])

        norm = np.sum(P_1d) * widths[keep_axis]
        if norm > 0:
            P_1d /= norm

        with np.errstate(divide='ignore', invalid='ignore'):
            pmf_1d = np.where(P_1d > 0, -np.log(P_1d) / self.beta, np.nan)
        pmf_1d -= np.nanmin(pmf_1d)

        return pmf_1d, centers[keep_axis]

    # ------------------------------------------------------------------
    # Conditional PMF
    # ------------------------------------------------------------------

    def conditional_pmf(self, fixed_coords):
        """
        Compute conditional PMF at one fixed coordinate.

        Parameters
        ----------
        fixed_coords : dict
            Exactly one key: value pair specifying the fixed coordinate.
            Supported keys: 'r', 'r_index', 'theta', 'theta_index',
            'n_cation', 'n_cation_index'.
            Value is either the coordinate value (nearest bin chosen) or a
            direct bin index.

        Returns
        -------
        pmf_cond  : np.ndarray   2D conditional PMF (kJ/mol, min=0)
        axis_labels : list of str
        centers0, centers1 : np.ndarray
        fixed_label : str
        """
        if self.P_3d is None:
            self.run_wham_3d()

        key, val = next(iter(fixed_coords.items()))

        axis_info = {
            'r':              (0, self.r_centers,      'r (nm)'),
            'r_index':        (0, self.r_centers,      'r (nm)'),
            'theta':          (1, self.theta_centers,  'θ (°)'),
            'theta_index':    (1, self.theta_centers,  'θ (°)'),
            'n_cation':       (2, self.cation_centers, f'n_{self._cation_label}'),
            'n_cation_index': (2, self.cation_centers, f'n_{self._cation_label}'),
        }
        if key not in axis_info:
            raise ValueError(
                f"Unknown key '{key}'. Use one of: {list(axis_info)}"
            )

        fixed_axis, centers_fixed, label_fixed = axis_info[key]

        if key.endswith('_index'):
            idx = int(val)
        else:
            idx = int(np.argmin(np.abs(centers_fixed - val)))

        free_axes  = [a for a in (0, 1, 2) if a != fixed_axis]
        all_ctrs   = [self.r_centers, self.theta_centers, self.cation_centers]
        all_widths = [self.r_width,   self.theta_width,   self.cation_width]
        all_labels = ['r (nm)', 'θ (°)', f'n_{self._cation_label}']

        # Slice
        slices = [slice(None), slice(None), slice(None)]
        slices[fixed_axis] = idx
        P_slice = self.P_3d[tuple(slices)]   # 2D

        # Normalise
        w0, w1 = all_widths[free_axes[0]], all_widths[free_axes[1]]
        norm   = np.sum(P_slice) * w0 * w1
        P_norm = P_slice / max(norm, 1e-300)

        with np.errstate(divide='ignore', invalid='ignore'):
            pmf_cond = np.where(P_norm > 0, -np.log(P_norm) / self.beta, np.nan)
        pmf_cond -= np.nanmin(pmf_cond)

        fixed_label = f"{label_fixed} = {centers_fixed[idx]:.2g}"

        return (
            pmf_cond,
            [all_labels[free_axes[0]], all_labels[free_axes[1]]],
            all_ctrs[free_axes[0]],
            all_ctrs[free_axes[1]],
            fixed_label,
        )

    # ------------------------------------------------------------------
    # Coupling free energy
    # ------------------------------------------------------------------

    def coupling_free_energy(self):
        """
        Compute ΔΔW(r,θ,n) = W(r,θ,n) − W(r) − W(θ) − W(n_cat).

        Negative values indicate positive coupling (e.g. flat orientation
        AND Na⁺ coordination both preferred near the surface).

        Returns
        -------
        ddW : np.ndarray  shape (n_r_bins, n_theta_bins, n_cation_bins)
        """
        if self.pmf_3d is None:
            self.run_wham_3d()
        return (
            self.pmf_3d
            - self.pmf_r[:, np.newaxis, np.newaxis]
            - self.pmf_theta[np.newaxis, :, np.newaxis]
            - self.pmf_cation[np.newaxis, np.newaxis, :]
        )

    # ------------------------------------------------------------------
    # Kd and thermodynamic observables
    # ------------------------------------------------------------------

    def _auto_z_cut(self):
        """
        Auto-detect the bound/bulk boundary in r.

        Returns the r value at the first local minimum of the 1D marginal
        W(r) followed by a rising slope (i.e. the well edge), or halfway
        along the r-axis if no clear minimum is found.
        """
        if self.pmf_r is None:
            self.run_wham_3d()
        pmf = np.nan_to_num(self.pmf_r, nan=np.nanmax(self.pmf_r))
        # Find the global minimum
        imin = int(np.nanargmin(pmf))
        # Walk right until PMF rises above 0 (bulk level, after correction)
        # or reaches the end
        for j in range(imin, len(pmf) - 1):
            if pmf[j] > 0.0:
                return float(self.r_centers[j])
        # Fallback: halfway
        return float(self.r_centers[len(self.r_centers) // 2])

    def kd_total(self, z_cut=None):
        """
        Compute total partition coefficient Kd.

        Kd = (1/V_1M) ∫∫∫_{r < z_cut} exp(−β W(r,θ,n)) dr dθ dn

        Parameters
        ----------
        z_cut : float or None
            Upper r-boundary of bound region (nm).  Auto-detected if None.

        Returns
        -------
        kd : float
        z_cut : float  [nm]
        """
        if self.pmf_3d is None:
            self.run_wham_3d()
        if z_cut is None:
            z_cut = self._auto_z_cut()

        z_mask = self.r_centers < z_cut
        B = np.exp(-self.beta * np.nan_to_num(self.pmf_3d, nan=0.0))

        integral = np.sum(
            B[z_mask, :, :]
        ) * self.r_width * self.theta_width * self.cation_width

        kd = integral / self.V_1M
        return kd, float(z_cut)

    def kd_cation_resolved(self, z_cut=None):
        """
        Compute cation-coordination-resolved Kd(n_cat).

        Returns
        -------
        cation_centers : np.ndarray
        kd_cat : np.ndarray
        """
        if self.pmf_3d is None:
            self.run_wham_3d()
        if z_cut is None:
            z_cut = self._auto_z_cut()

        z_mask = self.r_centers < z_cut
        B = np.exp(-self.beta * np.nan_to_num(self.pmf_3d, nan=0.0))

        kd_cat = np.zeros(self.n_cation_bins)
        for j in range(self.n_cation_bins):
            kd_cat[j] = (
                np.sum(B[z_mask, :, j]) * self.r_width * self.theta_width / self.V_1M
            )
        return self.cation_centers.copy(), kd_cat

    def kd_theta_resolved(self, z_cut=None):
        """
        Compute orientation-resolved Kd(θ).

        Returns
        -------
        theta_centers : np.ndarray
        kd_theta : np.ndarray
        """
        if self.pmf_3d is None:
            self.run_wham_3d()
        if z_cut is None:
            z_cut = self._auto_z_cut()

        z_mask = self.r_centers < z_cut
        B = np.exp(-self.beta * np.nan_to_num(self.pmf_3d, nan=0.0))

        kd_th = np.zeros(self.n_theta_bins)
        for j in range(self.n_theta_bins):
            kd_th[j] = (
                np.sum(B[z_mask, j, :]) * self.r_width * self.cation_width / self.V_1M
            )
        return self.theta_centers.copy(), kd_th

    def exchange_free_energy(self, z_cut=None, n_from=0, n_to=1, unit='kJ/mol'):
        """
        ΔG_exchange = −kT ln(Kd(n_from) / Kd(n_to)).

        Default: free energy cost of going from n_cat=0 to n_cat=1 at surface.

        Parameters
        ----------
        unit : str  'kJ/mol', 'kcal/mol', or 'kT'

        Returns
        -------
        delta_g : float
        """
        cat_centers, kd_cat = self.kd_cation_resolved(z_cut)
        i_from = int(np.argmin(np.abs(cat_centers - n_from)))
        i_to   = int(np.argmin(np.abs(cat_centers - n_to)))
        if kd_cat[i_to] == 0.0:
            warnings.warn(f"Kd(n={n_to}) = 0; exchange free energy is ill-defined.")
            return np.nan
        ratio    = kd_cat[i_from] / kd_cat[i_to]
        delta_g  = -self.K_B * self.T * np.log(ratio)   # kJ/mol
        return float(self._to_unit(np.array([delta_g]), unit)[0])

    def average_cation_number(self, z_cut=None):
        """
        Average cation coordination number at the binding site.

        Returns
        -------
        avg_n : float
        """
        cat_centers, kd_cat = self.kd_cation_resolved(z_cut)
        total = np.sum(kd_cat)
        if total == 0:
            return 0.0
        P_cat = kd_cat / total
        return float(np.sum(cat_centers * P_cat))

    def orientation_distribution(self, z_cut=None):
        """
        Probability density of tilt angle at the binding site.

        Returns
        -------
        theta_centers : np.ndarray
        P_theta : np.ndarray  (probability density per degree)
        """
        theta_centers, kd_th = self.kd_theta_resolved(z_cut)
        total = np.sum(kd_th) * self.theta_width
        if total == 0:
            return theta_centers, np.zeros_like(kd_th)
        return theta_centers, kd_th / total

    def average_tilt_angle(self, z_cut=None):
        """
        Average tilt angle at the binding site (degrees).
        """
        theta_centers, P_theta = self.orientation_distribution(z_cut)
        return float(np.sum(theta_centers * P_theta) * self.theta_width)

    def order_parameter(self, z_cut=None):
        """
        Orientational order parameter S = (3⟨cos²θ⟩ − 1) / 2.

        S = 1: flat; S = 0: isotropic; S = −0.5: perpendicular.
        """
        theta_centers, P_theta = self.orientation_distribution(z_cut)
        theta_rad = np.radians(theta_centers)
        cos2_avg  = float(np.sum(np.cos(theta_rad) ** 2 * P_theta) * self.theta_width)
        return (3 * cos2_avg - 1) / 2

    def binding_free_energy(self, z_cut=None, unit='kJ/mol'):
        """
        ΔG_bind = −kT ln(Kd).

        Returns
        -------
        delta_g : float  (kJ/mol by default)
        """
        kd, _ = self.kd_total(z_cut)
        if kd <= 0:
            return np.nan
        delta_g = -self.K_B * self.T * np.log(kd)   # kJ/mol
        return float(self._to_unit(np.array([delta_g]), unit)[0])

    def print_results(self, z_cut=None, unit='kJ/mol'):
        """
        Print all thermodynamic results in a formatted table.

        Parameters
        ----------
        z_cut : float or None
        unit  : str  Energy unit for reported values.
        """
        if self.pmf_3d is None:
            self.run_wham_3d()

        kd, z_cut_auto   = self.kd_total(z_cut)
        delta_g          = self.binding_free_energy(z_cut, unit=unit)
        avg_n            = self.average_cation_number(z_cut)
        dg_exchange      = self.exchange_free_energy(z_cut, unit=unit)
        avg_theta        = self.average_tilt_angle(z_cut)
        S                = self.order_parameter(z_cut)

        cat_centers, kd_cat = self.kd_cation_resolved(z_cut)
        total_kd_cat         = np.sum(kd_cat)

        u_sym = unit
        sep   = "=" * 70

        print(f"\n{sep}")
        print("3D WHAM RESULTS")
        print(sep)
        print(f"\n  Temperature   : {self.T} K")
        print(f"  Cation type   : {self._cation_label}")
        print(f"  Cation cutoff : {self.cation_cutoff} nm")
        print(f"  Bound cutoff  : r < {z_cut_auto:.3f} nm")

        print(f"\n{'-'*50}")
        print("THERMODYNAMICS")
        print(f"{'-'*50}")
        print(f"  Total Kd       : {kd:.3e}  (dimensionless)")
        print(f"  log10(Kd)      : {np.log10(kd):.2f}" if kd > 0 else "  log10(Kd) : N/A")
        print(f"  ΔG_bind        : {delta_g:.3f} {u_sym}")

        print(f"\n{'-'*50}")
        print(f"CATION ({self._cation_label}) COORDINATION")
        print(f"{'-'*50}")
        print(f"  ⟨n_{self._cation_label}⟩         : {avg_n:.2f}")
        print(f"  ΔG_exchange (n=0→1) : {dg_exchange:.3f} {u_sym}")
        print(f"\n  Coordination distribution:")
        for n_val, kd_val in zip(cat_centers, kd_cat):
            P = kd_val / total_kd_cat if total_kd_cat > 0 else 0.0
            print(f"    n = {int(n_val):2d}: Kd = {kd_val:.2e},  P = {P:.3f}")

        print(f"\n{'-'*50}")
        print("ORIENTATION")
        print(f"{'-'*50}")
        print(f"  ⟨θ⟩            : {avg_theta:.1f}°")
        print(f"  Order param S  : {S:.3f}")

        print(f"\n{sep}")
        print("Interpretation:")
        print("-" * 40)
        print(f"  {'✓' if delta_g < 0 else '✗'} Spontaneous adsorption (ΔG_bind {'< 0' if delta_g < 0 else '> 0'})")
        print(f"  {'✓' if avg_n > 0.5 else '○'} Cation present at binding site (⟨n⟩ = {avg_n:.2f})")
        print(f"  {'✓' if dg_exchange < 0 else '○'} Cation exchange favorable (ΔG_exch {'< 0' if dg_exchange < 0 else '> 0'})")
        if S > 0.5:
            print(f"  ✓ Highly ordered, flat adsorption (S = {S:.3f})")
        elif S < 0:
            print(f"  ○ Perpendicular orientation (S = {S:.3f})")
        else:
            print(f"  ○ Moderate orientational order (S = {S:.3f})")
        print(sep)

    # ------------------------------------------------------------------
    # Bootstrap error estimation
    # ------------------------------------------------------------------

    def bootstrap_errors_3d(self, n_bootstrap=100):
        """
        Estimate PMF uncertainties via Poisson bootstrap.

        Resamples every 3D histogram bin as Poisson(H) and re-runs WHAM.

        Sets
        ----
        self.pmf_3d_std, self.pmf_r_std, self.pmf_abs_std, self.pmf_signed_std

        Returns
        -------
        pmf_3d_std : np.ndarray  (n_r_bins, n_theta_bins, n_cation_bins)
        pmf_r_std  : np.ndarray  (n_r_bins,)
        """
        if self.pmf_3d is None:
            self.run_wham_3d()

        self._log(f"Bootstrap 3D: {n_bootstrap} samples…")

        hists_orig   = self.histograms_3d.copy()
        f_init       = self.f.copy()
        verbose_orig = self.verbose
        _fix_f       = getattr(self, '_fix_f', None)

        shape_3d = (n_bootstrap, self.n_r_bins, self.n_theta_bins, self.n_cation_bins)
        pmf_3d_samples = np.zeros(shape_3d)
        pmf_r_samples  = np.zeros((n_bootstrap, self.n_r_bins))

        self.verbose = False
        try:
            for b in tqdm(
                range(n_bootstrap), desc='Bootstrap 3D', disable=not verbose_orig
            ):
                self.histograms_3d = np.random.poisson(hists_orig).astype(float)
                self.f = f_init.copy()
                self.run_wham_3d(fix_f=_fix_f)
                pmf_3d_samples[b] = self.pmf_3d
                pmf_r_samples[b]  = self.pmf_r
        finally:
            self.verbose       = verbose_orig
            self.histograms_3d = hists_orig
            self.f             = f_init
        self.run_wham_3d(fix_f=_fix_f)

        self.pmf_3d_std = np.std(pmf_3d_samples, axis=0)
        self.pmf_r_std  = np.std(pmf_r_samples,  axis=0)
        self.pmf_abs_std = self.pmf_r_std
        std_neg = self.pmf_r_std[::-1][:-1]
        self.pmf_signed_std = np.concatenate([std_neg, self.pmf_r_std])

        self._log(
            f"Bootstrap done. "
            f"Max 3D σ = {np.nanmax(self.pmf_3d_std):.3f} kJ/mol; "
            f"max W(r) σ = {np.nanmax(self.pmf_r_std):.3f} kJ/mol."
        )
        return self.pmf_3d_std, self.pmf_r_std

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save_results(self, outdir='.', prefix='pmf3d'):
        """
        Save all PMF results to ``<outdir>/<prefix>.npz``.

        Returns
        -------
        path : str
        """
        if self.pmf_3d is None:
            raise RuntimeError("Run run_wham_3d() first.")
        os.makedirs(outdir, exist_ok=True)
        out = os.path.join(outdir, f'{prefix}.npz')
        save_kw = dict(
            r_centers      = self.r_centers,
            theta_centers  = self.theta_centers,
            cation_centers = self.cation_centers,
            pmf_3d         = self.pmf_3d,
            pmf_r          = self.pmf_r,
            pmf_theta      = self.pmf_theta,
            pmf_cation     = self.pmf_cation,
            P_3d           = self.P_3d,
            f              = self.f,
        )
        if self.pmf_3d_std is not None:
            save_kw['pmf_3d_std'] = self.pmf_3d_std
            save_kw['pmf_r_std']  = self.pmf_r_std
        np.savez(out, **save_kw)
        self._log(f"Results saved → {out}")
        return out

    def load_results(self, filepath):
        """
        Load previously saved results from a .npz file.

        Parameters
        ----------
        filepath : str

        Returns
        -------
        self
        """
        data = np.load(filepath)
        self.r_centers      = data['r_centers']
        self.theta_centers  = data['theta_centers']
        self.cation_centers = data['cation_centers']
        self.pmf_3d         = data['pmf_3d']
        self.pmf_r          = data['pmf_r']
        self.pmf_abs        = self.pmf_r
        self.pmf_theta      = data['pmf_theta']
        self.pmf_cation     = data['pmf_cation']
        self.P_3d           = data['P_3d']
        self.f              = data['f']
        self.n_r_bins       = len(self.r_centers)
        self.n_theta_bins   = len(self.theta_centers)
        self.n_cation_bins  = len(self.cation_centers)
        self.r_width        = float(self.r_centers[1] - self.r_centers[0]) if self.n_r_bins > 1 else 1.0
        self.theta_width    = float(self.theta_centers[1] - self.theta_centers[0]) if self.n_theta_bins > 1 else 1.0
        self.cation_width   = 1.0
        self.bin_centers_abs = self.r_centers
        r_vals  = self.r_centers
        z_neg   = -r_vals[::-1][:-1]
        pmf_neg = self.pmf_r[::-1][:-1]
        self._bin_centers_signed = np.concatenate([z_neg, r_vals])
        self.pmf_signed          = np.concatenate([pmf_neg, self.pmf_r])
        if 'pmf_3d_std' in data:
            self.pmf_3d_std  = data['pmf_3d_std']
            self.pmf_r_std   = data['pmf_r_std']
            self.pmf_abs_std = self.pmf_r_std
        self._log(f"Results loaded from {filepath}.")
        return self

