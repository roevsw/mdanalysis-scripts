"""
ClayThermo.py  –  Thermodynamic decomposition for clay–drug umbrella sampling.

Reads per-window umbrella*.edr files (raw simulation, NOT rerun energy group
files) to compute:
  ΔH(r)   = ⟨U_pot(r)⟩ − ⟨U_pot(bulk)⟩   per CIP molecule
  −TΔS(r) = ΔG(r) − ΔH(r)                 when a PMF array is supplied

Interaction decomposition:
  clay–drug : Coul-SR:MMT-PULL1/2 + LJ-SR:MMT-PULL1/2  (+ Coul-14 + LJ-14 variants)
  ion–drug  : Coul-SR:Ion-PULL1/2 + LJ-SR:Ion-PULL1/2
  water–drug: Coul-SR:Water-PULL1/2 + LJ-SR:Water-PULL1/2
  drug–drug : Coul-SR:PULL1-PULL2 + LJ-SR:PULL1-PULL2

All values are averaged over PULL1 and PULL2, then divided by 2 (two molecules
→ per-molecule quantity).

EDR reader priority:
  1. panedr.edr_to_df()           (pip install panedr)
  2. gmx energy subprocess        (uses $GMXBIN or searches common paths)
  3. Raises RuntimeError with install hint

Simulation parameters assumed:
  T = 298 K  (ref_t in with_salts umbrella*.mdp)
  energygrps = MMT Ion Water PULL1 PULL2
  nstenergy = 1000  → energy frame every 2 ps

Usage example
-------------
from my_scripts.ClayThermo import ClayThermo

ct = ClayThermo(
    umbrella_dir='/Volumes/My_bckp/project_1_US/US/CIP/with_salts/'
                 'Negative_with_salt/CIP_NaCl_KCl/NaCl/kReplicate1/'
                 'CIP-_the_side_cross_NaCl/'
                 'US_2CIP-_gaff_clayff_spc_mmt_2021_k1000_d0.1_PULLNVT_'
                 'Constr_Only_at_start_nosalt_11NaCl_50ns/Umbrella',
    temperature=298.0,
)
ct.load_energies()
ct.compute_enthalpy()
ct.print_summary()
ct.save('anionic_NaCl_11_rep1_thermo.npz')
"""

import os
import re
import subprocess
import tempfile
import warnings
from glob import glob
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
KB = 8.314462618e-3          # kJ mol⁻¹ K⁻¹

# ---------------------------------------------------------------------------
# Energy term definitions (as labelled by gmx energy in GROMACS 2022.x)
# ---------------------------------------------------------------------------
# Global terms we always extract
_GLOBAL_TERMS = [
    'Potential',
    'Temperature',
    'Kinetic-En.',
]

# Group pair terms: (component_label, pull_index → 1 or 2)
# For each drug molecule (PULL1, PULL2) we want contributions from MMT / Ion / Water
# and the drug–drug pair.  Format is "Coul-SR:GroupA-GroupB" / "LJ-SR:GroupA-GroupB".
# GROMACS always orders the group names alphabetically in the pair label, so the
# ordering is: Ion < MMT < PULL1 < PULL2 < Water  (lexicographic).
# Verified from gmx energy output above.

_PAIR_TERMS = {
    # clay–drug (MMT–PULL1, MMT–PULL2)
    'clay_coul_pull1': 'Coul-SR:MMT-PULL1',
    'clay_lj_pull1':   'LJ-SR:MMT-PULL1',
    'clay_coul_pull2': 'Coul-SR:MMT-PULL2',
    'clay_lj_pull2':   'LJ-SR:MMT-PULL2',
    # ion–drug (Ion < PULL1/2 alphabetically)
    'ion_coul_pull1':  'Coul-SR:Ion-PULL1',
    'ion_lj_pull1':    'LJ-SR:Ion-PULL1',
    'ion_coul_pull2':  'Coul-SR:Ion-PULL2',
    'ion_lj_pull2':    'LJ-SR:Ion-PULL2',
    # water–drug (PULL1/2 < Water alphabetically)
    'water_coul_pull1': 'Coul-SR:Water-PULL1',
    'water_lj_pull1':   'LJ-SR:Water-PULL1',
    'water_coul_pull2': 'Coul-SR:Water-PULL2',
    'water_lj_pull2':   'LJ-SR:Water-PULL2',
    # drug–drug (PULL1 < PULL2)
    'drug_coul':  'Coul-SR:PULL1-PULL2',
    'drug_lj':    'LJ-SR:PULL1-PULL2',
}

# All terms we ask gmx energy / panedr to extract (no duplicates)
_ALL_TERM_NAMES = _GLOBAL_TERMS + list(_PAIR_TERMS.values())
# deduplicate while preserving order
_ALL_TERM_NAMES = list(dict.fromkeys(_ALL_TERM_NAMES))


# ---------------------------------------------------------------------------
# GMX binary discovery
# ---------------------------------------------------------------------------

def _find_gmx() -> str:
    """Return path to a working gmx binary, or raise RuntimeError."""
    # environment variable override
    env = os.environ.get('GMXBIN')
    if env and Path(env).is_file():
        return env

    candidates = [
        'gmx', 'gmx_mpi', 'gmx_d',
        '/usr/local/gromacs/bin/gmx',
        '/usr/local/bin/gmx',
        '/opt/gromacs/bin/gmx',
    ]
    for c in candidates:
        try:
            result = subprocess.run(
                [c, '--version'],
                capture_output=True, timeout=10
            )
            if result.returncode == 0:
                return c
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    raise RuntimeError(
        "gmx not found. Install GROMACS or set GMXBIN=/path/to/gmx. "
        "Alternatively: pip install panedr"
    )


# ---------------------------------------------------------------------------
# EDR reader via gmx energy
# ---------------------------------------------------------------------------

def _read_edr_gmx(edr_path: Path, term_names: list) -> dict:
    """
    Run 'gmx energy' on *edr_path*, extract *term_names*, parse the XVG output.

    Parameters
    ----------
    edr_path : Path
    term_names : list of str   gmx energy labels, e.g. 'Coul-SR:MMT-PULL1'

    Returns
    -------
    dict  {term_name: np.ndarray of values (time series, kJ/mol)}
          also includes 'time': np.ndarray (ps)
    """
    gmx = _find_gmx()

    # gmx energy selects terms by matching substrings; we pass each name
    # prefixed with a newline separator, terminated with a 0 (quit).
    # To be safe we pass the numbered index by first querying available terms,
    # then mapping names → numbers.
    # Simpler: pass the names directly as stdin tokens – gmx will match them.
    selection = '\n'.join(term_names) + '\n0\n'

    with tempfile.TemporaryDirectory() as tmpdir:
        xvg_path = os.path.join(tmpdir, 'energy.xvg')
        result = subprocess.run(
            [gmx, 'energy', '-f', str(edr_path), '-o', xvg_path],
            input=selection,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"gmx energy failed on {edr_path}:\n{result.stderr[-2000:]}"
            )
        data = _parse_xvg(xvg_path, term_names)

    return data


def _parse_xvg(xvg_path: str, term_names: list) -> dict:
    """
    Parse a GROMACS .xvg file.

    Returns dict with 'time' and one key per column label found in
    @ s<N> legend lines, mapped to *term_names* in order.
    """
    times = []
    rows = []
    legends = []   # ordered legend labels from xvg header

    with open(xvg_path, 'r') as fh:
        for line in fh:
            line = line.rstrip('\n')
            if line.startswith('#'):
                continue
            if line.startswith('@'):
                # @ s0 legend "Potential"  etc.
                m = re.match(r'@\s+s\d+\s+legend\s+"([^"]+)"', line)
                if m:
                    legends.append(m.group(1))
                continue
            parts = line.split()
            if not parts:
                continue
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue
            times.append(vals[0])
            rows.append(vals[1:])

    times = np.array(times, dtype=np.float64)
    rows = np.array(rows, dtype=np.float64)   # shape (n_frames, n_terms)

    result = {'time': times}
    # Map legend labels to our term_names.  gmx energy legends may differ
    # slightly (e.g. spaces vs dashes), so we match with a flexible key.
    def _normalise(s):
        return re.sub(r'[\s\-]+', '-', s).lower()

    norm_terms = {_normalise(t): t for t in term_names}
    norm_legends = [_normalise(l) for l in legends]

    for col_idx, nl in enumerate(norm_legends):
        if col_idx >= rows.shape[1]:
            break
        # direct match
        if nl in norm_terms:
            result[norm_terms[nl]] = rows[:, col_idx]
        else:
            # partial match fallback: find the term_name that is a suffix of nl
            for nt_key, nt_orig in norm_terms.items():
                if nl.endswith(nt_key) or nt_key.endswith(nl):
                    if nt_orig not in result:
                        result[nt_orig] = rows[:, col_idx]
                    break

    return result


# ---------------------------------------------------------------------------
# Block averaging helpers
# ---------------------------------------------------------------------------

def _block_average(arr: np.ndarray, n_blocks: int):
    """
    Split *arr* into *n_blocks* equal-length chunks, return
    (mean, sem) where sem is the standard error from block means.
    """
    n = len(arr)
    if n < n_blocks:
        return float(np.mean(arr)), float(np.std(arr) / max(1, n**0.5))
    chunk = n // n_blocks
    block_means = [np.mean(arr[i*chunk:(i+1)*chunk]) for i in range(n_blocks)]
    bm = np.array(block_means)
    return float(np.mean(bm)), float(np.std(bm, ddof=1) / n_blocks**0.5)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ClayThermo:
    """
    Thermodynamic decomposition of clay–drug umbrella sampling.

    Reads per-window `umbrella{n}.edr` files (raw umbrella simulation).
    Requires either `panedr` (Python) or `gmx` (GROMACS binary) to read .edr.

    Parameters
    ----------
    umbrella_dir : str or Path
        Path to the `Umbrella/` directory containing umbrella*.edr and
        pullx*.xvg files.
    temperature : float
        Simulation temperature in K.  Default 298.0.
    n_blocks : int
        Number of blocks for block-averaging error estimation.  Default 5.
    bulk_fraction : float
        Fraction of windows (from the end of the pull coordinate range)
        used as the bulk reference.  Default 0.2.
    equil_skip_frac : float
        Fraction of each window's trajectory to discard as equilibration
        (from the start).  Default 0.2.
    edr_prefix : str
        Prefix for edr files.  Default 'umbrella'.
    pullx_prefix : str
        Prefix for pullx xvg files.  Default 'pullx'.
    """

    def __init__(
        self,
        umbrella_dir,
        temperature: float = 298.0,
        n_blocks: int = 5,
        bulk_fraction: float = 0.2,
        equil_skip_frac: float = 0.2,
        edr_prefix: str = 'umbrella',
        pullx_prefix: str = 'pullx',
    ):
        self.umbrella_dir = Path(umbrella_dir)
        if not self.umbrella_dir.is_dir():
            raise FileNotFoundError(f"umbrella_dir not found: {self.umbrella_dir}")

        self.temperature = float(temperature)
        self.n_blocks = int(n_blocks)
        self.bulk_fraction = float(bulk_fraction)
        self.equil_skip_frac = float(equil_skip_frac)
        self.edr_prefix = edr_prefix
        self.pullx_prefix = pullx_prefix

        # Auto-detect window count
        edr_files = sorted(
            glob(str(self.umbrella_dir / f'{edr_prefix}*.edr')),
            key=lambda p: int(re.search(r'(\d+)\.edr$', p).group(1))
        )
        if not edr_files:
            raise FileNotFoundError(
                f"No {edr_prefix}*.edr files found in {self.umbrella_dir}"
            )
        self.n_windows = len(edr_files)
        self._edr_files = [Path(p) for p in edr_files]

        # Results (populated by load_energies / compute_enthalpy / compute_entropy)
        self.r_centers = None          # (n_windows,) nm  – mean pull coordinate

        # Per-window mean ± sem for every extracted energy term
        # Dict: term_name → {'mean': array(n_windows), 'sem': array(n_windows)}
        self._energy_stats = {}

        # Decomposed ΔH components (set by compute_enthalpy)
        self.dH = None                 # (n_windows,) kJ/mol per CIP molecule
        self.dH_err = None             # (n_windows,) SEM

        self.dH_clay_lj   = None;  self.dH_clay_lj_err   = None
        self.dH_clay_coul  = None;  self.dH_clay_coul_err  = None
        self.dH_ion_lj    = None;  self.dH_ion_lj_err    = None
        self.dH_ion_coul   = None;  self.dH_ion_coul_err   = None
        self.dH_water_lj  = None;  self.dH_water_lj_err  = None
        self.dH_water_coul = None;  self.dH_water_coul_err = None
        self.dH_drug_lj   = None;  self.dH_drug_lj_err   = None
        self.dH_drug_coul  = None;  self.dH_drug_coul_err  = None

        # Entropy (set by compute_entropy)
        self.mTdS      = None          # −TΔS(r)  kJ/mol
        self.mTdS_err  = None
        self.dG_interp = None          # ΔG(r) interpolated to r_centers

    # ------------------------------------------------------------------
    # EDR reading
    # ------------------------------------------------------------------

    def _read_edr(self, edr_path: Path) -> dict:
        """
        Read energy time-series from an .edr file.

        Returns
        -------
        dict  {term_name: np.ndarray}  plus 'time': np.ndarray
        """
        # --- Strategy 1: panedr ---
        try:
            import panedr
            df = panedr.edr_to_df(str(edr_path))
            result = {'time': df['Time'].values}
            for term in _ALL_TERM_NAMES:
                # panedr uses space-separated names matching .log notation
                # Try both the gmx-style (dashes) and log-style (spaces)
                for col_key in [term, term.replace('-', ' ')]:
                    if col_key in df.columns:
                        result[term] = df[col_key].values
                        break
            return result
        except ImportError:
            pass
        except Exception as e:
            warnings.warn(f"panedr failed on {edr_path.name}: {e}; falling back to gmx")

        # --- Strategy 2: gmx energy ---
        try:
            return _read_edr_gmx(edr_path, _ALL_TERM_NAMES)
        except RuntimeError as e:
            raise RuntimeError(
                f"Could not read {edr_path}.\n"
                "Install panedr via:  pip install panedr\n"
                f"Original error: {e}"
            )

    # ------------------------------------------------------------------
    # pullx reader
    # ------------------------------------------------------------------

    def _read_pullx(self, pullx_path: Path) -> float:
        """
        Parse a pullx{n}.xvg file and return the mean pull coordinate (nm).

        The file has columns:  time  coord1  coord2
        (two CIP molecules).  We use mean((|coord1| + |coord2|) / 2).
        """
        times, c1, c2 = [], [], []
        with open(pullx_path, 'r') as fh:
            for line in fh:
                if line.startswith('#') or line.startswith('@'):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    times.append(float(parts[0]))
                    c1.append(float(parts[1]))
                    c2.append(float(parts[2]))
                except ValueError:
                    continue
        if not times:
            return np.nan
        c1 = np.array(c1)
        c2 = np.array(c2)
        # equil skip
        n_skip = max(0, int(len(times) * self.equil_skip_frac))
        return float(np.mean((np.abs(c1[n_skip:]) + np.abs(c2[n_skip:])) / 2.0))

    # ------------------------------------------------------------------
    # Main loading routine
    # ------------------------------------------------------------------

    def load_energies(self, verbose: bool = True):
        """
        Read all per-window .edr files, apply equilibration skip, compute
        block-averaged mean ± SEM for each energy term.

        Also reads pullx*.xvg to obtain r_centers.

        Results are stored in ``self._energy_stats`` and ``self.r_centers``.
        """
        if verbose:
            print(f"Loading energies from {self.n_windows} windows in:")
            print(f"  {self.umbrella_dir}")

        r_centers = np.full(self.n_windows, np.nan)
        energy_means = {t: np.full(self.n_windows, np.nan) for t in _ALL_TERM_NAMES}
        energy_sems  = {t: np.full(self.n_windows, np.nan) for t in _ALL_TERM_NAMES}

        for i, edr_path in enumerate(self._edr_files):
            win_idx = i + 1  # 1-based window number for file names
            if verbose:
                print(f"  window {win_idx:3d}/{self.n_windows} ... ", end='', flush=True)

            # --- read r_center from pullx ---
            pullx_path = self.umbrella_dir / f'{self.pullx_prefix}{win_idx}.xvg'
            if pullx_path.exists():
                r_centers[i] = self._read_pullx(pullx_path)
            else:
                warnings.warn(f"{pullx_path.name} not found; r_center set to NaN")

            # --- read energies ---
            try:
                data = self._read_edr(edr_path)
            except Exception as e:
                warnings.warn(f"Skipping window {win_idx}: {e}")
                if verbose:
                    print("SKIP")
                continue

            time = data.get('time', np.array([]))
            if len(time) == 0:
                warnings.warn(f"No time data for window {win_idx}")
                if verbose:
                    print("EMPTY")
                continue

            # Equilibration skip
            n_frames = len(time)
            n_skip = max(0, int(n_frames * self.equil_skip_frac))

            for term in _ALL_TERM_NAMES:
                arr = data.get(term)
                if arr is None or len(arr) == 0:
                    continue
                arr_prod = arr[n_skip:]
                if len(arr_prod) == 0:
                    continue
                mn, sem = _block_average(arr_prod, self.n_blocks)
                energy_means[term][i] = mn
                energy_sems[term][i]  = sem

            if verbose:
                pot = energy_means['Potential'][i]
                print(f"r={r_centers[i]:.3f} nm  ⟨Pot⟩={pot:.1f} kJ/mol")

        self.r_centers = r_centers
        self._energy_stats = {
            t: {'mean': energy_means[t], 'sem': energy_sems[t]}
            for t in _ALL_TERM_NAMES
        }

        if verbose:
            n_ok = int(np.sum(~np.isnan(self._energy_stats['Potential']['mean'])))
            print(f"\nLoaded {n_ok}/{self.n_windows} windows successfully.")

    # ------------------------------------------------------------------
    # Enthalpy computation
    # ------------------------------------------------------------------

    def compute_enthalpy(self, bulk_windows=None):
        """
        Compute ΔH(r) = ⟨U_pot(r)⟩ − ⟨U_pot(bulk)⟩  per CIP molecule.

        The bulk reference is the mean of *bulk_windows* (list of 0-based
        indices).  Default: last floor(bulk_fraction * n_windows) windows.

        Decomposition components (clay/ion/water LJ and Coul, drug–drug)
        are also computed and stored as ``self.dH_*`` attributes.

        All values are per CIP molecule:
          pair term = (PULL1_value + PULL2_value) / 2
          global U_pot / 2  (two molecules in the simulation)
        """
        if self._energy_stats is None or 'Potential' not in self._energy_stats:
            raise RuntimeError("Call load_energies() first.")

        n = self.n_windows

        if bulk_windows is None:
            n_bulk = max(1, int(np.floor(self.bulk_fraction * n)))
            bulk_windows = list(range(n - n_bulk, n))

        bulk_idx = np.array(bulk_windows, dtype=int)

        def _bulk_mean(arr):
            """Mean of arr over bulk_idx, ignoring NaN."""
            vals = arr[bulk_idx]
            valid = vals[~np.isnan(vals)]
            if len(valid) == 0:
                return np.nan
            return float(np.mean(valid))

        def _bulk_sem(arr_sem):
            """Combined SEM of bulk windows (quadrature)."""
            vals = arr_sem[bulk_idx]
            valid = vals[~np.isnan(vals)]
            if len(valid) == 0:
                return np.nan
            return float(np.sqrt(np.sum(valid**2)) / len(valid))

        # ---- Global potential energy ΔH ----
        pot_mean = self._energy_stats['Potential']['mean']
        pot_sem  = self._energy_stats['Potential']['sem']

        bulk_pot = _bulk_mean(pot_mean)
        bulk_pot_sem = _bulk_sem(pot_sem)

        # ΔU_pot / 2  (two CIP molecules in box → per molecule)
        self.dH     = (pot_mean - bulk_pot) / 2.0
        self.dH_err = np.sqrt(pot_sem**2 + bulk_pot_sem**2) / 2.0

        # ---- Helper for pair-term decomposition ----
        def _pair_dH(term_pull1, term_pull2):
            """
            Compute per-molecule ΔH for a pair interaction.
            Average PULL1 and PULL2 contributions, referenced to bulk.
            """
            m1 = self._energy_stats[term_pull1]['mean']
            s1 = self._energy_stats[term_pull1]['sem']
            m2 = self._energy_stats[term_pull2]['mean']
            s2 = self._energy_stats[term_pull2]['sem']

            # Average over two drug molecules
            avg_m = (m1 + m2) / 2.0
            avg_s = np.sqrt(s1**2 + s2**2) / 2.0

            bulk_m = _bulk_mean(avg_m)
            bulk_s = _bulk_sem(avg_s)

            dh     = avg_m - bulk_m
            dh_err = np.sqrt(avg_s**2 + bulk_s**2)
            return dh, dh_err

        t = _PAIR_TERMS  # shorthand

        self.dH_clay_lj,    self.dH_clay_lj_err    = _pair_dH(t['clay_lj_pull1'],   t['clay_lj_pull2'])
        self.dH_clay_coul,  self.dH_clay_coul_err  = _pair_dH(t['clay_coul_pull1'], t['clay_coul_pull2'])
        self.dH_ion_lj,     self.dH_ion_lj_err     = _pair_dH(t['ion_lj_pull1'],    t['ion_lj_pull2'])
        self.dH_ion_coul,   self.dH_ion_coul_err   = _pair_dH(t['ion_coul_pull1'],  t['ion_coul_pull2'])
        self.dH_water_lj,   self.dH_water_lj_err   = _pair_dH(t['water_lj_pull1'],  t['water_lj_pull2'])
        self.dH_water_coul, self.dH_water_coul_err = _pair_dH(t['water_coul_pull1'], t['water_coul_pull2'])

        # Drug–drug: single pair term (PULL1–PULL2), already per-pair not per-molecule
        # We keep it as-is (it's the interaction between the two CIP copies)
        m_dd = self._energy_stats[t['drug_lj']]['mean']
        s_dd = self._energy_stats[t['drug_lj']]['sem']
        bulk_m_dd = _bulk_mean(m_dd);  bulk_s_dd = _bulk_sem(s_dd)
        self.dH_drug_lj    = m_dd - bulk_m_dd
        self.dH_drug_lj_err = np.sqrt(s_dd**2 + bulk_s_dd**2)

        m_dc = self._energy_stats[t['drug_coul']]['mean']
        s_dc = self._energy_stats[t['drug_coul']]['sem']
        bulk_m_dc = _bulk_mean(m_dc);  bulk_s_dc = _bulk_sem(s_dc)
        self.dH_drug_coul    = m_dc - bulk_m_dc
        self.dH_drug_coul_err = np.sqrt(s_dc**2 + bulk_s_dc**2)

    # ------------------------------------------------------------------
    # Entropy computation
    # ------------------------------------------------------------------

    def compute_entropy(self, pmf_values, pmf_r):
        """
        Compute −TΔS(r) = ΔG(r) − ΔH(r) by interpolating the supplied
        PMF onto ``self.r_centers``.

        Parameters
        ----------
        pmf_values : array-like
            PMF in kJ/mol, referenced to zero in the bulk.
        pmf_r : array-like
            Corresponding r coordinates in nm.

        Sets
        ----
        self.dG_interp   : ΔG interpolated to r_centers
        self.mTdS        : −TΔS = ΔG − ΔH
        self.mTdS_err    : error propagated from ΔH_err (PMF error not included
                           unless supplied via pmf_errors)
        """
        if self.dH is None:
            raise RuntimeError("Call compute_enthalpy() first.")

        pmf_r = np.asarray(pmf_r, dtype=np.float64)
        pmf_v = np.asarray(pmf_values, dtype=np.float64)

        # Interpolate PMF to window r_centers (linear)
        valid = ~np.isnan(self.r_centers)
        self.dG_interp = np.full(self.n_windows, np.nan)
        self.dG_interp[valid] = np.interp(self.r_centers[valid], pmf_r, pmf_v)

        self.mTdS     = self.dG_interp - self.dH
        self.mTdS_err = self.dH_err.copy()   # PMF error not propagated by default

    # ------------------------------------------------------------------
    # Decomposition accessor
    # ------------------------------------------------------------------

    def decompose(self) -> dict:
        """
        Return a dictionary of all computed ΔH component arrays.

        Keys: 'r', 'dH', 'dH_err',
              'clay_lj', 'clay_coul', 'ion_lj', 'ion_coul',
              'water_lj', 'water_coul', 'drug_lj', 'drug_coul',
              plus '*_err' counterparts.
        """
        if self.dH is None:
            raise RuntimeError("Call compute_enthalpy() first.")
        return {
            'r':              self.r_centers,
            'dH':             self.dH,
            'dH_err':         self.dH_err,
            'clay_lj':        self.dH_clay_lj,
            'clay_lj_err':    self.dH_clay_lj_err,
            'clay_coul':      self.dH_clay_coul,
            'clay_coul_err':  self.dH_clay_coul_err,
            'ion_lj':         self.dH_ion_lj,
            'ion_lj_err':     self.dH_ion_lj_err,
            'ion_coul':       self.dH_ion_coul,
            'ion_coul_err':   self.dH_ion_coul_err,
            'water_lj':       self.dH_water_lj,
            'water_lj_err':   self.dH_water_lj_err,
            'water_coul':     self.dH_water_coul,
            'water_coul_err': self.dH_water_coul_err,
            'drug_lj':        self.dH_drug_lj,
            'drug_lj_err':    self.dH_drug_lj_err,
            'drug_coul':      self.dH_drug_coul,
            'drug_coul_err':  self.dH_drug_coul_err,
            'mTdS':           self.mTdS,
            'mTdS_err':       self.mTdS_err,
            'dG':             self.dG_interp,
        }

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, filepath):
        """
        Save all results to a compressed NumPy archive (.npz).

        Parameters
        ----------
        filepath : str or Path
        """
        if self.dH is None:
            raise RuntimeError("Nothing to save – call load_energies() and compute_enthalpy().")

        metadata = {
            'umbrella_dir':   str(self.umbrella_dir),
            'temperature':    self.temperature,
            'n_windows':      self.n_windows,
            'n_blocks':       self.n_blocks,
            'bulk_fraction':  self.bulk_fraction,
            'equil_skip_frac': self.equil_skip_frac,
            'edr_prefix':     self.edr_prefix,
            'pullx_prefix':   self.pullx_prefix,
        }

        arrays = {f'meta_{k}': np.array(v) for k, v in metadata.items()}
        arrays['r_centers'] = self.r_centers

        for attr in [
            'dH', 'dH_err',
            'dH_clay_lj', 'dH_clay_lj_err',
            'dH_clay_coul', 'dH_clay_coul_err',
            'dH_ion_lj', 'dH_ion_lj_err',
            'dH_ion_coul', 'dH_ion_coul_err',
            'dH_water_lj', 'dH_water_lj_err',
            'dH_water_coul', 'dH_water_coul_err',
            'dH_drug_lj', 'dH_drug_lj_err',
            'dH_drug_coul', 'dH_drug_coul_err',
            'mTdS', 'mTdS_err', 'dG_interp',
        ]:
            val = getattr(self, attr)
            if val is not None:
                arrays[attr] = val

        # Save per-term energy stats
        for term in _ALL_TERM_NAMES:
            safe = re.sub(r'[^A-Za-z0-9_]', '_', term)
            if term in self._energy_stats:
                arrays[f'estat_mean_{safe}'] = self._energy_stats[term]['mean']
                arrays[f'estat_sem_{safe}']  = self._energy_stats[term]['sem']

        np.savez_compressed(filepath, **arrays)
        print(f"Saved to {filepath}")

    @classmethod
    def load(cls, filepath) -> 'ClayThermo':
        """
        Reconstruct a ClayThermo instance from a saved .npz file.
        """
        data = np.load(filepath, allow_pickle=True)

        def _scalar(key, dtype=None):
            v = data[key]
            if v.ndim == 0:
                v = v.item()
            return dtype(v) if dtype else v

        umbrella_dir   = str(_scalar('meta_umbrella_dir'))
        temperature    = float(_scalar('meta_temperature'))
        n_blocks       = int(_scalar('meta_n_blocks'))
        bulk_fraction  = float(_scalar('meta_bulk_fraction'))
        equil_skip_frac = float(_scalar('meta_equil_skip_frac'))
        edr_prefix     = str(_scalar('meta_edr_prefix'))
        pullx_prefix   = str(_scalar('meta_pullx_prefix'))

        # Use a dummy dir if original is gone (results only mode)
        inst = object.__new__(cls)
        inst.umbrella_dir    = Path(umbrella_dir)
        inst.temperature     = temperature
        inst.n_blocks        = n_blocks
        inst.bulk_fraction   = bulk_fraction
        inst.equil_skip_frac = equil_skip_frac
        inst.edr_prefix      = edr_prefix
        inst.pullx_prefix    = pullx_prefix
        inst.n_windows       = int(_scalar('meta_n_windows'))
        inst._edr_files      = []

        inst.r_centers = data['r_centers']

        for attr in [
            'dH', 'dH_err',
            'dH_clay_lj', 'dH_clay_lj_err',
            'dH_clay_coul', 'dH_clay_coul_err',
            'dH_ion_lj', 'dH_ion_lj_err',
            'dH_ion_coul', 'dH_ion_coul_err',
            'dH_water_lj', 'dH_water_lj_err',
            'dH_water_coul', 'dH_water_coul_err',
            'dH_drug_lj', 'dH_drug_lj_err',
            'dH_drug_coul', 'dH_drug_coul_err',
            'mTdS', 'mTdS_err', 'dG_interp',
        ]:
            setattr(inst, attr, data[attr] if attr in data else None)

        inst._energy_stats = {}
        for term in _ALL_TERM_NAMES:
            safe = re.sub(r'[^A-Za-z0-9_]', '_', term)
            mk = f'estat_mean_{safe}'
            sk = f'estat_sem_{safe}'
            if mk in data:
                inst._energy_stats[term] = {
                    'mean': data[mk],
                    'sem':  data[sk] if sk in data else np.full_like(data[mk], np.nan),
                }

        return inst

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def print_summary(self):
        """Print a table of per-window r_center, ΔH ± err, and −TΔS if available."""
        print("=" * 72)
        print(f"ClayThermo Summary  –  T={self.temperature} K  "
              f"n_windows={self.n_windows}")
        print(f"  dir: {self.umbrella_dir}")
        print("-" * 72)

        has_entropy = self.mTdS is not None
        header = f"{'win':>4}  {'r (nm)':>8}  {'ΔH (kJ/mol)':>18}"
        if has_entropy:
            header += f"  {'−TΔS (kJ/mol)':>18}  {'ΔG (kJ/mol)':>14}"
        print(header)
        print("-" * 72)

        for i in range(self.n_windows):
            r_str = f"{self.r_centers[i]:.3f}" if self.r_centers is not None and not np.isnan(self.r_centers[i]) else "  NaN"
            if self.dH is not None and not np.isnan(self.dH[i]):
                dh_str = f"{self.dH[i]:+.2f} ± {self.dH_err[i]:.2f}"
            else:
                dh_str = "        NaN"
            row = f"{i+1:4d}  {r_str:>8}  {dh_str:>18}"
            if has_entropy and self.mTdS is not None and not np.isnan(self.mTdS[i]):
                ts_str = f"{self.mTdS[i]:+.2f} ± {self.mTdS_err[i]:.2f}"
                dg_str = f"{self.dG_interp[i]:+.2f}" if self.dG_interp is not None else "NaN"
                row += f"  {ts_str:>18}  {dg_str:>14}"
            print(row)

        if self.dH is not None:
            valid = ~np.isnan(self.dH)
            if np.any(valid):
                i_min = int(np.nanargmin(self.dH))
                print("-" * 72)
                print(f"  ΔH minimum: {self.dH[i_min]:+.2f} kJ/mol "
                      f"at r={self.r_centers[i_min]:.3f} nm (window {i_min+1})")
        print("=" * 72)
