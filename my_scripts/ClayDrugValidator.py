"""
ClayDrugValidator.py

Validation suite for neural network Potential of Mean Force (PMF) calculations
in pharmaceutical compound adsorption on clay minerals.

Compares a trained ClayPMFNeural / ClayPMFNeuralEnsemble against a reference
PMF (e.g. from ClayPMF3D WHAM) across all thermodynamically relevant metrics.

Extracted and adapted from Free_energy_extras/Improved_PMF.md.
Bug fixes applied:
  B1  _compute_desorption_barrier: replaced `is not np.nan` with np.isnan()
      (identity test on np.nan is always True, giving wrong barrier_error)
  B2  _analyze_cation_dependence: cache _get_nn_predictions() once before loop
      (original called it once per cation state = redundant forward passes)
  B3  _compute_error_statistics: fixed region mask indexing after `valid` filter
      (original sliced already-filtered `errors` with full-length boolean mask)
  B4  validate_against_experiment: fixed sign of predicted_Kd formula
      (was exp(+ΔG/RT) → always < 1; corrected to exp(-ΔG/RT) consistent with
      PMF_to_Kd.md convention K_d = [surface]/[bulk] so K_d > 1 = favorable)

Usage
-----
    from ClayDrugValidator import ClayDrugValidator

    validator = ClayDrugValidator(
        nn_pmf=ensemble,             # ClayPMFNeuralEnsemble
        reference_values=wham_flat,  # 1-D array of PMF values in kJ/mol
        reference_coords=coords,     # (N, 3): [r (nm), theta (deg), n_cat]
        drug_name="CIP+",
    )
    metrics = validator.compute_adsorption_metrics()
    validator.print_summary(metrics)
    validator.plot_validation_summary(save_path="validation.pdf")
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


class ClayDrugValidator:
    """
    Validation suite for neural network PMF in clay-drug systems.

    Parameters
    ----------
    nn_pmf : ClayPMFNeural or ClayPMFNeuralEnsemble
        Trained neural network PMF.  Must expose a ``predict(r, theta, n_cat)``
        method that accepts 1-D arrays and returns a 1-D array of PMF values
        in kJ/mol.
    reference_values : array-like, shape (N,)
        Reference PMF values in kJ/mol (from WHAM or another method).
    reference_coords : array-like, shape (N, 3)
        Coordinates for each reference point: [r (nm), theta (deg), n_cat].
    clay_surface_z : float, default=0.0
        Position of the clay surface; ``r`` in *reference_coords* is measured
        from this position.
    kT : float, default=2.478
        Thermal energy at 298 K in kJ/mol (= k_B * 298.15 K).
    drug_name : str, optional
        Name of the pharmaceutical compound (used in plots / summaries).
    """

    def __init__(self, nn_pmf, reference_values, reference_coords,
                 clay_surface_z=0.0, kT=2.478, drug_name=None):
        self.nn = nn_pmf
        self.ref_values = np.asarray(reference_values, dtype=float).ravel()
        self.ref_coords = np.asarray(reference_coords, dtype=float)
        self.z_surface = clay_surface_z
        self.kT = kT
        self.drug_name = drug_name or "pharmaceutical"

        # Distance from clay surface (column 0 is r)
        self.distances = self.ref_coords[:, 0]

        if len(self.ref_values) != len(self.ref_coords):
            raise ValueError(
                f"reference_values ({len(self.ref_values)}) and "
                f"reference_coords ({len(self.ref_coords)}) must have the "
                f"same length"
            )

        # Remove NaN/Inf reference values up-front so all internal arrays
        # are already clean (avoids repeated NaN propagation downstream)
        valid = np.isfinite(self.ref_values)
        self.ref_values = self.ref_values[valid]
        self.ref_coords = self.ref_coords[valid]
        self.distances = self.distances[valid]

        print(f"ClayDrugValidator initialised with {len(self.ref_values)} valid points")
        print(f"  Distance range : [{self.distances.min():.3f}, {self.distances.max():.3f}] nm")
        print(f"  PMF range      : [{self.ref_values.min():.2f}, {self.ref_values.max():.2f}] kJ/mol")

        # Adsorption regions (r_min nm, r_max nm, description)
        self.regions = {
            'bulk':         (3.0,  np.inf, 'Fully solvated, no clay influence'),
            'diffuse':      (1.5,  3.0,   'Weak electrostatic influence'),
            'stern':        (0.6,  1.5,   'Outer-sphere adsorption'),
            'inner_sphere': (0.3,  0.6,   'Direct surface contact'),
            'intercalated': (0.0,  0.3,   'Between clay layers'),
        }

        # Cation states
        self.cation_states = {
            0: 'no_cation',
            1: 'monovalent_bridge',
            2: 'divalent_bridge',
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_adsorption_metrics(self):
        """
        Compute all adsorption-relevant validation metrics.

        Returns
        -------
        dict
            Dictionary containing all validation metrics.
        """
        ref_1d, dist_1d = self._project_to_1d(self.ref_values, self.distances)
        nn_values = self._get_nn_predictions()
        nn_1d, _ = self._project_to_1d(nn_values, self.distances)

        metrics = {'drug': self.drug_name, 'kT': self.kT}
        metrics.update(self._compute_adsorption_energy(ref_1d, nn_1d, dist_1d))
        metrics.update(self._compute_well_characteristics(ref_1d, nn_1d, dist_1d))
        metrics.update(self._compute_desorption_barrier(ref_1d, nn_1d, dist_1d))
        metrics['secondary_minima'] = self._find_secondary_minima(ref_1d, nn_1d, dist_1d)
        metrics['cation_dependence'] = self._analyze_cation_dependence()
        metrics.update(self._compute_error_statistics(nn_values))
        metrics['pass'] = self._evaluate_pass(metrics)
        return metrics

    def validate_against_experiment(self, experimental_Kd, temperature=298.15):
        """
        Validate the NN PMF against an experimental dissociation/partition constant.

        Uses the convention K_d = [surface] / [bulk] so that K_d > 1 implies
        favorable adsorption (consistent with PMF_to_Kd.md Method 1):

            K_d = exp(−ΔG_ads / RT)
            ΔG_ads = −RT ln(K_d)

        Parameters
        ----------
        experimental_Kd : float
            Experimental dimensionless partition coefficient
            (K_d = [surface]/[bulk]; K_d > 1 means favorable adsorption).
        temperature : float, default=298.15
            Temperature in Kelvin.

        Returns
        -------
        dict
            Comparison between NN-predicted and experimental values.
        """
        R = 8.314e-3  # kJ mol⁻¹ K⁻¹
        metrics = self.compute_adsorption_metrics()
        nn_deltaG = metrics['DeltaG_ads_nn']   # negative for favorable adsorption

        # FIX B4: K_d = exp(−ΔG/RT).  Original had exp(+ΔG/RT) → K_d < 1 always.
        predicted_Kd = np.exp(-nn_deltaG / (R * temperature))

        # Experimental ΔG: ΔG = −RT ln(K_d)
        exp_deltaG = -R * temperature * np.log(experimental_Kd)

        return {
            'DeltaG_exp_kJ':  float(exp_deltaG),
            'DeltaG_nn_kJ':   float(nn_deltaG),
            'DeltaDeltaG_kJ': float(nn_deltaG - exp_deltaG),
            'Kd_exp':         float(experimental_Kd),
            'Kd_pred':        float(predicted_Kd),
            'Kd_ratio':       float(predicted_Kd / experimental_Kd),
            'pass':           0.5 < (predicted_Kd / experimental_Kd) < 2.0,
        }

    def plot_validation_summary(self, figsize=(14, 10), save_path=None):
        """
        Create a 6-panel validation figure.

        Panels
        ------
        (0,0) Reference vs NN scatter plot
        (0,1) Error distribution histogram
        (0,2) 1D PMF comparison (Boltzmann-projected)
        (1,0) Error vs distance
        (1,1) Region-specific MAE bar chart
        (1,2) Cation-dependent adsorption energies

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # Pre-compute everything once
        nn_values = self._get_nn_predictions()
        errors = nn_values - self.ref_values
        valid = np.isfinite(errors)
        err_stats = self._compute_error_statistics(nn_values)

        # Panel 1: Reference vs NN scatter
        ax = axes[0, 0]
        ax.scatter(self.ref_values[valid], nn_values[valid], alpha=0.3, s=5, c='steelblue')
        lims = [min(self.ref_values[valid].min(), nn_values[valid].min()),
                max(self.ref_values[valid].max(), nn_values[valid].max())]
        ax.plot(lims, lims, 'r--', lw=2, label='Perfect agreement')
        ax.set_xlabel('Reference PMF (kJ/mol)')
        ax.set_ylabel('NN PMF (kJ/mol)')
        ax.set_title(f'Scatter  |  R² = {err_stats["R2"]:.4f}')
        ax.legend()

        # Panel 2: Error histogram
        ax = axes[0, 1]
        err_flat = errors[valid]
        ax.hist(err_flat, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.axvline(np.median(err_flat), color='green', linestyle='-', linewidth=2,
                   label=f'Median: {np.median(err_flat):.3f}')
        ax.set_xlabel('Prediction error (kJ/mol)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Error distribution  |  MAE = {np.mean(np.abs(err_flat)):.3f}')
        ax.legend()

        # Panel 3: 1D PMF comparison
        ax = axes[0, 2]
        ref_1d, dist_1d = self._project_to_1d(self.ref_values, self.distances)
        nn_1d, _ = self._project_to_1d(nn_values, self.distances)
        ax.plot(dist_1d, ref_1d, 'k-', lw=2, label='Reference')
        ax.plot(dist_1d, nn_1d, 'r--', lw=2, label='NN')
        ax.axvline(0, color='grey', ls=':', alpha=0.7, label='Clay surface')
        ax.axhline(0, color='grey', ls=':', alpha=0.5)
        ax.set_xlabel('Distance from surface (nm)')
        ax.set_ylabel('W(r) (kJ/mol)')
        ax.set_title('1D PMF comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel 4: Error vs distance
        ax = axes[1, 0]
        ax.scatter(self.distances[valid], errors[valid], alpha=0.3, s=5, c='steelblue')
        ax.axhline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Distance from surface (nm)')
        ax.set_ylabel('Error (kJ/mol)')
        ax.set_title('Error vs distance')

        # Panel 5: Region-specific MAE
        ax = axes[1, 1]
        region_stats = err_stats['region_errors']
        regions = list(region_stats.keys())
        mae_values = [region_stats[r]['MAE'] for r in regions]
        colors = ['#2ecc71' if m < 2.0 else '#e74c3c' for m in mae_values]
        ax.bar(regions, mae_values, color=colors, edgecolor='black')
        ax.axhline(2.0, color='red', linestyle='--', linewidth=2, label='Threshold')
        ax.set_xlabel('Region')
        ax.set_ylabel('MAE (kJ/mol)')
        ax.set_title('Region-specific errors')
        ax.legend()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Panel 6: Cation dependence
        ax = axes[1, 2]
        cation_metrics = self._analyze_cation_dependence()
        cations, ref_dG, nn_dG = [], [], []
        for name, data in cation_metrics.items():
            if 'DeltaG_ads_ref' in data:
                cations.append(name.replace('_', ' ').title())
                ref_dG.append(data['DeltaG_ads_ref'])
                nn_dG.append(data['DeltaG_ads_nn'])
        if cations:
            x = np.arange(len(cations))
            width = 0.35
            ax.bar(x - width / 2, ref_dG, width, label='Reference', color='steelblue')
            ax.bar(x + width / 2, nn_dG, width, label='NN', color='tomato')
            ax.set_xlabel('Cation state')
            ax.set_ylabel('ΔG_ads (kJ/mol)')
            ax.set_title('Cation dependence of adsorption')
            ax.set_xticks(x)
            ax.set_xticklabels(cations)
            ax.legend()

        plt.suptitle(f'PMF Validation Summary: {self.drug_name}', fontsize=14, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved validation plot to {save_path}")

        return fig

    def print_summary(self, metrics=None):
        """Print a formatted validation summary to stdout."""
        if metrics is None:
            metrics = self.compute_adsorption_metrics()

        print("\n" + "=" * 60)
        print(f"PMF Validation Summary: {metrics['drug']}")
        print("=" * 60)

        print("\n[ADSORPTION THERMODYNAMICS]")
        print(f"  ΔG_ads (reference):  {metrics['DeltaG_ads_ref']:8.2f} kJ/mol")
        print(f"  ΔG_ads (NN):         {metrics['DeltaG_ads_nn']:8.2f} kJ/mol")
        print(f"  ΔΔG_ads:             {metrics['DeltaDeltaG_ads']:8.2f} kJ/mol")
        print(f"  Pass:                {abs(metrics['DeltaDeltaG_ads']) < 2.0}")

        print("\n[WELL CHARACTERISTICS]")
        print(f"  Well depth (ref):    {metrics['well_depth_ref']:8.2f} kJ/mol")
        print(f"  Well depth (NN):     {metrics['well_depth_nn']:8.2f} kJ/mol")
        print(f"  Well position (ref): {metrics['well_position_ref']:8.3f} nm")
        print(f"  Well position (NN):  {metrics['well_position_nn']:8.3f} nm")
        print(f"  Position error:      {metrics['well_position_error']:8.3f} nm")

        print("\n[DESORPTION BARRIER]")
        barrier_ref = metrics['desorption_barrier_ref']
        barrier_nn = metrics['desorption_barrier_nn']
        barrier_err = metrics['barrier_error']
        print(f"  Barrier (ref):       {barrier_ref:8.2f} kJ/mol" if not np.isnan(barrier_ref)
              else "  Barrier (ref):       n/a")
        print(f"  Barrier (NN):        {barrier_nn:8.2f} kJ/mol" if not np.isnan(barrier_nn)
              else "  Barrier (NN):        n/a")
        print(f"  Barrier error:       {barrier_err:8.2f} kJ/mol" if not np.isnan(barrier_err)
              else "  Barrier error:       n/a")

        print("\n[OVERALL STATISTICS]")
        print(f"  MAE:                 {metrics['overall_MAE']:8.3f} kJ/mol")
        print(f"  RMSE:                {metrics['overall_RMSE']:8.3f} kJ/mol")
        print(f"  Max error:           {metrics['overall_max_error']:8.3f} kJ/mol")
        print(f"  R²:                  {metrics['R2']:8.4f}")

        print("\n[REGION ERRORS]")
        for region, stats in metrics['region_errors'].items():
            status = "✓" if stats['MAE'] < 2.0 else "✗"
            print(f"  {region:15s} {status}  MAE = {stats['MAE']:.3f} kJ/mol  (n={stats['n_points']})")

        print("\n[CATION DEPENDENCE]")
        for name, data in metrics['cation_dependence'].items():
            if 'DeltaG_ads_ref' in data:
                status = "✓" if data['pass'] else "✗"
                print(f"  {name:20s} {status}  ΔΔG = {data['DeltaDeltaG']:6.2f} kJ/mol")

        print("\n[SECONDARY MINIMA]")
        sec = metrics['secondary_minima']
        print(f"  n_secondary (ref):   {sec['n_secondary_ref']}")
        print(f"  n_secondary (NN):    {sec['n_secondary_nn']}")

        print("\n" + "=" * 60)
        print(f"OVERALL VALIDATION: {'PASS' if metrics['pass'] else 'FAIL'}")
        print("=" * 60)

        return metrics['pass']

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_nn_predictions(self):
        """Return NN predictions at all (pre-filtered) reference coordinates."""
        try:
            return self.nn.predict(
                self.ref_coords[:, 0],
                self.ref_coords[:, 1],
                self.ref_coords[:, 2],
            )
        except AttributeError:
            # Fallback: nn is a plain callable
            return self.nn(self.ref_coords)

    def _project_to_1d(self, pmf_values, distances, n_bins=200):
        """
        Project 3D PMF onto the distance coordinate using Boltzmann weighting.

            W_1d(r) = −kT ln < exp(−W/kT) >_r

        Returns
        -------
        pmf_1d : ndarray (n_bins,)
        bin_centers : ndarray (n_bins,)
        """
        bins = np.linspace(distances.min(), distances.max(), n_bins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        pmf_1d = np.full(n_bins, np.nan)

        for i in range(n_bins):
            mask = (distances >= bins[i]) & (distances < bins[i + 1])
            if np.any(mask):
                boltz = np.exp(-pmf_values[mask] / self.kT)
                if np.sum(boltz) > 0:
                    pmf_1d[i] = -self.kT * np.log(np.mean(boltz))

        return pmf_1d, bin_centers

    def _compute_adsorption_energy(self, ref_1d, nn_1d, dist_1d):
        """ΔG_ads = W(min) − W(bulk), and their difference."""
        bulk_mask = (dist_1d > 3.5) & np.isfinite(ref_1d)
        if np.any(bulk_mask):
            bulk_ref = float(np.nanmean(ref_1d[bulk_mask]))
            bulk_nn = float(np.nanmean(nn_1d[bulk_mask]))
        else:
            bulk_ref = float(np.nanmean(ref_1d[-10:])) if len(ref_1d) > 10 else float(ref_1d[-1])
            bulk_nn = float(np.nanmean(nn_1d[-10:])) if len(nn_1d) > 10 else float(nn_1d[-1])

        min_ref = float(np.nanmin(ref_1d))
        min_nn = float(np.nanmin(nn_1d))

        return {
            'DeltaG_ads_ref':  min_ref - bulk_ref,
            'DeltaG_ads_nn':   min_nn - bulk_nn,
            'DeltaDeltaG_ads': (min_nn - bulk_nn) - (min_ref - bulk_ref),
            'bulk_ref_kJ':     bulk_ref,
            'bulk_nn_kJ':      bulk_nn,
        }

    def _compute_well_characteristics(self, ref_1d, nn_1d, dist_1d):
        """Depth and position of the primary adsorption well."""
        ref_minima = find_peaks(-ref_1d, prominence=1.0)[0]
        nn_minima = find_peaks(-nn_1d, prominence=1.0)[0]

        bulk_ref = float(np.nanmean(ref_1d[dist_1d > 3.5])) if np.any(dist_1d > 3.5) else 0.0
        bulk_nn = float(np.nanmean(nn_1d[dist_1d > 3.5])) if np.any(dist_1d > 3.5) else 0.0

        if len(ref_minima) > 0:
            ref_min_idx = ref_minima[np.argmin(ref_1d[ref_minima])]
        else:
            ref_min_idx = int(np.nanargmin(ref_1d))
        ref_well_depth = float(np.abs(ref_1d[ref_min_idx] - bulk_ref))
        ref_well_position = float(dist_1d[ref_min_idx])

        if len(nn_minima) > 0:
            nn_min_idx = nn_minima[np.argmin(nn_1d[nn_minima])]
        else:
            nn_min_idx = int(np.nanargmin(nn_1d))
        nn_well_depth = float(np.abs(nn_1d[nn_min_idx] - bulk_nn))
        nn_well_position = float(dist_1d[nn_min_idx])

        return {
            'well_depth_ref':      ref_well_depth,
            'well_depth_nn':       nn_well_depth,
            'well_depth_error':    nn_well_depth - ref_well_depth,
            'well_position_ref':   ref_well_position,
            'well_position_nn':    nn_well_position,
            'well_position_error': nn_well_position - ref_well_position,
        }

    def _compute_desorption_barrier(self, ref_1d, nn_1d, dist_1d):
        """Barrier from the adsorption well to the bulk region (r > 3.0 nm)."""
        well_idx_ref = int(np.nanargmin(ref_1d))
        well_idx_nn = int(np.nanargmin(nn_1d))
        bulk_start_idx = int(np.searchsorted(dist_1d, 3.0))

        if bulk_start_idx > well_idx_ref:
            segment_ref = ref_1d[well_idx_ref:bulk_start_idx + 1]
            barrier_ref = float(np.nanmax(segment_ref) - ref_1d[well_idx_ref])
        else:
            barrier_ref = np.nan

        if bulk_start_idx > well_idx_nn:
            segment_nn = nn_1d[well_idx_nn:bulk_start_idx + 1]
            barrier_nn = float(np.nanmax(segment_nn) - nn_1d[well_idx_nn])
        else:
            barrier_nn = np.nan

        # FIX B1: use np.isnan() — `is not np.nan` is an identity test that is
        # always True in CPython (np.nan is a singleton), so the original code
        # always computed barrier_error even when one side was nan.
        if not np.isnan(barrier_ref) and not np.isnan(barrier_nn):
            barrier_error = barrier_nn - barrier_ref
        else:
            barrier_error = np.nan

        return {
            'desorption_barrier_ref': barrier_ref,
            'desorption_barrier_nn':  barrier_nn,
            'barrier_error':          barrier_error,
        }

    def _find_secondary_minima(self, ref_1d, nn_1d, dist_1d):
        """Identify metastable adsorption states (all minima except the global one)."""
        ref_minima = find_peaks(-ref_1d, prominence=2.0)[0]
        nn_minima = find_peaks(-nn_1d, prominence=2.0)[0]

        ref_global_idx = int(np.argmin(ref_1d[ref_minima])) if len(ref_minima) > 0 else -1
        nn_global_idx = int(np.argmin(nn_1d[nn_minima])) if len(nn_minima) > 0 else -1

        ref_secondary = [ref_minima[i] for i in range(len(ref_minima)) if i != ref_global_idx]
        nn_secondary = [nn_minima[i] for i in range(len(nn_minima)) if i != nn_global_idx]

        return {
            'n_secondary_ref': len(ref_secondary),
            'n_secondary_nn':  len(nn_secondary),
            'positions_ref':   [float(dist_1d[i]) for i in ref_secondary],
            'positions_nn':    [float(dist_1d[i]) for i in nn_secondary],
            'pass':            len(ref_secondary) == len(nn_secondary),
        }

    def _analyze_cation_dependence(self):
        """Per-cation-state adsorption energies and NN vs reference comparison."""
        # FIX B2: compute NN predictions once here, not once per cation state
        all_nn = self._get_nn_predictions()

        cation_metrics = {}
        for n_cat, name in self.cation_states.items():
            mask = np.abs(self.ref_coords[:, 2] - n_cat) < 0.5
            if int(np.sum(mask)) < 10:
                cation_metrics[name] = {
                    'error': 'insufficient_samples',
                    'n_samples': int(np.sum(mask)),
                }
                continue

            ref_cat = self.ref_values[mask]
            dist_cat = self.distances[mask]
            nn_cat = all_nn[mask]   # slice from cached predictions

            bulk_mask = dist_cat > 3.5
            if np.any(bulk_mask):
                bulk_ref = float(np.mean(ref_cat[bulk_mask]))
                bulk_nn = float(np.mean(nn_cat[bulk_mask]))
            else:
                bulk_ref = float(np.percentile(ref_cat, 90))
                bulk_nn = float(np.percentile(nn_cat, 90))

            deltaG_ref = float(np.min(ref_cat)) - bulk_ref
            deltaG_nn = float(np.min(nn_cat)) - bulk_nn

            cation_metrics[name] = {
                'DeltaG_ads_ref': deltaG_ref,
                'DeltaG_ads_nn':  deltaG_nn,
                'DeltaDeltaG':    deltaG_nn - deltaG_ref,
                'n_samples':      int(np.sum(mask)),
                'pass':           abs(deltaG_nn - deltaG_ref) < 2.0,
            }

        return cation_metrics

    def _compute_error_statistics(self, nn_values):
        """Global and per-region MAE / RMSE / R² statistics."""
        errors = nn_values - self.ref_values
        valid = np.isfinite(errors)
        errors_valid = errors[valid]

        # FIX B3: keep mask in the original (N,) space and slice `errors`
        # (not `errors_valid`) to avoid shape mismatch after valid-filtering.
        region_errors = {}
        for region_name, (r_min, r_max, _) in self.regions.items():
            region_mask = (self.distances >= r_min) & (self.distances < r_max) & valid
            if np.any(region_mask):
                region_errors_i = errors[region_mask]   # shape = sum(region_mask)
                region_errors[region_name] = {
                    'MAE':      float(np.mean(np.abs(region_errors_i))),
                    'RMSE':     float(np.sqrt(np.mean(region_errors_i ** 2))),
                    'Max':      float(np.max(np.abs(region_errors_i))),
                    'n_points': int(np.sum(region_mask)),
                }

        ref_valid = self.ref_values[valid]
        ss_res = float(np.sum(errors_valid ** 2))
        ss_tot = float(np.sum((ref_valid - np.mean(ref_valid)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        return {
            'overall_MAE':       float(np.mean(np.abs(errors_valid))),
            'overall_RMSE':      float(np.sqrt(np.mean(errors_valid ** 2))),
            'overall_max_error': float(np.max(np.abs(errors_valid))),
            'region_errors':     region_errors,
            'R2':                float(r2),
        }

    def _evaluate_pass(self, metrics):
        """
        PASS if at least 4 out of 5 acceptance criteria are met.

        Primary (must pass): ΔΔG_ads < 2 kJ/mol, well position error < 0.1 nm,
                             well depth error < 2 kJ/mol.
        Secondary:           overall MAE < 2 kJ/mol, R² > 0.95.
        """
        checks = [
            abs(metrics.get('DeltaDeltaG_ads',    np.inf)) < 2.0,
            abs(metrics.get('well_position_error', np.inf)) < 0.1,
            abs(metrics.get('well_depth_error',    np.inf)) < 2.0,
            metrics.get('overall_MAE', np.inf) < 2.0,
            metrics.get('R2', 0.0) > 0.95,
        ]
        return sum(checks) >= 4
