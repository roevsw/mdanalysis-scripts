# mdanalysis-scripts

Python scripts for analysis of molecular dynamics (MD) trajectories from GROMACS simulations, focused on aqueous ion solvation and clay–organic–ion–water systems.

## Overview

### Ion Solvation Shell Analysis

Tools for characterising hydration shells around ions in aqueous electrolyte solutions from equilibrium MD trajectories.

- **`EquilibriumAnalysisOptimized`** — Coordination numbers, radial distribution functions (RDFs), hydration shell occupancy, and shell dynamics. Optimised for large trajectories with KDTree neighbour searches, vectorised calculations, and parallel processing.
- **`SaltComparison`** — Compares solvation shell structure and water dipole distributions across salt types and concentrations.
- **`ZDirectionalAnalysis`** — Density profiles along the z-axis, interface detection, and layer-resolved ion and water properties for interfacial systems (clays, membranes, bilayers).
- **`MolecularAnalysis`** — General-purpose trajectory analysis for multi-component systems: RDFs, coordination environments, and spatial distributions.

**Plotting** — each analysis class has a dedicated companion plotter for publication-ready figures:

- **`MolecularAnalysisPlotter`** — RDFs, ion binding and competition, coordination analysis, time series.
- **`ZDirectionalPlotter`** — z-density profiles and interface plots. Includes `PublicationExportMixin` for consistent figure export across all plotters.

### Clay–Organic–Ion–Water Analysis (Umbrella Sampling)

Tools for studying the adsorption of organic molecules on montmorillonite clay surfaces in ionic aqueous environments, using GROMACS umbrella sampling data.

- **`ClayOrganicIonWaterAnalysis`** — RDFs, coordination numbers, and spatial distribution analysis for clay–molecule–ion–water systems.
- **`ClayPMF`** — 1D WHAM potential of mean force (PMF) as a function of distance from the clay surface, from GROMACS `pullx*.xvg` files.
- **`ClayPMF2D`** — 2D WHAM PMF: distance × molecular tilt angle relative to the clay surface.
- **`ClayPMF3D`** — 3D WHAM PMF: distance × tilt angle × cation coordination number.
- **`ClayMBAR`** — MBAR-based PMF estimator using `pymbar`, as an alternative to WHAM.
- **`ClayMeanForce`** — Umbrella integration PMF from mean forces in GROMACS `pullf*.xvg` files (RFD, RBF, and Gaussian estimators).
- **`ClayThermo`** — Thermodynamic decomposition of the PMF into enthalpy (ΔH) and entropy (−TΔS) contributions from per-window EDR files.
- **`ClayConvergence`** — Block and cumulative WHAM convergence analysis to assess PMF convergence as a function of simulation length.

**Plotting**:

- **`ClayOrganicIonWaterAnalysisPlotter`** — Multi-component RDFs, competitive adsorption, bridge structures, stratified adsorption profiles, exchange kinetics, and selectivity coefficients.
- **`ClayPMFPlotter`** — PMF profiles (1D signed and symmetrised), umbrella window histograms, and sampling diagnostics.

## Usage

Add `my_scripts/` to your `PYTHONPATH` to import the analysis classes directly:

```bash
export PYTHONPATH="$(pwd)/my_scripts/:$PYTHONPATH"
```

Or add this line to your `.bashrc`.