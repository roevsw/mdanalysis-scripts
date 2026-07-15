# mdanalysis-scripts

Python scripts for analysis of molecular dynamics (MD) trajectories from GROMACS simulations, focused on aqueous ion solvation and clay–organic–ion–water systems.

## Overview

### Ion Solvation Shell Analysis

Tools for characterising hydration shells around ions in aqueous electrolyte solutions from equilibrium MD trajectories.

- **`EquilibriumAnalysisOptimized`** — Coordination numbers, radial distribution functions (RDFs), hydration shell occupancy, and shell dynamics. Optimised for large trajectories with KDTree neighbour searches, vectorised calculations, and parallel processing.
- **`SaltComparison`** — Compares solvation shell structure and water dipole distributions across salt types and concentrations.
- **`ZDirectionalAnalysis`** — Density profiles along the z-axis, interface detection, and layer-resolved ion and water properties for interfacial systems (clays, membranes, bilayers).
- **`MolecularAnalysis`** — General-purpose trajectory analysis for multi-component systems: RDFs, coordination environments, and spatial distributions.

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

### Clay–Organic–Ion–Water Analysis (Neural-Network PMF)

Neural-network representations of the 3D free-energy surface, building on `ClayPMF3D` WHAM results. Supports two backends: a pure-NumPy implementation and an optional PyTorch backend (`NeuralNetworkTorch`) for GPU-accelerated training.

- **`ClayPMFNeural`** — Fits a feedforward neural network to a single `ClayPMF3D` object. Two training strategies:
  - **Approach A** (`fit_smooth`): trains directly on the WHAM PMF grid to produce a smooth, differentiable W(r, θ, n_cat).
  - **Approach B** (`fit_reweighted`): computes per-frame WHAM weights from raw trajectory data, builds a finer-resolution unbiased P(r, θ, n_cat), and trains on that — avoiding coarse-binning artefacts of the original WHAM grid.
  - `predict` / `predict_b` / `predict_both` — evaluate the learned surface at arbitrary coordinates.
  - `tune_hyperparameters` — cross-validated grid search over network architecture and learning rate.
  - `save` / `load` — `.npz` serialisation for checkpointing.

- **`ClayPMFNeuralEnsemble`** — Pools data from multiple independent replicate `ClayPMF3D` runs into a single NN training session for a smoother, better-constrained free-energy surface. Also supports `fit_smooth_per_replicate` (one NN per replicate) for a deep ensemble with spatial uncertainty quantification via model spread.

- **`ClayDrugValidator`** — Validation suite comparing a trained neural PMF against a reference PMF (e.g. from WHAM). Computes adsorption and desorption barriers, cation dependence, error statistics by region, and experimental Kd comparison (`validate_against_experiment`). Produces a multi-panel validation summary figure.

### Plotting

Each analysis domain has a dedicated companion plotter for publication-ready figures. All plotters share a `PublicationExportMixin` for consistent figure export.

- **`MolecularAnalysisPlotter`** — RDFs, ion binding and competition, coordination analysis, time series. (Ion solvation shell analysis.)
- **`ZDirectionalPlotter`** — z-density profiles and interface plots. (Ion solvation / interfacial systems.)
- **`ClayOrganicIonWaterAnalysisPlotter`** — Multi-component RDFs, competitive adsorption, bridge structures, stratified adsorption profiles, exchange kinetics, and selectivity coefficients. (Clay–organic–ion–water systems.)
- **`ClayPMFPlotter`** — Unified PMF visualisation for all dimensionalities: 1D signed/symmetrised profiles and sampling diagnostics; 2D filled contour maps W(r,θ), marginals, coupling, conditional slices, and 3D surface views; 3D marginal panels W(r)/W(θ)/W(n_cat), fixed-axis contourf slices, and Kd-resolved plots; neural-ensemble uncertainty maps, loss curves, and ensemble coupling comparisons. (Umbrella sampling and neural-network PMF.)

#### Interactive 3D Figures

The `MolecularAnalysisPlotter.plot_spatial_binding_interactive()` method produces fully interactive 3D visualisations (rotate, zoom, pan) that can be exported as standalone HTML files.

- [Na⁺ spatial binding — carboxylic acid group](https://roevsw.github.io/mdanalysis-scripts/figures/spatial_binding_Na_carboxylic_acid.html)
- [Clay polyhedra — Si tetrahedra + Mg octahedra (CLAYFF MMT)](https://roevsw.github.io/mdanalysis-scripts/figures/clay_polyhedra_full.html)

## Usage

Add `my_scripts/` to your `PYTHONPATH` to import the analysis classes directly:

```bash
export PYTHONPATH="$(pwd)/my_scripts/:$PYTHONPATH"
```

Or add this line to your `.bashrc`.