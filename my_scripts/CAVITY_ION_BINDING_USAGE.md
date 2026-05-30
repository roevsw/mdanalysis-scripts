# Cavity-Specific Ion Binding Analysis

## Overview

This implementation adds the ability to analyze ion binding to specific Si hexagonal ring cavities in the clay surface. The analysis:

1. **Detects Si hexagonal rings** (6-membered rings) in each clay layer
2. **Counts ions** within cylindrical regions above each cavity
3. **Calculates statistics** (per-cavity timeseries, average occupancy, preferential sites)
4. **Visualizes results** with multiple plotting options

---

## 1. Analysis Method

### `analyze_cavity_ion_binding()`

Located in: `ClayOrganicIonWaterAnalysis.py`

#### Basic Usage

```python
from ClayOrganicIonWaterAnalysis import ClayOrganicIonWaterAnalysis

# Initialize analysis
analysis = ClayOrganicIonWaterAnalysis(
    trajectory='trajectory.xtc',
    topology='topology.tpr',
    ...
)

# Run cavity ion binding analysis
results = analysis.analyze_cavity_ion_binding(
    ion_types=['NA', 'MG'],           # Ions to analyze
    z_slice_centers=[-29.5, 26.5],    # Clay layer positions (Å)
    z_slice_width=2.0,                # Width of z-slice for Si selection
    cavity_radius=3.0,                # Radius of cylindrical region (Å)
    cavity_height=6.0,                # Height of cylindrical region (Å)
    si_si_threshold=4.5,              # Max Si-Si distance for ring detection
    step=1                            # Frame step size
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ion_types` | list of str | None | Ion names (e.g., ['NA', 'MG']). If None, uses all ions |
| `z_slice_centers` | list of float | None | Z-positions to analyze. If None, auto-detects clay layers |
| `z_slice_width` | float | 2.0 | Width of z-slice for Si atom selection (Å) |
| `cavity_radius` | float | 3.0 | Radius of cylindrical region above cavity (Å) |
| `cavity_height` | float | 6.0 | Height of cylindrical region above cavity (Å) |
| `si_si_threshold` | float | 4.5 | Maximum Si-Si distance for ring detection (Å) |
| `compute_per_cavity_timeseries` | bool | True | Calculate ion count timeseries for each cavity |
| `compute_avg_occupancy` | bool | True | Calculate average occupancy per cavity |
| `compute_preferential_sites` | bool | True | Identify cavities with highest occupancy |
| `compute_spatial_correlation` | bool | True | Analyze correlation between cavity position and ion density |
| `step` | int | 1 | Frame step size |
| `use_cache` | bool | True | Use cached results if available |
| `verbose` | bool | True | Print progress messages |

#### Output Structure

```python
results = {
    'z_slice_centers': np.ndarray,         # Z-positions analyzed
    'cavity_data': {                       # Per z-slice cavity information
        z_center: {
            'ring_centers': np.ndarray,    # XYZ coordinates of cavity centers
            'ring_indices': list,          # Si atom indices forming each ring
            'ring_radii': np.ndarray       # Average radius of each ring
        }
    },
    'ion_data': {                          # Per ion type binding data
        ion_type: {
            z_center: {
                'per_cavity_timeseries': np.ndarray,  # Shape (n_cavities, n_frames)
                'avg_occupancy': np.ndarray,          # Average ions per cavity
                'std_occupancy': np.ndarray,          # Standard deviation
                'max_occupancy': np.ndarray,          # Maximum occupancy
                'occupancy_fraction': np.ndarray      # Fraction of frames occupied
            }
        }
    },
    'preferential_sites': {                # Most occupied cavities
        ion_type: {
            z_center: {
                'cavity_indices': np.ndarray,      # Indices of top cavities
                'cavity_positions': np.ndarray,    # XY positions
                'avg_occupancy': np.ndarray        # Average occupancy values
            }
        }
    },
    'metadata': dict                       # Analysis parameters
}
```

---

## 2. Plotting Methods

Located in: `ClayOrganicIonWaterAnalysisPlotter.py`

### 2.1 `plot_cavity_ion_binding()`

**Overlays cavity markers on XY heatmaps, sized and colored by occupancy**

```python
from ClayOrganicIonWaterAnalysisPlotter import ClayOrganicIonWaterAnalysisPlotter

plotter = ClayOrganicIonWaterAnalysisPlotter(analysis)

plotter.plot_cavity_ion_binding(
    ion_types=['NA', 'MG'],
    z_slice_centers=[-29.5, 26.5],
    show_ion_density=True,              # Show background ion density
    ion_density_cmap='Purples',
    cavity_marker='o',                  # Marker style
    cavity_colormap='hot',              # Colormap for occupancy
    cavity_size_range=(50, 500),        # Min/max marker size
    show_empty_cavities=True,           # Show zero-occupancy cavities
    empty_cavity_color='gray',
    figsize=(15, 5),
    dpi=300,
    save_plots=True,
    filename='cavity_binding.png'
)
```

**Key Features:**
- Background ion density heatmap (optional)
- Cavity markers sized by average occupancy
- Cavity markers colored by occupancy (hot = high, cool = low)
- Empty cavities shown in gray (optional)
- Colorbar showing occupancy scale

### 2.2 `plot_cavity_occupancy_timeseries()`

**Shows ion occupancy time series for each cavity**

```python
plotter.plot_cavity_occupancy_timeseries(
    ion_types=['NA'],
    z_slice_centers=[26.5],
    max_cavities_per_plot=10,           # Show top 10 cavities
    show_average=True,                  # Show average across all cavities
    avg_linewidth=3,
    avg_color='red',
    individual_alpha=0.3,               # Transparency for individual lines
    colormap='tab10',
    figsize=(14, 8),
    save_plots=True,
    filename='cavity_timeseries.png'
)
```

**Key Features:**
- Individual cavity timeseries (top N by occupancy)
- Average occupancy across all cavities (bold red line)
- Time axis in picoseconds
- Grid for easy reading

### 2.3 `plot_preferential_binding_sites()`

**Spatial map highlighting top-occupied cavities**

```python
plotter.plot_preferential_binding_sites(
    ion_types=['NA', 'MG'],
    z_slice_centers=[-29.5, 26.5],
    top_n=5,                            # Highlight top 5 cavities
    marker='*',                         # Star markers
    marker_size=300,
    marker_color='gold',
    marker_edgecolor='red',
    show_labels=True,                   # Show occupancy values
    label_fontsize=9,
    figsize=(15, 5),
    save_plots=True,
    filename='preferential_sites.png'
)
```

**Key Features:**
- All cavities shown as gray background
- Top N cavities highlighted with gold stars
- Occupancy labels (e.g., "0.85") on each star
- Spatial distribution of preferential binding sites

---

## 3. Example Workflow

### Complete Analysis Pipeline

```python
from ClayOrganicIonWaterAnalysis import ClayOrganicIonWaterAnalysis
from ClayOrganicIonWaterAnalysisPlotter import ClayOrganicIonWaterAnalysisPlotter

# ============================================================================
# Step 1: Initialize Analysis
# ============================================================================
analysis = ClayOrganicIonWaterAnalysis(
    trajectory='trajectory.xtc',
    topology='topology.tpr',
    clay_selection='resname MMT',
    ion_dict={'NA': 'name NA', 'MG': 'name MG'},
    organic_dict={'CIP': 'resname CIP'},
    water_selection='resname SOL'
)

# ============================================================================
# Step 2: Run Cavity Ion Binding Analysis
# ============================================================================
print("Running cavity ion binding analysis...")

results = analysis.analyze_cavity_ion_binding(
    ion_types=['NA', 'MG'],
    z_slice_centers=[-29.5, 26.5, 29.5],  # Three clay layers
    z_slice_width=2.0,
    cavity_radius=3.0,                     # 3 Å radius around cavity center
    cavity_height=6.0,                     # 6 Å above cavity
    si_si_threshold=4.5,                   # Si-Si distance for ring detection
    step=1,                                # Analyze every frame
    verbose=True
)

# ============================================================================
# Step 3: Visualize Results
# ============================================================================
plotter = ClayOrganicIonWaterAnalysisPlotter(analysis)

# 3a. Cavity binding spatial map
print("\nPlotting cavity binding spatial map...")
plotter.plot_cavity_ion_binding(
    ion_types=['NA', 'MG'],
    show_ion_density=True,
    cavity_size_range=(50, 500),
    save_plots=True,
    filename='cavity_binding_spatial.png',
    dpi=600
)

# 3b. Occupancy timeseries
print("\nPlotting occupancy timeseries...")
plotter.plot_cavity_occupancy_timeseries(
    ion_types=['NA', 'MG'],
    max_cavities_per_plot=10,
    show_average=True,
    save_plots=True,
    filename='cavity_timeseries.png',
    dpi=300
)

# 3c. Preferential binding sites
print("\nPlotting preferential binding sites...")
plotter.plot_preferential_binding_sites(
    ion_types=['NA', 'MG'],
    top_n=5,
    show_labels=True,
    save_plots=True,
    filename='preferential_sites.png',
    dpi=300
)

print("\n✅ Analysis complete!")
```

---

## 4. Integration with Existing Workflow

### Use with `plot_ion_xy_heatmaps()` from ZDirectionalPlotter

You can now combine cavity analysis with the existing XY heatmap visualization:

```python
from ZDirectionalAnalysis import ZDirectionalAnalysis
from ZDirectionalPlotter import ZDirectionalPlotter

# Run Z-directional analysis
z_analysis = ZDirectionalAnalysis(...)
z_analysis.calculate_ion_density_profiles(...)

# Run cavity analysis
analysis.analyze_cavity_ion_binding(
    ion_types=['NA'],
    z_slice_centers=[-29.5, 26.5, 29.5],
    cavity_radius=3.0,
    cavity_height=6.0
)

# Plot XY heatmaps with clay overlays
plotter_z = ZDirectionalPlotter(z_analysis)
plotter_z.plot_ion_xy_heatmaps(
    clay_analysis_func=z_analysis.calculate_clay_spatial_distribution_xy,
    clay_analysis_params={
        'z_slice_centers': [-29.5, 26.5, 29.5],
        'z_slice_width': 2.0,
        'show_hexagonal_pattern': True,      # Shows Si atoms
        'si_connection_threshold': 6,
        'si_connection_style': 'lines',
        ...
    },
    ion_type='NA',
    save_plots=True
)

# Now overlay cavity binding data
plotter_cip = ClayOrganicIonWaterAnalysisPlotter(analysis)
plotter_cip.plot_cavity_ion_binding(
    ion_types=['NA'],
    z_slice_centers=[-29.5, 26.5, 29.5],
    show_ion_density=True,
    save_plots=True
)
```

---

## 5. Interpreting Results

### Cavity Detection

The method uses **graph-based ring detection**:
1. Builds neighbor graph with Si atoms as nodes
2. Connects Si atoms within `si_si_threshold` distance (~4.5 Å)
3. Finds all 6-membered cycles (hexagonal rings)
4. Calculates ring centers and radii

**Typical output:**
```
📍 Detecting Si hexagonal ring cavities...
   Si-Si threshold: 4.5 Å
   Z-slices: 3
   z =  -29.5 Å: Found 12 hexagonal ring cavities
   z =   26.5 Å: Found 11 hexagonal ring cavities
   z =   29.5 Å: Found 12 hexagonal ring cavities
```

### Ion Binding Criteria

Ions are considered "bound to cavity" if:
- **XY distance** from cavity center ≤ `cavity_radius` (default 3.0 Å)
- **Z position** is between cavity surface and `cavity_height` above (default 6.0 Å)
- Creates a **cylindrical binding region** above each cavity

### Occupancy Metrics

- **avg_occupancy**: Mean number of ions in cavity over all frames
- **occupancy_fraction**: Fraction of frames where cavity contains ≥1 ion
- **preferential_sites**: Cavities with highest avg_occupancy

**Example interpretation:**
```python
# Cavity 3 has avg_occupancy = 0.85
# → On average, 0.85 Na ions are in this cavity
# → Could mean: 85% of time occupied by 1 ion, 
#               or 42.5% of time occupied by 2 ions, etc.

# Cavity 3 has occupancy_fraction = 0.92
# → Cavity occupied (≥1 ion) in 92% of frames
# → High residence time at this site
```

---

## 6. Parameter Tuning Guide

### Si-Si Threshold (`si_si_threshold`)

**Controls ring detection sensitivity**

- **Too small** (e.g., 3.5 Å): May miss some rings, especially if Si atoms vibrate
- **Too large** (e.g., 6.0 Å): May detect spurious "rings" from non-adjacent Si atoms
- **Recommended**: 4.0-4.5 Å for typical clay structures

**Test:**
```python
for threshold in [3.5, 4.0, 4.5, 5.0]:
    results = analysis.analyze_cavity_ion_binding(
        si_si_threshold=threshold,
        verbose=True
    )
    print(f"Threshold {threshold}: {len(results['cavity_data'][z_center]['ring_centers'])} rings")
```

### Cavity Radius (`cavity_radius`)

**Defines XY extent of binding region**

- **Smaller** (e.g., 2.0 Å): More restrictive, only ions directly above cavity
- **Larger** (e.g., 4.0 Å): More permissive, includes nearby ions
- **Recommended**: 2.5-3.5 Å based on ion size and desired specificity

### Cavity Height (`cavity_height`)

**Defines Z extent of binding region**

- **Smaller** (e.g., 3.0 Å): Only first coordination shell
- **Larger** (e.g., 8.0 Å): Includes outer-sphere ions
- **Recommended**: 5.0-7.0 Å to capture inner + outer sphere

### Z-Slice Width (`z_slice_width`)

**Controls which Si atoms are included in ring detection**

- **Narrower** (e.g., 1.0 Å): More selective, may miss some Si atoms
- **Wider** (e.g., 4.0 Å): Includes more Si atoms, may mix layers
- **Recommended**: 2.0-3.0 Å to capture single clay layer

---

## 7. Future Enhancements (3D Polyhedral)

For later implementation, the 3D polyhedral visualization would:

1. **Use crystaltoolkit** for structure rendering
2. **Show SiO₄ tetrahedra** and **MgO₆ octahedra**
3. **Overlay ion density** in 3D
4. **Interactive visualization** with rotation/zoom

**Planned workflow:**
```python
# Future implementation
plotter.plot_cavity_ion_binding_3d(
    ion_types=['NA'],
    z_slice_centers=[26.5],
    show_polyhedra=True,              # Show SiO4/MgO6
    polyhedra_alpha=0.3,              # Transparent
    show_ion_density=True,            # 3D density cloud
    interactive=True,                 # Rotate/zoom
    backend='crystaltoolkit'          # or 'plotly'
)
```

---

## 8. Troubleshooting

### No cavities detected

**Problem:** `Found 0 hexagonal ring cavities`

**Solutions:**
1. Check `si_si_threshold` - increase to 4.5-5.0 Å
2. Check `z_slice_centers` - ensure they match clay layer positions
3. Check Si atom selection - verify `clay_selection` includes Si atoms
4. Try `z_slice_width=4.0` to include more Si atoms

### Occupancy values seem low

**Problem:** `avg_occupancy` is 0.1-0.2 for all cavities

**Solutions:**
1. Increase `cavity_radius` (e.g., 3.5-4.0 Å)
2. Increase `cavity_height` (e.g., 7.0-8.0 Å)
3. Check ion count - may have too few ions in system
4. Verify `z_slice_centers` match regions where ions are present

### Memory issues with large trajectories

**Problem:** Analysis runs out of memory

**Solutions:**
1. Increase `step` parameter (e.g., `step=5` to use every 5th frame)
2. Reduce number of z-slices
3. Set `compute_per_cavity_timeseries=False` if not needed

---

## 9. Citation

If you use this cavity ion binding analysis in your research, please cite:

```bibtex
@software{cavity_ion_binding,
  author = {Swai, R.},
  title = {Cavity-Specific Ion Binding Analysis for Clay Surfaces},
  year = {2026},
  url = {https://github.com/your-repo/solvation_shells}
}
```

---

## Contact

For questions or issues:
- **Author:** R. Swai
- **Date:** January 2026
- **Repository:** solvation_shells
