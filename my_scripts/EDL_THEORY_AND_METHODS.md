# Electrical Double Layer (EDL) Analysis: Theory and Methods

## Table of Contents
1. [Overview](#overview)
2. [Electric Double Layer Theory](#electric-double-layer-theory)
3. [Charge Density and Electrostatic Potential](#charge-density-and-electrostatic-potential)
4. [Stern Layer Identification](#stern-layer-identification)
5. [Debye Length Calculation](#debye-length-calculation)
6. [Adsorption Mode Classification](#adsorption-mode-classification)
7. [Physical Constants and Units](#physical-constants-and-units)
8. [References](#references)

---

## Overview

This document provides detailed theoretical background and methodology for the electrical double layer (EDL) analysis implemented in `ClayOrganicIonWaterAnalysis.py`. The analysis characterizes the ionic structure near charged surfaces (clay minerals) in aqueous solutions.

The EDL analysis includes:
- **Charge density profiles** from ion distributions
- **Electrostatic potential** via Poisson equation
- **Stern layer structure** (Inner and Outer Helmholtz Planes)
- **Debye screening length** (theoretical and fitted)
- **Ion adsorption modes** (inner-sphere, outer-sphere, diffuse)

---

## Electric Double Layer Theory

### Conceptual Framework

When a charged surface (e.g., clay mineral with negative surface charge) is immersed in an electrolyte solution, counterions accumulate near the surface to maintain electroneutrality. This creates a stratified ionic structure known as the **electrical double layer** (EDL).

The EDL consists of two main regions:

1. **Stern Layer (Compact Layer)**: 0-5 Å from surface
   - Ions are specifically adsorbed or bound to the surface
   - Strong surface-ion interactions dominate
   - NOT described by continuum electrostatics

2. **Diffuse Layer**: Beyond ~5 Å from surface
   - Ions are mobile and electrostatically screened
   - Exponential decay of potential and ionic concentrations
   - Described by Poisson-Boltzmann theory

### Gouy-Chapman-Stern Model

Our implementation follows the **Gouy-Chapman-Stern (GCS)** model:

```
Surface | Stern Layer | Diffuse Layer | Bulk
   σ₀   →    IHP  OHP  →   ψ decays   →  ψ ≈ 0
        |←-- d -->|←---- λ_D ----→|
```

Where:
- **σ₀**: Surface charge density
- **IHP**: Inner Helmholtz Plane (~2.5 Å from surface)
- **OHP**: Outer Helmholtz Plane (~5 Å from surface)
- **λ_D**: Debye screening length (characteristic decay length)

---

## Charge Density and Electrostatic Potential

### 1. Ion Density Profiles

Ion densities are calculated by histogramming ion positions along the z-axis (perpendicular to surface):

```
ρᵢ(z) = ⟨Nᵢ(z)⟩ / (ΔV)
```

Where:
- `ρᵢ(z)`: Number density of ion type i at position z (ions/Å³)
- `⟨Nᵢ(z)⟩`: Time-averaged number of ions in bin at z
- `ΔV = A × Δz`: Volume of bin (A = xy cross-sectional area)
- `Δz`: Bin width (default: 0.2 Å)

### 2. Charge Density Profile

Total charge density at each position:

```
ρ(z) = Σᵢ qᵢ ρᵢ(z)
```

Where:
- `qᵢ`: Charge of ion type i (in elementary charges e)
- Sum is over all ion types (cations and anions)

**Units**: e/Å³ (elementary charges per cubic Angstrom)

**Conversion to SI**:
```
ρ(z) [C/m³] = ρ(z) [e/Å³] × (1.602×10⁻¹⁹ C) × (10¹⁰ Å/m)³
```

### 3. Electrostatic Potential (Poisson Equation)

The electrostatic potential ψ(z) is obtained by solving the **one-dimensional Poisson equation**:

```
d²ψ/dz² = -ρ(z) / (ε₀ εᵣ)
```

Where:
- `ψ(z)`: Electrostatic potential (V or kT/e)
- `ρ(z)`: Charge density (C/m³)
- `ε₀ = 8.854×10⁻¹² F/m`: Vacuum permittivity
- `εᵣ = 78`: Relative permittivity of water (at 300 K)

**Solution Method**: Double integration of charge density
```
ψ(z) = -∫∫ ρ(z')/ε₀εᵣ dz' dz'
```

With boundary conditions:
- `ψ(z_bulk) = 0`: Potential is zero in bulk (reference point)
- `dψ/dz|_bulk = 0`: Electric field vanishes in bulk

**Output Units**: Reduced units (kT/e)
```
ψ [kT/e] = ψ [V] × e / (kT)
```

Where:
- `k = 1.381×10⁻²³ J/K`: Boltzmann constant
- `T = 300 K`: Temperature
- `kT ≈ 0.0259 eV = 25.9 meV` at room temperature

### 4. Electric Field

Electric field is the negative gradient of potential:

```
E(z) = -dψ/dz
```

Calculated numerically using numpy gradient with second-order accurate central differences.

**Units**: kT/(e·Å) or V/Å

---

## Stern Layer Identification

### Theory

The **Stern layer** represents the region where ions are specifically adsorbed to the surface. It is divided into:

1. **Inner Helmholtz Plane (IHP)**:
   - Location of specifically adsorbed ions (inner-sphere complexes)
   - Ions partially desolvated, direct contact with surface
   - Typically 2-3 Å from surface for hydrated ions
   - Corresponds to first peak in cation density

2. **Outer Helmholtz Plane (OHP)**:
   - Location of solvated ions (outer-sphere complexes)
   - Ions retain full hydration shell
   - Typically 4-6 Å from surface
   - Corresponds to first minimum in cation density after IHP

### Methodology

#### Manual Peak Mode (use_manual_peaks=True)

When you set `use_manual_peaks=True`, the code uses the peak positions you specified in `analyze_ion_peaks_manual()` to determine IHP and OHP. Here's the exact procedure:

##### Step 1: Find Dominant Cation

The code identifies which cation contributes most to surface charge screening:

```python
for each cation with manual peaks:
    weight = charge × first_peak_density
    # charge: +1 for Na⁺, +2 for Ca²⁺, etc.
    # first_peak_density: density at the first (closest to surface) peak
```

**Selection criteria:**
- Only considers **cations** (positive ions)
- Only considers ions with **at least one peak**
- Calculates **weight = q × ρ_peak** for each cation
- Dominant cation = highest weight

**Example:**
```
Na⁺: charge = +1, first peak density = 0.00390 ions/Å³
     → weight = 1 × 0.00390 = 0.00390

Ca²⁺: charge = +2, first peak density = 0.00150 ions/Å³
      → weight = 2 × 0.00150 = 0.00300

Dominant cation: Na⁺ (weight = 0.00390)
```

##### Step 2: Assign IHP Position

```python
IHP = first_peak_position_of_dominant_cation
```

The **first peak** is the one **closest to the surface**:
- For top surface at 26.6 Å: First peak might be at 23.9 Å
- For bottom surface at -26.6 Å: First peak might be at -23.9 Å

**Physical meaning:** The first peak represents the **highest concentration** of the dominant cation - this is where ions are most strongly adsorbed (Inner Helmholtz Plane).

##### Step 3: Calculate OHP Position

OHP is calculated **2.5 Å away from IHP** in the direction **away from the surface**:

**For top surface** (surface_position > 0, ions accumulate BELOW):
```python
OHP = IHP - 2.5 Å  # Move further from surface (toward bulk)
```

Example:
- Surface: 26.6 Å
- IHP: 23.9 Å (first Na⁺ peak, 2.7 Å from surface)
- OHP: 23.9 - 2.5 = 21.4 Å (5.2 Å from surface)

**For bottom surface** (surface_position < 0, ions accumulate ABOVE):
```python
OHP = IHP + 2.5 Å  # Move further from surface (toward bulk)
```

Example:
- Surface: -26.6 Å
- IHP: -23.9 Å (first Na⁺ peak, 2.7 Å from surface)
- OHP: -23.9 + 2.5 = -21.4 Å (5.2 Å from surface)

##### Step 4: Calculate Stern Layer Thickness

```python
stern_thickness = |OHP - IHP| = 2.5 Å
```

This is **fixed at 2.5 Å** when using manual peaks.

##### Complete Example

**Your system with manual peaks:**

```
Manual peaks for Na⁺: [23.9, 22.3, 20.1] Å

Step 1: Na⁺ is dominant cation (highest q × ρ)
Step 2: IHP = 23.9 Å (first peak)
Step 3: OHP = 23.9 - 2.5 = 21.4 Å (top surface)
Step 4: Stern thickness = |21.4 - 23.9| = 2.5 Å

Output:
  Stern layer from MANUAL peaks:
    Dominant cation: Na
    Inner Helmholtz Plane (IHP): 23.9 Å (from manual peak)
    Outer Helmholtz Plane (OHP): 21.4 Å (IHP - 2.5 Å)
    Stern layer thickness: 2.5 Å
```

##### Why This Approach?

1. **First peak = IHP**: The highest cation concentration indicates where ions are most strongly adsorbed to the surface (Inner Helmholtz Plane)

2. **OHP = IHP - 2.5 Å**: The Outer Helmholtz Plane is where solvated ions (with full hydration shell) are located. A hydration shell for monovalent cations is typically ~2.5 Å thick.

3. **Dominant cation selection**: On negatively charged clays, the cation that contributes most to charge screening (q × ρ) defines the EDL structure.

##### Key Point

**You control IHP directly** by setting the peak positions in `analyze_ion_peaks_manual()`:
```python
peaks = analyzer.analyze_ion_peaks_manual(
    peak_positions_dict={
        'Na': [23.9, 22.3, 20.1],  # First value (23.9) becomes IHP!
        'Cl': [20.5]
    }
)
```

Then when you run EDL analysis with `use_manual_peaks=True`:
```python
edl = analyzer.analyze_electrical_double_layer_complete(
    clay_surface_sel=clay_top,
    use_manual_peaks=True  # Uses 23.9 Å as IHP
)
```

#### Automatic Mode (use_manual_peaks=False, default)

If manual peaks not available or not requested:

1. **Define search region**:
   - Top surface (z > 0): Look below surface (z < z_surface)
   - Bottom surface (z < 0): Look above surface (z > z_surface)
   - Limit search to 15 Å to avoid opposite surface

2. **Calculate total cation density**:
   ```
   ρ_cation(z) = Σᵢ ρᵢ(z)  (for cations only)
   ```

3. **Find first minimum** (OHP):
   - Use scipy.signal.find_peaks on `-ρ_cation(z)` (inverted)
   - OHP is located at first minimum after surface

4. **Estimate IHP**:
   - Top surface: `IHP = z_surface - 2.5 Å`
   - Bottom surface: `IHP = z_surface + 2.5 Å`

### Physical Interpretation

**Stern Layer Thickness**: `d_Stern = |OHP - IHP|`

Typical values:
- ~2-3 Å for monovalent cations (Na⁺, K⁺)
- ~3-5 Å for divalent cations (Ca²⁺, Mg²⁺)

The IHP-OHP region contains:
- Specifically adsorbed counterions
- Tightly bound water molecules
- Surface complexes

Beyond OHP, the **diffuse layer** begins where exponential decay applies.

---

## Debye Length Calculation

The **Debye length (λ_D)** characterizes the thickness of the electrical double layer - the distance over which the electrostatic potential decays to 1/e (~37%) of its surface value.

### Theoretical Debye Length

Based on **Debye-Hückel theory** for dilute electrolyte solutions.

#### Formula

```
λ_D = √(ε₀ εᵣ kT / (e² × 2I))
```

Where:
- **ε₀ = 8.854×10⁻¹² F/m**: Vacuum permittivity
- **εᵣ = 78.0**: Relative permittivity of water at 300 K
- **k = 1.381×10⁻²³ J/K**: Boltzmann constant
- **T = 300 K**: Temperature (adjustable parameter)
- **e = 1.602×10⁻¹⁹ C**: Elementary charge
- **I**: Ionic strength (mol/m³ or ions/m³)

#### Ionic Strength Calculation

The ionic strength quantifies the total ionic content:

```
I = 0.5 × Σᵢ cᵢ zᵢ²
```

Where:
- `cᵢ`: Concentration of ion type i (ions/m³)
- `zᵢ`: Valence (charge number) of ion type i
- Sum over all ionic species

**Example** (for NaCl solution):
- Na⁺: c_Na = 1.0 M, z_Na = +1
- Cl⁻: c_Cl = 1.0 M, z_Cl = -1

```
I = 0.5 × (1.0 × 1² + 1.0 × 1²) = 1.0 M
```

#### Bulk Concentration Determination

1. **Define bulk region**:
   - For centered systems (|z_surface| > 10 Å):
     - Top surface: Bulk is between center and surface
     - Bottom surface: Bulk is between center and surface
   - Region at least `bulk_reference_distance` from surface
   - Default: `bulk_reference_distance = 10 Å`

2. **Calculate average ion density in bulk**:
   ```
   c_bulk,i = ⟨ρᵢ(z)⟩_bulk
   ```
   Average over all z-positions in bulk region

3. **Convert to molar concentration**:
   ```
   c_bulk,i [M] = c_bulk,i [ions/Å³] × (10¹⁰ Å/m)³ / N_A / 1000
   ```
   Where N_A = 6.022×10²³ is Avogadro's number

4. **Convert to ions/m³ for formula**:
   ```
   c_bulk,i [ions/m³] = c_bulk,i [ions/Å³] × (10¹⁰)³
   ```

#### Typical Values

| Concentration | Ionic Strength | Debye Length |
|--------------|----------------|--------------|
| 0.001 M (1 mM) | 0.001 M | 9.6 nm (96 Å) |
| 0.01 M (10 mM) | 0.01 M | 3.0 nm (30 Å) |
| 0.1 M (100 mM) | 0.1 M | 0.96 nm (9.6 Å) |
| 1.0 M | 1.0 M | 0.30 nm (3.0 Å) |

**Note**: For 1:1 electrolyte (NaCl), I = c. For multivalent ions, ionic strength is higher.

---

### Fitted Debye Length

The fitted Debye length is obtained by fitting the actual electrostatic potential decay in the simulation to an exponential function.

#### Theory

In the **diffuse layer**, far from specific adsorption effects, the potential should decay exponentially:

```
ψ(z) = ψ₀ exp(-|z - z_surface| / λ_D)
```

Where:
- `ψ(z)`: Potential at distance z from surface
- `ψ₀`: Potential at OHP (start of diffuse layer)
- `z_surface`: Surface position
- `λ_D`: Debye length (fitting parameter)

#### Methodology

1. **Define fitting region**:
   - Start: OHP position (from Stern layer analysis) or 5 Å default
   - End: OHP + 7 Å (captures 2-3 decay lengths)
   - Direction-aware:
     - Top surface: Look inward (decreasing z)
     - Bottom surface: Look outward (increasing z)

2. **Extract data points**:
   ```python
   z_fit = |z_centers - z_surface|  # Distance from surface
   psi_fit = |potential|             # Absolute potential values
   ```

3. **Check for meaningful decay**:
   ```
   Δψ = max(psi_fit) - min(psi_fit)
   ```
   - If Δψ < 0.01 kT/e: Potential too flat, skip fitting
   - Indicates insufficient screening or overlapping EDLs

4. **Exponential fitting**:
   ```python
   def exp_decay(z, psi0, lambda_d):
       return psi0 * exp(-z / lambda_d)
   
   # Fit using scipy.optimize.curve_fit
   popt, pcov = curve_fit(exp_decay, z_fit, psi_fit, 
                         bounds=([0, 0.5], [inf, 100.0]))
   lambda_D_fitted = popt[1]
   ```

5. **Constraints**:
   - `ψ₀`: 0 to ∞ (positive potential)
   - `λ_D`: 0.5 to 100 Å (physically reasonable range)

#### Quality Checks

1. **Fitted vs Theory Ratio**:
   ```
   Ratio = λ_D,theory / λ_D,fitted
   ```
   - Ratio ≈ 1.0: Excellent agreement
   - 0.7 < Ratio < 1.3: Good agreement (within 30%)
   - Ratio < 0.5 or > 2.0: Poor agreement, investigate

2. **Common Issues**:
   - **λ_D,fitted >> λ_D,theory**: Potential too flat, overlapping EDLs
   - **λ_D,fitted << λ_D,theory**: Ion-specific effects, correlation forces
   - **Ratio ≈ 1**: Ideal Debye-Hückel behavior confirmed

---

## Adsorption Mode Classification

Ion adsorption at charged surfaces occurs through different mechanisms with characteristic distances from the surface.

### Three Adsorption Modes

1. **Inner-Sphere Complexes** (ISC)
   - **Location**: Between surface and IHP
   - **Distance**: 0 to ~2.5 Å from surface
   - **Characteristics**:
     - Direct contact with surface
     - Partial or complete dehydration
     - Strong specific interactions (covalent/coordination bonds)
     - Low mobility, long residence times

2. **Outer-Sphere Complexes** (OSC)
   - **Location**: Between IHP and OHP
   - **Distance**: ~2.5 to ~5 Å from surface
   - **Characteristics**:
     - Separated by at least one water layer
     - Retain full hydration shell
     - Electrostatic attraction dominates
     - Moderate mobility

3. **Diffuse Layer Ions**
   - **Location**: Beyond OHP
   - **Distance**: > ~5 Å from surface
   - **Characteristics**:
     - Mobile, freely diffusing
     - Exponentially decreasing concentration
     - Purely electrostatic screening
     - Described by Poisson-Boltzmann theory

### Classification Methodology

#### 1. Ion Position Extraction

For each ion in each frame:
```python
z_ion = ion.position[2]  # z-coordinate
```

#### 2. Direction Detection

Determine if analyzing top or bottom surface:
```python
is_top_surface = (IHP < surface_position)
```

Logic:
- If IHP is below surface → top surface (ions accumulate downward)
- If IHP is above surface → bottom surface (ions accumulate upward)

#### 3. Region Definition (Direction-Aware)

**For Top Surface** (ions accumulate below surface):
```
Inner-sphere:  IHP < z < surface
Outer-sphere:  OHP < z < IHP  
Diffuse layer: z < OHP (within search limit)
```

**For Bottom Surface** (ions accumulate above surface):
```
Inner-sphere:  surface < z < IHP
Outer-sphere:  IHP < z < OHP
Diffuse layer: z > OHP (within search limit)
```

#### 4. Boundary Clipping

To avoid double-counting from opposite surface:

**Top Surface**:
```python
diffuse_outer_limit = max(OHP - 15.0, z_min + 2.0)
```

**Bottom Surface**:
```python
diffuse_outer_limit = min(OHP + 15.0, z_max - 2.0)
```

This ensures we only count ions associated with the analyzed surface.

#### 5. Ion Counting

For each ion type i and each frame:
```python
N_inner_i += count(IHP < z < surface)      # Inner-sphere
N_outer_i += count(OHP < z < IHP)          # Outer-sphere  
N_diffuse_i += count(OHP_limit < z < OHP)  # Diffuse layer
```

Average over all frames:
```python
⟨N_inner_i⟩ = Σ_frames N_inner_i / n_frames
```

#### 6. Percentage Calculation

```python
N_total_i = N_inner_i + N_outer_i + N_diffuse_i

% inner_i = 100 × N_inner_i / N_total_i
% outer_i = 100 × N_outer_i / N_total_i
% diffuse_i = 100 × N_diffuse_i / N_total_i
```

### Physical Interpretation

#### Surface Charge Compensation

For negatively charged surface (clay):
- Inner-sphere cations directly neutralize surface charge
- Outer-sphere + diffuse layer ions provide long-range screening

Charge balance:
```
σ_surface + σ_inner + σ_outer + σ_diffuse = 0
```

#### Ion Specificity

Different ions show different adsorption preferences:

**Strongly Adsorbing** (more inner-sphere):
- Ca²⁺, Mg²⁺, Ba²⁺ (divalent cations)
- Large polarizable ions (Cs⁺)

**Weakly Adsorbing** (more diffuse):
- Na⁺, K⁺ (small monovalent cations)
- Cl⁻, NO₃⁻ (anions on negative surface)

#### Typical Distributions

**For montmorillonite clay at moderate ionic strength:**

| Ion | Inner-Sphere | Outer-Sphere | Diffuse |
|-----|--------------|--------------|---------|
| Na⁺ | 0-5% | 5-15% | 85-95% |
| Ca²⁺| 10-30% | 20-40% | 40-70% |
| Cl⁻ | 0% | 0-1% | 99-100% |

---

## Physical Constants and Units

### Fundamental Constants

| Constant | Symbol | Value | Units |
|----------|--------|-------|-------|
| Boltzmann constant | k_B | 1.381×10⁻²³ | J/K |
| Elementary charge | e | 1.602×10⁻¹⁹ | C |
| Vacuum permittivity | ε₀ | 8.854×10⁻¹² | F/m |
| Avogadro's number | N_A | 6.022×10²³ | mol⁻¹ |

### System Parameters (Adjustable)

| Parameter | Default | Description |
|-----------|---------|-------------|
| Temperature (T) | 300 K | System temperature |
| Dielectric constant (εᵣ) | 78.0 | Water at 300 K |
| z_bin_width | 0.2 Å | Spatial resolution |
| bulk_reference_distance | 10 Å | Minimum distance to bulk |

### Derived Quantities

**Thermal energy at 300 K**:
```
kT = 1.381×10⁻²³ J/K × 300 K = 4.143×10⁻²¹ J
   = 0.0259 eV = 25.9 meV
```

**Reduced potential unit**:
```
kT/e = 25.9 mV
```

**Conversion factors**:
```
1 Å = 10⁻¹⁰ m
1 M = 1 mol/L = 6.022×10²⁶ ions/m³
1 e/Å³ = 1.602×10¹¹ C/m³
```

---

## Unit Conversions

### Length
```
1 Å = 0.1 nm = 10⁻¹⁰ m
```

### Concentration
```
1 ion/Å³ = 10³⁰ ions/m³
         = 1.66054 × 10³ M
         = 1660.54 M

1 M = 6.022 × 10²⁶ ions/m³
    = 6.022 × 10⁻⁴ ions/Å³
```

### Charge Density
```
1 e/Å³ = 1.602 × 10⁻¹⁹ C × (10¹⁰ m⁻¹)³
       = 1.602 × 10¹¹ C/m³

1 e/Å² = 1.602 × 10⁻¹⁹ C × (10¹⁰ m⁻¹)²
       = 1.602 C/m²
```

### Potential
```
1 V = 38.68 kT/e  (at 300 K)
1 kT/e = 25.9 mV  (at 300 K)
```

---

## Validation and Quality Checks

### 1. Charge Neutrality

Total integrated charge should be zero:
```
Q_total = ∫ ρ(z) dz ≈ 0
```

Check:
```python
Q_total = trapz(charge_density, dx=z_bin_width)
if |Q_total| < 0.001:  # Within 0.001 e
    System is charge neutral ✓
```

### 2. Bulk Potential

Potential should approach zero in bulk:
```
ψ(z_bulk) ≈ 0 ± 0.01 kT/e
```

### 3. Debye Length Consistency

Theory vs fitted should agree within factor of 2:
```
0.5 < λ_D,theory / λ_D,fitted < 2.0
```

Larger deviations indicate:
- Ion-ion correlations (high concentration)
- Non-ideality, ion pairing
- Overlapping EDLs (confined system)
- Surface charge regulation

### 4. Stern Layer Positions

Physically reasonable values:
```
IHP: 2-4 Å from surface
OHP: 4-6 Å from surface
Stern thickness: 2-4 Å
```

### 5. Adsorption Mode Totals

Sum of all modes should equal 100%:
```
% inner + % outer + % diffuse ≈ 100%
```

---

## Comparing EDL Across Different Systems

When analyzing multiple systems with different ionic strengths or compositions, systematic comparison requires appropriate metrics that account for concentration dependence.

### Best Metrics for EDL Comparison (in order of preference)

#### 1. Debye Length (λ_D) - Most Fundamental

**Why:** The characteristic length scale of the EDL that is **already normalized** for concentration.

**Key Properties:**
- **Theory:** λ_D ∝ 1/√(I), so it naturally accounts for ionic strength
- Compare both λ_D,theory and λ_D,fitted
- **Ideal for:** Determining if systems follow Debye-Hückel predictions

**Example Comparison:**
```
System A: 0.1 M NaCl → λ_D ≈ 10 Å
System B: 1.0 M NaCl → λ_D ≈ 3 Å
Ratio: λ_D(A) / λ_D(B) = 3.3
```

**Interpretation:** The ratio directly tells you how much "tighter" or "looser" the EDL is. System B has a 3.3× more compact EDL due to 10× higher ionic strength.

**Theoretical Relationship:**
```
λ_D [Å] ≈ 3.04 / √(I [M])    for 1:1 electrolyte at 298 K
```

#### 2. Stern Layer Thickness (d_Stern = |OHP - IHP|)

**Why:** Should be **concentration-independent** - determined by ion size and hydration shell.

**Typical Values:**
- **Monovalent cations** (Na⁺, K⁺): 2-3 Å
- **Divalent cations** (Ca²⁺, Mg²⁺): 3-5 Å
- **Large hydrated ions**: 4-6 Å

**Interpretation:**
- **Constant across concentrations**: Normal behavior ✓
- **Changes significantly**: Surface chemistry effects or ion-specific interactions
- **Ideal for**: Checking consistency of surface interactions and ion hydration

**Use Case:**
```
System A (0.1 M NaCl): d_Stern = 2.5 Å
System B (1.0 M NaCl): d_Stern = 2.6 Å
→ Consistent! Surface interactions unchanged
```

#### 3. Adsorption Mode Percentages

**Why:** Shows **how** ions distribute, not just the EDL length scale.

**Components:**
- **% inner-sphere**: Direct surface binding (should be low, ~0-5% for Na⁺)
- **% outer-sphere**: Hydrated adsorption (may increase with concentration)
- **% diffuse**: Free ions (should decrease with concentration)

**Expected Concentration Trend:**
```
Low concentration:  Inner ~0-2%, Outer ~5-10%,  Diffuse ~90%
High concentration: Inner ~0-5%, Outer ~20-40%, Diffuse ~60-80%
```

Higher concentration → more outer-sphere, less diffuse (saturation effects)

**Interpretation:**
- **Increasing outer-sphere %**: Surface sites becoming saturated
- **Constant inner-sphere %**: Specific adsorption not changing
- **Ideal for**: Understanding adsorption mechanisms and surface capacity

#### 4. Surface Charge Compensation Distance (d_90 or d_95)

**Why:** Practical measure of "effective EDL thickness" - where bulk neutrality is achieved.

**Definition:**
Distance d from surface where integrated charge compensates X% of surface charge:
```
∫(surface to d) ρ(z) dz = -X% × σ_surface
```

Typically use X = 90% or 95%

**Calculation:**
```python
cumulative_charge = cumsum(charge_density * dz)
d_90 = z[where cumulative_charge reaches -0.9 × surface_charge]
```

**Interpretation:**
- **d_90 ≈ 3 × λ_D**: Expected for Debye-Hückel behavior
- **d_90 >> 3 × λ_D**: Long-range interactions, weak screening
- **d_90 << 3 × λ_D**: Strong screening, ion correlations

**Use Case:**
```
System A: d_90 = 30 Å, λ_D = 10 Å → d_90 / λ_D = 3.0 ✓
System B: d_90 = 50 Å, λ_D = 10 Å → d_90 / λ_D = 5.0 (anomalous)
```

---

### Recommended Comparison Strategy

#### Primary Comparison: Debye Length

Use **λ_D** as the main metric because it's:
- Theoretically grounded
- Concentration-aware
- Directly comparable across systems

**Analysis:**
1. Plot λ_D vs √(I) on log-log scale → should be linear
2. Compare λ_D,fitted / λ_D,theory ratio across systems
3. Deviations indicate non-ideal behavior

#### Secondary Checks:

1. **Stern layer thickness**: Should remain constant (~2-3 Å for Na⁺)
   - Varying? → Surface chemistry or ion-specific effects

2. **λ_D,fitted / λ_D,theory ratio**: Checks for non-ideality
   - Ratio ≈ 1.0: Ideal Debye-Hückel ✓
   - Ratio < 0.7 or > 1.5: Non-ideal effects important

3. **Adsorption percentages**: Shows mechanism changes
   - Track % outer-sphere vs concentration
   - Identify surface saturation

#### Recommended Visualizations:

**Plot 1: Debye Length Scaling**
```
x-axis: √(Ionic Strength) [√M]
y-axis: λ_D [Å]
Expected: Linear with slope = -3.04 Å/√M
```

**Plot 2: Non-Ideality Check**
```
x-axis: Concentration [M]
y-axis: λ_D,fitted / λ_D,theory
Expected: ≈ 1.0, deviations show where theory breaks down
```

**Plot 3: Adsorption Saturation**
```
x-axis: Concentration [M]
y-axis: % outer-sphere
Expected: Sigmoid curve showing surface saturation
```

**Plot 4: EDL Compactness**
```
x-axis: Concentration [M]
y-axis: d_90 / λ_D ratio
Expected: ≈ 3.0 for ideal systems
```

---

### Comparison Table Template

For systematic comparison across systems:

| System | [Ion] (M) | I (M) | λ_D,theory (Å) | λ_D,fitted (Å) | Ratio | d_Stern (Å) | % Outer | % Diffuse |
|--------|-----------|-------|----------------|----------------|-------|-------------|---------|-----------|
| A | 0.1 | 0.1 | 9.6 | 10.2 | 0.94 | 2.5 | 8% | 90% |
| B | 0.5 | 0.5 | 4.3 | 4.8 | 0.90 | 2.6 | 18% | 78% |
| C | 1.0 | 1.0 | 3.0 | 3.5 | 0.86 | 2.5 | 28% | 68% |

**Key Observations:**
- λ_D decreases with √(I) as expected
- Ratio slightly < 1: Screening slightly stronger than theory
- d_Stern constant: Surface chemistry unchanged
- Outer-sphere increases: Surface saturation with concentration

---

## Assumptions and Limitations

### Assumptions

1. **Mean-field approximation**: Ion-ion correlations neglected
2. **Continuum electrostatics**: Valid for distances > ~5 Å from surface
3. **Constant dielectric**: εᵣ = 78 everywhere (ignores interfacial water structure)
4. **Planar geometry**: Surface is flat on EDL length scales
5. **Equilibrium**: System is at thermodynamic equilibrium

### Limitations

1. **High concentration** (> 1 M): Ion-ion correlations important, Debye-Hückel breaks down
2. **Multivalent ions**: Charge inversion, overcharging possible
3. **Specific ion effects**: Not captured by continuum models
4. **Water structure**: Discrete water layers near surface ignored
5. **Dynamic effects**: Diffusion, residence times not calculated
6. **Overlapping EDLs**: Confined systems (separation < 2×λ_D)

### When to be Cautious

- **λ_D,fitted >> λ_D,theory**: Likely overlapping EDLs or non-equilibrium
- **λ_D,fitted << λ_D,theory**: Strong ion-ion correlations or specific interactions
- **Flat potential**: Insufficient ionic strength or box too small
- **High % inner-sphere**: May indicate numerical artifacts if > 50%

---

## Interpretation Guidelines

### Experimental Comparison

When comparing with experimental EDL measurements (e.g., surface force apparatus, electrophoresis):

1. **Zeta potential**: Related to potential at hydrodynamic slip plane (~OHP)
   ```
   ζ ≈ ψ(OHP)
   ```

2. **Surface charge density**: Compare with titration or potentiometric data
   ```
   σ₀ [C/m²] = surface_charge_density [e/Å²] × 1.602
   ```

3. **Debye length**: Compare with conductivity measurements
   ```
   λ_D [nm] = 0.304 / √(I [M])  (for 1:1 electrolyte at 298 K)
   ```

### Physical Insights

**Large Debye Length** (λ_D > 10 Å):
- Low ionic strength
- Weak screening
- Long-range electrostatic interactions

**Small Debye Length** (λ_D < 3 Å):
- High ionic strength
- Strong screening
- Short-range interactions dominate

**Thick Stern Layer** (> 5 Å):
- Large hydrated ions (Ca²⁺, Mg²⁺)
- Strong specific adsorption

**Thin Stern Layer** (< 3 Å):
- Small ions or partial dehydration
- Weak specific adsorption

---

## References

### Textbooks

1. **Israelachvili, J. N.** (2011). *Intermolecular and Surface Forces* (3rd ed.). Academic Press.
   - Chapter 14: Electrostatic Forces between Surfaces in Liquids
   - Chapter 15: Interactions involving Polar Molecules

2. **Hunter, R. J.** (2001). *Foundations of Colloid Science* (2nd ed.). Oxford University Press.
   - Chapter 6: The Electrical Double Layer

3. **Bockris, J. O'M., & Reddy, A. K. N.** (2000). *Modern Electrochemistry* (Vol. 2A). Plenum Press.
   - Chapter 7: Ion-Solvent Interactions
   - Chapter 8: Ion-Ion Interactions

### Key Papers

4. **Chapman, D. L.** (1913). A contribution to the theory of electrocapillarity. *The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science*, 25(148), 475-481.

5. **Stern, O.** (1924). Zur theorie der elektrolytischen doppelschicht. *Zeitschrift für Elektrochemie und angewandte physikalische Chemie*, 30(21‐22), 508-516.

6. **Grahame, D. C.** (1947). The electrical double layer and the theory of electrocapillarity. *Chemical Reviews*, 41(3), 441-501.

7. **Verwey, E. J. W., & Overbeek, J. T. G.** (1948). *Theory of the Stability of Lyophobic Colloids*. Elsevier.

### Clay-Specific References

8. **Tournassat, C., et al.** (2016). Modeling the acid-base properties of montmorillonite edge surfaces. *Environmental Science & Technology*, 50(24), 13436-13445.

9. **Marry, V., et al.** (2008). Microscopic simulations of interlayer structure and dynamics in bihydrated heteroionic montmorillonites. *The Journal of Physical Chemistry B*, 112(32), 9854-9862.

10. **Carretero, M. I., & Pozo, M.** (2009). Clay and non-clay minerals in the pharmaceutical industry: Part I. Excipients and medical applications. *Applied Clay Science*, 46(1), 73-80.

---

## Changelog

**Version 1.0** (January 2026)
- Initial comprehensive documentation
- All EDL analysis methods documented
- Theory and implementation details
- Validation guidelines added

---

*This document accompanies the `ClayOrganicIonWaterAnalysis.py` module.*
*For questions or corrections, please contact the development team.*
