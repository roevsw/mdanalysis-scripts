#!/usr/bin/env python3
"""
_add_mbar_cells.py
------------------
Insert Section 9 MBAR cells into ClayFreeEnergy_ExtraAnalysis.ipynb,
then update the reload-stub cell to include ClayMBAR.

Cells inserted after cell 44 (clay_path.save) and before
cell 45 (## 8. Summary of Saved Results).
"""

import json, sys, os

NB_PATH = 'ClayFreeEnergy_ExtraAnalysis.ipynb'

def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [src] if isinstance(src, str) else src,
    }

def md(src):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [src] if isinstance(src, str) else src,
    }


# ── New MBAR cells ─────────────────────────────────────────────────────────

mbar_cells = [

    # --- 9a: Section header ---
    md(
        "---\n"
        "## 9. MBAR — Multistate Bennett Acceptance Ratio\n\n"
        "MBAR is an asymptotically optimal, bin-free estimator of umbrella "
        "sampling free energies (Shirts & Chodera 2008).  Unlike WHAM, MBAR "
        "does not require binning during optimisation; the 1-D W(r) PMF is "
        "recovered by reweighting all samples to the unbiased ensemble.\n\n"
        "> **Prerequisite**: `pymbar >= 4.0` must be installed:\n"
        "> ```\n"
        "> conda install -c conda-forge pymbar\n"
        "> ```\n"
        "> If pymbar is not installed, `ClayMBAR.__init__` will raise an "
        "`ImportError` with instructions."
    ),

    # --- 9b: Create + run MBAR ---
    code(
        "from ClayMBAR import ClayMBAR\n\n"
        "# Requires pmf.load_data() to have been called (run_wham is optional but\n"
        "# recommended so that the WHAM curve is available for comparison).\n"
        "# K = 2 * N_WINDOWS pseudo-states (both CIP molecules per window).\n"
        "clay_mbar = ClayMBAR(\n"
        "    pmf=pmf,\n"
        "    n_bins=200,\n"
        "    bulk_fraction=0.2,\n"
        ")\n"
        "print(clay_mbar)\n\n"
        "# Run MBAR  (may take 30–120 s depending on dataset size)\n"
        "clay_mbar.run_mbar(uncertainty_method='analytical', verbose=True)"
    ),

    # --- 9c: reference_to_bulk + adsorption_energy ---
    code(
        "# Reference both WHAM and MBAR PMFs to the bulk plateau (high-r end).\n"
        "clay_mbar.reference_to_bulk()\n\n"
        "# Adsorption free energy from MBAR\n"
        "dG_mbar, r_min_mbar = clay_mbar.adsorption_energy(r_surface=0.5)\n"
        "print(f'MBAR  ΔG_ads = {dG_mbar:.2f} kJ/mol  (minimum at r = {r_min_mbar:.3f} nm)')\n\n"
        "# Compare with WHAM adsorption energy (if available)\n"
        "try:\n"
        "    dG_wham, r_min_wham = pmf.get_adsorption_energy()\n"
        "    print(f'WHAM  ΔG_ads = {dG_wham:.2f} kJ/mol  (minimum at r = {r_min_wham:.3f} nm)')\n"
        "    print(f'Δ(MBAR−WHAM) = {dG_mbar - dG_wham:.2f} kJ/mol')\n"
        "except Exception as e:\n"
        "    print(f'WHAM adsorption energy: {e}')"
    ),

    # --- 9d: Comparison plot ---
    code(
        "# Overlay WHAM and MBAR W(r) on the same axes.\n"
        "# plotter1d is the ClayPMFPlotter instantiated in Section 3.\n"
        "fig_mbar, ax_mbar = plotter1d.plot_mbar_comparison(\n"
        "    clay_mbar=clay_mbar,\n"
        "    show_wham=True,\n"
        "    show_mbar=True,\n"
        "    show_error_wham=True,\n"
        "    show_error_mbar=True,\n"
        "    annotate_well=True,\n"
        "    unit='kJ/mol',\n"
        "    title='WHAM vs MBAR  —  W(r)',\n"
        "    figsize=(9, 5),\n"
        ")\n\n"
        "ClayPMFPlotter.save_figure(\n"
        "    fig_mbar,\n"
        "    os.path.join(OUTPUT_DIR, 'mbar_comparison.png'),\n"
        "    dpi=300,\n"
        ")"
    ),

    # --- 9e: print_summary ---
    code(
        "clay_mbar.print_summary()"
    ),

    # --- 9f: save ---
    code(
        "clay_mbar.save(os.path.join(OUTPUT_DIR, 'clay_mbar.npz'))"
    ),
]


# ── Load notebook ──────────────────────────────────────────────────────────
with open(NB_PATH) as f:
    nb = json.load(f)

cells = nb['cells']
n_before = len(cells)
print(f"Notebook before: {n_before} cells")

# ── Insert MBAR cells after cell 44 (index 44), before cell 45 ────────────
insert_at = 45
cells[insert_at:insert_at] = mbar_cells
print(f"Inserted {len(mbar_cells)} MBAR cells at position {insert_at}")

# ── Update the reload-stub cell (was cell 48, now shifted) ────────────────
# The reload stub is now at index 48 + len(mbar_cells) = 48 + 6 = 54
reload_idx = insert_at + len(mbar_cells) + 3   # 45+6+3 = 54
# Verify it's the right cell
cell_src = ''.join(cells[reload_idx]['source'])
if 'ClayMeanForce.load' not in cell_src:
    # Search for the reload stub in the remaining cells
    for i in range(insert_at + len(mbar_cells), len(cells)):
        if 'ClayMeanForce.load' in ''.join(cells[i]['source']):
            reload_idx = i
            break
    else:
        print("WARNING: Could not find reload stub cell — skipping update.")
        reload_idx = None

if reload_idx is not None:
    existing = ''.join(cells[reload_idx]['source'])
    mbar_stub = (
        "# mbar2   = ClayMBAR.load(os.path.join(OUTPUT_DIR, 'clay_mbar.npz'))\n"
    )
    # Prepend MBAR stub if not already present
    if 'ClayMBAR.load' not in existing:
        new_src = mbar_stub + existing
        cells[reload_idx]['source'] = [new_src]
        print(f"Updated reload stub at cell index {reload_idx}")
    else:
        print(f"Reload stub at cell {reload_idx} already contains ClayMBAR.load")

# ── Write back ────────────────────────────────────────────────────────────
nb['cells'] = cells
with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook after:  {len(cells)} cells")
print(f"Written: {NB_PATH}  ({os.path.getsize(NB_PATH) // 1024} kB)")
