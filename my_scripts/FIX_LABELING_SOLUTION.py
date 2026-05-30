"""
SOLUTION: How to fix the labeling and color matching issue

Problem:
--------
1. Your rdf_api_clay was calculated with OLD labeling logic (before the fix)
   - Keys are: 'Ob-C2_O1_O3', 'Ob-N16', 'Ob-C4', etc.
   
2. Your colors dictionary expects NEW labels (after fix)
   - Keys are: 'Ob-carboxylic_acid', 'Ob-piperazine', 'Ob-quinolone'
   
This causes mismatches!

Solution Option 1: RECALCULATE RDFs (Recommended)
--------------------------------------------------
"""

# Step 1: Reload the module to get the new labeling logic
import sys
import importlib

if 'ClayOrganicIonWaterAnalysis' in sys.modules:
    del sys.modules['ClayOrganicIonWaterAnalysis']
    
from ClayOrganicIonWaterAnalysis import ClayOrganicIonWaterAnalysis

# Step 2: Recreate the analysis object (quick, just object creation)
occ_analysis = ClayOrganicIonWaterAnalysis(
    top='nvt.tpr',
    traj='nvt.trr',
    solute_sel='resname api',
    solvent_sel='resname SOL or resname WAT',
    cation_sel={
        'Na': 'name NA',
        'K': 'name K',
        'Mg': 'name MG',
        'Ca': 'name CA',
    },
    anion_sel='name CL',
    center_method='COM'
)

# Step 3: Redefine custom selections (the new code stores _custom_name)
occ_analysis.define_selections({
    'CIP_parts': {
        'quinolone': 'resname api and (name N6 or name C10 or name C11 or name C12 or name C19 or name C21 or name C22 or name C23 or name C4 or name C5)',
        'piperazine': 'resname api and (name N13 or name N16 or name C14 or name C15 or name C17 or name C18)',
        'carboxylic_acid':'resname api and (name O1 or name O3 or name C2)',
        'cyclopropyl': 'resname api and (name C7 or name C8 or name C9)',
        'O_ketone': 'resname api and name O24',
        'fluoride': 'resname api and name F20'
    },
    'solvent': {
        'water_oxygen': 'resname SOL WAT and (name OW or name Ow)',
        'water_hydrogen': 'resname SOL WAT and (name HW1 or name HW2 or name Hw1 or name Hw2)'
    },
    'MMT_surface': {
        'surface_oxygen':'resname MMT and name Ob',
        'octahedral_hydroxyl': 'resname MMT and (name Ohmg or name H)',
        'surface_silicon': 'resname MMT and name Si',
        'octahedral_al': 'resname MMT and (name Al or name AL)',
        'octahedral_mg': 'resname MMT and (name Mgo or name MGO)'
    }
})

# Step 4: Extract variables
quinolone = occ_analysis.custom_selections['CIP_parts']['quinolone']
carboxylic_acid = occ_analysis.custom_selections['CIP_parts']['carboxylic_acid']
piperazine = occ_analysis.custom_selections['CIP_parts']['piperazine']
surface_oxygen = occ_analysis.custom_selections['MMT_surface']['surface_oxygen']
surface_silicon = occ_analysis.custom_selections['MMT_surface']['surface_silicon']
octahedral_hydroxyl = occ_analysis.custom_selections['MMT_surface']['octahedral_hydroxyl']
octahedral_mg = occ_analysis.custom_selections['MMT_surface']['octahedral_mg']

# Step 5: RECALCULATE with force_rerun (will use new labeling logic)
rdf_api_clay_NEW = occ_analysis.molecular_rdf(
    center_method='atomistic',
    bin_width=0.1,
    group1_sel=[surface_oxygen, surface_silicon, octahedral_mg, octahedral_hydroxyl],
    group2_sel=[carboxylic_acid, piperazine, quinolone],
    range=(0, 20),
    force_rerun=True,  # IMPORTANT: Force recalculation with new labels
    njobs=1,
    step=1
)

# Step 6: Check the new labels
print("\n✓ NEW LABELS:")
for key in rdf_api_clay_NEW.keys():
    print(f"  {key}")

# Step 7: Now plot with matching colors
plotter.plot_multiple_rdfs(
    rdf_api_clay_NEW,  # Use the NEW dictionary
    xlim=(0, 15),
    colors={
        'Ob-carboxylic_acid': 'darkred',
        'Si-carboxylic_acid': 'red',
        # ... rest of your colors
    },
    custom_labels={
        'Ob-carboxylic_acid': r'O$_b$-COOH',
        # ... rest of your custom labels
    }
)


"""
Solution Option 2: MAP OLD LABELS TO NEW COLORS (Temporary workaround)
-----------------------------------------------------------------------
If you don't want to recalculate, map your old labels to colors manually:
"""

# Print actual keys to see what you have
print("ACTUAL KEYS IN rdf_api_clay:")
for key in rdf_api_clay.keys():
    print(f"  {key}")

# Create color dictionary matching ACTUAL keys (old labels)
colors_for_old_labels = {
    # Old carboxylic acid labels (C2_O1_O3)
    'Ob-C2_O1_O3': 'darkred',
    'Si-C2_O1_O3': 'red',
    'H_Ohmg-C2_O1_O3': 'orange',
    'Mgo-C2_O1_O3': 'orangered',
    # Old piperazine labels (N16)
    'Ob-N16': 'darkblue',
    'Si-N16': 'blue',
    'H_Ohmg-N16': 'cyan',
    'Mgo-N16': 'dodgerblue',
    # Old quinolone labels (C4)
    'Ob-C4': 'darkgreen',
    'Si-C4': 'green',
    'H_Ohmg-C4': 'limegreen',
    'Mgo-C4': 'lightgreen',
}

# Map old labels to better display names
custom_labels_for_old = {
    'Ob-C2_O1_O3': r'O$_b$-COOH',
    'Si-C2_O1_O3': r'Si-COOH',
    'H_Ohmg-C2_O1_O3': r'OH-COOH',
    'Mgo-C2_O1_O3': r'Mg$_o$-COOH',
    'Ob-N16': r'O$_b$-piperazine',
    'Si-N16': r'Si-piperazine',
    'H_Ohmg-N16': r'OH-piperazine',
    'Mgo-N16': r'Mg$_o$-piperazine',
    'Ob-C4': r'O$_b$-quinolone',
    'Si-C4': r'Si-quinolone',
    'H_Ohmg-C4': r'OH-quinolone',
    'Mgo-C4': r'Mg$_o$-quinolone',
}

# Now plot using OLD labels but NEW display names
plotter.plot_multiple_rdfs(
    rdf_api_clay,  # Use old dictionary
    xlim=(0, 15),
    colors=colors_for_old_labels,
    custom_labels=custom_labels_for_old
)
