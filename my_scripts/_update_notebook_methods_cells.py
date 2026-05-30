import json
from pathlib import Path

nb_path = Path('/Volumes/SwaiData/conference/US/CIP/density_profile/wth_salt/anionic/mgcl2/DensityProfile_CPXN-1_gaff_clayff_spc_kao_2021_NVT_Start_at_center_6MgCl2/NVT/StructureZprofile.ipynb')
nb = json.loads(nb_path.read_text())

imports_code = '''# Import reusable methods writer utilities from my_scripts
from pathlib import Path
import importlib
import methods_writer

# Reload so notebook picks up edits to my_scripts/methods_writer.py without restart
importlib.reload(methods_writer)
from methods_writer import write_water_spatial_methods
'''

call_code = '''# Call this any time after the calculation cell
methods_text, methods_docx_path = write_water_spatial_methods(
    results_obj=results,
    params=water_spatial_kwargs,
    output_dir=Path.cwd(),
    show_in_notebook=True,
    save_docx=True,
)

print("Methods text generated and displayed in notebook.")
'''

updated = 0
for c in nb.get('cells', []):
    if c.get('id') == '942bcbce':
        c['cell_type'] = 'code'
        c.setdefault('metadata', {})['language'] = 'python'
        c['source'] = imports_code.splitlines(keepends=True)
        updated += 1

if updated != 1:
    raise RuntimeError(f'Expected to update 1 import cell, updated {updated}')

# Ensure there is a dedicated call cell right after the import cell.
import_idx = next(i for i, c in enumerate(nb['cells']) if c.get('id') == '942bcbce')
next_idx = import_idx + 1
next_cell_src = ''.join(nb['cells'][next_idx].get('source', [])) if next_idx < len(nb['cells']) else ''

if '# Call this any time after the calculation cell' not in next_cell_src:
    new_call_cell = {
        'cell_type': 'code',
        'id': '8f3a2d1c',
        'metadata': {'language': 'python'},
        'source': call_code.splitlines(keepends=True),
    }
    nb['cells'].insert(next_idx, new_call_cell)
else:
    nb['cells'][next_idx]['cell_type'] = 'code'
    nb['cells'][next_idx].setdefault('metadata', {})['language'] = 'python'
    nb['cells'][next_idx]['source'] = call_code.splitlines(keepends=True)

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print('Updated import cell and ensured dedicated call cell is present.')
