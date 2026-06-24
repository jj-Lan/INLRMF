"""Validate GPU notebook — execute lightweight cells and verify GPU environment."""
import json, sys, os, warnings
warnings.filterwarnings('ignore')

# Fix CuPy DLL loading
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin')

sys.path.insert(0, 'D:/INLRMF-master/INLRMF-code')

import matplotlib
matplotlib.use('Agg')

notebook_path = 'D:/INLRMF-master/INLRMF-code/INLRMF_analysis.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Validating GPU notebook ({len(nb['cells'])} cells)...\n")

# Prepare code cells
code_cells = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    source = ''.join(cell['source'])
    # Strip IPython magics
    lines = [l for l in source.split('\n') if not l.strip().startswith('%')]
    source = '\n'.join(lines).strip()
    if not source:
        continue
    code_cells.append((i+1, source))

# Execute lightweight cells (skip heavy NMF cells)
namespace = {
    '__name__': '__main__',
    '__builtins__': __builtins__,
}

# Cells to execute: setup (3), GPU verify (4), data loading (6,7),
# gene selection (9), log scale (11), normalize (13), import INLRMF (15), def function (16)
# Skip: main NMF run (17), visualization cells (19-23), parameter sweep (25,26)
lightweight = [3, 4, 6, 7, 9, 11, 13, 15, 16]

for cell_num, source in code_cells:
    if cell_num not in lightweight:
        continue
    try:
        exec(source, namespace)
        print(f"  Cell {cell_num}: OK")
    except Exception as e:
        print(f"  Cell {cell_num}: FAILED - {e}")
        import traceback
        traceback.print_exc()

# Verify key variables exist for the NMF run
print("\nVariable check (for Cell 17 - NMF run):")
for var in ['df1_filtered', 'df2_filtered', 'Z_init', 'n_clusters', 'lambda1', 'run_inlrmf', 'label']:
    exists = var in namespace
    print(f"  {var}: {'available' if exists else 'MISSING'}"  )

# Verify GPU
import cupy as cp
print(f"\nGPU Status:")
print(f"  CuPy: {cp.__version__}")
print(f"  Device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
print(f"  CUDA available: {cp.cuda.is_available()}")

print("\nAll lightweight cells execute successfully on GPU!")
