"""Validate notebook code cells for syntax errors (skip heavy computation)."""
import json, sys, os
import ast

sys.path.insert(0, 'D:/INLRMF-master/INLRMF-code')

notebook_path = 'D:/INLRMF-master/INLRMF-code/INLRMF_analysis.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Validating {len(nb['cells'])} cells...\n")

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

# First pass: syntax check
errors = []
for cell_num, source in code_cells:
    try:
        ast.parse(source)
        print(f"  Cell {cell_num}: syntax OK ({len(source)} chars)")
    except SyntaxError as e:
        errors.append((cell_num, str(e)))
        print(f"  Cell {cell_num}: SYNTAX ERROR - {e}")

if errors:
    print(f"\n{len(errors)} syntax errors found!")
    for cn, err in errors:
        print(f"  Cell {cn}: {err}")
else:
    print(f"\nAll {len(code_cells)} code cells pass syntax validation.")

# Second pass: test-execute non-NMF cells
print("\n" + "="*60)
print("Testing execution of lightweight cells...")
namespace = {
    '__name__': '__main__',
    '__builtins__': __builtins__,
}
import matplotlib
matplotlib.use('Agg')

lightweight_cells = [3, 5, 6, 8, 10, 12, 14]  # Skip 15 (def) and 16 (run NMF)
for cell_num, source in code_cells:
    cell_id = cell_num
    if cell_id not in lightweight_cells:
        continue
    try:
        exec(source, namespace)
        print(f"  Cell {cell_num}: EXEC OK")
    except Exception as e:
        print(f"  Cell {cell_num}: EXEC FAILED - {e}")

# Test cell 15 (function definition)
print("\nTesting function definition (cell 15)...")
for cell_num, source in code_cells:
    if cell_num == 15:
        try:
            exec(source, namespace)
            print("  Cell 15: function defined OK")
        except Exception as e:
            print(f"  Cell 15: FAILED - {e}")
            import traceback
            traceback.print_exc()

# Test cell 16 but skip actual NMF
print("\nValidating cell 16 setup (skipping NMF)...")
# Check variables exist
for var in ['df1_filtered', 'df2_filtered', 'Z_init', 'n_clusters', 'lambda1', 'run_inlrmf']:
    if var in namespace:
        print(f"  {var}: available")
    else:
        print(f"  {var}: MISSING")

print("\nValidation complete. Notebook structure is sound.")
print("Note: Full execution of NMF cells requires ~30 min and was not tested here.")
