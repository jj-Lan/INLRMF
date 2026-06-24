"""Test runner: Execute the INLRMF_analysis.ipynb notebook cells directly.

This script reads the .ipynb JSON, extracts code cells, and executes them
sequentially to verify the notebook runs without errors.
"""
import json, sys, os, traceback

# Add project path
sys.path.insert(0, 'D:/INLRMF-master/INLRMF-code')

notebook_path = 'D:/INLRMF-master/INLRMF-code/INLRMF_analysis.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Loaded notebook with {len(nb['cells'])} cells")
print("=" * 60)

# Setup matplotlib for non-interactive backend
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for testing

# Create a shared namespace
namespace = {
    '__name__': '__main__',
    '__builtins__': __builtins__,
}

# Handle IPython magic
import re

def strip_magics(source_lines):
    """Remove IPython magic commands like %matplotlib inline."""
    return [l for l in source_lines if not l.strip().startswith('%')]

success_count = 0
fail_count = 0
skip_count = 0

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue

    source = ''.join(cell['source'])
    # Strip IPython magics
    lines = source.split('\n')
    lines = strip_magics(lines)
    source = '\n'.join(lines).strip()

    if not source:
        skip_count += 1
        continue

    print(f"\n[Cell {i+1}] Executing ({len(source)} chars)...")
    print("-" * 40)

    try:
        exec(source, namespace)
        success_count += 1
        print(f"  -> OK")
    except Exception as e:
        fail_count += 1
        print(f"  -> FAILED: {e}")
        traceback.print_exc()
        # Continue with next cell for testing purposes

print("\n" + "=" * 60)
print(f"Results: {success_count} OK, {fail_count} FAILED, {skip_count} SKIPPED")
if fail_count > 0:
    print("Some cells failed. Review the errors above.")
    sys.exit(1)
else:
    print("All code cells executed successfully!")
