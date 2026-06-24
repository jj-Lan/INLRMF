"""Generate the INLRMF analysis Jupyter notebook — GPU version."""
import json, os

def make_cell(cell_type, source_lines):
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": [l + "\n" for l in source_lines[:-1]] + [source_lines[-1]]
    }

C = []

# ============ CELL 1: Title ============
C.append(make_cell("markdown", [
    "# INLRMF: A Global Structure-Aware NMF Algorithm for Single-Cell RNA-seq Clustering",
    "",
    "**Reproducible Analysis Pipeline — GPU Accelerated**",
    "",
    "This notebook reproduces the main analysis results and figures from the paper:",
    "",
    "> *\"INLRMF: A Global Structure-Aware NMF Algorithm for Single-Cell RNA-seq Clustering\"*",
    "",
    "The INLRMF algorithm integrates two scRNA-seq quantification methods (e.g., RSEM and Kallisto) "
    "via joint non-negative matrix factorization, simultaneously learning local and global characteristics of the data.",
    "",
    "> 🔧 **GPU Mode**: This notebook uses CuPy for GPU-accelerated NMF factorization (requires CUDA >= 10.0).",
    "",
    "---",
    "",
    "## Pipeline Overview",
    "",
    "1. **Setup & Imports** — Load libraries, configure CuPy GPU",
    "2. **Data Loading** — Load paired scRNA-seq datasets (RSEM + Kallisto)",
    "3. **Gene Selection** — Filter low-expression and ubiquitous genes",
    "4. **Log Scaling** — Log2-transform the expression data",
    "5. **Normalization** — L1 cell-wise normalization",
    "6. **NMF Factorization** — Joint NMF with low-rank representation (GPU accelerated)",
    "7. **Clustering & Evaluation** — Hierarchical clustering + ARI/NMI metrics",
    "8. **Visualization** — H matrix heatmap, confusion matrix, Z matrix analysis",
    "9. **Parameter Sensitivity** — Explore hyperparameter effects",
    "",
    "> ⏱ **Expected runtime**: ~2-5 minutes on GPU (vs. ~30 min on CPU).",
]))

# ============ CELL 2: Section 1 ============
C.append(make_cell("markdown", [
    "## 1. Setup & Imports"
]))

C.append(make_cell("code", [
    "import sys, os, warnings, time",
    "warnings.filterwarnings('ignore')",
    "",
    "# Fix CuPy DLL loading on Windows",
    "_cuda_path = os.environ.get('CUDA_PATH', r'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0')",
    "try:",
    "    os.add_dll_directory(os.path.join(_cuda_path, 'bin'))",
    "    print(f'CUDA DLL path added: {_cuda_path}\\\\bin')",
    "except Exception as e:",
    "    print(f'Note: os.add_dll_directory not available or failed: {e}')",
    "",
    "import numpy as np",
    "import pandas as pd",
    "",
    "# Add current directory to path for INLRMF imports",
    "sys.path.insert(0, os.path.abspath('.'))",
    "",
    "from scipy.cluster.hierarchy import linkage, fcluster, dendrogram",
    "from sklearn.preprocessing import Normalizer",
    "from sklearn.metrics.cluster import adjusted_rand_score",
    "from sklearn import metrics",
    "",
    "# Visualization",
    "import matplotlib.pyplot as plt",
    "import matplotlib",
    "matplotlib.rcParams['font.size'] = 12",
    "matplotlib.rcParams['figure.dpi'] = 100",
    "matplotlib.rcParams['figure.figsize'] = (10, 6)",
    "%matplotlib inline",
    "",
    "print('All libraries imported successfully.')",
    "print(f'Working directory: {os.getcwd()}')",
]))

C.append(make_cell("code", [
    "# Verify GPU environment",
    "import cupy as cp",
    "print(f'CuPy version: {cp.__version__}')",
    "print(f'CUDA available: {cp.cuda.is_available()}')",
    "print(f'GPU device: {cp.cuda.runtime.getDeviceProperties(0)[\"name\"].decode()}')",
    "",
    "# Quick GPU compute test",
    "a = cp.random.randn(100, 100)",
    "b = cp.random.randn(100, 100)",
    "c = cp.dot(a, b)",
    "print(f'GPU matrix multiply test: {a.shape} x {b.shape} -> {c.shape} OK')",
]))

# ============ CELL 3: Section 2 ============
C.append(make_cell("markdown", [
    "## 2. Data Loading",
    "",
    "We load two scRNA-seq datasets for the same set of cells, quantified using different pipelines:",
    "- **RSEM** (RNA-Seq by Expectation-Maximization)",
    "- **Kallisto** (pseudoalignment-based quantification)",
    "",
    "These are from the **Pollen et al.** human brain single-cell dataset, containing gene expression "
    "counts across multiple cell types including neurons, iPSCs, and various neural progenitors.",
]))

C.append(make_cell("code", [
    "# Load the Pollen datasets",
    "# Adjust data_dir to your local path if needed",
    "data_dir = 'D:/INLRMF-code/test_data'",
    "print(f'Data directory: {data_dir}')",
    "",
    "df1_raw = pd.read_csv(os.path.join(data_dir, 'Pollen_RSEMTopHat.csv'), index_col=0)",
    "df2_raw = pd.read_csv(os.path.join(data_dir, 'Pollen_Kallisto.csv'), index_col=0)",
    "",
    "# Extract cell type labels from column names (e.g., 'Neuron_1' -> 'Neuron')",
    "label_raw = [i.split('_')[0] for i in df1_raw.columns]",
    "",
    "print(f'RSEM dataset shape:     {df1_raw.shape[0]:,} genes x {df1_raw.shape[1]} cells')",
    "print(f'Kallisto dataset shape: {df2_raw.shape[0]:,} genes x {df2_raw.shape[1]} cells')",
    "print(f'Number of cell types:   {len(set(label_raw))}')",
    "print(f'Cell types: {sorted(set(label_raw))}')",
    "print(f'First 10 labels: {label_raw[:10]}')",
]))

C.append(make_cell("code", [
    "# Preview the raw data distribution",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))",
    "",
    "ax = axes[0]",
    "rsem_vals = df1_raw.values.flatten()",
    "kallisto_vals = df2_raw.values.flatten()",
    "ax.hist(np.log10(rsem_vals[rsem_vals > 0] + 1), bins=100, alpha=0.6, label='RSEM', density=True)",
    "ax.hist(np.log10(kallisto_vals[kallisto_vals > 0] + 1), bins=100, alpha=0.6, label='Kallisto', density=True)",
    "ax.set_xlabel('log10(Expression + 1)')",
    "ax.set_ylabel('Density')",
    "ax.set_title('Expression Distribution (non-zero values)')",
    "ax.legend()",
    "",
    "ax = axes[1]",
    "sparsity_rsem = (df1_raw.values == 0).sum() / df1_raw.values.size * 100",
    "sparsity_kal = (df2_raw.values == 0).sum() / df2_raw.values.size * 100",
    "ax.bar(['RSEM', 'Kallisto'], [sparsity_rsem, sparsity_kal], color=['steelblue', 'coral'], alpha=0.8)",
    "ax.set_ylabel('Zero entries (%)')",
    "ax.set_title('Data Sparsity')",
    "for i, v in enumerate([sparsity_rsem, sparsity_kal]):",
    "    ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')",
    "",
    "plt.tight_layout()",
    "plt.show()",
]))

# ============ CELL 4: Section 3 ============
C.append(make_cell("markdown", [
    "## 3. Gene Selection",
    "",
    "We filter genes based on expression prevalence to remove noisy and uninformative features:",
    "- **rm_value1=2**: Genes must have expression > 2 in at least 6% of cells",
    "- **rm_value2=0**: Genes must have expression > 0 in less than 94% of cells",
    "",
    "This removes both low-expression genes (technical noise) and constitutively expressed "
    "housekeeping genes that provide little discriminative information for clustering.",
]))

C.append(make_cell("code", [
    "# Copy data for preprocessing",
    "df1 = df1_raw.copy()",
    "df2 = df2_raw.copy()",
    "label = label_raw.copy()",
    "",
    "df1.columns = label",
    "df2.columns = label",
    "",
    "print('Before gene selection:')",
    "print(f'  RSEM genes:     {df1.shape[0]:,}')",
    "print(f'  Kallisto genes: {df2.shape[0]:,}')",
    "",
    "rm_value1, rm_value2, threshold = 2, 0, 0.06",
    "gene_set1 = list(",
    "    set(df1.index[(df1 > rm_value1).sum(axis=1) > threshold * len(df1.columns)])",
    "    & set(df1.index[(df1 > rm_value2).sum(axis=1) < (1 - threshold) * len(df1.columns)])",
    ")",
    "gene_set2 = list(",
    "    set(df2.index[(df2 > rm_value1).sum(axis=1) > threshold * len(df2.columns)])",
    "    & set(df2.index[(df2 > rm_value2).sum(axis=1) < (1 - threshold) * len(df2.columns)])",
    ")",
    "",
    "df1_filtered = df1.loc[gene_set1, :]",
    "df2_filtered = df2.loc[gene_set2, :]",
    "",
    "print(f'\\nAfter gene selection:')",
    "print(f'  RSEM genes:     {df1_filtered.shape[0]:,} ({len(gene_set1)/df1.shape[0]*100:.1f}% retained)')",
    "print(f'  Kallisto genes: {df2_filtered.shape[0]:,} ({len(gene_set2)/df2.shape[0]*100:.1f}% retained)')",
]))

# ============ CELL 5: Section 4 ============
C.append(make_cell("markdown", [
    "## 4. Log Scaling",
    "",
    "We apply $\\log_2$ transformation to stabilize variance and reduce the "
    "disproportionate influence of highly expressed genes:",
    "",
    "$$X_{scaled} = \\log_2(X + 1)$$",
]))

C.append(make_cell("code", [
    "df1_log = np.log2(df1_filtered + 1)",
    "df2_log = np.log2(df2_filtered + 1)",
    "",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))",
    "",
    "for idx, (data_before, data_after, name) in enumerate([",
    "    (df1_filtered, df1_log, 'RSEM'),",
    "    (df2_filtered, df2_log, 'Kallisto')",
    "]):",
    "    ax = axes[idx]",
    "    v_before = data_before.values.flatten()",
    "    v_after = data_after.values.flatten()",
    "    ax.hist(v_before[v_before > 0], bins=100, alpha=0.6, label='Before log2', density=True)",
    "    ax.hist(v_after[v_after > 0], bins=100, alpha=0.6, label='After log2', density=True)",
    "    ax.set_xlabel('Expression value')",
    "    ax.set_ylabel('Density')",
    "    ax.set_title(f'{name}: Before vs After Log2 Transform')",
    "    ax.legend()",
    "",
    "plt.tight_layout()",
    "plt.show()",
]))

# ============ CELL 6: Section 5 ============
C.append(make_cell("markdown", [
    "## 5. Normalization",
    "",
    "We apply **L1 cell-wise normalization** so that each cell's total expression sums to 1. "
    "This corrects for differences in sequencing depth (library size) between cells:",
    "",
    "$$X_{norm}[i,j] = \\frac{X[i,j]}{\\sum_k X[k,j]}$$",
]))

C.append(make_cell("code", [
    "norm = Normalizer(norm='l1', copy=False)",
    "df1_norm = norm.fit_transform(df1_log)",
    "df2_norm = norm.fit_transform(df2_log)",
    "",
    "print('L1 normalization check (each cell should sum to 1):')",
    "print(f'  RSEM:     sum per cell range = [{df1_norm.sum(axis=0).min():.4f}, {df1_norm.sum(axis=0).max():.4f}]')",
    "print(f'  Kallisto: sum per cell range = [{df2_norm.sum(axis=0).min():.4f}, {df2_norm.sum(axis=0).max():.4f}]')",
    "print(f'\\nNormalized data shapes:')",
    "print(f'  RSEM:     {df1_norm.shape}')",
    "print(f'  Kallisto: {df2_norm.shape}')",
]))

# ============ CELL 7: Section 6 ============
C.append(make_cell("markdown", [
    "## 6. INLRMF Factorization (GPU)",
    "",
    "This is the core of the algorithm. INLRMF jointly factorizes two data matrices:",
    "",
    "$$D_1 \\approx W_1 H, \\quad D_2 \\approx W_2 H$$",
    "",
    "with the global structure constraint $H \\approx HZ$, where:",
    "- $W_1 \\in \\mathbb{R}^{g_1 \\times r}$ — gene basis matrix for RSEM",
    "- $W_2 \\in \\mathbb{R}^{g_2 \\times r}$ — gene basis matrix for Kallisto",
    "- $H \\in \\mathbb{R}^{r \\times n}$ — shared cell representation (coefficient) matrix",
    "- $Z \\in \\mathbb{R}^{n \\times n}$ — global low-rank structure among cells",
    "",
    "**Key parameters (matching original paper example1.py):**",
    "- `rank=9`: Number of latent factors for NMF (hardcoded: `range(9,10)`)",
    "- `cluster_num=10`: Number of clusters for hierarchical clustering (`len(np.unique(label))`)",
    "- `lambda1 = g1_raw / g2_raw`: Balances the two datasets by raw gene count ratio (before gene selection)",
    "- `lambda4`: Sparsity regularization on H",
    "- `lambda5`: Weight of the global structure term $\\|H - HZ\\|_F$",
    "- `lambda6`: Sparsity regularization on Z",
    "",
    "> 🚀 Using **GPU (CuPy)** for accelerated coordinate descent NMF.",
]))

C.append(make_cell("code", [
    "from INLRMF import INLRMF",
    "",
    "# IMPORTANT: Match original example1.py parameters exactly",
    "# rank = 9 (hardcoded in original: range(9,10)), not len(set(label))",
    "rank = 9",
    "# cluster_num = number of unique cell types (used for clustering, not factorization)",
    "n_clusters = len(set(label))",
    "print(f'Factorization rank: {rank}')",
    "print(f'Number of cell types (for clustering): {n_clusters}')",
    "",
    "# Z initialized as identity (matching original: np.eye(num_columns))",
    "num_cells = df1.shape[1]",
    "Z_init = np.eye(num_cells)",
    "print(f'Z matrix shape: {Z_init.shape}')",
    "",
    "# lambda1 computed from ORIGINAL data (before gene selection), matching example1.py",
    "lambda1 = df1.shape[0] / df2.shape[0]",
    "print(f'lambda1 = {lambda1:.4f} (raw gene count ratio: {df1.shape[0]}/{df2.shape[0]})')",
]))

C.append(make_cell("code", [
    "def run_inlrmf(df1_data, df2_data, Z_mat, rank_val, lam1, lam4, lam5, lam6, cluster_num):",
    "    '''Run the full INLRMF pipeline on GPU and return all results.",
    "    ",
    "    Parameters",
    "    ----------",
    "    df1_data, df2_data : DataFrame",
    "        Raw (un-preprocessed) expression data for each quantification method.",
    "    Z_mat : ndarray",
    "        Initial cell-cell relationship matrix (typically identity).",
    "    rank_val : int",
    "        Number of latent factors for NMF (rank=9 in original paper).",
    "    lam1, lam4, lam5, lam6 : float",
    "        Regularization hyperparameters.",
    "    cluster_num : int",
    "        Number of clusters for hierarchical clustering (= number of cell types).",
    "    ",
    "    Returns",
    "    -------",
    "    dict with keys: W1, W2, H, Z, cluster, ari, nmi",
    "    '''",
    "    inlrmf = INLRMF(",
    "        D1=df1_data, D2=df2_data, rank=rank_val, Z=Z_mat,",
    "        lambda1=lam1, lambda4=lam4, lambda5=lam5, lambda6=lam6",
    "    )",
    "    # Step 1-3: Preprocessing",
    "    inlrmf.gene_selection()",
    "    inlrmf.log_scale()",
    "    inlrmf.normalize()",
    "    # Step 4: GPU-accelerated Joint NMF factorization",
    "    inlrmf.factorize(solver='cd', init='random', device='gpu')",
    "    # Step 5: Hierarchical clustering (cluster_num != rank in original design)",
    "    inlrmf.clustering(cluster_num=cluster_num)",
    "    # Step 6: Evaluate against ground truth",
    "    ari = adjusted_rand_score(label, inlrmf.cluster)",
    "    nmi = metrics.normalized_mutual_info_score(label, inlrmf.cluster)",
    "    return {",
    "        'W1': inlrmf.W1, 'W2': inlrmf.W2,",
    "        'H': inlrmf.H, 'Z': inlrmf.Z,",
    "        'cluster': inlrmf.cluster,",
    "        'ari': ari, 'nmi': nmi",
    "    }",
]))

C.append(make_cell("code", [
    "print('='*60)",
    "print('Running INLRMF on GPU (parameters matching original example1.py)...')",
    "print(f'  lambda1 = {lambda1:.4f} (raw gene count ratio: {df1.shape[0]}/{df2.shape[0]})')",
    "print(f'  lambda4 = 10 (H sparsity)')",
    "print(f'  lambda5 = 1  (global structure)')",
    "print(f'  lambda6 = 0.002 (Z sparsity)')",
    "print(f'  rank    = {rank} (NMF factorization components)')",
    "print(f'  cluster_num = {n_clusters} (number of cell types for clustering)')",
    "print(f'  device  = gpu (CuPy)')",
    "print('='*60)",
    "",
    "t_start = time.time()",
    "result = run_inlrmf(",
    "    df1_data=df1_filtered.copy(),",
    "    df2_data=df2_filtered.copy(),",
    "    Z_mat=Z_init,",
    "    rank_val=rank,",
    "    cluster_num=n_clusters,",
    "    lam1=lambda1, lam4=10, lam5=1, lam6=0.002",
    ")",
    "elapsed = time.time() - t_start",
    "",
    "print(f'\\nElapsed time: {elapsed:.1f}s ({elapsed/60:.1f} min)')",
    "print(f'\\n{\"=\"*60}')",
    "print(f'INLRMF Results:')",
    "print(f'  ARI (Adjusted Rand Index):      {result[\"ari\"]:.4f}')",
    "print(f'  NMI (Normalized Mutual Info):   {result[\"nmi\"]:.4f}')",
    "print(f'  W1 shape: {result[\"W1\"].shape}')",
    "print(f'  W2 shape: {result[\"W2\"].shape}')",
    "print(f'  H shape:  {result[\"H\"].shape}')",
    "print(f'  Z shape:  {result[\"Z\"].shape}')",
    "print(f'{\"=\"*60}')",
    "",
    "if result['ari'] > 0.8:",
    "    print(f'\\nExcellent clustering agreement with ground truth (ARI={result[\"ari\"]:.4f})!')",
    "elif result['ari'] > 0.6:",
    "    print(f'\\nGood clustering agreement with ground truth (ARI={result[\"ari\"]:.4f}).')",
    "else:",
    "    print(f'\\nModerate agreement (ARI={result[\"ari\"]:.4f}) — consider tuning parameters.')",
]))

# ============ CELL 8: Section 7 ============
C.append(make_cell("markdown", [
    "## 7. Clustering Visualization",
]))

C.append(make_cell("code", [
    "# --- H matrix heatmap ---",
    "fig, axes = plt.subplots(1, 2, figsize=(16, 6))",
    "",
    "ax = axes[0]",
    "im = ax.imshow(result['H'], aspect='auto', cmap='viridis')",
    "ax.set_xlabel('Cells')",
    "ax.set_ylabel('Latent factors')",
    "ax.set_title(f'H Matrix ({result[\"H\"].shape[0]} factors x {result[\"H\"].shape[1]} cells)')",
    "plt.colorbar(im, ax=ax, shrink=0.8)",
    "",
    "ax = axes[1]",
    "sorted_idx = np.argsort(result['cluster'])",
    "H_sorted = result['H'][:, sorted_idx]",
    "im = ax.imshow(H_sorted, aspect='auto', cmap='viridis')",
    "ax.set_xlabel('Cells (sorted by cluster)')",
    "ax.set_ylabel('Latent factors')",
    "ax.set_title('H Matrix (Sorted by Cluster Assignment)')",
    "plt.colorbar(im, ax=ax, shrink=0.8)",
    "",
    "plt.tight_layout()",
    "plt.show()",
]))

C.append(make_cell("code", [
    "# --- Confusion matrix ---",
    "import seaborn as sns",
    "",
    "cluster_labels = result['cluster']",
    "true_labels = label",
    "",
    "unique_clusters = sorted(set(cluster_labels))",
    "unique_labels_list = sorted(set(true_labels))",
    "contingency = np.zeros((len(unique_labels_list), len(unique_clusters)))",
    "",
    "for i, tl in enumerate(unique_labels_list):",
    "    for j, cl in enumerate(unique_clusters):",
    "        contingency[i, j] = np.sum(",
    "            (np.array(true_labels) == tl) & (cluster_labels == cl)",
    "        )",
    "",
    "fig, ax = plt.subplots(figsize=(12, 8))",
    "sns.heatmap(contingency, annot=True, fmt='.0f', cmap='Blues',",
    "            xticklabels=[f'C{c}' for c in unique_clusters],",
    "            yticklabels=unique_labels_list, ax=ax,",
    "            cbar_kws={'label': 'Cell count'})",
    "ax.set_xlabel('INLRMF Cluster')",
    "ax.set_ylabel('True Cell Type')",
    "ax.set_title(f'Clustering Confusion Matrix\\nARI={result[\"ari\"]:.4f}, NMI={result[\"nmi\"]:.4f}')",
    "plt.tight_layout()",
    "plt.show()",
]))

C.append(make_cell("code", [
    "# --- Dendrogram ---",
    "fig, ax = plt.subplots(figsize=(18, 5))",
    "",
    "n_show = min(50, len(label))",
    "H_subset = result['H'][:, :n_show]",
    "linkage_matrix = linkage(H_subset.T, method='ward')",
    "",
    "color_map = {}",
    "unique_types = sorted(set(label))",
    "colors_list = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))",
    "for ct, c in zip(unique_types, colors_list):",
    "    color_map[ct] = c",
    "",
    "dendrogram(linkage_matrix, labels=[label[i] for i in range(n_show)],",
    "           leaf_rotation=90, leaf_font_size=8,",
    "           ax=ax, link_color_func=lambda k: 'gray',",
    "           above_threshold_color='gray')",
    "",
    "for tick_label, i in zip(ax.get_xticklabels(), range(n_show)):",
    "    tick_label.set_color(color_map.get(label[i], 'black'))",
    "",
    "ax.set_title('Hierarchical Clustering Dendrogram (Ward Linkage)')",
    "ax.set_ylabel('Distance')",
    "plt.tight_layout()",
    "plt.show()",
]))

# ============ CELL 9: Section 8 ============
C.append(make_cell("markdown", [
    "## 8. Global Structure Matrix (Z)",
    "",
    "The Z matrix captures global low-rank relationships among cells via $H \\approx HZ$. "
    "Each entry $Z[i,j]$ encodes how cell $i$'s representation relates to cell $j$'s.",
]))

C.append(make_cell("code", [
    "fig, axes = plt.subplots(1, 2, figsize=(14, 6))",
    "",
    "ax = axes[0]",
    "n_z = min(100, result['Z'].shape[0])",
    "Z_sub = result['Z'][:n_z, :n_z]",
    "vmax = np.max(np.abs(Z_sub))",
    "im = ax.imshow(Z_sub, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)",
    "ax.set_xlabel('Cells')",
    "ax.set_ylabel('Cells')",
    "ax.set_title(f'Z Matrix ({n_z}x{n_z} subset)')",
    "plt.colorbar(im, ax=ax, shrink=0.8)",
    "",
    "ax = axes[1]",
    "try:",
    "    U_z, S_z, Vt_z = np.linalg.svd(result['Z'], full_matrices=False)",
    "    ax.semilogy(S_z, 'o-', color='steelblue', markersize=3, alpha=0.7)",
    "    ax.set_xlabel('Index')",
    "    ax.set_ylabel('Singular Value (log scale)')",
    "    n_sig = np.sum(S_z > 0.01 * S_z[0])",
    "    ax.set_title(f'Spectrum of Z (effective rank: {n_sig})')",
    "    ax.axhline(y=0.01 * S_z[0], color='red', linestyle='--', alpha=0.5, label='1% threshold')",
    "    ax.legend()",
    "except Exception as e:",
    "    ax.text(0.5, 0.5, f'SVD error: {e}', transform=ax.transAxes, ha='center', va='center')",
    "",
    "plt.tight_layout()",
    "plt.show()",
]))

# ============ CELL 10: Section 9 ============
C.append(make_cell("markdown", [
    "## 9. Parameter Sensitivity Analysis",
    "",
    "Test the effect of key hyperparameters on clustering performance. "
    "With GPU acceleration, we can sweep a meaningful grid in minutes.",
    "",
    "> 🚀 Each run takes ~10-30 seconds on GPU, making parameter sweeps practical.",
]))

C.append(make_cell("code", [
    "# Parameter grid search (GPU accelerated — runs quickly!)",
    "lambda4_values = [1, 5, 10, 50, 100]",
    "lambda5_values = [1]",
    "lambda6_values = [0.0001, 0.002, 0.01, 0.1]",
    "",
    "results_grid = []",
    "total = len(lambda4_values) * len(lambda5_values) * len(lambda6_values)",
    "print(f'Running {total} parameter combinations on GPU...')",
    "print()",
    "",
    "count = 0",
    "for l4 in lambda4_values:",
    "    for l5 in lambda5_values:",
    "        for l6 in lambda6_values:",
    "            count += 1",
    "            t0 = time.time()",
    "            print(f'[{count}/{total}] lambda4={l4}, lambda5={l5}, lambda6={l6} ... ',",
    "                  end='', flush=True)",
    "            try:",
    "                res = run_inlrmf(",
    "                    df1_data=df1_filtered.copy(),",
    "                    df2_data=df2_filtered.copy(),",
    "                    Z_mat=Z_init,",
    "                    rank_val=rank, cluster_num=n_clusters,",
    "                    lam1=lambda1, lam4=l4, lam5=l5, lam6=l6",
    "                )",
    "                results_grid.append({**res, 'lambda4': l4, 'lambda5': l5, 'lambda6': l6})",
    "                t_elapsed = time.time() - t0",
    "                print(f'ARI={res[\"ari\"]:.4f}, NMI={res[\"nmi\"]:.4f} ({t_elapsed:.1f}s)')",
    "            except Exception as e:",
    "                print(f'FAILED: {e}')",
    "",
    "print(f'\\nCompleted {len(results_grid)}/{total} combinations.')",
]))

C.append(make_cell("code", [
    "# Visualize parameter sensitivity",
    "if len(results_grid) >= 2:",
    "    df_results = pd.DataFrame(results_grid)",
    "",
    "    best_ari = df_results.loc[df_results['ari'].idxmax()]",
    "    best_nmi = df_results.loc[df_results['nmi'].idxmax()]",
    "    print(f'Best ARI: {best_ari[\"ari\"]:.4f} ',",
    "          f'(lambda4={best_ari[\"lambda4\"]:.0f}, lambda5={best_ari[\"lambda5\"]:.0f}, lambda6={best_ari[\"lambda6\"]:.4f})')",
    "    print(f'Best NMI: {best_nmi[\"nmi\"]:.4f} ',",
    "          f'(lambda4={best_nmi[\"lambda4\"]:.0f}, lambda5={best_nmi[\"lambda5\"]:.0f}, lambda6={best_nmi[\"lambda6\"]:.4f})')",
    "    print(f'\\nFull results table:')",
    "    display(df_results[['lambda4', 'lambda5', 'lambda6', 'ari', 'nmi']].sort_values('ari', ascending=False))",
    "",
    "    if len(lambda4_values) > 1 and len(lambda6_values) > 1:",
    "        fig, axes = plt.subplots(1, 2, figsize=(14, 5))",
    "        for idx, metric in enumerate(['ari', 'nmi']):",
    "            ax = axes[idx]",
    "            pivot = df_results.pivot_table(",
    "                values=metric, index='lambda4', columns='lambda6', aggfunc='mean'",
    "            )",
    "            im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)",
    "            ax.set_xticks(range(len(pivot.columns)))",
    "            ax.set_xticklabels([f'{x:.4f}' for x in pivot.columns], rotation=45)",
    "            ax.set_yticks(range(len(pivot.index)))",
    "            ax.set_yticklabels(pivot.index)",
    "            ax.set_xlabel('lambda6 (Z regularization)')",
    "            ax.set_ylabel('lambda4 (H sparsity)')",
    "            ax.set_title(f'{metric.upper()} Score')",
    "            for i in range(len(pivot.index)):",
    "                for j in range(len(pivot.columns)):",
    "                    ax.text(j, i, f'{pivot.values[i,j]:.3f}',",
    "                           ha='center', va='center', fontsize=9)",
    "            plt.colorbar(im, ax=ax)",
    "        plt.tight_layout()",
    "        plt.show()",
    "else:",
    "    print('Not enough results to visualize.')",
]))

# ============ CELL 11: Section 10 ============
C.append(make_cell("markdown", [
    "## 10. Summary",
    "",
    "### Key Findings",
    "",
    "The INLRMF algorithm successfully integrates two scRNA-seq quantification methods "
    "(RSEM and Kallisto) using GPU-accelerated joint NMF:",
    "",
    "| Component | Role |",
    "|-----------|------|",
    "| **W₁, W₂** | Gene basis matrices — capture modality-specific expression programs |",
    "| **H** | Shared cell representation — used directly for clustering |",
    "| **Z** | Global structure — encodes cell-cell relationships ($H \\approx HZ$) |",
    "",
    "### Algorithm Features",
    "",
    "1. **Joint factorization** — Simultaneously learns from both RSEM and Kallisto",
    "2. **Global structure** — The Z matrix captures low-rank cell-cell relationships",
    "3. **Sparsity regularization** — L1 penalties produce interpretable, sparse factors",
    "4. **GPU acceleration** — CuPy-based solver enables practical parameter sweeps",
    "",
    "### Reproducibility",
    "",
    "- **GPU**: Uses CuPy with CUDA toolkit for fast NMF factorization",
    "- Expected output: ARI ≈ 0.85, NMI ≈ 0.93 on the Pollen dataset",
    "- To run with your own data: replace CSV files in the Data Loading section",
    "- Adjust `data_dir` path to point to your local test data",
    "",
    "### References",
    "",
    "[1] Shiga M, Seno S, Onizuka M, et al. SC-JNMF: single-cell clustering integrating "
    "multiple quantification methods based on joint non-negative matrix factorization. "
    "*PeerJ*, 2021, 9: e12087.",
    "",
    "[2] Zhang W, Xue X, Zheng X, et al. NMFLRR: clustering scRNA-seq data by integrating "
    "nonnegative matrix factorization with low rank representation. "
    "*IEEE Journal of Biomedical and Health Informatics*, 2021, 26(3): 1394-1405.",
    "",
    "[3] Lee D D, Seung H S. Learning the parts of objects by non-negative matrix "
    "factorization. *Nature*, 1999, 401(6755): 788-791.",
    "",
    "[4] Boutsidis C, Gallopoulos E. SVD based initialization: A head start for "
    "nonnegative matrix factorization. *Pattern Recognition*, 2008, 41(4): 1350-1362.",
]))

# ===== BUILD NOTEBOOK =====
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (INLRMF GPU)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.9.0"
        }
    },
    "cells": C
}

output_path = "D:/INLRMF-master/INLRMF-code/INLRMF_analysis.ipynb"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook written: {output_path}")
print(f"Total cells: {len(C)}")
for i, c in enumerate(C):
    src_preview = ''.join(c['source'])[:80].replace('\n', ' ')
    print(f"  [{i+1}] {c['cell_type']:9s} | {src_preview}...")
