# INLRMF
A Global Structure-Aware NMF Algorithm for Single-Cell RNA-seq Clustering

## Introduction
This project constructed a non negative matrix factorization model integrating two scRNA-seq. This model is called the integrated non negative low rank matrix factorization algorithm (inlrmf), which can simultaneously learn the local and global characteristics of scRNA-seq data

The code in this project will reproduce the results in our paper, "INLRMF: A Global Structure-Aware NMF Algorithm for Single-Cell RNA-seq Clustering".

## Requirement
- CUDA>=10.0 (gpu required)
- Pandas>=2.2
- Python>=3.6
- NumPy >= 1.13.3
- Cupy >= 12.0
- SciPy >= 0.19.1
- Scikit-Learn >= 0.23

##  Usage
You can reproduce the clustering analysis in two ways:

- **PyCharm / Command line**: Run the sample script located in `INLRMF-code\example\example1.py` to obtain and view the clustering results.
- **Jupyter Notebook**: Open and run `INLRMF-code\INLRMF_analysis.ipynb` for a step-by-step interactive pipeline with visualizations, parameter sensitivity analysis, and CSV export of all factor matrices (W1, W2, H, Z).

## Data
The Input data is located at `INLRMF-code\test_data`. The `.csv` files are the input datasets of the INLRMF method.

### Dropout Noise
The script `INLRMF-code\example\Dropout noise.py` simulates dropout events common in scRNA-seq data by randomly setting a fraction (default 30%) of expression values to zero. It reads the original dataset and outputs a noisy version, which can be used to evaluate the robustness of INLRMF under realistic single-cell dropout conditions.

## References
<div id="svdinit">
[1] Shiga M, Seno S, Onizuka M, et al. SC-JNMF: single-cell clustering integrating multiple quantification methods based on joint non-negative matrix factorization[J]. PeerJ, 2021, 9: e12087.

[2] Zhang W, Xue X, Zheng X, et al. NMFLRR: clustering scRNA-seq data by integrating nonnegative matrix factorization with low rank representation[J]. IEEE Journal of Biomedical and Health Informatics, 2021, 26(3): 1394-1405.

[3] Lee D D, Seung H S. Learning the parts of objects by non-negative matrix factorization[J]. nature, 1999, 401(6755): 788-791.

[4] Boutsidis C, Gallopoulos E. SVD based initialization: A head start for nonnegative matrix factorization[J]. Pattern recognition, 2008, 41(4): 1350-1362.
</div>
