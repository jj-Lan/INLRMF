from INLRMF import INLRMF
import numpy as np
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import metrics

df1 = pd.read_csv("../test_data/Pollen_RSEMTopHat.csv", index_col=0)
df2 = pd.read_csv("../test_data/Pollen_Kallisto.csv", index_col=0)
label = [i.split('_')[0] for i in df1.columns]
df1.columns = label
df2.columns = label
num_columns =df1.shape[1]
Z = np.eye(num_columns)

A=[10]
B=[1]
C=[0.002]
for l4 in A:
    for l5 in B:
        for l6 in C:
            for r in range(9,10):
                INLRMF = INLRMF(D1=df1, D2=df2, rank=r,Z=Z,
                                  lambda1=df1.shape[0] / df2.shape[0],
                                  lambda4=l4,lambda5=l5,lambda6=l6)
                INLRMF.gene_selection()
                INLRMF.log_scale()
                INLRMF.normalize()
                INLRMF.factorize()
                INLRMF.clustering(cluster_num=len(np.unique(label)))
                nmi=metrics.normalized_mutual_info_score(label, INLRMF.cluster)
                ari = adjusted_rand_score(label, INLRMF.cluster)
                print('L4='f'{l4}','L5='f'{l5}','L6='f'{l6}', 'rank='f'{r}', "ARI:", ari, "NMI:", nmi)
