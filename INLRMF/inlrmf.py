import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import Normalizer
from ._joint_nmf_gpu import Joint_NMF_GPU
from .SparseNMF import SparseNMF,init_nmf

class INLRMF:

    def __init__(self, D1, D2, Z, rank, lambda1=1., lambda2=0.,
                 lambda3=0., lambda4=1., lambda5=1., lambda6=1., W1=None, W2=None, H=None,
                 geneset1=None, geneset2=None, cluster=None):
        self.D1 = D1.astype(np.float32)
        self.D2 = D2.astype(np.float32)
        self.W1 = W1
        self.W2 = W2
        self.H = H
        self.Z = Z
        self.rank = rank
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.lambda5 = lambda5
        self.lambda6 = lambda6
        self.geneset1 = geneset1
        self.geneset2 = geneset2
        self.cluster = cluster

    def gene_selection(self, rm_value1=2, rm_value2=0, threshold=0.06):

        self.gene_set1 = list(set(self.D1.index[(self.D1 > rm_value1).sum(axis=1) > threshold * len(self.D1.columns)])
                              & set(self.D1.index[(self.D1 > rm_value2).sum(axis=1) < (1 - threshold) * len(self.D1.columns)]))
        self.gene_set2 = list(set(self.D2.index[(self.D2 > rm_value1).sum(axis=1) > threshold * len(self.D2.columns)])
                              & set(self.D2.index[(self.D2 > rm_value2).sum(axis=1) < (1 - threshold) * len(self.D2.columns)]))
        self.D1 = self.D1.loc[self.gene_set1, :]
        self.D2 = self.D2.loc[self.gene_set2, :]

    def log_scale(self):

        self.D1 = np.log2(self.D1 + 1)
        self.D2 = np.log2(self.D2 + 1)

    def normalize(self, norm='l1', normalize='cell'):

        norm = Normalizer(norm=norm, copy=False)
        if normalize == 'cell':
            self.D1 = norm.fit_transform(self.D1)
            self.D2 = norm.fit_transform(self.D2)
        elif normalize == 'gene':
            self.D1 = norm.fit_transform(self.D1.T).T
            self.D2 = norm.fit_transform(self.D2.T).T

    def factorize(self, solver='cd', init='random', device='gpu'):

        if solver == 'cd':
            if init == 'random':
                self.D = np.vstack((self.D1, self.D2))
                len = self.D1.shape[0]
                self.W1, self.W2, self.H = init_nmf(
                    X=self.D, n_components=self.rank, len=len)
            elif self.W1 is None or self.W2 is None or self.H is None:
                print("select 'random' or set the value of factorized matrix.")

            if device == 'gpu':
                j_nmf = Joint_NMF_GPU(
                    self.D1, self.D2, self.W1, self.W2, self.H, self.Z,
                    self.lambda1, self.lambda2, self.lambda3, self.lambda4, self.lambda5, self.lambda6,
                    iter_num=10000, conv_judge=1e-5, calc_log=[])
                self.W1, self.W2, self.H, self.Z = j_nmf.calc()
            else:
                raise ValueError(f"Unknown device '{device}'. This project uses GPU only (device='gpu').")


    def clustering(self, method='hierarchical', cluster_num=None):


        if method == 'hierarchical':
            self.cluster = linkage(self.H.T, method='ward')
            self.cluster = fcluster(self.cluster,
                                    t=cluster_num,
                                    criterion="maxclust")
