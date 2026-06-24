import numpy as np
from math import sqrt
from sklearn.decomposition import NMF
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import svd_flip, squared_norm


class SparseNMF(NMF):

    def __init__(self, n_components=None, solver='cd',
                 beta_loss='frobenius', tol=1e-4, max_iter=200,
                 random_state=None,  l1_ratio=0., verbose=0,
                 shuffle=False):
        super(SparseNMF, self).__init__(n_components=n_components, init='custom',
                                        solver=solver, beta_loss=beta_loss, tol=tol,
                                        max_iter=max_iter, random_state=random_state,
                                        l1_ratio=l1_ratio, verbose=verbose,
                                        shuffle=shuffle)

    def fit_transform(self, X, y=None, W=None, H=None):


        W, H = init_nmf(X, self.n_components)
        return super(SparseNMF, self).fit_transform(X, W=W, H=H)

    def fit(self, X, y=None, **params):

        self.fit_transform(X, **params)
        return self


def norm(x):

    return sqrt(squared_norm(x))


def init_nmf(X, n_components,len, eps=1e-6):

    U, S, V = svds(X, n_components)
    S = S[::-1]
    U, V = svd_flip(U[:, ::-1], V[::-1])
    W, H = np.zeros(U.shape), np.zeros(V.shape)

    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]


        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))


        x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
        x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm


        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0
    H[H < eps] = 0

    W1 = W[:len, :]
    W2 = W[len:, :]

    return W1,W2, H

