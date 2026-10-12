__version__ = "1.0.0"

import numpy as np
from sklearn.neighbors import NearestNeighbors


class NoResampling:
    name = "imbalanced"

    def fit_resample(self, X, y):
        return X, y


class SMOTE:
    name = "smote"

    def __init__(self, k_neighbors=5, random_state=None):
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y):
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        X_min = X[y == 1]
        n_new = int((y == 0).sum() - len(X_min))
        if n_new <= 0 or len(X_min) < 2:
            return X, y
        k = min(self.k_neighbors, len(X_min) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
        neigh = nn.kneighbors(X_min, return_distance=False)[:, 1:]  # drop self

        base = rng.integers(0, len(X_min), n_new)
        partner = neigh[base, rng.integers(0, k, n_new)]
        gap = rng.random((n_new, 1))
        X_syn = X_min[base] + gap * (X_min[partner] - X_min[base])

        return (np.vstack([X, X_syn]),
                np.concatenate([y, np.ones(n_new, dtype=y.dtype)]))


class ADASYN:
    name = "adasyn"

    def __init__(self, n_neighbors=5, beta=1.0, random_state=None):
        self.n_neighbors = n_neighbors
        self.beta = beta
        self.random_state = random_state

    def fit_resample(self, X, y):
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        min_idx = np.flatnonzero(y == 1)
        X_min = X[min_idx]
        G = int(((y == 0).sum() - len(X_min)) * self.beta)
        if G <= 0 or len(X_min) < 2:
            return X, y

        K = self.n_neighbors
        nn_all = NearestNeighbors(n_neighbors=K + 1).fit(X)
        neigh_all = nn_all.kneighbors(X_min, return_distance=False)[:, 1:]
        r = (y[neigh_all] == 0).sum(axis=1) / K
        if r.sum() == 0:                     
            return SMOTE(K, self.random_state).fit_resample(X, y)  
        r_hat = r / r.sum()
        g = np.rint(r_hat * G).astype(int)
        n_new = int(g.sum())
        if n_new == 0:
            return X, y

        k = min(K, len(X_min) - 1)
        nn_min = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
        neigh_min = nn_min.kneighbors(X_min, return_distance=False)[:, 1:]
        base = np.repeat(np.arange(len(X_min)), g)
        partner = neigh_min[base, rng.integers(0, k, n_new)]
        gap = rng.random((n_new, 1))
        X_syn = X_min[base] + gap * (X_min[partner] - X_min[base])

        return (np.vstack([X, X_syn]),
                np.concatenate([y, np.ones(n_new, dtype=y.dtype)]))


def get_sampler(name, random_state=None, k=5):
    name = name.lower()
    if name == "imbalanced":
        return NoResampling()
    if name == "smote":
        return SMOTE(k_neighbors=k, random_state=random_state)
    if name == "adasyn":
        return ADASYN(n_neighbors=k, random_state=random_state)
    raise ValueError(f"Unknown sampling '{name}'")
