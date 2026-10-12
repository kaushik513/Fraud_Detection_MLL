"""
resampling.py — SMOTE and ADASYN implemented from scratch.

Version: 1.0.0
Changelog:
  1.0.0 (2026-10-10) — vectorised SMOTE (Chawla et al., 2002) and ADASYN
                       (He et al., 2008) built on sklearn NearestNeighbors;
                       'imbalanced' no-op sampler; factory function.

Why from scratch? imbalanced-learn is optional; writing both oversamplers on top
of a k-nearest-neighbour search makes the Week-8 kNN material (Euclidean
distance, choosing k) do real work in the project, and lets the report explain
exactly how the synthetic fraud points are generated. If imbalanced-learn is
installed, `tests/test_core.py` cross-checks class counts against it.

Both methods are applied to the TRAINING data only (see experiment.py),
after one-hot encoding and scaling, mirroring the reference workflow [1].
"""

__version__ = "1.0.0"

import numpy as np
from sklearn.neighbors import NearestNeighbors


class NoResampling:
    """'imbalanced' setting: data passes through untouched."""
    name = "imbalanced"

    def fit_resample(self, X, y):
        return X, y


class SMOTE:
    """Synthetic Minority Over-sampling TEchnique.

    For every synthetic sample:
      1. pick a minority point x_i at random,
      2. pick one of its k nearest *minority* neighbours x_nn,
      3. x_new = x_i + u * (x_nn - x_i),  u ~ Uniform(0, 1)
    i.e. a random point on the line segment between two nearby fraud cases.
    Generates enough points to make the classes exactly 1:1.
    """
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
    """ADAptive SYNthetic sampling.

    Like SMOTE, but the number of synthetic points made around each minority
    sample is proportional to how "hard" it is to learn:
      r_i = (# majority points among its K nearest neighbours in the whole set) / K
      g_i = round( r_i / sum(r) * G ),  G = n_majority - n_minority
    so fraud cases surrounded by legitimate transactions get more synthetic
    neighbours. Classes end up *approximately* balanced (rounding).
    """
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

        # Step 1: difficulty ratio r_i using neighbours from BOTH classes.
        K = self.n_neighbors
        nn_all = NearestNeighbors(n_neighbors=K + 1).fit(X)
        neigh_all = nn_all.kneighbors(X_min, return_distance=False)[:, 1:]
        r = (y[neigh_all] == 0).sum(axis=1) / K
        if r.sum() == 0:                       # minority perfectly separated:
            return SMOTE(K, self.random_state).fit_resample(X, y)  # degrade to SMOTE
        r_hat = r / r.sum()
        g = np.rint(r_hat * G).astype(int)
        n_new = int(g.sum())
        if n_new == 0:
            return X, y

        # Step 2: interpolate towards minority neighbours, g_i times for x_i.
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
