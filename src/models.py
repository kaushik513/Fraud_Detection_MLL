"""
models.py — model factory, the probability-based kNN, and a from-scratch SGD
logistic regression.

Version: 1.0.0
Changelog:
  1.0.0 (2026-10-10) — factory for LR / kNN / SVM / XGBoost / probability-based
                       kNN (+ Naive Bayes, CART, SGD-LR extensions); XGBoost
                       fallback to sklearn HistGradientBoosting when the
                       xgboost package is missing; stratified-subsampled SVC.

Lab-manual links:
  Week 5  Logistic Regression (sigmoid, threshold 0.5, sklearn LogisticRegression)
  Week 6  Cost functions: binary cross-entropy / log loss, SGD update
          b <- b - alpha * (prediction - y) * x  → LogisticRegressionSGD
  Week 7  Naive Bayes (GaussianNB for continuous data)
  Week 8  kNN: choice of k, Euclidean / Manhattan / Minkowski distance
  Week 9  CART (Gini) vs C4.5-style (entropy / information gain) trees, max_depth
  Week 12 SVM: kernels (linear, poly, rbf, sigmoid), C, soft margin, hinge loss
"""

__version__ = "1.0.0"

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import config

_SKLEARN_GE_18 = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)

try:
    from xgboost import XGBClassifier
    XGB_BACKEND = "xgboost"
except Exception:  # package not installed
    XGBClassifier = None
    XGB_BACKEND = "sklearn-HistGradientBoosting (fallback: pip install xgboost)"


# ---------------------------------------------------------------------------
# Proposed method of the reference paper
# ---------------------------------------------------------------------------
class ProbabilityKNN(BaseEstimator, ClassifierMixin):
    """Probability-based kNN using Logistic-Regression probabilities [1, §3.2.5].

    The reference paper augments kNN's similarity with the fraud probability
    estimated by logistic regression. Our implementation (the paper's exact
    formula could not be retrieved — see PROJECT_NOTES.md) uses

        d(x, z) = ( (1/F) * sum_i |x_i - z_i|^p  +  lambda^p * |P(x) - P(z)|^p )^(1/p)

    where P(.) = LR's P(fraud | features), F = number of features and
    lambda = `prob_weight`. Dividing the feature term by F puts it on the scale
    of an *average* feature, so lambda = 1 makes the LR probability count as much
    as a typical feature; lambda = 0 recovers ordinary kNN (useful sanity check).

    Implemented by rescaling X by F^(-1/p) and appending lambda * P(x) as an extra
    column, so sklearn's fast neighbour search is reused unchanged.
    """

    def __init__(self, n_neighbors=5, prob_weight=1.0, lr_C=1.0, p=2):
        self.n_neighbors = n_neighbors
        self.prob_weight = prob_weight
        self.lr_C = lr_C
        self.p = p

    def _augment(self, X):
        X = np.asarray(X, dtype=float)
        prob = self.lr_.predict_proba(X)[:, [1]]
        scale = X.shape[1] ** (-1.0 / self.p)
        return np.hstack([X * scale, self.prob_weight * prob])

    def fit(self, X, y):
        self.lr_ = LogisticRegression(C=self.lr_C, max_iter=1000)
        self.lr_.fit(X, y)
        self.knn_ = KNeighborsClassifier(n_neighbors=self.n_neighbors, p=self.p,
                                         n_jobs=-1)
        self.knn_.fit(self._augment(X), y)
        self.classes_ = self.knn_.classes_
        return self

    def predict(self, X):
        return self.knn_.predict(self._augment(X))

    def predict_proba(self, X):
        return self.knn_.predict_proba(self._augment(X))


# ---------------------------------------------------------------------------
# Week 5 / Week 6: logistic regression trained with mini-batch SGD, from scratch
# ---------------------------------------------------------------------------
class LogisticRegressionSGD(BaseEstimator, ClassifierMixin):
    """Binary logistic regression minimising log loss with mini-batch SGD.

        z = Xw + b,  p = sigmoid(z)
        loss = -mean( y log p + (1-y) log(1-p) ) + (l2/2)||w||^2
        grad_w = X^T (p - y) / m + l2 * w,  grad_b = mean(p - y)

    `loss_history_` stores the training log loss after each epoch so the
    error-vs-iteration curve from the Week-6 exercises can be plotted.
    """

    def __init__(self, learning_rate=0.05, epochs=30, batch_size=256, l2=0.0,
                 random_state=config.RANDOM_STATE):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.random_state = random_state

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def _log_loss(self, X, y):
        p = np.clip(self._sigmoid(X @ self.coef_ + self.intercept_), 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.random_state)
        n, d = X.shape
        self.coef_ = np.zeros(d)
        self.intercept_ = 0.0
        self.loss_history_ = []
        for _ in range(self.epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                err = self._sigmoid(X[idx] @ self.coef_ + self.intercept_) - y[idx]
                self.coef_ -= self.learning_rate * (X[idx].T @ err / len(idx)
                                                    + self.l2 * self.coef_)
                self.intercept_ -= self.learning_rate * err.mean()
            self.loss_history_.append(self._log_loss(X, y))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p1 = self._sigmoid(np.asarray(X, dtype=float) @ self.coef_ + self.intercept_)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# SVC with a training-size cap (see config.SVM_MAX_TRAIN)
# ---------------------------------------------------------------------------
class SubsampledSVC(BaseEstimator, ClassifierMixin):
    def __init__(self, kernel="rbf", C=1.0, gamma="scale", tol=1e-3,
                 class_weight=None, max_train=config.SVM_MAX_TRAIN,
                 random_state=config.RANDOM_STATE):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.class_weight = class_weight
        self.max_train = max_train
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.n_train_used_ = len(y)
        if self.max_train and len(y) > self.max_train:
            rng = np.random.default_rng(self.random_state)
            keep = []
            for cls in np.unique(y):           # stratified: keep class ratio
                idx = np.flatnonzero(y == cls)
                n_keep = max(1, int(round(len(idx) * self.max_train / len(y))))
                keep.append(rng.choice(idx, n_keep, replace=False))
            keep = np.concatenate(keep)
            X, y = X[keep], y[keep]
            self.n_train_used_ = len(y)
        self.svc_ = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma, tol=self.tol,
                        class_weight=self.class_weight, cache_size=500,
                        max_iter=config.SVM_MAX_ITER, random_state=self.random_state)
        self.svc_.fit(X, y)
        self.classes_ = self.svc_.classes_
        return self

    def predict(self, X):
        return self.svc_.predict(X)

    def decision_function(self, X):
        return self.svc_.decision_function(X)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_model(name, params=None, seed=config.RANDOM_STATE):
    """Instantiate model `name` with hyperparameters `params`
    (defaults to the baseline in config.BASELINE_PARAMS)."""
    p = dict(config.BASELINE_PARAMS[name])
    if params:
        p.update(params)

    if name == "logreg":
        # scikit-learn >= 1.8 deprecates `penalty` in favour of `l1_ratio`;
        # config keeps the readable 'l1'/'l2' names and we translate here.
        if _SKLEARN_GE_18 and "penalty" in p:
            p["l1_ratio"] = {"l1": 1.0, "l2": 0.0}[p.pop("penalty")]
        return LogisticRegression(random_state=seed, **p)
    if name == "knn":
        return KNeighborsClassifier(n_jobs=-1, **p)
    if name == "svm":
        return SubsampledSVC(random_state=seed, **p)
    if name == "xgboost":
        if XGBClassifier is not None:
            return XGBClassifier(eval_metric="logloss", tree_method="hist",
                                 n_jobs=-1, random_state=seed, **p)
        # Fallback keeps the same knobs where sklearn has an equivalent.
        return HistGradientBoostingClassifier(
            max_iter=p["n_estimators"], max_depth=p["max_depth"],
            learning_rate=p["learning_rate"],
            max_features=p["colsample_bytree"],
            min_samples_leaf=max(1, int(p["min_child_weight"]) * 4),
            class_weight={0: 1.0, 1: float(p["scale_pos_weight"])},
            early_stopping=False, random_state=seed)
    if name == "prob_knn":
        return ProbabilityKNN(**p)
    if name == "naive_bayes":
        return GaussianNB(**p)
    if name == "cart":
        return DecisionTreeClassifier(random_state=seed, **p)
    if name == "logreg_sgd":
        return LogisticRegressionSGD(random_state=seed, **p)
    raise ValueError(f"Unknown model '{name}'")


def positive_scores(model, X):
    """Continuous fraud score for ROC / PR curves."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            pass
    return model.decision_function(X)
