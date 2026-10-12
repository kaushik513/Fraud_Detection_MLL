"""
test_core.py — quick sanity checks (run:  python -m pytest -q  or  python tests/test_core.py)

Version: 1.0.0
Changelog:
  1.0.0 (2026-10-10) — SMOTE/ADASYN balance + interpolation checks, prob-kNN
                       reduces to kNN at lambda=0, SGD-LR agrees with sklearn,
                       strict protocol keeps real fraud rate in test folds,
                       every model fits and predicts.
"""

__version__ = "1.0.0"

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from src.data_loader import make_synthetic, prepare_frame  # noqa: E402
from src.experiment import evaluate_on_folds, prepare_folds  # noqa: E402
from src.models import (LogisticRegressionSGD, ProbabilityKNN,  # noqa: E402
                        make_model)
from src.resampling import ADASYN, SMOTE  # noqa: E402


def _toy(n=2000, rate=0.05, d=6, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rate).astype(int)
    X = rng.normal(size=(n, d)) + 1.5 * y[:, None]
    return X, y


def test_smote_balances_and_interpolates():
    X, y = _toy()
    Xr, yr = SMOTE(5, random_state=1).fit_resample(X, y)
    assert (yr == 1).sum() == (yr == 0).sum()
    assert np.allclose(Xr[:len(X)], X)                    # originals untouched
    syn = Xr[len(X):]
    lo, hi = X[y == 1].min(0), X[y == 1].max(0)
    assert np.all(syn >= lo - 1e-9) and np.all(syn <= hi + 1e-9)  # inside minority hull box


def test_adasyn_roughly_balances():
    X, y = _toy()
    Xr, yr = ADASYN(5, random_state=1).fit_resample(X, y)
    n0, n1 = (yr == 0).sum(), (yr == 1).sum()
    assert abs(n1 - n0) / n0 < 0.02


def test_against_imblearn_if_available():
    try:
        from imblearn.over_sampling import SMOTE as ImbSMOTE
    except ImportError:
        return
    X, y = _toy()
    _, y1 = SMOTE(5, random_state=0).fit_resample(X, y)
    _, y2 = ImbSMOTE(k_neighbors=5, random_state=0).fit_resample(X, y)
    assert np.bincount(y1).tolist() == np.bincount(y2).tolist()


def test_prob_knn_lambda_zero_equals_knn():
    X, y = _toy(800)
    a = ProbabilityKNN(n_neighbors=5, prob_weight=0.0).fit(X, y).predict(X)
    b = make_model("knn", {"n_neighbors": 5}).fit(X, y).predict(X)
    assert (a == b).all()


def test_sgd_logreg_matches_sklearn():
    X, y = _toy(3000, rate=0.3)
    sgd = LogisticRegressionSGD(learning_rate=0.1, epochs=60, batch_size=64).fit(X, y)
    sk = make_model("logreg", {"C": 1e6}).fit(X, y)
    agree = (sgd.predict(X) == sk.predict(X)).mean()
    assert agree > 0.97
    assert sgd.loss_history_[-1] < sgd.loss_history_[0]


def test_strict_protocol_keeps_real_rate_in_test_folds():
    X, y, _ = prepare_frame(make_synthetic(4000))
    folds = prepare_folds(X, y, "smote", "strict", n_folds=3)
    for f in folds:
        assert abs(f.y_test.mean() - y.mean()) < 0.01   # test fold not resampled
        assert abs(f.y_train.mean() - 0.5) < 1e-9       # training fold balanced


def test_every_model_runs():
    X, y, _ = prepare_frame(make_synthetic(3000))
    folds = prepare_folds(X, y, "smote", "strict", n_folds=2)
    for m in config.ALL_MODELS:
        res = evaluate_on_folds(folds, m)
        assert 0.0 <= res["f1_mean"] <= 1.0, m


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
