"""
experiment.py — cross-validation and hold-out evaluation under a fixed protocol.

Version: 1.0.0
Changelog:
  1.0.0 (2026-10-10) — prepared (cached) CV folds per sampling setting so every
                       configuration is scored on identical data; "strict" and
                       "paper" protocols; hold-out evaluation; fit/predict timing.

Key design choice — preventing data leakage
-------------------------------------------
In the "strict" protocol each fold does:
    fit scaler + one-hot encoder on the training fold   (Week 3 / Week 5 practice)
    oversample the training fold only (SMOTE / ADASYN)
    train the model, then test on the untouched fold (real ~1.7 % fraud rate)
Oversampling before splitting lets synthetic copies of a test-fold fraud case
leak into training, inflating recall/precision. The "paper" protocol does
exactly that, on purpose, so the reproduction can be compared with published
numbers and the size of the leakage effect can be reported.
"""

__version__ = "1.0.0"

import time

import numpy as np
from sklearn.model_selection import StratifiedKFold

import config
from src.metrics import aggregate, compute_metrics
from src.models import make_model, positive_scores
from src.preprocessing import build_preprocessor
from src.resampling import get_sampler


class Fold:
    __slots__ = ("X_train", "y_train", "X_test", "y_test")

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test


def prepare_folds(X, y, sampling, protocol=config.DEFAULT_PROTOCOL,
                  n_folds=config.CV_FOLDS, seed=config.RANDOM_STATE,
                  k=config.SMOTE_K):
    """Build the K preprocessed (and, if requested, resampled) folds once.
    Every model / hyperparameter configuration is then evaluated on these
    same arrays, so differences come from the hyperparameters alone."""
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []

    if protocol == "paper":
        prep = build_preprocessor(X).fit(X)
        Xt = prep.transform(X)
        Xr, yr = get_sampler(sampling, seed, k).fit_resample(Xt, y)
        for tr, te in skf.split(Xr, yr):
            folds.append(Fold(Xr[tr], yr[tr], Xr[te], yr[te]))
        return folds

    if protocol != "strict":
        raise ValueError("protocol must be 'strict' or 'paper'")
    for i, (tr, te) in enumerate(skf.split(X, y)):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        prep = build_preprocessor(X_tr).fit(X_tr)
        Xtr_t, Xte_t = prep.transform(X_tr), prep.transform(X_te)
        Xtr_r, ytr_r = get_sampler(sampling, seed + i, k).fit_resample(Xtr_t, y[tr])
        folds.append(Fold(Xtr_r, ytr_r, Xte_t, y[te]))
    return folds


def evaluate_on_folds(folds, model_name, params=None, seed=config.RANDOM_STATE,
                      keep_predictions=False):
    """Train/test `model_name` with `params` on every fold.
    Returns aggregated metrics (+ pooled out-of-fold scores if requested)."""
    per_fold, all_y, all_s, all_p = [], [], [], []
    for f in folds:
        model = make_model(model_name, params, seed)
        t0 = time.perf_counter()
        model.fit(f.X_train, f.y_train)
        t1 = time.perf_counter()
        pred = model.predict(f.X_test)
        scores = positive_scores(model, f.X_test)
        t2 = time.perf_counter()
        m = compute_metrics(f.y_test, pred, scores)
        m["fit_time"], m["predict_time"] = t1 - t0, t2 - t1
        per_fold.append(m)
        if keep_predictions:
            all_y.append(f.y_test); all_s.append(scores); all_p.append(pred)
    out = aggregate(per_fold)
    out["n_train_per_fold"] = int(np.mean([len(f.y_train) for f in folds]))
    if keep_predictions:
        out["oof_y"] = np.concatenate(all_y)
        out["oof_scores"] = np.concatenate(all_s)
        out["oof_pred"] = np.concatenate(all_p)
    return out


def evaluate_holdout(X_dev, y_dev, X_test, y_test, model_name, params, sampling,
                     seed=config.RANDOM_STATE, k=config.SMOTE_K):
    """Final, unbiased check: fit preprocessing + resampling + model on the whole
    development set, score once on the never-touched hold-out set."""
    prep = build_preprocessor(X_dev).fit(X_dev)
    Xd, Xt = prep.transform(X_dev), prep.transform(X_test)
    Xd, yd = get_sampler(sampling, seed, k).fit_resample(Xd, np.asarray(y_dev))
    model = make_model(model_name, params, seed)
    t0 = time.perf_counter()
    model.fit(Xd, yd)
    t1 = time.perf_counter()
    pred = model.predict(Xt)
    scores = positive_scores(model, Xt)
    t2 = time.perf_counter()
    m = compute_metrics(np.asarray(y_test), pred, scores)
    m["fit_time"], m["predict_time"] = t1 - t0, t2 - t1
    m["y_true"], m["scores"], m["pred"] = np.asarray(y_test), scores, pred
    m["model"] = model
    return m
