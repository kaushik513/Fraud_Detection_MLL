__version__ = "1.0.0"

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

METRIC_KEYS = ["precision", "recall", "f1", "accuracy", "specificity",
               "roc_auc", "pr_auc"]


def compute_metrics(y_true, y_pred, scores=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    if scores is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, scores)
        out["pr_auc"] = average_precision_score(y_true, scores)
    return {k: (float(v) if not isinstance(v, int) else v) for k, v in out.items()}


def aggregate(fold_metrics):
    agg = {}
    for k in METRIC_KEYS + ["fit_time", "predict_time"]:
        vals = np.array([m[k] for m in fold_metrics], dtype=float)
        agg[f"{k}_mean"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
        agg[f"{k}_std"] = float(np.nanstd(vals)) if np.isfinite(vals).any() else np.nan
    for k in ["tp", "fp", "tn", "fn"]:
        agg[k] = int(sum(m[k] for m in fold_metrics))   # pooled confusion matrix
    return agg
