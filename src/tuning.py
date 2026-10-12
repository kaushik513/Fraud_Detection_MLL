"""
tuning.py — systematic hyperparameter variation.

Version: 1.0.0
Changelog:
  1.0.0 (2026-10-10) — random search over config.SEARCH_SPACES (no duplicate
                       configurations, invalid combos skipped), best-config
                       selection by mean CV F1, one-at-a-time sensitivity sweeps.

Two complementary views of "does changing the hyperparameters help?":
  * Random search  — explores combinations, picks the best by CV F1 (fraud class).
  * Sensitivity    — changes ONE hyperparameter around the baseline and records
                     how F1/recall/precision respond (like the Week-6 exercise of
                     sweeping beta and plotting MSE).
Model selection uses only the development folds; the hold-out set is used once,
afterwards, so the "tuned" score is not optimistically biased.
"""

__version__ = "1.0.0"

import itertools
import json

import numpy as np

import config
from src.experiment import evaluate_on_folds


def _is_valid(model_name, params):
    if model_name == "svm" and params.get("kernel") == "linear" \
            and params.get("gamma") not in (None, "scale"):
        return False      # gamma is ignored by the linear kernel → duplicate config
    return True


def sample_configs(model_name, n_iter, seed=config.RANDOM_STATE):
    """Draw up to n_iter distinct configurations from the search space.
    If the whole grid is smaller than n_iter, the full grid is returned."""
    space = config.SEARCH_SPACES[model_name]
    keys = list(space)
    grid_size = int(np.prod([len(space[k]) for k in keys]))
    baseline = config.BASELINE_PARAMS[model_name]

    if grid_size <= n_iter:
        candidates = [dict(zip(keys, vals)) for vals in itertools.product(*space.values())]
    else:
        rng = np.random.default_rng(seed)
        seen, candidates, attempts = set(), [], 0
        while len(candidates) < n_iter and attempts < 50 * n_iter:
            attempts += 1
            cfg = {k: space[k][rng.integers(len(space[k]))] for k in keys}
            key = json.dumps(cfg, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(cfg)

    out = []
    for cfg in candidates:
        full = dict(baseline)
        full.update(cfg)
        if _is_valid(model_name, full) and full != baseline:
            out.append(full)
    return out


def tune(folds, model_name, n_iter=config.DEFAULT_N_ITER, metric=config.SELECTION_METRIC,
         log=print, should_stop=lambda: False):
    """Evaluate the baseline plus n_iter candidate configurations on `folds`.
    Returns (records, best_record). Each record: {'params', 'variant', metrics...}."""
    records = []
    base = evaluate_on_folds(folds, model_name, config.BASELINE_PARAMS[model_name])
    records.append({"variant": "baseline",
                    "params": dict(config.BASELINE_PARAMS[model_name]), **base})
    log(f"    baseline  {metric}={base[metric + '_mean']:.4f}")

    for i, cfg in enumerate(sample_configs(model_name, n_iter), 1):
        if should_stop():
            log("    stop requested — ending search early")
            break
        try:
            res = evaluate_on_folds(folds, model_name, cfg)
        except Exception as exc:                       # e.g. a solver/penalty clash
            log(f"    config {i} skipped ({type(exc).__name__}: {exc})")
            continue
        records.append({"variant": "candidate", "params": cfg, **res})
        log(f"    config {i:>2}  {metric}={res[metric + '_mean']:.4f}  {_short(cfg)}")

    best = max(records, key=lambda r: (np.nan_to_num(r[metric + "_mean"], nan=-1),
                                       np.nan_to_num(r["pr_auc_mean"], nan=-1)))
    return records, best


def sensitivity(folds, model_name, log=print, should_stop=lambda: False):
    """One-at-a-time sweep defined in config.SENSITIVITY_SWEEPS."""
    if model_name not in config.SENSITIVITY_SWEEPS:
        return None
    param, values = config.SENSITIVITY_SWEEPS[model_name]
    rows = []
    for v in values:
        if should_stop():
            break
        cfg = dict(config.BASELINE_PARAMS[model_name])
        cfg[param] = v
        res = evaluate_on_folds(folds, model_name, cfg)
        rows.append({"param": param, "value": v, **{k: res[k] for k in res
                                                     if k.endswith("_mean") or k.endswith("_std")}})
        log(f"    {param}={v}: f1={res['f1_mean']:.4f} recall={res['recall_mean']:.4f}")
    return rows


def _short(cfg):
    return ", ".join(f"{k}={v}" for k, v in cfg.items())
