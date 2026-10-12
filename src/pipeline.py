"""
pipeline.py — end-to-end study: split → (per sampling) CV folds → baseline +
random search + sensitivity per model → hold-out comparison → plots + report.

Version: 1.0.0
Changelog:
  1.0.0 (2026-10-10) — first full pipeline; shared by main.py (CLI) and gui.py.
"""

__version__ = "1.0.0"

import json
import os
import time

import pandas as pd

import config
from src import reporting
from src.data_loader import holdout_split
from src.experiment import evaluate_holdout, prepare_folds
from src.metrics import METRIC_KEYS
from src.models import XGB_BACKEND
from src.tuning import sensitivity, tune


def _jsonable(params):
    return json.dumps(params, sort_keys=True, default=str)


def run_study(X, y, info, models=None, samplings=None, protocol=config.DEFAULT_PROTOCOL,
              n_iter=config.DEFAULT_N_ITER, do_sensitivity=True,
              out_dir=config.RESULTS_DIR, synthetic=False, log=print,
              should_stop=lambda: False, progress=lambda done, total: None):
    models = models or config.PAPER_MODELS
    samplings = samplings or config.SAMPLINGS
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    settings = {"models": models, "samplings": samplings, "protocol": protocol,
                "n_iter": n_iter, "sensitivity": do_sensitivity, "synthetic": synthetic,
                "xgb_backend": XGB_BACKEND, "cv_folds": config.CV_FOLDS,
                "holdout_fraction": config.HOLDOUT_FRACTION,
                "svm_max_train": config.SVM_MAX_TRAIN, "seed": config.RANDOM_STATE,
                "baseline_params": {m: config.BASELINE_PARAMS[m] for m in models},
                "search_spaces": {m: config.SEARCH_SPACES[m] for m in models},
                "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config_version": config.__version__}
    with open(os.path.join(out_dir, "run_config.json"), "w") as fh:
        json.dump(settings, fh, indent=2, default=str)

    X_dev, X_test, y_dev, y_test = holdout_split(X, y)
    log(f"Development set: {len(y_dev):,} rows ({y_dev.sum()} fraud) | "
        f"hold-out: {len(y_test):,} rows ({y_test.sum()} fraud)")
    log(f"XGBoost backend: {XGB_BACKEND}")

    cv_rows, final_rows, best_params, plot_paths = [], [], {}, []
    t_start = time.perf_counter()
    total_steps, done_steps = len(samplings) * len(models), 0
    progress(0, total_steps)

    for sampling in samplings:
        if should_stop():
            break
        log(f"\n=== Sampling: {sampling.upper()} (protocol: {protocol}) ===")
        folds = prepare_folds(X_dev, y_dev, sampling, protocol)
        log(f"  folds ready: ~{len(folds[0].y_train):,} training rows per fold, "
            f"{folds[0].y_train.mean():.1%} fraud after resampling")
        holdout_runs = []

        for model in models:
            if should_stop():
                break
            log(f"  [{config.MODEL_LABELS[model]}]  (step {done_steps + 1}/{total_steps}, "
                f"{time.perf_counter() - t_start:.0f} s elapsed)")
            records, best = tune(folds, model, n_iter, log=log, should_stop=should_stop)
            for r in records:
                cv_rows.append({"sampling": sampling, "model": model, "variant": r["variant"],
                                "is_best": r is best, "params": _jsonable(r["params"]),
                                **{k: v for k, v in r.items()
                                   if k.endswith("_mean") or k.endswith("_std")
                                   or k in ("tp", "fp", "tn", "fn", "n_train_per_fold")}})
            best_params.setdefault(sampling, {})[model] = best["params"]
            log(f"    best: {best['variant']}  CV F1={best['f1_mean']:.4f}")

            if do_sensitivity and not should_stop():
                sens = sensitivity(folds, model, log=log, should_stop=should_stop)
                if sens:
                    pd.DataFrame(sens).to_csv(
                        os.path.join(out_dir, f"sensitivity_{model}_{sampling}.csv"), index=False)
                    p = reporting.plot_sensitivity(sens, model, sampling, plots_dir)
                    if p:
                        plot_paths.append(p)

            # Hold-out comparison: baseline vs best configuration.
            base_rec = records[0]
            for variant, rec in [("baseline", base_rec), ("tuned", best)]:
                hm = evaluate_holdout(X_dev, y_dev, X_test, y_test, model,
                                      rec["params"], sampling)
                holdout_runs.append((model, variant, hm))
                final_rows.append({
                    "sampling": sampling, "model": model, "variant": variant,
                    "params": _jsonable(rec["params"]),
                    **{f"holdout_{k}": hm[k] for k in METRIC_KEYS + ["fit_time", "predict_time"]},
                    **{f"holdout_{k}": hm[k] for k in ("tp", "fp", "tn", "fn")},
                    **{f"cv_{k}_mean": rec[f"{k}_mean"]
                       for k in METRIC_KEYS + ["fit_time", "predict_time"]},
                    "params_same_as_baseline": rec["params"] == base_rec["params"],
                    **{f"cv_{k}_std": rec[f"{k}_std"] for k in METRIC_KEYS},
                })
                if model == "logreg_sgd" and variant == "tuned":
                    p = reporting.plot_sgd_loss(hm["model"], plots_dir, sampling)
                    if p:
                        plot_paths.append(p)
            b, t = final_rows[-2], final_rows[-1]
            log(f"    hold-out F1: baseline {b['holdout_f1']:.4f} → tuned {t['holdout_f1']:.4f}")
            done_steps += 1
            progress(done_steps, total_steps)

        for fn in (reporting.plot_confusion_grid, reporting.plot_curves):
            p = fn(holdout_runs, plots_dir, sampling)
            if p:
                plot_paths.append(p)

    cv_df = pd.DataFrame(cv_rows)
    final_df = pd.DataFrame(final_rows)
    cv_df.to_csv(os.path.join(out_dir, "cv_all_configurations.csv"), index=False)
    final_df.to_csv(os.path.join(out_dir, "final_comparison.csv"), index=False)
    with open(os.path.join(out_dir, "best_params.json"), "w") as fh:
        json.dump(best_params, fh, indent=2, default=str)

    if not final_df.empty:
        plot_paths += reporting.plot_baseline_vs_tuned(final_df, plots_dir, "holdout")
        plot_paths.append(reporting.plot_times(final_df, plots_dir))
        reporting.write_summary(final_df, info, settings, os.path.join(out_dir, "summary.md"))

    log(f"\nDone in {time.perf_counter() - t_start:.1f} s — results in {os.path.abspath(out_dir)}")
    return final_df, cv_df, plot_paths
