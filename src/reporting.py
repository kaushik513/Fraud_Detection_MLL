__version__ = "1.0.0"

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

import config

BASE_C, TUNED_C = "#8C8C8C", "#2A9D8F"


def _save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_baseline_vs_tuned(final_df, out_dir, source="holdout"):
    paths = []
    for metric in ["f1", "recall", "precision"]:
        samplings = list(final_df["sampling"].unique())
        fig, axes = plt.subplots(1, len(samplings), figsize=(5.2 * len(samplings), 4.4),
                                 sharey=True)
        axes = np.atleast_1d(axes)
        for ax, s in zip(axes, samplings):
            sub = final_df[final_df.sampling == s]
            models = list(dict.fromkeys(sub["model"]))
            x = np.arange(len(models))
            col = f"{source}_{metric}"
            b = [sub[(sub.model == m) & (sub.variant == "baseline")][col].mean() for m in models]
            t = [sub[(sub.model == m) & (sub.variant == "tuned")][col].mean() for m in models]
            ax.bar(x - 0.2, b, 0.4, color=BASE_C, label="baseline")
            ax.bar(x + 0.2, t, 0.4, color=TUNED_C, label="tuned")
            ax.set_xticks(x)
            ax.set_xticklabels([config.MODEL_LABELS[m].replace(" (", "\n(") for m in models],
                               rotation=45, ha="right", fontsize=8)
            ax.set_title(s.upper())
            ax.set_ylim(0, 1)
        axes[0].set_ylabel(f"{metric.upper()} ({source})")
        axes[0].legend()
        fig.suptitle(f"Baseline vs tuned — fraud-class {metric.upper()} on {source} data")
        paths.append(_save(fig, out_dir, f"compare_{metric}_{source}.png"))
    return paths


def plot_times(final_df, out_dir):
    df = final_df.copy()
    df["label"] = df["model"].map(config.MODEL_LABELS) + " / " + df["sampling"]
    df["total_time"] = df["cv_fit_time_mean"] + df["cv_predict_time_mean"]
    piv = df.pivot_table(index="label", columns="variant", values="total_time")
    piv = piv.sort_values(piv.columns[0])
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(piv) + 1.5))
    y = np.arange(len(piv))
    if "baseline" in piv:
        ax.barh(y - 0.2, piv["baseline"], 0.4, color=BASE_C, label="baseline")
    if "tuned" in piv:
        ax.barh(y + 0.2, piv["tuned"], 0.4, color=TUNED_C, label="tuned")
    ax.set_yticks(y); ax.set_yticklabels(piv.index, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Mean fit + predict time per CV fold (s, log scale)")
    ax.set_title("Computational cost (training + scoring one test fold)"); ax.legend()
    return _save(fig, out_dir, "compare_training_time.png")


def plot_sensitivity(sens_rows, model, sampling, out_dir):
    if not sens_rows:
        return None
    df = pd.DataFrame(sens_rows)
    param = df["param"].iloc[0]
    labels = [str(v) for v in df["value"]]
    pos = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, c in [("f1", "#E76F51"), ("recall", "#2A9D8F"), ("precision", "#264653")]:
        ax.errorbar(pos, df[f"{m}_mean"], yerr=df[f"{m}_std"], marker="o", capsize=3,
                    label=m, color=c)
    base_v = config.BASELINE_PARAMS[model][param]
    if base_v in list(df["value"]):
        ax.axvline(list(df["value"]).index(base_v), ls="--", c="grey", lw=1,
                   label="baseline value")
    ax.set_xticks(pos); ax.set_xticklabels(labels)
    ax.set_xlabel(param); ax.set_ylabel("5-fold CV score"); ax.set_ylim(0, 1)
    ax.set_title(f"{config.MODEL_LABELS[model]} — sensitivity to {param} ({sampling})")
    ax.legend(fontsize=8)
    return _save(fig, out_dir, f"sensitivity_{model}_{sampling}.png")


def plot_confusion_grid(holdout_runs, out_dir, sampling):
    runs = [r for r in holdout_runs if r[1] == "tuned"]
    if not runs:
        return None
    n = len(runs)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (model, _, m) in zip(axes, runs):
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred 0", "pred 1"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["true 0", "true 1"])
        ax.set_title(config.MODEL_LABELS[model], fontsize=9)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"Hold-out confusion matrices — tuned models ({sampling})")
    return _save(fig, out_dir, f"confusion_tuned_{sampling}.png")


def plot_curves(holdout_runs, out_dir, sampling):
    runs = [r for r in holdout_runs if r[1] == "tuned"]
    if not runs:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for model, _, m in runs:
        p, r, _ = precision_recall_curve(m["y_true"], m["scores"])
        fpr, tpr, _ = roc_curve(m["y_true"], m["scores"])
        lab = config.MODEL_LABELS[model]
        axes[0].plot(r, p, lw=1.4, label=f"{lab} (AP={m['pr_auc']:.3f})")
        axes[1].plot(fpr, tpr, lw=1.4, label=f"{lab} (AUC={m['roc_auc']:.3f})")
    base_rate = float(np.mean(runs[0][2]["y_true"]))
    axes[0].axhline(base_rate, ls="--", c="grey", lw=1, label=f"chance ({base_rate:.3f})")
    axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision–Recall (hold-out)")
    axes[1].plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    axes[1].set_xlabel("False positive rate"); axes[1].set_ylabel("True positive rate")
    axes[1].set_title("ROC (hold-out)")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.suptitle(f"Tuned models — {sampling}")
    return _save(fig, out_dir, f"curves_tuned_{sampling}.png")


def plot_sgd_loss(model, out_dir, sampling):
    hist = getattr(model, "loss_history_", None)
    if not hist:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(hist) + 1), hist, "o-", ms=3, c="#E76F51")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Training log loss (binary cross-entropy)")
    ax.set_title(f"From-scratch SGD logistic regression — {sampling}")
    return _save(fig, out_dir, f"sgd_logloss_{sampling}.png")


def write_summary(final_df, info, settings, out_path):
    lines = [
        "# Experiment summary",
        "",
        f"* Data: {info['rows']:,} rows, {info['features']} features, "
        f"{info['fraud']:,} fraud ({info['fraud_rate']:.2%})"
        + ("  **— SYNTHETIC DEMO DATA, do not report**" if settings.get("synthetic") else ""),
        f"* Protocol: `{settings['protocol']}`, {config.CV_FOLDS}-fold stratified CV on the "
        f"development split, final scores on a {int(config.HOLDOUT_FRACTION * 100)} % "
        "stratified hold-out set (never resampled).",
        f"* Tuning: random search, {settings['n_iter']} configurations per model, "
        f"selection metric = CV {config.SELECTION_METRIC.upper()} (fraud class).",
        f"* XGBoost backend: {settings['xgb_backend']}",
        "",
        "## Hold-out results (fraud class)",
        "",
        "| Sampling | Model | Variant | Precision | Recall | F1 | Accuracy | PR-AUC | CV F1 (mean ± sd) | Fit + predict / fold (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in final_df.iterrows():
        lines.append(
            f"| {r.sampling} | {config.MODEL_LABELS[r.model]} | {r.variant} | "
            f"{r.holdout_precision:.3f} | {r.holdout_recall:.3f} | {r.holdout_f1:.3f} | "
            f"{r.holdout_accuracy:.3f} | {r.holdout_pr_auc:.3f} | "
            f"{r.cv_f1_mean:.3f} ± {r.cv_f1_std:.3f} | "
            f"{r.cv_fit_time_mean + r.cv_predict_time_mean:.3f} |")

    lines += ["", "## Effect of tuning (hold-out F1, tuned − baseline)", "",
              "| Sampling | Model | ΔF1 | ΔRecall | ΔPrecision | Best hyperparameters |",
              "|---|---|---|---|---|---|"]
    for (s, m), g in final_df.groupby(["sampling", "model"], sort=False):
        if set(g.variant) >= {"baseline", "tuned"}:
            b = g[g.variant == "baseline"].iloc[0]
            t = g[g.variant == "tuned"].iloc[0]
            best = ("baseline retained (no candidate beat it on CV F1)"
                    if t.get("params_same_as_baseline", False) else f"`{t.params}`")
            lines.append(f"| {s} | {config.MODEL_LABELS[m]} | {t.holdout_f1 - b.holdout_f1:+.3f} | "
                         f"{t.holdout_recall - b.holdout_recall:+.3f} | "
                         f"{t.holdout_precision - b.holdout_precision:+.3f} | {best} |")
    lines += ["", "Notes:",
              "* Accuracy stays high for almost every model because ~98 % of rows are "
              "legitimate; compare models on F1, recall, precision and PR-AUC.",
              "* A 'tuned' row can score below its baseline on the hold-out set: the "
              "configuration was chosen on CV folds, and small fraud counts make F1 noisy.",
              ""]
    if settings["protocol"] == "paper":
        lines.insert(-1, "* **Paper protocol:** CV scores were computed on data that was "
                         "oversampled *before* splitting, so synthetic copies of test-fold fraud "
                         "cases were in training. The gap between the CV F1 column and the "
                         "hold-out F1 column measures that leakage.")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return out_path
