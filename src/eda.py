"""
eda.py — exploratory data analysis: distributions, categorical relationships,
correlations, outliers, PCA view and a k-means clustering view.

Version: 1.0.0
Changelog:
  1.0.0 (2026-10-10) — class balance, histograms (sqrt(n) bin rule), fraud rate
                       per category, correlation heatmap, IQR / 3-sigma outlier
                       table + boxplots, PCA scatter + explained variance,
                       k-means elbow + fraud rate per cluster.

Lab-manual links:
  Week 2  Matplotlib bar / histogram / scatter plots; number of bins = sqrt(n)
  Week 3  Outlier identification with lower/upper limits (std-dev method)
  Week 3 (PCA additional) StandardScaler + PCA, 2-D PCA scatter, explained variance
  Week 10 K-means (k-means++ init), SSE / elbow method
"""

__version__ = "1.0.0"

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import config
from src.data_loader import categorical_columns, numeric_columns
from src.preprocessing import build_preprocessor

LEGIT_C, FRAUD_C = "#4C72B0", "#DD5144"


def _save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def run_eda(X, y, out_dir, log=print, sample_for_heavy=6000):
    os.makedirs(out_dir, exist_ok=True)
    y = np.asarray(y)
    df = X.copy()
    df["Target"] = y
    nums, cats = numeric_columns(X), categorical_columns(X)
    paths, summary = [], {}

    # 1. class balance --------------------------------------------------------
    counts = pd.Series(y).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Legitimate (0)", "Fraud (1)"], counts.values, color=[LEGIT_C, FRAUD_C])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v / len(y):.2%})", ha="center", va="bottom")
    ax.set_ylabel("Records"); ax.set_title("Class distribution")
    ax.set_ylim(0, counts.max() * 1.18)
    paths.append(_save(fig, out_dir, "01_class_distribution.png"))
    summary["class_counts"] = {int(k): int(v) for k, v in counts.items()}
    log("  class distribution done")

    # 2. numeric distributions by class (bins = sqrt(n), Week-2 rule) ---------
    bins = int(np.sqrt(len(df)))
    cols = 3
    rows = int(np.ceil(len(nums) / cols)) or 1
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.0 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, nums):
        for cls, colr, lab in [(0, LEGIT_C, "legit"), (1, FRAUD_C, "fraud")]:
            ax.hist(df.loc[df.Target == cls, c], bins=min(bins, 60), density=True,
                    alpha=0.55, color=colr, label=lab)
        ax.set_title(c, fontsize=9)
    for ax in axes[len(nums):]:
        ax.axis("off")
    axes[0].legend(fontsize=8)
    fig.suptitle("Numeric feature distributions by class (density)", y=1.0)
    paths.append(_save(fig, out_dir, "02_numeric_histograms.png"))
    log("  histograms done")

    # 3. categorical relationships: fraud rate per category -------------------
    if cats:
        rows = int(np.ceil(len(cats) / 2))
        fig, axes = plt.subplots(rows, 2, figsize=(12, 3.4 * rows))
        axes = np.atleast_1d(axes).ravel()
        overall = y.mean()
        cat_tables = {}
        for ax, c in zip(axes, cats):
            rate = df.groupby(c)["Target"].agg(["mean", "size"]).sort_values("mean")
            cat_tables[c] = rate.round(4).to_dict()
            ax.barh(rate.index.astype(str), rate["mean"] * 100, color=FRAUD_C, alpha=0.8)
            ax.axvline(overall * 100, color="k", ls="--", lw=1, label="overall")
            ax.set_xlabel("Fraud rate (%)"); ax.set_title(c, fontsize=10)
            ax.tick_params(axis="y", labelsize=8)
        for ax in axes[len(cats):]:
            ax.axis("off")
        axes[0].legend(fontsize=8)
        paths.append(_save(fig, out_dir, "03_categorical_fraud_rate.png"))
        summary["categorical_fraud_rates"] = cat_tables
    log("  categorical relationships done")

    # 4. correlation heatmap (numeric + target) -------------------------------
    corr = df[nums + ["Target"]].corr()
    fig, ax = plt.subplots(figsize=(0.55 * len(corr) + 3, 0.5 * len(corr) + 2.5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=60,
                                                       ha="right", fontsize=8)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Pearson correlation (numeric features + Target)")
    paths.append(_save(fig, out_dir, "04_correlation_heatmap.png"))
    corr.round(4).to_csv(os.path.join(out_dir, "correlation_matrix.csv"))
    summary["top_target_correlations"] = (corr["Target"].drop("Target").abs()
                                          .sort_values(ascending=False).head(8)
                                          .round(4).to_dict())
    log("  correlations done")

    # 5. outliers: IQR rule and 3-sigma rule ----------------------------------
    rows_out = []
    for c in nums:
        s = df[c]
        if s.nunique() <= 2:          # binary flags are not "outliers"
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        mu, sd = s.mean(), s.std()
        iqr_mask = (s < lo) | (s > hi)
        sig_mask = (s < mu - 3 * sd) | (s > mu + 3 * sd)
        rows_out.append({"feature": c, "iqr_outliers": int(iqr_mask.sum()),
                         "sigma3_outliers": int(sig_mask.sum()),
                         "fraud_rate_in_iqr_outliers":
                             float(df.loc[iqr_mask, "Target"].mean()) if iqr_mask.any() else np.nan})
    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(os.path.join(out_dir, "outlier_summary.csv"), index=False)
    cont = [r["feature"] for r in rows_out]
    if cont:
        fig, axes = plt.subplots(1, len(cont), figsize=(2.2 * len(cont) + 1, 4))
        axes = np.atleast_1d(axes)
        for ax, c in zip(axes, cont):
            ax.boxplot([df.loc[df.Target == 0, c], df.loc[df.Target == 1, c]],
                       tick_labels=["0", "1"], flierprops={"markersize": 2})
            ax.set_title(c, fontsize=8)
        fig.suptitle("Box plots by class (points = IQR outliers)")
        paths.append(_save(fig, out_dir, "05_boxplots_outliers.png"))
    summary["outliers"] = rows_out
    log("  outliers done")

    # Heavy steps run on a stratified-ish random sample to stay fast.
    rng = np.random.default_rng(config.RANDOM_STATE)
    idx = np.arange(len(df))
    if len(idx) > sample_for_heavy:
        fraud_idx = idx[y == 1]
        legit_idx = rng.choice(idx[y == 0], sample_for_heavy - len(fraud_idx), replace=False)
        idx = np.concatenate([legit_idx, fraud_idx])
    Xs, ys = X.iloc[idx], y[idx]
    Z = build_preprocessor(Xs).fit_transform(Xs)

    # 6. PCA -------------------------------------------------------------------
    pca = PCA().fit(Z)
    Z2 = pca.transform(Z)[:, :2]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].scatter(Z2[ys == 0, 0], Z2[ys == 0, 1], s=4, alpha=0.3, c=LEGIT_C, label="legit")
    axes[0].scatter(Z2[ys == 1, 0], Z2[ys == 1, 1], s=10, alpha=0.8, c=FRAUD_C, label="fraud")
    axes[0].set_xlabel("PC 1"); axes[0].set_ylabel("PC 2"); axes[0].legend()
    axes[0].set_title("2-D PCA projection (all fraud + sample of legit)")
    ev = pca.explained_variance_ratio_
    n_show = min(20, len(ev))
    axes[1].bar(range(1, n_show + 1), ev[:n_show], color=LEGIT_C, label="individual")
    axes[1].plot(range(1, n_show + 1), np.cumsum(ev)[:n_show], "o-", c=FRAUD_C,
                 label="cumulative")
    axes[1].axhline(0.9, ls="--", c="grey", lw=1)
    axes[1].set_xlabel("Principal component"); axes[1].set_ylabel("Explained variance ratio")
    axes[1].set_title("Scree plot"); axes[1].legend()
    paths.append(_save(fig, out_dir, "06_pca.png"))
    summary["pca_components_for_90pct_variance"] = int(np.searchsorted(np.cumsum(ev), 0.9) + 1)
    summary["encoded_feature_count"] = int(Z.shape[1])
    log("  PCA done")

    # 7. k-means: elbow + fraud concentration per cluster ----------------------
    ks, sse = list(range(1, 11)), []
    for k in ks:
        sse.append(KMeans(n_clusters=k, init="k-means++", n_init=5,
                          random_state=config.RANDOM_STATE).fit(Z).inertia_)
    # elbow = point furthest from the straight line joining the first and last
    # points of the (min-max normalised) k-vs-SSE curve
    kn = (np.array(ks) - ks[0]) / (ks[-1] - ks[0])
    sn = (np.array(sse) - min(sse)) / (max(sse) - min(sse) + 1e-12)
    dists = np.abs(sn - (1 - kn)) / np.sqrt(2)   # line from (0,1) to (1,0)
    k_best = ks[int(np.argmax(dists))]
    labels = KMeans(n_clusters=k_best, init="k-means++", n_init=10,
                    random_state=config.RANDOM_STATE).fit_predict(Z)
    cl = pd.DataFrame({"cluster": labels, "fraud": ys}).groupby("cluster")["fraud"] \
        .agg(["mean", "size"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(ks, sse, "o-", c=LEGIT_C)
    axes[0].plot(k_best, sse[k_best - 1], "o", ms=12, mfc="none", mec=FRAUD_C, mew=2)
    axes[0].set_xlabel("k"); axes[0].set_ylabel("SSE (inertia)")
    axes[0].set_title(f"K-means elbow (chosen k = {k_best})")
    axes[1].bar(cl.index.astype(str), cl["mean"] * 100, color=FRAUD_C)
    for i, (rate, size) in enumerate(zip(cl["mean"], cl["size"])):
        axes[1].text(i, rate * 100, f"n={size}", ha="center", va="bottom", fontsize=8)
    axes[1].set_xlabel("Cluster"); axes[1].set_ylabel("Fraud rate in sample (%)")
    axes[1].set_title("Fraud concentration per cluster")
    paths.append(_save(fig, out_dir, "07_kmeans_clusters.png"))
    summary["kmeans"] = {"elbow_k": k_best, "sse": [round(float(s), 1) for s in sse],
                         "cluster_fraud_rate": cl["mean"].round(4).to_dict(),
                         "cluster_size": cl["size"].to_dict(),
                         "note": "computed on a sample containing ALL fraud rows, so rates "
                                 "are higher than the population rate"}
    log("  k-means done")

    X.describe(include="all").T.to_csv(os.path.join(out_dir, "descriptive_statistics.csv"))
    with open(os.path.join(out_dir, "eda_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return paths, summary
