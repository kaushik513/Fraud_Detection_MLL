# Credit-Card Fraud Detection — Reproduction & Hyperparameter Optimisation

**Version 1.0.0** (2026-10-10) · Changelog: 1.0.0 — initial release.

ML Lab mini-project (Kaushik Nanduru, Yashita Kotturi, Soham Saxena). Reproduces the
workflow of Karunya et al. (2025) — EDA → one-hot encoding + scaling → imbalanced /
SMOTE / ADASYN data → Logistic Regression, kNN, SVM, XGBoost and the proposed
probability-based kNN → 5-fold CV — then varies the models' hyperparameters and compares
the published-style baseline with the tuned configurations.

Full documentation (methods, papers, lab-manual concepts with locations, concepts from
outside the manuals): **[PROJECT_NOTES.md](PROJECT_NOTES.md)**.

## Setup

```bash
# Windows
setup_env.bat
# macOS / Linux
bash setup_env.sh
```
(or `pip install -r requirements.txt` in any Python 3.10+ environment). Put the dataset
CSV in `data/` (see `data/README.md`).

## Run

**GUI** — `python gui.py`
1. *Data*: browse to the CSV (or pick the synthetic demo) → **Load data**
2. *Exploratory analysis*: **Run EDA** and click through the figures
3. *Experiments*: tick models / data settings, choose the protocol and search budget → **Run study**
4. *Results*: hold-out comparison table + every plot

**Command line**

```bash
python main.py eda --data data/credit_card.csv
python main.py run --data data/credit_card.csv                 # 5 paper models, 3 settings
python main.py run --data data/credit_card.csv --protocol paper  # leaky, paper-style CV
python main.py all --data data/credit_card.csv --models logreg knn svm xgboost prob_knn \
                   naive_bayes cart logreg_sgd --n-iter 20
python main.py all --synthetic --n-iter 3 --no-sensitivity     # 2-minute smoke test
python -m pytest -q                                            # sanity tests
```

## Outputs (`results/`)

| File | Contents |
|---|---|
| `summary.md` | Hold-out table + tuned-minus-baseline deltas, ready for the Results section |
| `final_comparison.csv` | Baseline vs tuned per model × setting: hold-out and CV metrics, timings |
| `cv_all_configurations.csv` | Every configuration tried, with 5-fold mean ± sd |
| `best_params.json` | Chosen hyperparameters per model × setting |
| `sensitivity_<model>_<setting>.csv` | One-at-a-time sweeps |
| `run_config.json` | Exact settings of the run (for reproducibility) |
| `plots/` | Baseline-vs-tuned bars (F1/recall/precision), training time, sensitivity curves, confusion matrices, PR/ROC curves, SGD loss curve |
| `eda/` | Seven EDA figures + descriptive statistics, correlation and outlier tables |

## Before you report numbers

* `config.py → BASELINE_PARAMS` holds the reproduction settings. Check them against the
  paper's tables (the full text could not be retrieved while this was written).
* Never report results produced with `--synthetic`; that data only exercises the pipeline.
