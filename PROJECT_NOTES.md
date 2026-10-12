# Project Notes — what was used, and where it came from

**Version 1.0.0** (2026-10-10)
Changelog: 1.0.0 — first version, written alongside code v1.0.0.

Project: *Reproduction and Hyperparameter Optimization of Machine Learning Models for
Credit Card Fraud Detection* — Kaushik Nanduru (240962636), Yashita Kotturi (240962578),
Soham Saxena (240962582).

This file records (1) what the code does and how the synopsis maps onto it, (2) the papers
it relies on, (3) every concept taken from the ML lab manuals, with its location, and
(4) the concepts that come from outside the lab manuals. Sections 3–5 can be lifted into
the report's *Methodology* and *Implementation Details* sections.

---

## 1. What was built

```
fraud_hpo/
├── config.py            all settings: baseline params, search spaces, sweeps, protocol
├── main.py              command-line entry point (eda / run / all)
├── gui.py               Tkinter GUI (Data · EDA · Experiments · Results tabs)
├── src/
│   ├── data_loader.py   CSV loading, cleaning, imputation, stratified hold-out, synthetic demo data
│   ├── preprocessing.py one-hot encoding + standard scaling (fitted on training data only)
│   ├── resampling.py    SMOTE and ADASYN written from scratch
│   ├── models.py        model factory, probability-based kNN, SGD logistic regression, capped SVC
│   ├── metrics.py       precision / recall / F1 / accuracy / specificity / ROC-AUC / PR-AUC
│   ├── experiment.py    cached CV folds, strict vs paper protocol, hold-out evaluation, timing
│   ├── tuning.py        random search + one-at-a-time sensitivity sweeps
│   ├── eda.py           exploratory analysis figures and tables
│   ├── reporting.py     comparison plots and the auto-generated summary.md
│   └── pipeline.py      orchestrates the full study (shared by CLI and GUI)
├── tests/test_core.py   sanity tests
└── data/                put the dataset CSV here
```

Every source file starts with a version number and a changelog so you can tell which
version to roll back to.

### 1.1 Synopsis objectives → code

| Synopsis item | Where it is implemented |
|---|---|
| EDA: distributions, categorical relationships, correlations, outliers | `src/eda.py` → `results/eda/01…07_*.png`, `outlier_summary.csv`, `correlation_matrix.csv` |
| One-hot encoding + standardisation | `src/preprocessing.py` (`ColumnTransformer` of `OneHotEncoder` + `StandardScaler`) |
| Imbalanced / SMOTE / ADASYN settings | `src/resampling.py`, selected per run (`config.SAMPLINGS`) |
| LR, kNN, SVM, XGBoost, probability-based kNN | `src/models.py → make_model()`, `ProbabilityKNN` |
| Reproduce reference hyperparameters (baseline) | `config.BASELINE_PARAMS` |
| Vary k, regularisation strength, tolerance, kernel settings, tree depth, number of estimators, learning rate | `config.SEARCH_SPACES` (random search) and `config.SENSITIVITY_SWEEPS` (one-at-a-time) |
| Five-fold CV; precision, recall, F1, accuracy, computational time | `src/experiment.py`, `src/metrics.py` (fit and predict time recorded per fold) |
| Compare baseline vs modified across balancing strategies | `src/pipeline.py` → `final_comparison.csv`, `summary.md`, `plots/compare_*.png` |
| Probability-based kNN vs XGBoost efficiency | `plots/compare_training_time.png`, `cv_fit_time_mean` / `holdout_predict_time` columns |

### 1.2 How one experiment runs

```
full dataset ──stratified 80/20──► development set ──► 5 stratified folds
                                   │                     for each fold:
                                   │                       fit one-hot + scaler on train fold
                                   │                       SMOTE / ADASYN on train fold only
                                   │                       fit model, score on untouched test fold
                                   │                     → baseline config + N random configs
                                   │                     → best config = highest mean CV F1 (fraud)
                                   │                     → sensitivity sweep around the baseline
                                   └──► hold-out set (20 %, never resampled, used once)
                                        baseline and best config each refit on the whole
                                        development set and scored here → final comparison
```

* **Same folds for every configuration.** Folds (including the resampled training data) are
  built once per data setting and reused, so differences between configurations come from
  the hyperparameters only.
* **Two protocols.** `strict` (default) resamples only training folds. `paper` resamples the
  whole dataset first and then runs CV on the balanced data — the usual source of very high
  published SMOTE/ADASYN scores, because synthetic copies of test-fold fraud cases end up
  in training. Running both lets the report reproduce the paper's style of numbers *and*
  quantify the leakage effect. The hold-out set is never resampled in either protocol.
* **Selection metric = F1 of the fraud class.** Accuracy is reported but never used for
  selection: predicting "legitimate" for everyone already scores ≈ 98 % accuracy here.
* **Hold-out set** is used once, after tuning, so the tuned score is not optimistically
  biased by having been chosen on the same data it is reported on.

### 1.3 Hyperparameters that are varied

| Model | Baseline (`config.BASELINE_PARAMS`) | Varied in random search | Sensitivity sweep |
|---|---|---|---|
| Logistic Regression | C=1, L2, lbfgs, tol=1e-4 | C 0.001–100, L1/L2, tol 1e-5–1e-2, class_weight | C |
| kNN | k=5, uniform, Euclidean | k 1–31, uniform/distance weighting, Manhattan/Euclidean | k |
| SVM | RBF, C=1, gamma='scale', tol=1e-3 | kernel linear/poly/rbf/sigmoid, C 0.1–100, gamma, tol, class_weight | C |
| XGBoost | 100 trees, depth 6, lr 0.3 | trees 100–400, depth 3–8, lr 0.03–0.3, subsample, colsample, min_child_weight, scale_pos_weight | max_depth |
| Probability-based kNN | k=5, λ=1, LR C=1 | k, λ (weight of LR probability), LR C, distance p | λ (0 = plain kNN) |
| Naive Bayes *(ext.)* | var_smoothing=1e-9 | var_smoothing | — |
| Decision tree *(ext.)* | Gini, unlimited depth | Gini/entropy, depth, min_samples_leaf, class_weight | max_depth |
| SGD logistic regression *(ext.)* | α=0.05, 30 epochs | α, epochs, batch size, L2 | — |

*(ext.)* = extension models drawn from the lab manuals; not part of the reference paper.
Add `--models naive_bayes cart logreg_sgd` (CLI) or tick them in the GUI to include them.

### 1.4 Assumptions you must check before reporting

1. **Baseline values.** The full text of [1] could not be retrieved while the code was written
   (the publisher blocks automated access), so `BASELINE_PARAMS` uses library defaults —
   the most likely settings for an unspecified configuration, and k = 5 is also the value the
   Week-8 manual recommends. If [1] reports other values, edit only that block.
2. **Probability-based kNN formula.** From the abstract and section headings of [1], the method
   uses logistic-regression probabilities inside kNN's similarity ("kNN with logistic
   regression-based distance", §3.2.5). Our implementation uses
   `d(x,z) = ( (1/F)·Σ|xᵢ−zᵢ|ᵖ + λᵖ·|P(x)−P(z)|ᵖ )^(1/p)`, with `P` the LR fraud probability.
   If §3.2.5 gives a different formula, `ProbabilityKNN._augment` in `src/models.py` is the
   only place to change. λ = 0 reproduces ordinary kNN (checked by a test).
3. **Dataset.** Use the exact Kaggle link from [1]'s data-availability statement (the
   synopsis describes 25,134 records, 19 features, 420 positives). The loader drops `ID`,
   removes duplicates and imputes missing values; the Data tab / CLI prints what it did.
   If duplicates are removed, the row count will differ slightly from the paper — say so.
4. **XGBoost backend.** With the `xgboost` package installed (it is in `requirements.txt`) the
   real `XGBClassifier` is used. Without it, the code falls back to scikit-learn's
   `HistGradientBoostingClassifier` and prints that in the log and `summary.md` — results
   from the fallback must not be labelled "XGBoost" in the report.
5. **SVM training cap.** SVC training time grows roughly with the square of the sample count,
   so SVM training folds are stratified-subsampled to `SVM_MAX_TRAIN` = 5,000 rows (same for
   baseline and tuned). Mention this in Implementation Details, or raise the cap if you have
   time to wait.

---

## 2. Referenced papers

### 2.1 Papers from the synopsis reference list and how the project uses them

| # | Paper (short) | Used for |
|---|---|---|
| [1] | Karunya R.V. et al., *Credit Card Fraud Data Analysis and Prediction Using ML Algorithms*, Security and Privacy, 2025 | **Reference study being reproduced**: workflow, model set, SMOTE/ADASYN comparison, probability-based kNN (§3.2.5), efficiency comparison with XGBoost |
| [2] | Ileberi, Sun, Wang, *Performance Evaluation of ML Methods for CCFD Using SMOTE and AdaBoost*, IEEE Access, 2021 | Motivates SMOTE as the standard oversampler for fraud data |
| [3] | Kaur & Gosain, *Issues and Challenges of Class Imbalance Problem in Classification*, IJIT, 2022 | Why accuracy misleads on imbalanced data → F1/recall/precision/PR-AUC used for selection |
| [4] | Rtayli & Enneya, *Enhanced CCFD based on SVM-RFE and hyper-parameters optimization*, JISA, 2020 | Precedent for tuning SVM hyperparameters in fraud detection |
| [5] | Cherif et al., *CCFD in the era of disruptive technologies: a systematic review*, JKSU-CIS, 2023 | Background / literature survey |
| [6] | Dang et al., *ML Based on Resampling Approaches and Deep RL for CCFD Systems*, Appl. Sci., 2021 | Resampling strategies compared across classifiers |
| [7] | Benchaji et al., *Enhanced CCFD based on attention mechanism and LSTM*, J. Big Data, 2021 | Literature survey (sequential models; not implemented) |
| [8] | Ileberi, Sun, Wang, *ML based CCFD using the GA algorithm for feature selection*, J. Big Data, 2022 | Literature survey (feature selection; future scope) |
| [9] | Dastidar et al., *NAG: neural feature aggregation framework for CCFD*, KAIS, 2022 | Literature survey (feature aggregation; future scope) |
| [10] | Alarfaj et al., *CCFD Using State-of-the-Art ML and DL Algorithms*, IEEE Access, 2022 | Literature survey; baseline ML vs DL context |
| [11] | Mienye & Sun, *A DL Ensemble With Data Resampling for CCFD*, IEEE Access, 2023 | Resampling + ensembles; context for XGBoost |
| [12] | Ileberi & Sun, *Advancing Model Performance with ADASYN and RFECV …*, IEEE Access, 2024 | Motivates ADASYN as the second balancing strategy |
| [13] | Salam et al., *Federated Learning Model for CCFD With Data Balancing Techniques*, NCA, 2024 | Literature survey (federated setting; future scope) |
| [14] | Chhabra et al., *A Voting Ensemble ML-Based CCFD Using Highly Imbalanced Data*, MTA, 2024 | Ensembles on highly imbalanced data; future scope |
| [15] | Habibpour et al., *Uncertainty-aware CCFD using deep learning*, EAAI, 2023 | Literature survey; probability outputs / thresholds |
| [16] | Mim et al., *A soft voting ensemble learning approach for CCFD*, Heliyon, 2024 | Literature survey; ensembles |
| [17] | Sulaiman et al., *CCFD Using Improved Deep Learning Models*, CMC, 2024 | Precedent that optimised configurations improve fraud detection |
| [18]–[22] | Huang et al. 2024; Tang & Liang 2024; Charizanos et al. 2024; Xie et al. 2023; Rehman Khalid et al. 2024 | In the synopsis reference list but **not cited in its text** — the guidelines require every reference to be cited, so cite them in the report's Literature Survey (undersampling, graph learning, online fuzzy detection, GAN augmentation, ensembles) |

### 2.2 Algorithm sources used in the code (outside the synopsis list)

These are the original descriptions of methods the code implements. Several are conference
papers or software papers, so they suit the Implementation Details section rather than the
journal-only reference list the guidelines ask for — check with your guide.

* N. V. Chawla, K. W. Bowyer, L. O. Hall, W. P. Kegelmeyer, "SMOTE: Synthetic Minority
  Over-sampling Technique," *Journal of Artificial Intelligence Research*, vol. 16,
  pp. 321–357, 2002. → `resampling.SMOTE`
* H. He, Y. Bai, E. A. Garcia, S. Li, "ADASYN: Adaptive synthetic sampling approach for
  imbalanced learning," *Proc. IEEE IJCNN*, pp. 1322–1328, 2008. → `resampling.ADASYN`
* T. Chen, C. Guestrin, "XGBoost: A Scalable Tree Boosting System," *Proc. ACM KDD*,
  pp. 785–794, 2016. → XGBoost model
* J. Bergstra, Y. Bengio, "Random Search for Hyper-Parameter Optimization," *Journal of
  Machine Learning Research*, vol. 13, pp. 281–305, 2012. → `tuning.sample_configs`
* S. Kaufman, S. Rosset, C. Perlich, O. Stitelman, "Leakage in data mining: Formulation,
  detection, and avoidance," *ACM Trans. Knowledge Discovery from Data*, vol. 6, no. 4, 2012.
  → the `strict` vs `paper` protocol discussion
* F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine
  Learning Research*, vol. 12, pp. 2825–2830, 2011. → library used throughout

---

## 3. Concepts used from the lab manuals (with location)

Locations are given as *manual file → section heading (question number)*. The PDFs' text
layer has no page numbers, so headings are used; they appear in the same words in each PDF.

### Week 01 — NumPy (`Week-01.pdf`)
| Concept | Location in manual | Used in |
|---|---|---|
| ndarray, shape/axis, element-wise arithmetic "without using for-loops" | *Learning Objectives*; *The Basics*; *Basic Operations* | All numeric code is vectorised: synthetic-point generation in `resampling.py` (`X_min[base] + gap * (X_min[partner] - X_min[base])`) |
| Matrix product with `@` | *Basic Operations* (A @ B) | `LogisticRegressionSGD.fit` (`X[idx].T @ err`) |
| Random number generator `np.random.default_rng` | *Basic Operations* (`rg = np.random.default_rng(1)`) | All random sampling (`resampling.py`, `tuning.py`, `data_loader.make_synthetic`) |

### Week 02 — Data analysis & visualisation (`Week-02-Data_Analysis_and_visualisation_with_Python-v2-aln (1).pdf`)
| Concept | Location | Used in |
|---|---|---|
| Pandas DataFrame, `read_csv`, `drop(axis='columns')`, `loc`/`iloc` | *Pandas*; *DataFrame*; *Load Files Into a DataFrame*; *Deleting columns* | `data_loader.load_csv / prepare_frame` |
| Data cleaning: empty cells, wrong data, duplicates | *Data Cleaning* | `prepare_frame`: `drop_duplicates`, imputation, binary-target check |
| Bar plot | *2. Bar plot* | Class distribution, fraud rate per category (`eda.py` figs 01, 03), comparison charts (`reporting.py`) |
| Histogram; number of bins = √n | *3. Hist plot* → *Additional way to determine the number of bins* | `eda.py` fig 02 (bins = √n, capped at 60 for readability) |
| Scatter plot | *4. Scatter plot* | PCA scatter (`eda.py` fig 06) |

### Week 03 — Data pre-processing & regression (`Week-03-Data-Preprocessing and Regression Techniques.pdf`)
| Concept | Location | Used in |
|---|---|---|
| Missing values: dropping, numerical imputation with the median, categorical imputation with the most frequent value | *Missing values* → *Dropping*, *Numerical imputation*, *Categorical imputation* | `prepare_frame` (median for numeric, `value_counts().idxmax()` for categorical — the manual's exact recipe) |
| Outliers via lower/upper limits from the standard deviation | *Outliers* | `eda.py` outlier table (3-σ rule, plus the IQR rule) |
| Overfitting and regularisation | *Overfitting* | Motivation for tuning C (LR/SVM), max_depth / min_child_weight (trees), k (kNN) |
| One-hot encoding of non-numeric data | *Data with no numerical values* | `preprocessing.build_preprocessor` (`OneHotEncoder`) |
| `train_test_split` with `random_state` and `stratify` | *How to split a Dataset into Train and Test Sets* (parameter 6, `stratify`) | `data_loader.holdout_split` (stratified 80/20 hold-out) |
| scikit-learn cross-validation | *SCIKIT LEARN* → *Features of Scikit learn* ("Cross validation") | `experiment.prepare_folds` (`StratifiedKFold`, 5 folds) |
| Cost function minimised by gradient descent | *Cost function* | Background for `LogisticRegressionSGD` |

### Week 03 additional — PCA (`PCA MACHINE LEARNING ALGORITHM-week03-additional.pdf`)
| Concept | Location | Used in |
|---|---|---|
| StandardScaler: μ = 0, σ = 1 before distance-based methods | *3. Perform Data Preprocessing in Python* | `preprocessing.py` (`StandardScaler`) — essential for kNN, SVM, SMOTE/ADASYN distances |
| PCA with `n_components`, 2-D PCA scatter | *4. Perform Dimension Reduction using PCA*; *6. Plot the 2D PCA Graph* | `eda.py` fig 06 (left) |
| Explained-variance bar plot / scree plot, keeping ≥ 90 % variance | *How Many Dimensions Should You Reduce Your Data with PCA?*; Exercises 1–2 | `eda.py` fig 06 (right); `eda_summary.json → pca_components_for_90pct_variance` |

### Week 04 — "parametric methods" file, content *Polynomial and Multiple Regression* (`Week-04-parametric methods.pdf`)
| Concept | Location | Used in |
|---|---|---|
| TP / FP / TN / FN and the confusion matrix | *Machine Learning Model evaluation metrics*; *Confusion matrix* | `metrics.compute_metrics`; `plots/confusion_tuned_*.png` |
| Accuracy is misleading under class imbalance (the 90/100 cancer example) | *Accuracy* | Why selection uses F1, and why accuracy is reported with a warning in `summary.md` |
| Precision, recall/sensitivity, specificity, F1 (harmonic mean) | *Precision*; *Recall/Sensitivity/True Positive Rate*; *Specificity*; *F1 score* | `metrics.py` (fraud class as positive) |
| PR curve, ROC curve, and *PR vs ROC* (PR preferred when the positive class is rare) | *PR curve*; *ROC curve*; *PR vs ROC curve* | `plots/curves_tuned_*.png`; PR-AUC used as tie-breaker in `tuning.tune` |

### Week 05 — Logistic regression & SGD (`Week-05-LOGISTIC REGRESSION AND STOCHASTIC GRADIENT DESCENT.pdf`)
| Concept | Location | Used in |
|---|---|---|
| Sigmoid, probability output, 0.5 threshold, log-odds, MLE | *Logistic Function (Sigmoid Function)*; *Terminologies*; *How does Logistic Regression work?* | `logreg` model; probabilities feed the probability-based kNN; `LogisticRegressionSGD._sigmoid` / `predict` |
| Fit `StandardScaler` on training data, apply to test data | Python program (steps 3–4: `scaler.fit_transform(X_train)`, `scaler.transform(X_test)`) | `experiment.prepare_folds` fits preprocessing per training fold |
| `LogisticRegression(solver='liblinear')`, `confusion_matrix`, `classification_report` | Python program | `config.SEARCH_SPACES['logreg']` uses liblinear (supports L1 and L2) |
| ROC curve and AUC from predicted probabilities | *Additional Questions* Q3 | `metrics.py` ROC-AUC; `reporting.plot_curves` |

### Week 06 — Cost functions (`Week-06-Cost functions-modified.pdf`)
| Concept | Location | Used in |
|---|---|---|
| Binary cross-entropy / log loss | *2. Binary Classification Cost Functions*; *4. Log Loss* | Loss minimised and logged by `LogisticRegressionSGD` |
| Gradient descent, learning rate α, convergence | *Gradient Descent: Minimizing the cost function* | `LogisticRegressionSGD` (`learning_rate` is tuned) |
| SGD update for logistic regression — gradient = (prediction − y)·x | *Derivation of the update equation for logistic regression using stochastic gradient descent* | `LogisticRegressionSGD.fit` (mini-batch version of the same update) |
| Error vs iteration plot | Questions 2a, Additional Q2–Q3 | `plots/sgd_logloss_<setting>.png` |
| Sweeping one parameter and plotting the error (β vs MSE) | Question 1a–c | Idea behind the sensitivity sweeps (`tuning.sensitivity`, `plots/sensitivity_*.png`) |

### Week 07 — Naive Bayes (`WEEK -07-NAÏVE BAYES CLASSIFIER.pdf`)
| Concept | Location | Used in |
|---|---|---|
| Bayes' theorem; posterior = likelihood × prior / evidence | *BAYES' THEOREM*; *Naming the Terms in the Theorem* | Basis of the Naive Bayes extension model |
| Gaussian NB for continuous data | *Key Concepts* → 3. *Multinomial and Gaussian Naive Bayes* | `naive_bayes` extension (`GaussianNB`, `var_smoothing` tuned) |
| Evaluate with accuracy, precision, recall, F1; tune hyperparameters | *Practical Implementation* steps 4–5 | Same metric set for every model |
| Limitation: "Poor with Rare Events" | *Limitations* → 3 | Discussion point: NB on 1.7 % fraud with and without oversampling |

### Week 08 — kNN & ID3 (`WEEK -08-K-Nearest Neighbour (K-NN) & ID-3 Decision Tree.pdf`)
| Concept | Location | Used in |
|---|---|---|
| kNN algorithm steps (choose K, Euclidean distance, majority vote); lazy learner | *Working of a K-NN* (Steps 1–6); *Introduction* | `knn` model; `ProbabilityKNN`; neighbour search inside SMOTE / ADASYN |
| Selecting K: K = 5 most preferred; very small K is noisy | *Selection of the value K in the K-NN Algorithm* | Baseline k = 5; search space k ∈ {1…31}; k sensitivity sweep |
| Manhattan (L1), Euclidean (L2), Minkowski (Lp) distances | *Other Distance Measures used in K-NN* | `p ∈ {1, 2}` in kNN and probability-based kNN search spaces; `ProbabilityKNN` distance is an Lp form |
| Pseudo-code of kNN | *Pseudo Code of KNN* | Mirrors `ProbabilityKNN.predict` |
| Entropy, information gain (ID3) | *ID-3 Decision Tree* → *Metrics in ID3* | `criterion='entropy'` option of the CART extension |

### Week 09 — Decision trees, C4.5 & CART (`Week 09 Decision Tree - C4.5 and CART.pdf`)
| Concept | Location | Used in |
|---|---|---|
| Gini index (CART) vs entropy / information gain (C4.5) | *Attribute Selection Measures*; *C4.5, AND CART DECISION TREES* | `cart` extension, `criterion ∈ {gini, entropy}` |
| Regularising hyperparameters: `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_leaf_nodes` | *CART* (bullet list of hyperparameters) | `cart` search space; same role as `max_depth` / `min_child_weight` in XGBoost |
| Pruning / overfitting of large trees | *Pruning: Getting an Optimal Decision tree* | Depth sweeps for `cart` and `xgboost` |
| "Decision trees are biased with imbalance dataset … balance out the dataset" | *Decision Tree Cons* | Motivation for evaluating tree models on SMOTE/ADASYN data and for `class_weight` / `scale_pos_weight` |
| Variance reduced "by bagging and boosting" | *Decision Tree Cons* | Link from single trees to XGBoost (boosted trees) |

### Week 10 — Clustering (`Week 10 Data Clustering .pdf`)
| Concept | Location | Used in |
|---|---|---|
| K-means, `init="k-means++"`, `n_init`, `random_state` | *K-Means Clustering*; *Method 2: Using k-means++* | `eda.py` fig 07 |
| SSE vs K and the elbow method | Questions 1c–1d | `eda.py` fig 07 (elbow chosen automatically) |
| Clustering for "finding unusual data objects (outlier detection)" | *Why Clustering?* | Fraud concentration per cluster in fig 07 |
| PCA to visualise high-dimensional data | *Visualizing the K-means clustering* | `eda.py` fig 06 |
| "Create a pipeline to train multiple … algorithms and hyper-parameters" | *How To Make Clustering in ML* → step 5 | Overall design of `pipeline.py` / `tuning.py` |

### Week 12 — SVM (`Week-12 SVM.pdf`)
| Concept | Location | Used in |
|---|---|---|
| Maximum-margin hyperplane, support vectors | *How does SVM work?*; *Support Vector Machine Terminology* | `svm` model |
| Soft margin, hinge loss, regularisation parameter C | *Support Vector Machine Terminology* → *Soft Margin*, *C*, *Hinge Loss* | C is tuned and swept (`config.SEARCH_SPACES['svm']`, `SENSITIVITY_SWEEPS['svm']`) |
| Kernels: linear, polynomial, RBF, sigmoid | *Popular kernel functions in SVM*; *Additional Questions* Q2a | `kernel ∈ {linear, poly, rbf, sigmoid}` |
| `SVC(kernel="rbf", gamma=…, C=…)` | *SVM implementation in Python* | `SubsampledSVC` wraps `SVC`; `gamma` tuned |
| Dual problem — training cost depends on the number of samples | *Dual Problem* | Why SVM training folds are capped (`SVM_MAX_TRAIN`) |

---

## 4. Concepts used from outside the lab manuals

| Concept | Why it is needed | Where |
|---|---|---|
| **SMOTE** (Chawla et al. 2002) | One of the three data settings in [1]; interpolates new fraud points between a fraud case and one of its k nearest fraud neighbours | `resampling.SMOTE` (from scratch) |
| **ADASYN** (He et al. 2008) | Second balancing strategy in [1]; generates more synthetic points around fraud cases surrounded by legitimate ones (ratio rᵢ of majority neighbours) | `resampling.ADASYN` (from scratch) |
| **Gradient-boosted trees / XGBoost** (Chen & Guestrin 2016) | Model in [1]; sequential trees each fitting the previous ensemble's errors, with depth, learning rate (shrinkage), number of trees, row/column subsampling and `scale_pos_weight` as hyperparameters | `xgboost` model (`HistGradientBoostingClassifier` fallback) |
| **Probability-based kNN** ([1] §3.2.5) | The reference paper's proposed method; LR probability added to the kNN distance | `models.ProbabilityKNN` |
| **Stratified k-fold CV** | Keeps the ~1.7 % fraud rate in every fold; plain k-fold can produce folds with almost no fraud | `experiment.prepare_folds` |
| **Data leakage / resampling inside CV folds** (Kaufman et al. 2012) | Oversampling before splitting leaks information into the test fold | `strict` vs `paper` protocol |
| **Hold-out test set after model selection** | Gives an unbiased estimate of the tuned model; CV scores of the selected config are optimistically biased | `pipeline.run_study` |
| **Random search HPO** (Bergstra & Bengio 2012) | Covers large search spaces with a fixed budget better than a full grid | `tuning.sample_configs` |
| **One-at-a-time sensitivity analysis** | Shows *how* each key hyperparameter moves F1/recall/precision, not just the best value | `tuning.sensitivity` |
| **Class weighting** (`class_weight='balanced'`, `scale_pos_weight`) | Cost-sensitive alternative to resampling — tuned as a hyperparameter | LR, SVM, CART, XGBoost search spaces |
| **L1 (lasso) penalty** | Regularisation that can zero out weak features | LR search space (`penalty='l1'`) |
| **PR-AUC (average precision)** | Threshold-free summary that ignores true negatives, suited to rare positives | `metrics.py`; tie-breaker in selection |
| **IQR outlier rule** | Robust to skewed features such as income, complementing the manual's σ-rule | `eda.py` outlier table |
| **`ColumnTransformer`** | Applies one-hot encoding and scaling to different columns in one fitted object | `preprocessing.py` |
| **Training-set subsampling for SVC** | Keeps kernel-SVM training tractable after oversampling | `models.SubsampledSVC` |
| **Wall-clock timing** (`time.perf_counter`) | "Computational efficiency" objective — fit and predict time per fold | `experiment.py` |
| **Tkinter GUI with a worker thread + queue** | Keeps the window responsive while experiments run; widgets updated only from the main thread | `gui.py` |

---

## 5. Notes for the report

* **Methodology section**: §1.2 (pipeline), §1.3 (hyperparameter table), the protocol note, and
  the formulas for SMOTE, ADASYN and the probability-based kNN distance (docstrings in
  `resampling.py` and `models.py`).
* **Implementation Details**: Python 3, scikit-learn, xgboost, matplotlib, Tkinter; from-scratch
  SMOTE/ADASYN/SGD-LR; SVM cap; seed 42; 5-fold stratified CV; 20 % hold-out.
* **Results and Discussion**: paste `results/summary.md`; use `plots/compare_f1_holdout.png`,
  `compare_recall_holdout.png`, `compare_training_time.png`, the sensitivity plots and the PR
  curves. Discuss (a) baseline vs tuned per setting, (b) imbalanced vs SMOTE vs ADASYN,
  (c) probability-based kNN vs XGBoost in F1 *and* training time, (d) `strict` vs `paper`
  protocol — how much of the published-style score comes from leakage. (In a test on the
  synthetic demo data, kNN + SMOTE scored CV F1 ≈ 0.95 under `paper` but only ≈ 0.11 on
  the untouched hold-out set — expect a similar gap on the real data.)
* **Plagiarism (≤ 15 % Turnitin)**: the code comments and this file paraphrase the manuals and
  papers; still write the report text in your own words.

### Expected run time
Measured while testing: all 8 models × 3 settings with 4 random configurations plus the
sensitivity sweeps took about 42 minutes on a 2-core machine, and roughly 60 % of that was
the scikit-learn *fallback* for XGBoost (the real `xgboost` package was not installable
there). The default 12 configurations per model take proportionally longer. Use the GUI's
*Quick preset* (4 configurations, no sweeps) or `--n-iter 4 --no-sensitivity` while
developing, then do one full run for the report. A run can be stopped from the GUI;
whatever finished is still saved.

### Efficiency comparison
kNN and the probability-based kNN are "lazy" learners: training is almost free and the cost
is paid when scoring. Time is therefore reported as **fit + predict per fold**
(`compare_training_time.png`, `summary.md`); `final_comparison.csv` keeps both parts
separately (`cv_fit_time_mean`, `cv_predict_time_mean`, `holdout_fit_time`,
`holdout_predict_time`).
