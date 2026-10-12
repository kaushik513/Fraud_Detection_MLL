__version__ = "1.0.0"

RANDOM_STATE = 42

# Data
TARGET_COLUMN = "Target" # binary: 1 = fraud, 0 = legitimate
ID_LIKE_COLUMNS = ["ID", "Id", "id", "Ind_ID", "applicant_id"]  
HOLDOUT_FRACTION = 0.20           
CV_FOLDS = 5                      

# Class-balancing settings
SAMPLINGS = ["imbalanced", "smote", "adasyn"]
SMOTE_K = 5                       
ADASYN_K = 5                      

DEFAULT_PROTOCOL = "strict"
SVM_MAX_TRAIN = 5000
SVM_MAX_ITER = 50000

SELECTION_METRIC = "f1"

BASELINE_PARAMS = {
    "logreg": {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "tol": 1e-4,
               "max_iter": 1000, "class_weight": None},
    "knn":    {"n_neighbors": 5, "weights": "uniform", "p": 2},
    "svm":    {"kernel": "rbf", "C": 1.0, "gamma": "scale", "tol": 1e-3,
               "class_weight": None},
    "xgboost": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.3,
                "subsample": 1.0, "colsample_bytree": 1.0,
                "min_child_weight": 1, "scale_pos_weight": 1.0},
    "prob_knn": {"n_neighbors": 5, "prob_weight": 1.0, "lr_C": 1.0, "p": 2},
    # ---- extension models drawn from the lab manuals (not in the paper) ----
    "naive_bayes": {"var_smoothing": 1e-9},
    "cart": {"criterion": "gini", "max_depth": None, "min_samples_leaf": 1,
             "class_weight": None},
    "logreg_sgd": {"learning_rate": 0.05, "epochs": 30, "batch_size": 256,
                   "l2": 0.0},
}

SEARCH_SPACES = {
    "logreg": {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],            
        "tol": [1e-5, 1e-4, 1e-3, 1e-2],
        "max_iter": [1000],
        "class_weight": [None, "balanced"],
    },
    "knn": {
        "n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21, 31],
        "weights": ["uniform", "distance"],
        "p": [1, 2],                        
    },
    "svm": {
        "kernel": ["linear", "rbf", "poly", "sigmoid"],   
        "C": [0.1, 1.0, 10.0, 100.0],
        "gamma": ["scale", 0.01, 0.1, 1.0],
        "tol": [1e-4, 1e-3, 1e-2],
        "class_weight": [None, "balanced"],
    },
    "xgboost": {
        "n_estimators": [100, 200, 400],
        "max_depth": [3, 4, 6, 8],
        "learning_rate": [0.03, 0.1, 0.3],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [1, 5],
        "scale_pos_weight": [1.0, 5.0, 20.0, 58.0],  
    },
    "prob_knn": {
        "n_neighbors": [3, 5, 7, 11, 15, 21],
        "prob_weight": [0.5, 1.0, 2.0, 5.0, 10.0],
        "lr_C": [0.01, 0.1, 1.0, 10.0],
        "p": [1, 2],
    },
    "naive_bayes": {"var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5, 1e-3]},
    "cart": {
        "criterion": ["gini", "entropy"], # CART vs C4.5-style impurity 
        "max_depth": [3, 5, 8, 12, None],
        "min_samples_leaf": [1, 5, 20],
        "class_weight": [None, "balanced"],
    },
    "logreg_sgd": {
        "learning_rate": [0.005, 0.01, 0.05, 0.1],
        "epochs": [10, 30, 60],
        "batch_size": [64, 256],
        "l2": [0.0, 1e-4, 1e-2],
    },
}

DEFAULT_N_ITER = 12

SENSITIVITY_SWEEPS = {
    "logreg":   ("C", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]),
    "knn":      ("n_neighbors", [1, 3, 5, 7, 11, 15, 21, 31]),
    "svm":      ("C", [0.1, 1.0, 10.0, 100.0]),
    "xgboost":  ("max_depth", [2, 3, 4, 6, 8, 10]),
    "prob_knn": ("prob_weight", [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]),
    "cart":     ("max_depth", [2, 3, 5, 8, 12, 20]),
}

PAPER_MODELS = ["logreg", "knn", "svm", "xgboost", "prob_knn"]
EXTENSION_MODELS = ["naive_bayes", "cart", "logreg_sgd"]
ALL_MODELS = PAPER_MODELS + EXTENSION_MODELS

MODEL_LABELS = {
    "logreg": "Logistic Regression",
    "knn": "kNN",
    "svm": "SVM",
    "xgboost": "XGBoost",
    "prob_knn": "Probability-based kNN",
    "naive_bayes": "Gaussian Naive Bayes",
    "cart": "Decision Tree (CART)",
    "logreg_sgd": "Logistic Regression (SGD, from scratch)",
}

RESULTS_DIR = "results"
