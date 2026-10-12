"""
preprocessing.py — one-hot encoding + standardisation, fitted on training data only.

Version: 1.0.0
Changelog:
  1.0.0 (2026-10-10) — ColumnTransformer: OneHotEncoder for categorical columns,
                       StandardScaler for numeric columns.

Lab-manual links: Week 3 "Data with no numerical values" (one-hot encoding),
Week 3 PCA-additional "Perform Data Preprocessing" (StandardScaler, mu=0, sigma=1),
Week 5 example (scaler.fit_transform on train, transform on test).
"""

__version__ = "1.0.0"

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import categorical_columns, numeric_columns


def build_preprocessor(X):
    """Return an unfitted transformer for the columns of X.

    Scaling matters here: kNN, SVM and Logistic Regression are all distance- or
    gradient-based, so a raw Total_income (~10^5) would otherwise swamp Age (~10^1).
    handle_unknown='ignore' keeps rare categories that appear only in a test fold
    from crashing the pipeline.
    """
    cats = categorical_columns(X)
    nums = numeric_columns(X)
    transformers = []
    if nums:
        transformers.append(("num", StandardScaler(), nums))
    if cats:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore",
                                                  sparse_output=False), cats))
    return ColumnTransformer(transformers, remainder="drop")


def feature_names(preprocessor):
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:  # pragma: no cover - older sklearn
        return None
