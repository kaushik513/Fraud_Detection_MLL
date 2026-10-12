__version__ = "1.0.0"

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import categorical_columns, numeric_columns


def build_preprocessor(X):
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
    except Exception:  
        return None
