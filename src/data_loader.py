__version__ = "1.0.0"

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config


def load_csv(path, target=config.TARGET_COLUMN):
    df = pd.read_csv(path)
    return prepare_frame(df, target)


def prepare_frame(df, target=config.TARGET_COLUMN):
    df = df.copy()
    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found. Columns are: {list(df.columns)}")

    dropped = [c for c in config.ID_LIKE_COLUMNS if c in df.columns]
    df = df.drop(columns=dropped)

    n_before = len(df)
    df = df.drop_duplicates()
    n_dupes = n_before - len(df)

    missing_before = int(df.isnull().sum().sum())
    for col in df.columns:
        if col == target or not df[col].isnull().any():
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].value_counts().idxmax())
    df = df.dropna(subset=[target])

    y = df[target].astype(int).to_numpy()
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("Target must be binary 0/1 (1 = fraud).")
    X = df.drop(columns=[target])

    info = {
        "rows": int(len(df)),
        "features": int(X.shape[1]),
        "fraud": int(y.sum()),
        "legit": int((y == 0).sum()),
        "fraud_rate": float(y.mean()),
        "dropped_id_columns": dropped,
        "duplicates_removed": int(n_dupes),
        "missing_values_imputed": missing_before,
        "categorical_columns": categorical_columns(X),
        "numeric_columns": numeric_columns(X),
    }
    return X, y, info


def categorical_columns(X):
    return [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]


def numeric_columns(X):
    return [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]


def holdout_split(X, y, fraction=config.HOLDOUT_FRACTION, seed=config.RANDOM_STATE):
    return train_test_split(X, y, test_size=fraction, stratify=y, random_state=seed)


def make_synthetic(n=25134, fraud_rate=420 / 25134, seed=config.RANDOM_STATE):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)

    def pick(options, probs_legit, probs_fraud):
        out = np.empty(n, dtype=object)
        out[y == 0] = rng.choice(options, (y == 0).sum(), p=probs_legit)
        out[y == 1] = rng.choice(options, (y == 1).sum(), p=probs_fraud)
        return out

    age = np.clip(rng.normal(43, 11, n) - 6 * y, 21, 69).round(1)
    unemployed = rng.binomial(1, 0.17 + 0.08 * y)
    years_emp = np.where(unemployed == 1, 0.0,
                         np.clip(rng.exponential(7, n) * (1 - 0.4 * y), 0.1, 43)).round(2)
    income = np.round(np.exp(rng.normal(12.0, 0.45, n) - 0.25 * y), -2)
    acct = np.clip(rng.normal(27, 16, n) - 12 * y, 0, 60).round()
    pocket = (y == 1) & (rng.random(n) < 0.35)
    age[pocket] = rng.uniform(21, 29, pocket.sum()).round(1)
    acct[pocket] = rng.integers(0, 6, pocket.sum())
    children = rng.poisson(0.45, n)
    family = children + rng.choice([1, 2], n, p=[0.25, 0.75])

    df = pd.DataFrame({
        "ID": np.arange(5008804, 5008804 + n),
        "Gender": pick(["F", "M"], [0.65, 0.35], [0.55, 0.45]),
        "Own_car": rng.binomial(1, 0.38 - 0.12 * y),
        "Own_property": rng.binomial(1, 0.66 - 0.15 * y),
        "Work_phone": rng.binomial(1, 0.23, n),
        "Phone": rng.binomial(1, 0.30, n),
        "Email": rng.binomial(1, 0.09 + 0.06 * y),
        "Unemployed": unemployed,
        "Num_children": children,
        "Num_family": family,
        "Account_length": acct,
        "Total_income": income,
        "Age": age,
        "Years_employed": years_emp,
        "Income_type": pick(["Working", "Commercial associate", "State servant",
                             "Pensioner", "Student"],
                            [0.52, 0.24, 0.08, 0.159, 0.001],
                            [0.47, 0.25, 0.06, 0.219, 0.001]),
        "Education_type": pick(["Secondary / secondary special", "Higher education",
                                "Incomplete higher", "Lower secondary",
                                "Academic degree"],
                               [0.67, 0.27, 0.04, 0.019, 0.001],
                               [0.63, 0.27, 0.06, 0.039, 0.001]),
        "Family_status": pick(["Married", "Single / not married", "Civil marriage",
                               "Separated", "Widow"],
                              [0.69, 0.13, 0.08, 0.06, 0.04],
                              [0.62, 0.17, 0.09, 0.06, 0.06]),
        "Housing_type": pick(["House / apartment", "With parents", "Municipal apartment",
                              "Rented apartment", "Office apartment", "Co-op apartment"],
                             [0.89, 0.05, 0.03, 0.015, 0.01, 0.005],
                             [0.84, 0.07, 0.05, 0.02, 0.01, 0.01]),
        "Occupation_type": pick(["Laborers", "Core staff", "Sales staff", "Managers",
                                 "Drivers", "Other"],
                                [0.25, 0.15, 0.14, 0.12, 0.09, 0.25],
                                [0.28, 0.12, 0.15, 0.08, 0.12, 0.25]),
        "Target": y,
    })
    return df
