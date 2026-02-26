"""
Preprocessing utilities for the German Credit Risk dataset.

Responsibilities:
- Load raw data from CSV.
- Handle missing values.
- Encode categorical variables.
- Scale numerical features.
- Train/test split with reproducible random_state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42


@dataclass
class PreprocessedData:
    """Container for preprocessed train/test data and the fitted transformer."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    transformer: ColumnTransformer


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw German credit data CSV."""
    return pd.read_csv(path)


def build_preprocessing_pipeline(
    df: pd.DataFrame,
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Build a ColumnTransformer for numeric and categorical preprocessing."""
    target_col = "Risk"
    feature_cols = [c for c in df.columns if c != target_col]

    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [
        c for c in feature_cols if c not in numeric_features
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


def preprocess_and_split(
    df: pd.DataFrame, test_size: float = 0.2
) -> PreprocessedData:
    """
    Full preprocessing pipeline:
    - Encode target Risk as 0 (good) and 1 (bad).
    - Build transformers.
    - Fit on train, transform train and test.
    """
    df = df.copy()

    # Encode target: good -> 0, bad -> 1
    df["Risk"] = df["Risk"].map({"good": 0, "bad": 1})

    X = df.drop(columns=["Risk"])
    y = df["Risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor, _, _ = build_preprocessing_pipeline(df)

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Convert back to DataFrame for easier inspection if needed
    X_train_df = pd.DataFrame(X_train_transformed.toarray() if hasattr(X_train_transformed, "toarray") else X_train_transformed)
    X_test_df = pd.DataFrame(X_test_transformed.toarray() if hasattr(X_test_transformed, "toarray") else X_test_transformed)

    return PreprocessedData(
        X_train=X_train_df,
        X_test=X_test_df,
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        transformer=preprocessor,
    )




