"""
Dataset loading utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_dataset_path(path_value: str) -> Path:
    if not path_value:
        raise ValueError("Dataset path is empty. Please set 'path' in the dataset configuration.")

    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    return path


def load_breast_cancer_dataset(params):
    data = load_breast_cancer()
    X = data.data
    y = data.target

    return train_test_split(
        X,
        y,
        test_size=params.get("test_size", 0.2),
        random_state=params.get("random_state", 42),
        shuffle=True,
        stratify=y,
    )


def _resolve_header_value(header_value: Any):
    if isinstance(header_value, str) and header_value.lower() == "none":
        return None
    return header_value


def _resolve_column_name(df: pd.DataFrame, column_reference: Any) -> Any:
    if isinstance(column_reference, int):
        return df.columns[column_reference]
    return column_reference


def _encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    encoded = df.copy()
    for column in encoded.columns:
        if pd.api.types.is_bool_dtype(encoded[column]):
            encoded[column] = encoded[column].astype(int)
            continue
        if pd.api.types.is_numeric_dtype(encoded[column]):
            if encoded[column].isna().any():
                encoded[column] = encoded[column].fillna(encoded[column].median())
            continue

        series = encoded[column].astype("string").fillna("__missing__")
        encoded[column] = pd.Categorical(series).codes

    return encoded


def _encode_target(y: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(y):
        if y.isna().any():
            y = y.fillna(y.mode().iloc[0])
        return y.to_numpy()
    return pd.Categorical(y.astype("string").fillna("__missing__")).codes


def load_csv_classification_dataset(params):
    path = _resolve_dataset_path(str(params.get("path", "")))
    target_column = params.get("target_column", "target")
    fraction = float(params.get("fraction", 1.0))
    random_state = int(params.get("random_state", 42))
    test_size = float(params.get("test_size", 0.2))
    stratify = bool(params.get("stratify", True))
    header = _resolve_header_value(params.get("header", "infer"))
    skipinitialspace = bool(params.get("skipinitialspace", False))
    n_rows = params.get("n_rows")
    drop_columns = params.get("drop_columns", [])

    df = pd.read_csv(path, header=header, nrows=n_rows, skipinitialspace=skipinitialspace)
    if "column_names" in params:
        df.columns = list(params["column_names"])

    if fraction < 1.0:
        df = df.sample(frac=fraction, random_state=random_state)

    target_column_name = _resolve_column_name(df, target_column)
    if target_column_name not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataset: {path}")

    drop_column_names = [_resolve_column_name(df, column) for column in drop_columns]
    feature_columns = params.get("feature_columns")
    if feature_columns:
        resolved_feature_columns = [_resolve_column_name(df, column) for column in feature_columns]
        feature_df = df[resolved_feature_columns].copy()
    else:
        feature_df = df.drop(columns=[target_column_name, *drop_column_names], errors="ignore").copy()

    feature_df = _encode_categorical_features(feature_df)
    y = _encode_target(df[target_column_name])
    X = feature_df.to_numpy(dtype=np.float32, copy=False)

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=y if stratify else None,
    )


def load_fraud_dataset(params):
    merged_params = {
        "target_column": "Class",
        **params,
    }
    return load_csv_classification_dataset(merged_params)
