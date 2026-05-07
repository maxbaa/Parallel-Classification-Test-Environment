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


def _resolve_requested_train_size(params: dict[str, Any], available_samples: int) -> int | None:
    train_size = params.get("train_size")
    train_fraction = params.get("train_fraction")

    if train_size is not None and train_fraction is not None:
        raise ValueError("Please configure either 'train_size' or 'train_fraction', not both.")

    if train_size is not None:
        requested = int(train_size)
    elif train_fraction is not None:
        fraction = float(train_fraction)
        if not 0 < fraction <= 1:
            raise ValueError(f"'train_fraction' must be in the range (0, 1], got {fraction}.")
        requested = int(round(available_samples * fraction))
    else:
        return None

    if requested <= 0:
        raise ValueError(f"Requested train subset size must be positive, got {requested}.")
    if requested > available_samples:
        raise ValueError(
            f"Requested train subset size {requested} exceeds available training samples {available_samples}."
        )

    return requested


def _nested_stratified_subset_indices(y: np.ndarray, subset_size: int, random_state: int) -> np.ndarray:
    if subset_size >= len(y):
        return np.arange(len(y))

    classes, y_inverse = np.unique(y, return_inverse=True)
    if subset_size < len(classes):
        raise ValueError(
            f"Requested train subset size {subset_size} is too small for {len(classes)} classes."
        )

    rng = np.random.default_rng(random_state)
    class_indices = [np.flatnonzero(y_inverse == class_index) for class_index in range(len(classes))]
    class_counts = np.array([len(indices) for indices in class_indices], dtype=int)
    fractional_targets = class_counts * (subset_size / len(y))
    base_counts = np.floor(fractional_targets).astype(int)
    base_counts = np.minimum(base_counts, class_counts)
    base_counts = np.where(base_counts == 0, 1, base_counts)

    assigned = int(base_counts.sum())
    remainders = fractional_targets - np.floor(fractional_targets)
    order = np.argsort(-remainders)

    while assigned < subset_size:
        updated = False
        for class_index in order:
            if base_counts[class_index] < class_counts[class_index]:
                base_counts[class_index] += 1
                assigned += 1
                updated = True
                if assigned == subset_size:
                    break
        if not updated:
            break

    while assigned > subset_size:
        updated = False
        for class_index in reversed(order):
            if base_counts[class_index] > 1:
                base_counts[class_index] -= 1
                assigned -= 1
                updated = True
                if assigned == subset_size:
                    break
        if not updated:
            raise ValueError(
                "Unable to reduce stratified subset counts without dropping an entire class."
            )

    selected_indices: list[np.ndarray] = []
    for class_index, indices in enumerate(class_indices):
        shuffled_indices = rng.permutation(indices)
        selected_indices.append(shuffled_indices[: base_counts[class_index]])

    combined = np.concatenate(selected_indices)
    return np.sort(combined)


def _apply_train_subset(X_train: np.ndarray, y_train: np.ndarray, params: dict[str, Any]):
    requested_train_size = _resolve_requested_train_size(params, len(y_train))
    if requested_train_size is None or requested_train_size == len(y_train):
        return X_train, y_train

    subset_random_state = int(params.get("train_subset_random_state", params.get("random_state", 42)))
    subset_strategy = str(params.get("train_subset_strategy", "stratified_nested"))

    if subset_strategy == "stratified_nested":
        subset_indices = _nested_stratified_subset_indices(y_train, requested_train_size, subset_random_state)
    elif subset_strategy == "stratified_shuffle":
        subset_indices, _ = train_test_split(
            np.arange(len(y_train)),
            train_size=requested_train_size,
            random_state=subset_random_state,
            shuffle=True,
            stratify=y_train,
        )
        subset_indices = np.sort(subset_indices)
    else:
        raise ValueError(
            f"Unknown train subset strategy '{subset_strategy}'. "
            "Supported values: 'stratified_nested', 'stratified_shuffle'."
        )

    return X_train[subset_indices], y_train[subset_indices]


def load_breast_cancer_dataset(params):
    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params.get("test_size", 0.2),
        random_state=params.get("random_state", 42),
        shuffle=True,
        stratify=y,
    )
    X_train, y_train = _apply_train_subset(X_train, y_train, params)
    return X_train, X_test, y_train, y_test


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

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=y if stratify else None,
    )
    X_train, y_train = _apply_train_subset(X_train, y_train, params)
    return X_train, X_test, y_train, y_test


def load_fraud_dataset(params):
    merged_params = {
        "target_column": "Class",
        **params,
    }
    return load_csv_classification_dataset(merged_params)
