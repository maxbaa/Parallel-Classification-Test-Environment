"""
loaders.py
This module contains data loading utilities for different datasets.
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def load_breast_cancer_dataset(params):
    data = load_breast_cancer()
    X = data.data
    y = data.target

    return train_test_split(
        X, y,
        test_size=params.get("test_size", 0.2),
        random_state=42,
        shuffle=True,
        stratify=y
    )

def load_fraud_dataset(params):
    path = ""

    df = pd.read_csv(path)

    fraction = params.get("fraction", 1.0)
    if fraction < 1.0:
        df = df.sample(frac=fraction, random_state=42)
    
    X = df.drop("Class", axis=1).values
    y = df["Class"].values

    return train_test_split(
        X, y,
        test_size=params.get("test_size", 0.2),
        random_state=42,
        shuffle=True,
        stratify=y
    )