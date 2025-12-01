"""
metrics.py
This module contains functions to evaluate the performance of machine learning models for classification tasks.
"""

from __future__ import annotations
from typing import Optional, Any
from dataclasses import dataclass
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

@dataclass
class QualityMetrics:
    """Holds classification quality metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float] 
    confusion_matrix: np.ndarray

def compute_quality_metrics(model: Any, X_test, y_test) -> QualityMetrics:
    """Compute classification quality metrics for a given model and test data."""

    y_pred = model.predict(X_test)

    roc_auc = None
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            roc_auc = roc_auc_score(y_test, scores)
    except Exception:
        roc_auc = None

    confusion_matrix = confusion_matrix(y_test, y_pred)

    return QualityMetrics(
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, zero_division=0),
        recall=recall_score(y_test, y_pred, zero_division=0),
        f1=f1_score(y_test, y_pred, zero_division=0),
        roc_auc=roc_auc,
        confusion_matrix=confusion_matrix
    )