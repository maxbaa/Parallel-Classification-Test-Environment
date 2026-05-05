"""
metrics.py
This module contains functions to evaluate the performance of machine learning models for classification tasks.
"""

from __future__ import annotations
import json
from typing import Optional, Any
from dataclasses import dataclass
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

@dataclass
class QualityMetrics:
    """Holds classification quality metrics."""
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]
    confusion_matrix_json: str
    classification_report_json: str

def compute_quality_metrics(model: Any, X_test, y_test) -> QualityMetrics:
    """Compute classification quality metrics for a given model and test data."""

    y_pred = model.predict(X_test)
    y_true = np.asarray(y_test)
    labels = np.unique(y_true)
    is_binary = len(labels) == 2

    roc_auc = None
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            if is_binary:
                roc_auc = roc_auc_score(y_true, y_proba[:, 1])
            else:
                roc_auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            if is_binary:
                roc_auc = roc_auc_score(y_true, scores)
            else:
                roc_auc = roc_auc_score(y_true, scores, multi_class="ovr", average="weighted")
    except Exception:
        roc_auc = None

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    confusion_payload = {
        "labels": labels.tolist(),
        "matrix": matrix.tolist(),
    }

    return QualityMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, average="weighted", zero_division=0),
        recall=recall_score(y_true, y_pred, average="weighted", zero_division=0),
        f1=f1_score(y_true, y_pred, average="weighted", zero_division=0),
        roc_auc=roc_auc,
        confusion_matrix_json=json.dumps(confusion_payload, sort_keys=True),
        classification_report_json=json.dumps(report, sort_keys=True),
    )
