"""Model evaluation utilities."""
from typing import Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


def evaluate_classifier(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """Evaluate classifier and return metrics.

    Args:
        model: Trained classifier with predict() and predict_proba()
        X_test: Test features
        y_test: Test labels

    Returns:
        Dict with accuracy, precision, recall, f1, roc_auc

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> X = np.random.rand(50, 10)
        >>> y = np.array([0] * 25 + [1] * 25)
        >>> model = RandomForestClassifier(random_state=42).fit(X, y)
        >>> metrics = evaluate_classifier(model, X, y)
        >>> "accuracy" in metrics
        True
    """
    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    return metrics


def print_evaluation_report(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> None:
    """Print detailed evaluation report.

    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test labels
    """
    y_pred = model.predict(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["Implausible", "Plausible"]))

    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("\n=== Metrics Summary ===")
    metrics = evaluate_classifier(model, X_test, y_test)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
