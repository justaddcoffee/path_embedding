"""Tests for model evaluation."""
import numpy as np
from path_embedding.model.evaluation import evaluate_classifier, print_evaluation_report
from path_embedding.model.classifier import train_classifier


def test_evaluate_classifier():
    """Test evaluating classifier with metrics.

    >>> from sklearn.ensemble import RandomForestClassifier
    >>> X = np.random.rand(50, 10)
    >>> y = np.array([0] * 25 + [1] * 25)
    >>> model = RandomForestClassifier(random_state=42)
    >>> model.fit(X, y)
    RandomForestClassifier(random_state=42)
    >>> metrics = evaluate_classifier(model, X, y)
    >>> "accuracy" in metrics
    True
    """
    # Create simple dataset
    X_train = np.random.rand(100, 10)
    y_train = np.array([0] * 50 + [1] * 50)

    model = train_classifier(X_train, y_train)

    X_test = np.random.rand(50, 10)
    y_test = np.array([0] * 25 + [1] * 25)

    metrics = evaluate_classifier(model, X_test, y_test)

    # Should have all expected metrics
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics

    # All metrics should be between 0 and 1
    for metric_name, value in metrics.items():
        assert 0 <= value <= 1, f"{metric_name} out of range"


def test_evaluate_classifier_perfect():
    """Test with perfect predictions."""
    # Create separable data
    X_train = np.vstack([
        np.random.rand(50, 10) - 1,  # Class 0
        np.random.rand(50, 10) + 1,  # Class 1
    ])
    y_train = np.array([0] * 50 + [1] * 50)

    model = train_classifier(X_train, y_train)

    # Test on training data (should be nearly perfect)
    metrics = evaluate_classifier(model, X_train, y_train)

    # Should have high accuracy
    assert metrics["accuracy"] > 0.8


def test_print_evaluation_report(capsys):
    """Test that print_evaluation_report outputs to stdout."""
    X_train = np.random.rand(100, 10)
    y_train = np.array([0] * 50 + [1] * 50)

    model = train_classifier(X_train, y_train)

    X_test = np.random.rand(50, 10)
    y_test = np.array([0] * 25 + [1] * 25)

    print_evaluation_report(model, X_test, y_test)

    captured = capsys.readouterr()
    assert "Classification Report" in captured.out
    assert "Confusion Matrix" in captured.out
    assert "Metrics Summary" in captured.out
