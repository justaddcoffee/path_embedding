"""Tests for classifier training."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from path_embedding.model.classifier import train_classifier


def test_train_classifier():
    """Test training Random Forest classifier.

    >>> X = np.random.rand(100, 10)
    >>> y = np.random.randint(0, 2, 100)
    >>> model = train_classifier(X, y)
    >>> isinstance(model, RandomForestClassifier)
    True
    """
    # Create synthetic data
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)

    model = train_classifier(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)

    # Should be able to make predictions
    predictions = model.predict(X_train)
    assert predictions.shape[0] == 100


def test_train_classifier_predictions():
    """Test that trained model can make predictions."""
    X_train = np.random.rand(50, 10)
    y_train = np.random.randint(0, 2, 50)

    model = train_classifier(X_train, y_train)

    X_test = np.random.rand(10, 10)
    predictions = model.predict(X_test)

    assert predictions.shape[0] == 10
    # Predictions should be 0 or 1
    assert all(p in [0, 1] for p in predictions)


def test_train_classifier_probabilities():
    """Test that model can output probabilities."""
    X_train = np.random.rand(50, 10)
    y_train = np.random.randint(0, 2, 50)

    model = train_classifier(X_train, y_train)

    X_test = np.random.rand(10, 10)
    probs = model.predict_proba(X_test)

    assert probs.shape == (10, 2)
    # Probabilities should sum to 1
    assert np.allclose(probs.sum(axis=1), 1.0)
