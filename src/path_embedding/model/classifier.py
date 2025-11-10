"""Classifier training and prediction."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42
) -> RandomForestClassifier:
    """Train Random Forest classifier.

    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training labels (n_samples,)
        random_state: Random seed for reproducibility

    Returns:
        Trained RandomForestClassifier

    Example:
        >>> X = np.random.rand(50, 10)
        >>> y = np.random.randint(0, 2, 50)
        >>> model = train_classifier(X, y)
        >>> isinstance(model, RandomForestClassifier)
        True
    """
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model
