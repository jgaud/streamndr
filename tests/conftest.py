"""Shared fixtures for streamndr tests."""
import numpy as np
import pytest
from river.datasets import synth


@pytest.fixture(scope="session")
def dataset():
    """Generate the standard RandomRBF dataset used across all model tests.

    Returns a dict with X_train, y_train, X_test, y_test, and known_classes.
    Uses the same parameters as the README quickstart example.
    """
    ds = synth.RandomRBF(
        seed_model=42, seed_sample=42, n_classes=4, n_features=5, n_centroids=10
    )

    offline_size = 1000
    online_size = 5000
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for x, y in ds.take(10 * (offline_size + online_size)):
        if len(y_train) < offline_size:
            if y == 0 or y == 1:
                X_train.append(np.array(list(x.values())))
                y_train.append(y)
        elif len(y_test) < online_size:
            X_test.append(x)
            y_test.append(y)
        else:
            break

    return {
        "X_train": np.array(X_train),
        "y_train": np.array(y_train),
        "X_test": X_test,
        "y_test": y_test,
        "known_classes": [0, 1],
    }
