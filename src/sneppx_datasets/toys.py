"""Small built-in toy datasets (NumPy)."""

import numpy as np


def xor_classification(n=64, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, 2)).astype(np.float64)
    y = (X[:, 0] != X[:, 1]).astype(np.float64)
    return X, y


def sine_regression(n=64, seed=0, noise=0.05):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2 * np.pi, n)
    y = np.sin(x) + noise * rng.standard_normal(n)
    return x.reshape(-1, 1), y
