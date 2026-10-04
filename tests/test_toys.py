import numpy as np

from sneppx_datasets.toys import sine_regression, xor_classification


def test_xor_shapes():
    X, y = xor_classification(n=32)
    assert X.shape == (32, 2)
    assert y.shape == (32,)
    assert set(np.unique(y)) <= {0.0, 1.0}


def test_sine_range():
    x, y = sine_regression(n=64)
    assert x.shape == (64, 1)
    assert np.all(np.abs(y) < 1.5)
