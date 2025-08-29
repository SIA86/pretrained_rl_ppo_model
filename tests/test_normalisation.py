import numpy as np
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from scr.normalisation import NormalizationStats


def test_zscore_normalisation():
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    norm = NormalizationStats(kind="zscore").fit(X)
    Xt = norm.transform(X)
    assert np.allclose(Xt.mean(axis=0), 0.0, atol=1e-7)
    assert np.allclose(Xt.std(axis=0), 1.0, atol=1e-7)


def test_minmax_normalisation():
    X = np.array([[0, 1], [2, 3], [4, 5]], dtype=float)
    norm = NormalizationStats(kind="minmax").fit(X)
    Xt = norm.transform(X)
    assert np.allclose(Xt.min(axis=0), 0.0)
    assert np.allclose(Xt.max(axis=0), 1.0)


def test_robust_normalisation():
    X = np.array([[0, 1], [2, 3], [4, 5]], dtype=float)
    norm = NormalizationStats(kind="robust").fit(X)
    Xt = norm.transform(X)
    assert np.allclose(np.median(Xt, axis=0), 0.0)
    q1 = np.percentile(Xt, 25, axis=0)
    q3 = np.percentile(Xt, 75, axis=0)
    assert np.allclose(q3 - q1, 1.0)


def test_imputation_removes_nans():
    X_train = np.array([[1, np.nan], [3, 4]], dtype=float)
    norm = NormalizationStats(kind="zscore").fit(X_train)
    X = np.array([[np.nan, 5]], dtype=float)
    Xt = norm.transform(X)
    assert np.all(np.isfinite(Xt))
