"""Offline robustness: the pure-NumPy IRLS backend must run with zero
third-party ML dependencies and agree with scikit-learn to within tolerance.
"""

from __future__ import annotations

import numpy as np

from equiforge.data.synthesizer import make_dataset
from equiforge.eval.metrics import accuracy
from equiforge.models.logreg import LogisticRegressionBackend


def _fit_and_score(backend: str, seed: int = 7):
    ds = make_dataset(name="t", n_samples=2000, seed=seed)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ds.y))
    rng.shuffle(idx)
    nte = int(0.3 * len(ds.y))
    te, tr = idx[:nte], idx[nte:]
    m = LogisticRegressionBackend(backend=backend, random_state=seed).fit(ds.X[tr], ds.y[tr])
    p = m.predict_proba(ds.X[te])[:, 1]
    pred = (p >= 0.5).astype(float)
    return m, p, accuracy(ds.y[te], pred)


def test_numpy_backend_runs_and_is_finite():
    m, p, _ = _fit_and_score("numpy")
    assert np.all(np.isfinite(p))
    assert np.all((p >= 0.0) & (p <= 1.0))
    assert np.all(np.isfinite(m._w))
    assert np.isfinite(m._b)


def test_numpy_agrees_with_sklearn():
    if not __import__("equiforge.models.logreg", fromlist=["available_sklearn"]).available_sklearn():
        import pytest

        pytest.skip("scikit-learn not available")
    _, p_sk, acc_sk = _fit_and_score("sklearn")
    _, p_np, acc_np = _fit_and_score("numpy")
    # Predictions and accuracy should agree closely on the same data.
    assert np.mean(np.abs(p_sk - p_np)) < 0.05
    assert abs(acc_sk - acc_np) < 0.05
