"""Logistic-regression backend with a pure-NumPy IRLS fallback.

The Tier-0 backend is scikit-learn's ``LogisticRegression`` (deterministic with
a fixed ``random_state``). When scikit-learn is unavailable (offline / minimal
environments) we fall back to a hand-written Newton–Raphson (IRLS) solver with
L2 regularisation — also fully deterministic — so the toolkit runs with *zero*
third-party runtime dependencies beyond NumPy.
"""

from __future__ import annotations

import numpy as np

from equiforge.core.errors import ModelError


def available_sklearn() -> bool:
    """Return True if scikit-learn can be imported."""
    try:
        import sklearn  # noqa: F401

        return True
    except ImportError:  # pragma: no cover - environment dependent
        return False


class LogisticRegressionBackend:
    """A small, deterministic logistic-regression wrapper.

    Args:
        backend: ``"auto"`` (sklearn if present else numpy), ``"sklearn"`` or
            ``"numpy"``.
        random_state: Seed for the sklearn solver (no effect on the numpy path).
        max_iter: Maximum Newton / IRLS iterations.
        C: Inverse regularisation strength.
    """

    def __init__(
        self,
        backend: str = "auto",
        random_state: int = 0,
        max_iter: int = 2000,
        C: float = 1.0,
    ) -> None:
        if backend == "auto":
            backend = "sklearn" if available_sklearn() else "numpy"
        if backend not in ("sklearn", "numpy"):
            raise ModelError(f"unknown backend: {backend}")
        self.backend = backend
        self.random_state = random_state
        self.max_iter = max_iter
        self.C = float(C)
        self._w: np.ndarray | None = None
        self._b: float | None = None
        self._sk = None

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        if X.shape[0] == 0:
            raise ModelError("cannot fit on empty data")
        if self.backend == "sklearn":
            self._fit_sklearn(X, y, sample_weight)
        else:
            self._fit_numpy(X, y, sample_weight)
        return self

    def _fit_sklearn(self, X, y, sample_weight):
        from sklearn.linear_model import LogisticRegression

        sw = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        self._sk = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver="lbfgs",
            random_state=self.random_state,
        )
        self._sk.fit(X, y, sample_weight=sw)
        self._w = np.asarray(self._sk.coef_[0], dtype=np.float64)
        self._b = float(self._sk.intercept_[0])

    def _fit_numpy(self, X, y, sample_weight):
        n, d = X.shape
        lam = 1.0 / (self.C * n) if self.C > 0 else 0.0
        sw = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        sw = sw / sw.mean()  # keep weights centred so the regulariser is stable
        w = np.zeros(d, dtype=np.float64)
        b = 0.0
        for _ in range(self.max_iter):
            z = X @ w + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
            p = np.clip(p, 1e-12, 1.0 - 1e-12)
            r = p - y
            grad_w = (X * (r * sw)[:, None]).sum(axis=0) + lam * w
            grad_b = float((r * sw).sum())
            wp = p * (1.0 - p) * sw
            H = (X * wp[:, None]).T @ X + lam * np.eye(d)
            off = (wp[:, None] * X).sum(axis=0)
            Hfull = np.zeros((d + 1, d + 1), dtype=np.float64)
            Hfull[:d, :d] = H
            Hfull[:d, d] = off
            Hfull[d, :d] = off
            Hfull[d, d] = float(wp.sum())
            g = np.concatenate([grad_w, [grad_b]])
            try:
                delta = np.linalg.solve(Hfull, g)
            except np.linalg.LinAlgError:  # pragma: no cover - numerical safety
                delta = g  # gradient-descent fallback
            w = w - delta[:d]
            b = b - float(delta[d])
            if np.max(np.abs(delta)) < 1e-9:
                break
        self._w = w
        self._b = b

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self._w is None or self._b is None:
            raise ModelError("model not fitted")
        z = X @ self._w + self._b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
