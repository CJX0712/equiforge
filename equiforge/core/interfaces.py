"""Lightweight protocols for classifiers and fairness methods."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class BaseClassifier(Protocol):
    """Minimal classifier contract used by fairness methods."""

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None) -> BaseClassifier: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...
