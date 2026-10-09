"""Invariant tests for the fairness metrics.

These encode the mathematical guarantees a reader can independently verify:
* Expected accuracy reduces to the conventional fraction-correct for hard
  predictions.
* A perfectly group-balanced assignment has zero DPD and zero EOD.
* The expectation formulation is exact for randomised predictions.
"""

from __future__ import annotations

import numpy as np

from equiforge.eval.metrics import (
    accuracy,
    demographic_parity_difference,
    equalized_odds_difference,
    evaluate,
)


def test_accuracy_hard_equals_fraction_correct():
    y = np.array([1, 0, 1, 0, 1, 0])
    pred = np.array([1, 0, 0, 0, 1, 1])  # 4/6 correct
    assert abs(accuracy(y, pred) - 4 / 6) < 1e-12


def test_accuracy_expected_equals_formula():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.2, 0.4, 0.7])  # expected correct = .9 + .8 + .6 + .3
    expected = np.mean(p * y + (1 - p) * (1 - y))
    assert abs(accuracy(y, p) - expected) < 1e-12


def test_dpd_zero_when_balanced():
    rng = np.random.default_rng(0)
    s = rng.integers(0, 2, size=500)
    y = rng.integers(0, 2, size=500)
    # A constant prediction gives identical selection rates across groups.
    pred = np.full(len(y), 0.5, dtype=float)
    assert demographic_parity_difference(y, pred, s) < 1e-9


def test_eod_zero_when_balanced():
    rng = np.random.default_rng(1)
    s = rng.integers(0, 2, size=500)
    y = rng.integers(0, 2, size=500)
    pred = np.full(len(y), 0.5, dtype=float)
    assert equalized_odds_difference(y, pred, s) < 1e-9


def test_eod_positive_when_groups_differ():
    y = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    s = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    pred = np.array([1, 1, 0, 0, 1, 1, 1, 0])  # group 1 has worse FPR
    assert equalized_odds_difference(y, pred, s) > 0


def test_evaluate_smoke_continuous():
    y = np.array([1, 0, 1, 0, 1, 0])
    s = np.array([0, 0, 1, 1, 0, 1])
    pred = np.array([0.8, 0.1, 0.6, 0.3, 0.9, 0.2])
    res = evaluate(y, pred, s, method="x")
    assert res.method == "x"
    assert 0.0 <= res.accuracy <= 1.0
    assert 0.0 <= res.equalized_odds_difference <= 1.0
    assert set(res.per_group.keys()) == {0, 1}
