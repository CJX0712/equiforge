"""Invariant tests for the Kamiran & Calders (2012) reweighing preprocessing.

The reweighing weights must (1) be centred to mean 1 and (2) make the
reweighted joint distribution of ``(s, y)`` statistically independent, i.e.
``P_w(s, y) ≈ P_w(s) · P_w(y)`` for every cell. We verify the latter up to a
small tolerance.
"""

from __future__ import annotations

import numpy as np

from equiforge.fairness.methods import reweighing_weights


def _weighted_joint_independence(s, y, w, tol=1e-6):
    w = w / w.sum()
    cells = {}
    for g in np.unique(s):
        for c in np.unique(y):
            m = (s == g) & (y == c)
            cells[(int(g), int(c))] = float(w[m].sum())
    p_s = {int(g): float(w[s == g].sum()) for g in np.unique(s)}
    p_y = {int(c): float(w[y == c].sum()) for c in np.unique(y)}
    worst = 0.0
    for (g, c), p_sy in cells.items():
        worst = max(worst, abs(p_sy - p_s[g] * p_y[c]))
    return worst, cells, p_s, p_y


def test_weights_mean_one():
    rng = np.random.default_rng(3)
    s = rng.integers(0, 2, size=2000)
    y = rng.integers(0, 2, size=2000)
    w = reweighing_weights(s, y)
    assert abs(w.mean() - 1.0) < 1e-9


def test_reweighted_independence():
    rng = np.random.default_rng(4)
    s = rng.integers(0, 2, size=4000)
    y = rng.integers(0, 2, size=4000)
    w = reweighing_weights(s, y)
    worst, _, _, _ = _weighted_joint_independence(s, y, w)
    assert worst < 1e-9


def test_unweighted_not_independent_when_correlated():
    # When s and y are correlated, the unweighted joint is NOT independent,
    # so reweighing must actually change the weights away from 1.
    rng = np.random.default_rng(5)
    s = rng.integers(0, 2, size=4000)
    y = ((s == 1) & (rng.random(4000) < 0.8)) | ((s == 0) & (rng.random(4000) < 0.2))
    y = y.astype(int)
    w = reweighing_weights(s, y)
    assert np.std(w) > 1e-3  # weights are non-trivial
    worst_unweighted, _, _, _ = _weighted_joint_independence(s, y, np.ones(len(s)))
    worst_weighted, _, _, _ = _weighted_joint_independence(s, y, w)
    assert worst_weighted < worst_unweighted
