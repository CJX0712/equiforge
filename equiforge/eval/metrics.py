"""Group-fairness metrics.

All metrics are computed *exactly* (in expectation) on the supplied (test) set
— no sampling — so results are deterministic. Convention: higher disparity is
worse, higher accuracy is better.

* Demographic Parity Difference (DPD): gap in selection rates across groups.
* Equalized Odds Difference (EOD): average of the TPR gap and FPR gap across
  groups. This is the headline fairness metric and the one the flagship targets.

Predictions may be either hard binary labels (``0/1``) or *continuous expected*
probabilities ``P(pred = 1)`` produced by the randomised-threshold postprocessor
(Hardt et al., 2016). All metrics are written so that they evaluate correctly
under expectation:

* ``accuracy``  is the *expected* accuracy
  ``E[1(pred = y)] = mean(p·y + (1-p)·(1-y))``.
* ``tpr`` / ``fpr`` / ``selection_rate`` are the expected positive-selection
  rates within the positive / negative / full subgroups, i.e. ``mean(p)`` over
  the respective mask. For hard predictions these reduce to the standard counts.

This single formulation keeps the unconstrained baseline (hard 0/1) and the
postprocessed method (continuous) directly comparable and bit-identical across
runs.
"""

from __future__ import annotations

import numpy as np

from equiforge.core.types import FairnessResult, GroupMetrics


def _group_rates(y_true: np.ndarray, pred: np.ndarray, s: np.ndarray, g: int):
    mask = s == g
    if int(mask.sum()) == 0:
        return 0.0, 0.0, 0.0, 0.0
    y = y_true[mask]
    p = pred[mask].astype(np.float64)
    pos = y == 1
    neg = y == 0
    tpr = float(p[pos].mean()) if int(pos.sum()) > 0 else 1.0
    fpr = float(p[neg].mean()) if int(neg.sum()) > 0 else 0.0
    sr = float(p.mean())
    # Expected accuracy for this group: E[1(pred=y)] = p·y + (1-p)·(1-y) averaged.
    acc = float(np.mean(p * y + (1.0 - p) * (1.0 - y)))
    return sr, tpr, fpr, acc


def accuracy(y_true: np.ndarray, pred: np.ndarray) -> float:
    """Expected accuracy under the (possibly randomised) prediction policy.

    For hard predictions this is the usual fraction correct; for continuous
    expected probabilities it is the expectation of the 0/1 loss, which is the
    deterministic, sampling-free quantity reported throughout EquiForge.
    """
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    return float(np.mean(p * y + (1.0 - p) * (1.0 - y)))


def selection_rate_overall(pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(pred, dtype=np.float64)))


def demographic_parity_difference(y_true, pred, s) -> float:
    s = np.asarray(s)
    pred = np.asarray(pred, dtype=np.float64)
    groups = np.unique(s)
    if len(groups) < 2:
        return 0.0
    srs = [_group_rates(y_true, pred, s, g)[0] for g in groups]
    return float(max(srs) - min(srs))


def equalized_odds_difference(y_true, pred, s) -> float:
    s = np.asarray(s)
    pred = np.asarray(pred, dtype=np.float64)
    groups = np.unique(s)
    if len(groups) < 2:
        return 0.0
    tprs = [_group_rates(y_true, pred, s, g)[1] for g in groups]
    fprs = [_group_rates(y_true, pred, s, g)[2] for g in groups]
    return 0.5 * (abs(max(tprs) - min(tprs)) + abs(max(fprs) - min(fprs)))


def evaluate(y_true, pred, s, method: str = "", notes: str = "") -> FairnessResult:
    """Compute a full :class:`FairnessResult` for one prediction array.

    ``pred`` may be hard 0/1 labels or continuous expected probabilities.
    """
    y_true = np.asarray(y_true)
    pred = np.asarray(pred, dtype=np.float64)
    s = np.asarray(s)
    per_group: dict[int, GroupMetrics] = {}
    for g in np.unique(s):
        sr, tpr, fpr, acc = _group_rates(y_true, pred, s, int(g))
        per_group[int(g)] = GroupMetrics(accuracy=acc, selection_rate=sr, tpr=tpr, fpr=fpr)
    return FairnessResult(
        method=method,
        accuracy=accuracy(y_true, pred),
        selection_rate_overall=selection_rate_overall(pred),
        demographic_parity_difference=demographic_parity_difference(y_true, pred, s),
        equalized_odds_difference=equalized_odds_difference(y_true, pred, s),
        per_group=per_group,
        notes=notes,
    )
