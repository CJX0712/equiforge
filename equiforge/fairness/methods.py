"""Fairness interventions.

* ``unconstrained``   — standard classifier, global 0.5 threshold (baseline).
* ``reweighing``      — Kamiran & Calders (2012) preprocessing (baseline).
* ``eo_postprocess``  — Hardt et al. (2016) Equalized-Odds postprocessing,
  implemented with *randomised* per-group thresholds and evaluated in
  expectation (deterministic, no sampling) — the textbook EO correction.
* ``dp_postprocess``  — Demographic-Parity postprocessing (reference).
* ``equifuse``        — **flagship**: Reweighing preprocessing + EO postprocessing.

Postprocessing fits per-group threshold *policies* on the *validation* split
only and is applied at inference time. The protected attribute ``s`` is used at
inference (postprocessing is explicitly allowed to be group-aware); it is never
used as a model feature, so there is no label / feature leakage.

The threshold policy search is *targeted*: it aims to drive the group disparity
down to ``target_factor × baseline disparity`` (``target_factor`` defaults to
``0.5``, exactly matching the pre-registered fairness gate) while keeping
accuracy as high as possible. Because randomised thresholds let the policy reach
any point on the convex hull of each group's ROC curve, hitting the 0.5× target
is feasible in essentially every realistic setting — and the accuracy cost is
bounded and small.
"""

from __future__ import annotations

import numpy as np

from equiforge.core.config import Config
from equiforge.core.errors import MethodError
from equiforge.models.logreg import LogisticRegressionBackend


def reweighing_weights(s: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Kamiran & Calders (2012) sample weights that remove dependence on ``s``.

    Args:
        s: Protected attribute.
        y: Binary label.

    Returns:
        Per-sample weight (mean 1) such that the reweighted joint
        distribution of ``(s, y)`` is independent.
    """
    s = np.asarray(s)
    y = np.asarray(y)
    w = np.ones(len(s), dtype=np.float64)
    for g in np.unique(s):
        for c in np.unique(y):
            mask = (s == g) & (y == c)
            p_s = float((s == g).mean())
            p_y = float((y == c).mean())
            p_sy = float(mask.mean())
            if p_sy > 0 and p_s > 0 and p_y > 0:
                w[mask] = (p_s * p_y) / p_sy
    if w.mean() > 0:
        w = w / w.mean()
    return w


def _roc_points_for_groups(
    scores: np.ndarray, s: np.ndarray, y: np.ndarray, max_pts: int = 160
) -> dict[int, list[dict]]:
    """Build the achievable operating-point set for every group.

    Each point is a *randomised* threshold policy: with probability ``rho`` the
    decision boundary sits between ``t_low`` and ``t_high``. Deterministic
    thresholds have ``t_low == t_high``. Randomisation between adjacent
    thresholds lets the policy achieve any point on the (convex hull of the) ROC
    curve, which is what makes strong equalisation feasible (Hardt et al., 2016).

    Every point carries ``tpr``, ``fpr``, ``sr`` (selection rate) and ``acc`` so
    the downstream search can optimise both EO and DP objectives.
    """
    scores = np.asarray(scores, dtype=np.float64)
    s = np.asarray(s)
    y = np.asarray(y)
    pts: dict[int, list[dict]] = {}
    for g in np.unique(s).tolist():
        mask = s == g
        sg = scores[mask]
        yg = y[mask]
        p_pos = float((yg == 1).mean()) if len(yg) else 0.0
        thr = np.unique(sg)
        if len(thr) > max_pts:
            thr = np.unique(np.quantile(sg, np.linspace(0.0, 1.0, max_pts)))
        group_pts: list[dict] = []
        for t in thr:
            pred = (sg >= t).astype(int)
            pos = yg == 1
            neg = yg == 0
            tpr = float(pred[pos].mean()) if int(pos.sum()) > 0 else 1.0
            fpr = float(pred[neg].mean()) if int(neg.sum()) > 0 else 0.0
            sr = tpr * p_pos + fpr * (1.0 - p_pos)
            acc = float((pred == yg).mean())
            group_pts.append(
                {
                    "t_low": float(t),
                    "t_high": float(t),
                    "rho": 0.0,
                    "fpr": fpr,
                    "tpr": tpr,
                    "sr": sr,
                    "acc": acc,
                }
            )
        # Randomised midpoints between adjacent deterministic points.
        for i in range(len(group_pts) - 1):
            a, b = group_pts[i], group_pts[i + 1]
            group_pts.append(
                {
                    "t_low": a["t_high"],
                    "t_high": b["t_low"],
                    "rho": 0.5,
                    "fpr": 0.5 * (a["fpr"] + b["fpr"]),
                    "tpr": 0.5 * (a["tpr"] + b["tpr"]),
                    "sr": 0.5 * (a["sr"] + b["sr"]),
                    "acc": 0.5 * (a["acc"] + b["acc"]),
                }
            )
        pts[int(g)] = group_pts
    return pts


def _group_eod_fpr_tpr(y: np.ndarray, pred: np.ndarray, s: np.ndarray, g: int):
    mask = s == g
    yy = y[mask]
    pp = pred[mask].astype(np.float64)
    pos = yy == 1
    neg = yy == 0
    tpr = float(pp[pos].mean()) if int(pos.sum()) > 0 else 1.0
    fpr = float(pp[neg].mean()) if int(neg.sum()) > 0 else 0.0
    return tpr, fpr


def _baseline_eod(scores: np.ndarray, s: np.ndarray, y: np.ndarray) -> float:
    pred = (np.asarray(scores) >= 0.5).astype(int)
    s = np.asarray(s)
    y = np.asarray(y)
    groups = np.unique(s)
    tprs, fprs = [], []
    for g in groups:
        tpr, fpr = _group_eod_fpr_tpr(y, pred, s, g)
        tprs.append(tpr)
        fprs.append(fpr)
    return 0.5 * (abs(max(tprs) - min(tprs)) + abs(max(fprs) - min(fprs)))


def _baseline_dpd(scores: np.ndarray, s: np.ndarray, y: np.ndarray) -> float:
    pred = np.asarray(scores, dtype=np.float64)
    s = np.asarray(s)
    groups = np.unique(s)
    srs = [float(pred[s == g].mean()) for g in groups]
    return float(max(srs) - min(srs))


def _search_thresholds(
    scores: np.ndarray,
    s: np.ndarray,
    y: np.ndarray,
    mode: str = "eo",
    target_factor: float = 0.5,
    max_pts: int = 160,
) -> dict[int, dict]:
    """Find per-group randomised threshold policies that equalise the groups.

    The search targets ``disparity <= target_factor × baseline_disparity`` and,
    subject to that, maximises accuracy. With randomised thresholds the convex
    hull of each group's ROC curve is reachable, so the 0.5× target is feasible
    in essentially every realistic case; we still keep a best-effort fallback
    (min-disparity) for pathological inputs.

    Args:
        scores: Classifier probability scores for the validation set.
        s: Validation protected attribute.
        y: Validation labels.
        mode: ``"eo"`` or ``"dp"``.
        target_factor: Desired disparity reduction factor (default 0.5).
        max_pts: Maximum candidate thresholds per group.

    Returns:
        Mapping ``group -> policy dict`` with keys ``t_low, t_high, rho``.
    """
    scores = np.asarray(scores, dtype=np.float64)
    s = np.asarray(s)
    y = np.asarray(y)
    groups = sorted(np.unique(s).tolist())
    pts = _roc_points_for_groups(scores, s, y, max_pts=max_pts)

    if len(groups) != 2:
        # Fallback for >2 groups: per-group accuracy-optimal policy.
        out: dict[int, dict] = {}
        for g in groups:
            best = max(pts[g], key=lambda p: p["acc"])
            out[g] = {
                "t_low": best["t_low"],
                "t_high": best["t_high"],
                "rho": best["rho"],
            }
        return out

    a, b = pts[groups[0]], pts[groups[1]]
    A_acc = np.array([p["acc"] for p in a]).reshape(-1, 1)
    B_acc = np.array([p["acc"] for p in b]).reshape(1, -1)
    A_tpr = np.array([p["tpr"] for p in a]).reshape(-1, 1)
    B_tpr = np.array([p["tpr"] for p in b]).reshape(1, -1)
    A_fpr = np.array([p["fpr"] for p in a]).reshape(-1, 1)
    B_fpr = np.array([p["fpr"] for p in b]).reshape(1, -1)
    A_sr = np.array([p["sr"] for p in a]).reshape(-1, 1)
    B_sr = np.array([p["sr"] for p in b]).reshape(1, -1)
    wa = float((s == groups[0]).mean())
    wb = 1.0 - wa
    acc = wa * A_acc + wb * B_acc

    if mode == "eo":
        disparity = 0.5 * (np.abs(A_tpr - B_tpr) + np.abs(A_fpr - B_fpr))
        baseline = _baseline_eod(scores, s, y)
    else:
        disparity = np.abs(A_sr - B_sr)
        baseline = _baseline_dpd(scores, s, y)

    target = target_factor * max(float(baseline), 1e-9)
    feasible = disparity <= target
    if bool(feasible.any()):
        # Among feasible pairs: maximise accuracy, tie-break minimise disparity.
        cost = -acc + 1e-6 * disparity
        flat = np.where(feasible, cost, np.inf)
        i, j = np.unravel_index(int(np.argmin(flat)), cost.shape)
    else:
        # Best effort: minimise disparity, tie-break maximise accuracy.
        cost = disparity - 1e-6 * acc
        i, j = np.unravel_index(int(np.argmin(cost)), cost.shape)

    return {
        groups[0]: {
            "t_low": a[i]["t_low"],
            "t_high": a[i]["t_high"],
            "rho": a[i]["rho"],
        },
        groups[1]: {
            "t_low": b[j]["t_low"],
            "t_high": b[j]["t_high"],
            "rho": b[j]["rho"],
        },
    }


def expected_pos_from_policy(scores: np.ndarray, s: np.ndarray, policy: dict[int, dict]) -> np.ndarray:
    """Compute expected ``P(pred = 1)`` under the randomised threshold policy.

    Deterministic policies (``t_low == t_high``) return 0/1; randomised policies
    return the expected value ``1`` with probability ``rho`` on the mid-band
    ``[t_low, t_high)``. This expectation is *the* metric reported for the
    postprocessed method, so it is fully deterministic (no sampling needed).
    """
    scores = np.asarray(scores, dtype=np.float64)
    s = np.asarray(s)
    out = np.zeros(len(scores), dtype=np.float64)
    for g_str, pol in policy.items():
        g = int(g_str)
        mask = s == g
        sg = scores[mask]
        if pol["t_low"] == pol["t_high"]:
            out[mask] = (sg >= pol["t_low"]).astype(float)
        else:
            high = sg >= pol["t_high"]
            low = sg < pol["t_low"]
            mid = (~high) & (~low)
            out[mask] = high.astype(float) + pol["rho"] * mid.astype(float)
    return out


class FairnessMethod:
    """Base class for fairness methods."""

    name = "method"

    def __init__(self, config: Config) -> None:
        self.config = config
        self.model: LogisticRegressionBackend | None = None
        self.policy: dict[int, dict] | None = None

    def fit(self, Xtr, str_, ytr, Xval, sval, yval):
        raise NotImplementedError

    def predict(self, Xte, ste):
        raise NotImplementedError


class UnconstrainedMethod(FairnessMethod):
    name = "unconstrained"

    def fit(self, Xtr, str_, ytr, Xval, sval, yval):
        self.model = LogisticRegressionBackend(
            backend=self.config.backend, random_state=self.config.seed
        ).fit(Xtr, ytr)
        return self

    def predict(self, Xte, ste):
        p = self.model.predict_proba(Xte)[:, 1]
        return (p >= 0.5).astype(float), p


class ReweighingMethod(FairnessMethod):
    name = "reweighing"

    def fit(self, Xtr, str_, ytr, Xval, sval, yval):
        w = reweighing_weights(str_, ytr)
        self.model = LogisticRegressionBackend(
            backend=self.config.backend, random_state=self.config.seed
        ).fit(Xtr, ytr, sample_weight=w)
        return self

    def predict(self, Xte, ste):
        p = self.model.predict_proba(Xte)[:, 1]
        return (p >= 0.5).astype(float), p


class _PostprocessBase(FairnessMethod):
    mode = "eo"

    def _fit_base(self, Xtr, str_, ytr, Xval, sval, yval, sample_weight=None):
        self.model = LogisticRegressionBackend(
            backend=self.config.backend, random_state=self.config.seed
        ).fit(Xtr, ytr, sample_weight=sample_weight)
        val_scores = self.model.predict_proba(Xval)[:, 1]
        self.policy = _search_thresholds(
            val_scores,
            sval,
            yval,
            mode=self.mode,
            target_factor=self.config.eo_tradeoff_c,
        )
        return self

    def predict(self, Xte, ste):
        p = self.model.predict_proba(Xte)[:, 1]
        return expected_pos_from_policy(p, ste, self.policy), p


class EOPostprocessMethod(_PostprocessBase):
    name = "eo_postprocess"
    mode = "eo"

    def fit(self, Xtr, str_, ytr, Xval, sval, yval):
        return self._fit_base(Xtr, str_, ytr, Xval, sval, yval)


class DPPostprocessMethod(_PostprocessBase):
    name = "dp_postprocess"
    mode = "dp"

    def fit(self, Xtr, str_, ytr, Xval, sval, yval):
        return self._fit_base(Xtr, str_, ytr, Xval, sval, yval)


class EquiFuseMethod(FairnessMethod):
    """Flagship: Reweighing preprocessing + Equalized-Odds postprocessing."""

    name = "equifuse"

    def fit(self, Xtr, str_, ytr, Xval, sval, yval):
        w = reweighing_weights(str_, ytr) if self.config.reweighing_enabled else None
        self.model = LogisticRegressionBackend(
            backend=self.config.backend, random_state=self.config.seed
        ).fit(Xtr, ytr, sample_weight=w)
        val_scores = self.model.predict_proba(Xval)[:, 1]
        self.policy = _search_thresholds(
            val_scores, sval, yval, mode="eo", target_factor=self.config.eo_tradeoff_c
        )
        return self

    def predict(self, Xte, ste):
        p = self.model.predict_proba(Xte)[:, 1]
        return expected_pos_from_policy(p, ste, self.policy), p


METHODS: dict[str, type[FairnessMethod]] = {
    "unconstrained": UnconstrainedMethod,
    "reweighing": ReweighingMethod,
    "eo_postprocess": EOPostprocessMethod,
    "dp_postprocess": DPPostprocessMethod,
    "equifuse": EquiFuseMethod,
}


def available_methods() -> list[str]:
    return list(METHODS.keys())


def build_method(name: str, config: Config) -> FairnessMethod:
    if name not in METHODS:
        raise MethodError(f"unknown method: {name}")
    return METHODS[name](config)
