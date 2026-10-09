"""Shared dataclasses for EquiForge."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FairnessDataset:
    """A binary-classification dataset with a protected attribute.

    Attributes:
        X: Feature matrix ``(n_samples, n_features)``.
        s: Protected / sensitive attribute, integer-coded groups.
        y: Binary ground-truth label (0/1).
        feature_names: Optional human-readable feature names.
        name: Identifier used in reports.
    """

    X: np.ndarray
    s: np.ndarray
    y: np.ndarray
    feature_names: list[str] = field(default_factory=list)
    name: str = "dataset"


@dataclass
class GroupMetrics:
    """Per-group classification statistics."""

    accuracy: float
    selection_rate: float
    tpr: float
    fpr: float


@dataclass
class FairnessResult:
    """Result of evaluating one fairness method on one dataset split."""

    method: str
    accuracy: float
    selection_rate_overall: float
    demographic_parity_difference: float
    equalized_odds_difference: float
    per_group: dict[int, GroupMetrics] = field(default_factory=dict)
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "accuracy": self.accuracy,
            "selection_rate_overall": self.selection_rate_overall,
            "demographic_parity_difference": self.demographic_parity_difference,
            "equalized_odds_difference": self.equalized_odds_difference,
            "per_group": {
                str(g): {
                    "accuracy": gm.accuracy,
                    "selection_rate": gm.selection_rate,
                    "tpr": gm.tpr,
                    "fpr": gm.fpr,
                }
                for g, gm in self.per_group.items()
            },
            "notes": self.notes,
        }
