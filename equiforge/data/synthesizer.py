"""Synthetic fairness data generators.

The generators build tabular binary-classification datasets that contain a
*protected* attribute ``s`` and a *merit* feature ``z`` that is the only
signal for the label. Disparity is introduced in two controllable ways:

* ``group_gap``  — the merit distribution is shifted between groups
  (a qualification / base-rate gap, the classic source of *spurious*
  statistical disparity).
* ``direct_effect`` — the label depends directly on the group, so even a
  perfectly merit-based decision cannot fully equalise error rates
  (a genuine fairness–accuracy trade-off).

Label noise (``label_noise``) makes the task imperfectly separable, which is
what makes equalised-odds correction cost real accuracy instead of being free.

All randomness is seeded; the same ``seed`` always yields the same dataset.
"""

from __future__ import annotations

import numpy as np

from equiforge.core.types import FairnessDataset


def make_dataset(
    name: str = "adult_like",
    n_samples: int = 4000,
    group_gap: float = 0.8,
    direct_effect: float = 0.4,
    label_noise: float = 0.8,
    p_s: float = 0.5,
    n_noise: int = 4,
    seed: int = 7,
) -> FairnessDataset:
    """Generate a single synthetic fairness dataset.

    Args:
        name: Dataset identifier.
        n_samples: Total number of samples.
        group_gap: Mean merit shift between the two groups (selection gap).
        direct_effect: Direct contribution of the group to the label logit.
        label_noise: Std-dev of additive logistic noise on the label.
        p_s: Prior probability of group ``s == 1``.
        n_noise: Number of uninformative Gaussian noise features.
        seed: RNG seed.

    Returns:
        A :class:`FairnessDataset`.
    """
    rng = np.random.default_rng(seed)
    s = (rng.random(n_samples) < p_s).astype(int)
    mu = np.where(s == 1, group_gap / 2.0, -group_gap / 2.0)
    z = rng.normal(mu, 1.0, size=n_samples)
    noise = rng.normal(0.0, 1.0, size=(n_samples, n_noise))
    X = np.column_stack([z, noise]).astype(np.float64)

    logit = z + direct_effect * s
    y = (logit + rng.normal(0.0, label_noise, size=n_samples) > 0.0).astype(int)

    feature_names = ["merit"] + [f"noise_{i}" for i in range(n_noise)]
    return FairnessDataset(X=X, s=s, y=y, feature_names=feature_names, name=name)


# Pre-tuned difficulty presets. Each is a *real* fairness scenario: a base-rate
# gap plus a mild direct effect, with imperfect separability. The "sweet spot"
# (baseline clearly below ceiling) was found by sweeping group_gap / direct_effect
# / label_noise so that Unconstrained shows a clear EO violation that the
# flagship can substantially reduce without destroying accuracy.
_PRESETS: dict[str, dict] = {
    "credit_like": {
        "group_gap": 0.9,
        "direct_effect": 0.35,
        "label_noise": 0.85,
        "p_s": 0.5,
    },
    "adult_like": {
        "group_gap": 0.7,
        "direct_effect": 0.35,
        "label_noise": 0.80,
        "p_s": 0.5,
    },
    "law_school_like": {
        "group_gap": 1.0,
        "direct_effect": 0.20,
        "label_noise": 0.70,
        "p_s": 0.5,
    },
}


def make_benchmark_suite(n_samples: int = 4000, seed: int = 7) -> list[FairnessDataset]:
    """Build the default multi-dataset benchmark suite.

    Args:
        n_samples: Samples per dataset.
        seed: Base seed; each dataset gets ``seed + offset`` for independence.

    Returns:
        A list of :class:`FairnessDataset` (one per preset).
    """
    suite: list[FairnessDataset] = []
    for offset, (key, preset) in enumerate(_PRESETS.items()):
        suite.append(
            make_dataset(
                name=key,
                n_samples=n_samples,
                seed=seed + offset * 101,
                **preset,
            )
        )
    return suite
