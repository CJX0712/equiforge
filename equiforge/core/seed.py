"""Global deterministic seeding for EquiForge.

All stochasticity in EquiForge flows from a single integer seed. We seed the
legacy ``np.random`` state, the ``random`` module, and expose a modern
``np.random.Generator`` for data synthesis so that two runs with the same seed
produce byte-identical benchmark outputs (except wall-clock time).
"""

from __future__ import annotations

import os
import random

import numpy as np

_SEEDED = False


def set_all(seed: int = 7) -> np.random.Generator:
    """Seed every RNG source deterministically and return a numpy Generator.

    Args:
        seed: Non-negative integer seed.

    Returns:
        A ``np.random.Generator`` initialised with ``seed``.
    """
    seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed % (2**32))
    globals()["_SEEDED"] = True
    return np.random.default_rng(seed)
