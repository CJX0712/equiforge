"""Configuration with environment overrides and schema validation."""

from __future__ import annotations

import os
from dataclasses import dataclass

from equiforge.core.errors import ConfigError

_INT_KEYS = {"SEED", "N_SAMPLES", "N_JOBS"}
_FLOAT_KEYS = {"TEST_SIZE", "VAL_SIZE", "EO_TRADEOFF_C"}


@dataclass
class Config:
    """EquiForge runtime configuration.

    All fields can be overridden through ``EQUIFORGE_*`` environment variables
    (e.g. ``EQUIFORGE_SEED=42``).
    """

    seed: int = 7
    test_size: float = 0.30
    val_size: float = 0.20
    n_samples: int = 4000
    backend: str = "auto"  # auto | sklearn | numpy
    eo_tradeoff_c: float = 0.45  # target disparity reduction factor in postprocessing:
    #   the policy drives EOD/DPD down to <= eo_tradeoff_c * baseline disparity.
    #   Default 0.45 is *tighter* than the 0.5 pre-registered gate, giving a
    #   robust margin so the gate clears on held-out test splits.
    reweighing_enabled: bool = True
    n_jobs: int = 1

    @classmethod
    def from_env(cls) -> Config:
        cfg = cls()
        for key, value in os.environ.items():
            if not key.startswith("EQUIFORGE_"):
                continue
            name = key[len("EQUIFORGE_") :].lower()
            if key in _INT_KEYS:
                setattr(cfg, name, int(value))
            elif key in _FLOAT_KEYS:
                setattr(cfg, name, float(value))
            elif name == "backend":
                setattr(cfg, name, str(value))
            elif name == "reweighing_enabled":
                setattr(cfg, name, str(value).lower() in ("1", "true", "yes"))
        return cfg

    def validate(self) -> None:
        if not (0.0 < self.test_size < 1.0):
            raise ConfigError("test_size must be in (0, 1)")
        if not (0.0 < self.val_size < 1.0):
            raise ConfigError("val_size must be in (0, 1)")
        if self.test_size + self.val_size >= 1.0:
            raise ConfigError("test_size + val_size must be < 1.0")
        if self.backend not in ("auto", "sklearn", "numpy"):
            raise ConfigError(f"unknown backend: {self.backend}")
        if self.n_samples <= 0:
            raise ConfigError("n_samples must be positive")
