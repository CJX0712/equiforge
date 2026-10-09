"""End-to-end gate and flagship-behaviour tests.

* The pre-registered fairness gate (per dataset: EOD <= 0.5 * Unconstrained EOD
  AND ACC >= Unconstrained ACC - 0.04) must pass for the flagship ``equifuse``.
* The flagship must reduce EOD relative to the unconstrained baseline on every
  dataset.
"""

from __future__ import annotations

from equiforge.core.config import Config
from equiforge.data.synthesizer import make_benchmark_suite
from equiforge.pipeline.pipeline import benchmark


def test_gate_passes():
    cfg = Config()
    cfg.validate()
    suite = make_benchmark_suite(n_samples=4000, seed=cfg.seed)
    res = benchmark(suite, seeds=[7, 11, 23], config=cfg)
    assert res["gate"]["passed"] is True
    for ds, v in res["gate"]["per_dataset"].items():
        assert v["passed"] is True, f"gate failed on {ds}: {v}"


def test_equifuse_reduces_eod_everywhere():
    cfg = Config()
    suite = make_benchmark_suite(n_samples=4000, seed=cfg.seed)
    res = benchmark(suite, methods=["unconstrained", "equifuse"], seeds=[7, 11, 23], config=cfg)
    for ds in res["datasets"]:
        unc = res["per_dataset"]["unconstrained"][ds]["eod"]
        fus = res["per_dataset"]["equifuse"][ds]["eod"]
        assert fus < unc, f"equifuse did not reduce EOD on {ds} ({unc} -> {fus})"


def test_equifuse_accuracy_within_floor():
    cfg = Config()
    suite = make_benchmark_suite(n_samples=4000, seed=cfg.seed)
    res = benchmark(suite, methods=["unconstrained", "equifuse"], seeds=[7, 11, 23], config=cfg)
    for ds in res["datasets"]:
        unc = res["per_dataset"]["unconstrained"][ds]["accuracy"]
        fus = res["per_dataset"]["equifuse"][ds]["accuracy"]
        assert fus >= unc - 0.04, f"accuracy floor breached on {ds}"
