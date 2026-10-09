"""Determinism: two benchmark runs with the same configuration and seeds must
produce byte-identical per-cell metrics (excluding wall-clock time).
"""

from __future__ import annotations

from equiforge.core.config import Config
from equiforge.data.synthesizer import make_benchmark_suite
from equiforge.pipeline.pipeline import benchmark


def test_benchmark_bit_identical():
    cfg = Config()
    cfg.validate()
    suite = make_benchmark_suite(n_samples=2000, seed=cfg.seed)
    r1 = benchmark(suite, seeds=[7, 11, 23], config=cfg)
    r2 = benchmark(suite, seeds=[7, 11, 23], config=cfg)
    max_delta = 0.0
    for ds in r1["datasets"]:
        for m in r1["methods"]:
            for key in ("accuracy", "dpd", "eod"):
                a = r1["per_cell"][ds][m][key]
                b = r2["per_cell"][ds][m][key]
                max_delta = max(max_delta, max(abs(x - y) for x, y in zip(a, b)))
    assert max_delta == 0.0


def test_single_run_reproducible():
    cfg = Config()
    suite = make_benchmark_suite(n_samples=1500, seed=42)
    a = benchmark(suite, methods=["equifuse"], seeds=[7], config=cfg)
    b = benchmark(suite, methods=["equifuse"], seeds=[7], config=cfg)
    assert a["per_cell"]["credit_like"]["equifuse"]["eod"] == b["per_cell"]["credit_like"]["equifuse"]["eod"]
