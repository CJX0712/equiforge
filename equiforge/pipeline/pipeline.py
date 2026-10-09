"""End-to-end fairness pipeline and benchmark harness.

The pipeline wires data -> model -> fairness method -> evaluation in a single
direction with no cycles::

    run_one  :  data -> method.fit -> method.predict -> evaluate
    benchmark:  repeat run_one over a suite x methods x seeds

Determinism: a fixed ``seed`` reproduces identical metrics (wall-clock time
excluded). The benchmark aggregates over >=3 seeds and reports ``mean +/- std``.
"""

from __future__ import annotations

import numpy as np

from equiforge.core.config import Config
from equiforge.core.seed import set_all
from equiforge.core.types import FairnessDataset, FairnessResult
from equiforge.eval.metrics import evaluate
from equiforge.fairness.methods import available_methods, build_method


def split_dataset(ds: FairnessDataset, config: Config, seed: int):
    """Deterministic train/val/test split (no stratification needed for the
    synthetic generator, but seeded for reproducibility)."""
    rng = np.random.default_rng(seed)
    n = len(ds.y)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = round(n * config.test_size)
    n_val = round(n * config.val_size)
    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]
    return train_idx, val_idx, test_idx


def run_one(ds: FairnessDataset, method_name: str, config: Config, seed: int) -> FairnessResult:
    """Run a single method on a single dataset at a single seed."""
    set_all(seed)  # keeps any legacy RNG use deterministic per run
    train_idx, val_idx, test_idx = split_dataset(ds, config, seed)
    Xtr, str_, ytr = ds.X[train_idx], ds.s[train_idx], ds.y[train_idx]
    Xval, sval, yval = ds.X[val_idx], ds.s[val_idx], ds.y[val_idx]
    Xte, ste, yte = ds.X[test_idx], ds.s[test_idx], ds.y[test_idx]

    method = build_method(method_name, config)
    method.fit(Xtr, str_, ytr, Xval, sval, yval)
    pred, _ = method.predict(Xte, ste)
    return evaluate(yte, pred, ste, method=method_name)


def _mean_std(values: list[float]):
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def benchmark(
    suite: list[FairnessDataset],
    methods: list[str] | None = None,
    seeds: list[int] | None = None,
    config: Config | None = None,
) -> dict:
    """Run the full benchmark.

    Args:
        suite: List of datasets.
        methods: Method names (defaults to all available).
        seeds: RNG seeds (defaults to ``[7, 11, 23]``).
        config: Runtime config.

    Returns:
        A nested dict with per-(dataset, method) ``mean +/- std`` metrics plus a
        fairness gate evaluation for the flagship.
    """
    config = config or Config()
    config.validate()
    methods = methods or available_methods()
    seeds = seeds or [7, 11, 23]

    per_cell: dict[str, dict[str, dict[str, list[float]]]] = {}
    for ds in suite:
        per_cell[ds.name] = {}
        for m in methods:
            acc, dpd, eod = [], [], []
            for sd in seeds:
                res = run_one(ds, m, config, sd)
                acc.append(res.accuracy)
                dpd.append(res.demographic_parity_difference)
                eod.append(res.equalized_odds_difference)
            per_cell[ds.name][m] = {
                "accuracy": acc,
                "dpd": dpd,
                "eod": eod,
            }

    # Per-dataset aggregate (mean +/- std over seeds) and cross-dataset mean.
    per_dataset: dict[str, dict[str, dict[str, float]]] = {}
    agg: dict[str, dict[str, float]] = {}
    for m in methods:
        acc_all, dpd_all, eod_all = [], [], []
        per_dataset[m] = {}
        for ds in suite:
            a_m, a_s = _mean_std(per_cell[ds.name][m]["accuracy"])
            d_m, d_s = _mean_std(per_cell[ds.name][m]["dpd"])
            e_m, e_s = _mean_std(per_cell[ds.name][m]["eod"])
            per_dataset[m][ds.name] = {
                "accuracy": a_m,
                "accuracy_std": a_s,
                "dpd": d_m,
                "dpd_std": d_s,
                "eod": e_m,
                "eod_std": e_s,
            }
            acc_all.append(a_m)
            dpd_all.append(d_m)
            eod_all.append(e_m)
        agg[m] = {
            "accuracy": float(np.mean(acc_all)),
            "dpd": float(np.mean(dpd_all)),
            "eod": float(np.mean(eod_all)),
        }

    # Gate (per-dataset, all must hold): flagship EOD <= 0.5 * Unconstrained EOD
    # AND flagship accuracy >= Unconstrained accuracy - 0.04.
    gate = _evaluate_gate(per_dataset)
    return {
        "config": {
            "seed": config.seed,
            "backend": config.backend,
            "test_size": config.test_size,
            "val_size": config.val_size,
            "eo_tradeoff_c": config.eo_tradeoff_c,
        },
        "seeds": list(seeds),
        "datasets": [ds.name for ds in suite],
        "methods": list(methods),
        "per_cell": per_cell,
        "per_dataset": per_dataset,
        "aggregate": agg,
        "gate": gate,
    }


def _evaluate_gate(per_dataset: dict[str, dict[str, dict[str, float]]]) -> dict:
    unc = per_dataset.get("unconstrained")
    flag = per_dataset.get("equifuse")
    if unc is None or flag is None:
        return {"passed": False, "reason": "missing baseline or flagship"}
    per_ds = {}
    all_pass = True
    for ds in unc:
        u_eod = unc[ds]["eod"]
        f_eod = flag[ds]["eod"]
        u_acc = unc[ds]["accuracy"]
        f_acc = flag[ds]["accuracy"]
        eod_pass = f_eod <= 0.5 * max(u_eod, 1e-9)
        acc_pass = f_acc >= u_acc - 0.04
        ok = bool(eod_pass and acc_pass)
        all_pass = all_pass and ok
        per_ds[ds] = {
            "eod_pass": eod_pass,
            "acc_pass": acc_pass,
            "passed": ok,
            "unconstrained_eod": u_eod,
            "equifuse_eod": f_eod,
            "unconstrained_acc": u_acc,
            "equifuse_acc": f_acc,
        }
    return {
        "passed": bool(all_pass),
        "per_dataset": per_ds,
        "eod_ratio_aggregate": float(
            np.mean([v["equifuse_eod"] / max(v["unconstrained_eod"], 1e-9) for v in per_ds.values()])
        ),
        "unconstrained_eod": float(np.mean([v["unconstrained_eod"] for v in per_ds.values()])),
        "equifuse_eod": float(np.mean([v["equifuse_eod"] for v in per_ds.values()])),
        "unconstrained_acc": float(np.mean([v["unconstrained_acc"] for v in per_ds.values()])),
        "equifuse_acc": float(np.mean([v["equifuse_acc"] for v in per_ds.values()])),
    }
