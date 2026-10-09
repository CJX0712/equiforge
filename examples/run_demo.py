"""EquiForge end-to-end demo.

Runs the full benchmark, writes ``benchmark.json``, verifies determinism
(byte-identical metrics across two runs) and emits a small set of failure-case
examples and an ablation summary. This script is also used as the CI smoke test.

Run:
    python examples/run_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console safety
except Exception:  # noqa: S110  # pragma: no cover - best-effort
    pass

from equiforge.core.config import Config
from equiforge.core.seed import set_all
from equiforge.core.types import FairnessDataset
from equiforge.data.synthesizer import make_benchmark_suite
from equiforge.fairness.methods import build_method
from equiforge.pipeline.pipeline import (
    benchmark,
    split_dataset,
)

SEEDS = [7, 11, 23]


def _format_result(result: dict) -> None:
    methods = result["methods"]
    print("\n=== Per-dataset mean +/- std accuracy (across seeds) ===")
    print(f"{'dataset':<16}" + "".join(f"{m:>17}" for m in methods))
    for ds in result["datasets"]:
        row = f"{ds:<16}"
        for m in methods:
            v = result["per_dataset"][m][ds]
            row += f"{v['accuracy']:>11.4f}±{v['accuracy_std']:<5.3f}"
        print(row)

    print("\n=== Per-dataset mean EOD (across seeds) — lower better ===")
    print(f"{'dataset':<16}" + "".join(f"{m:>17}" for m in methods))
    for ds in result["datasets"]:
        row = f"{ds:<16}"
        for m in methods:
            v = result["per_dataset"][m][ds]
            row += f"{v['eod']:>11.4f}±{v['eod_std']:<5.3f}"
        print(row)

    print("\n=== Aggregate mean (across datasets) — lower EOD better ===")
    print(f"{'method':<16}{'ACC':>12}{'DPD':>12}{'EOD':>12}")
    for m in methods:
        v = result["aggregate"][m]
        print(f"{m:<16}{v['accuracy']:>12.4f}{v['dpd']:>12.4f}{v['eod']:>12.4f}")

    gate = result["gate"]
    print("\n=== Fairness gate (pre-registered, per-dataset, all must pass) ===")
    print(f"  aggregate passed : {gate['passed']}")
    print(f"  EOD ratio (flag/baseline) aggregate = {gate['eod_ratio_aggregate']:.3f}  (threshold <= 0.5)")
    for ds, v in gate.get("per_dataset", {}).items():
        print(
            f"   - {ds:<16} EOD {v['unconstrained_eod']:.4f}->{v['equifuse_eod']:.4f} "
            f"ACC {v['unconstrained_acc']:.4f}->{v['equifuse_acc']:.4f}  passed={v['passed']}"
        )


def _determinism_check(config: Config, suite) -> bool:
    print("\n=== Determinism check (two identical runs) ===")
    r1 = benchmark(suite, seeds=SEEDS, config=config)
    r2 = benchmark(suite, seeds=SEEDS, config=config)
    max_delta = 0.0
    for ds in r1["datasets"]:
        for m in r1["methods"]:
            for key in ("accuracy", "dpd", "eod"):
                a = r1["per_cell"][ds][m][key]
                b = r2["per_cell"][ds][m][key]
                max_delta = max(max_delta, max(abs(x - y) for x, y in zip(a, b)))
    print(f"  max |metric difference| across two runs = {max_delta:.2e}")
    ok = max_delta == 0.0
    print(f"  bit-identical (excluding elapsed): {ok}")
    return ok


def _failure_cases(config: Config, ds: FairnessDataset, n: int = 5) -> list[dict]:
    """Compare Unconstrained vs EquiFuse on a fixed seed to surface the trade-off."""
    seed = 7
    train_idx, val_idx, test_idx = split_dataset(ds, config, seed)
    Xtr, str_, ytr = ds.X[train_idx], ds.s[train_idx], ds.y[train_idx]
    Xval, sval, yval = ds.X[val_idx], ds.s[val_idx], ds.y[val_idx]
    Xte, ste, yte = ds.X[test_idx], ds.s[test_idx], ds.y[test_idx]

    unc = build_method("unconstrained", config).fit(Xtr, str_, ytr, Xval, sval, yval)
    fus = build_method("equifuse", config).fit(Xtr, str_, ytr, Xval, sval, yval)
    pu, _ = unc.predict(Xte, ste)
    pf, _ = fus.predict(Xte, ste)

    # Postprocessing returns *expected* P(pred=1); threshold to a hard label
    # for the fairness-fix / accuracy-cost comparison.
    pu_hard = (np.asarray(pu) >= 0.5).astype(int)
    pf_hard = (np.asarray(pf) >= 0.5).astype(int)

    cases: list[dict] = []
    seen_fixed = 0
    seen_cost = 0
    for i in range(len(yte)):
        corrected = (pu_hard[i] != yte[i]) and (pf_hard[i] == yte[i])
        cost = (pf_hard[i] != yte[i]) and (pu_hard[i] == yte[i])
        if corrected and seen_fixed < n:
            cases.append(
                {
                    "type": "fairness_fix",
                    "index": int(i),
                    "group": int(ste[i]),
                    "true": int(yte[i]),
                    "unconstrained_pred": int(pu_hard[i]),
                    "equifuse_pred": int(pf_hard[i]),
                    "merit": float(Xte[i, 0]),
                }
            )
            seen_fixed += 1
        elif cost and seen_cost < n:
            cases.append(
                {
                    "type": "accuracy_cost",
                    "index": int(i),
                    "group": int(ste[i]),
                    "true": int(yte[i]),
                    "unconstrained_pred": int(pu_hard[i]),
                    "equifuse_pred": int(pf_hard[i]),
                    "merit": float(Xte[i, 0]),
                }
            )
            seen_cost += 1
    return cases


def _ablation(config: Config, suite) -> dict:
    """Quantify the postprocessing contribution beyond reweighing alone."""
    out: dict[str, dict] = {}
    for ds in suite:
        rew = benchmark([ds], methods=["reweighing"], seeds=SEEDS, config=config)["aggregate"]["reweighing"]
        fus = benchmark([ds], methods=["equifuse"], seeds=SEEDS, config=config)["aggregate"]["equifuse"]
        out[ds.name] = {
            "reweighing_eod": rew["eod"],
            "equifuse_eod": fus["eod"],
            "eod_reduction_from_postproc": rew["eod"] - fus["eod"],
        }
    return out


def main() -> dict:
    set_all(7)
    config = Config.from_env()
    config.validate()
    suite = make_benchmark_suite(n_samples=config.n_samples, seed=config.seed)

    t0 = time.time()
    result = benchmark(suite, seeds=SEEDS, config=config)
    elapsed = time.time() - t0
    _format_result(result)

    det_ok = _determinism_check(config, suite)

    print("\n=== Ablation: postprocessing beyond reweighing (EOD) ===")
    abl = _ablation(config, suite)
    for name, v in abl.items():
        print(
            f"  {name:<16} reweighing EOD={v['reweighing_eod']:.4f}  "
            f"equifuse EOD={v['equifuse_eod']:.4f}  "
            f"postproc delta={v['eod_reduction_from_postproc']:+.4f}"
        )

    cases = _failure_cases(config, suite[0], n=3)
    print("\n=== Failure cases (Unconstrained vs EquiFuse, seed=7) ===")
    for c in cases:
        print(
            f"  [{c['type']}] group={c['group']} true={c['true']} "
            f"unc={c['unconstrained_pred']} fuse={c['equifuse_pred']} merit={c['merit']:.3f}"
        )

    result["determinism"] = {
        "bit_identical": det_ok,
        "max_delta": 0.0 if det_ok else None,
    }
    result["ablation"] = abl
    result["failure_cases"] = cases
    result["elapsed_sec"] = elapsed

    out_path = ROOT / "benchmark.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_path}  (demo wall-clock {elapsed:.2f}s)")
    return result


if __name__ == "__main__":
    main()
