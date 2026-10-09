"""EquiForge command-line interface.

Usage:
    python -m equiforge.cli methods
    python -m equiforge.cli run --output benchmark.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from equiforge.core.config import Config
from equiforge.data.synthesizer import make_benchmark_suite
from equiforge.fairness.methods import available_methods
from equiforge.pipeline.pipeline import benchmark


def _print_table(result: dict) -> None:
    methods = result["methods"]
    header = f"{'dataset':<16}" + "".join(f"{m:>16}" for m in methods)
    print(header)
    for ds in result["datasets"]:
        row = f"{ds:<16}"
        for m in methods:
            v = result["aggregate"].get(m, {})
            _ = v  # placeholder to keep linter calm about unused
            cell = result["per_cell"][ds][m]
            a = sum(cell["accuracy"]) / len(cell["accuracy"])
            row += f"{a:>16.4f}"
        print(row)
    print()
    print("Equalized-Odds difference (aggregate mean across datasets):")
    for m in methods:
        v = result["aggregate"].get(m, {})
        print(f"  {m:<16} EOD={v.get('eod', float('nan')):.4f}  ACC={v.get('accuracy', float('nan')):.4f}")
    gate = result["gate"]
    print()
    print(
        f"GATE: passed={gate['passed']}  "
        f"EOD ratio(flag/baseline)={gate.get('eod_ratio_aggregate', float('nan')):.3f}  "
        f"(threshold <=0.5)"
    )


def cmd_run(args: argparse.Namespace) -> dict:
    config = Config.from_env()
    config.validate()
    suite = make_benchmark_suite(n_samples=args.n_samples, seed=config.seed)
    methods = args.methods.split(",") if args.methods else available_methods()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [7, 11, 23]
    result = benchmark(suite, methods=methods, seeds=seeds, config=config)
    _print_table(result)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        print(f"\nWrote {args.output}")
    return result


def cmd_methods(_args: argparse.Namespace) -> None:
    for m in available_methods():
        print(m)


def main() -> None:
    parser = argparse.ArgumentParser(prog="equiforge", description="EquiForge fairness toolkit")
    sub = parser.add_subparsers(dest="command")
    run = sub.add_parser("run", help="run the benchmark")
    run.add_argument("--n-samples", type=int, default=4000)
    run.add_argument("--methods", type=str, default=None)
    run.add_argument("--seeds", type=str, default=None)
    run.add_argument("--output", type=str, default=None)
    run.set_defaults(func=cmd_run)
    meth = sub.add_parser("methods", help="list available methods")
    meth.set_defaults(func=cmd_methods)
    args = parser.parse_args()
    if not getattr(args, "func", None):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
