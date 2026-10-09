"""CLI smoke tests: the command-line entrypoint lists methods and runs the
benchmark, writing a well-formed JSON report.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "equiforge.cli", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_methods_lists_all():
    proc = _run_cli(["methods"])
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    for name in (
        "unconstrained",
        "reweighing",
        "eo_postprocess",
        "dp_postprocess",
        "equifuse",
    ):
        assert name in out


def test_cli_run_writes_report(tmp_path):
    out_file = tmp_path / "bench.json"
    # Use the validated default configuration (n_samples=4000) so the gate is
    # exercised under the same conditions as the benchmark demo.
    proc = _run_cli(
        [
            "run",
            "--methods",
            "unconstrained,equifuse",
            "--seeds",
            "7,11,23",
            "--output",
            str(out_file),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert "gate" in data
    assert "per_dataset" in data
    assert data["gate"]["passed"] is True
