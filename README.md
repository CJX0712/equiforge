# EquiForge

> 世界级算法公平性工具箱 · World-class algorithmic-fairness toolkit.

**EquiForge** is a deterministic, CPU-only Python toolkit for *group fairness* in
binary classification. Its flagship method **EquiFuse** combines

* **Reweighing** (Kamiran & Calders, 2012) — preprocessing that removes the
  dependence between the protected attribute and the label, and
* **Equalized-Odds postprocessing** (Hardt et al., 2016) — per-group, *randomised*
  decision thresholds evaluated **in expectation** (no sampling), the textbook
  equalised-odds correction.

Everything is built for **reproducibility**: one integer seed reproduces byte-identical
metrics, and the toolkit runs with **zero third-party ML dependencies** (a pure-NumPy
Newton–Raphson logistic-regression solver is the offline fallback).

- Author / 作者: **晨星 (Morning Star)**
- License: MIT
- Python: ≥ 3.10

---

## Why EquiFuse

| Property | EquiFuse | Unconstrained |
| --- | --- | --- |
| Equalized-Odds Difference (EOD) | **≤ 0.5 × baseline** ✅ | baseline |
| Accuracy | within 0.04 of baseline ✅ | baseline |
| Deterministic | yes (bit-identical across runs) | yes |
| Third-party ML deps | none required | sklearn optional |

The flagship **pre-registered fairness gate** (per dataset, all must hold) is:

```
EOD(equifuse) <= 0.5 * EOD(unconstrained)
AND
ACC(equifuse) >= ACC(unconstrained) - 0.04
```

## Install

```bash
pip install -r requirements.txt        # needs only numpy; sklearn optional
pip install -e .                        # installs the `equiforge` CLI
```

The pure-NumPy fallback means you can run EquiForge even on a machine with no
scikit-learn installed:

```bash
EQUIFORGE_BACKEND=numpy python examples/run_demo.py
```

## Quick start

```bash
# List available methods
python -m equiforge.cli methods

# Run the benchmark and write a JSON report
python -m equiforge.cli run --output benchmark.json

# Or the full demo (benchmark + determinism check + ablation + failure cases)
python examples/run_demo.py
```

From Python:

```python
from equiforge.data.synthesizer import make_benchmark_suite
from equiforge.pipeline.pipeline import benchmark
from equiforge.core.config import Config

cfg = Config()
suite = make_benchmark_suite(n_samples=4000, seed=cfg.seed)
result = benchmark(suite, seeds=[7, 11, 23], config=cfg)
print(result["gate"])          # pre-registered fairness gate result
```

## Methods

| Method | Reference | What it does |
| --- | --- | --- |
| `unconstrained` | — | Standard logistic regression, global 0.5 threshold (the baseline). |
| `reweighing` | Kamiran & Calders (2012) | Sample weights that decorrelate `s` and `y`. |
| `eo_postprocess` | Hardt et al. (2016) | Equalized-Odds postprocessing (randomised thresholds, expectation). |
| `dp_postprocess` | (reference) | Demographic-Parity postprocessing. |
| **`equifuse`** | **this work** | **Reweighing + Equalized-Odds postprocessing (flagship).** |

## How the postprocessing works

For each group we build the set of achievable operating points on its ROC curve,
including **randomised midpoints** between adjacent deterministic thresholds.
A randomised threshold returns `1` with probability `ρ` on a small score band,
which lets the policy reach *any* point on the convex hull of the ROC curve —
this is what makes strong equalisation feasible. We then pick, **per group**, the
policy that drives the group disparity down to `≤ eo_tradeoff_c × baseline`
(default `0.45`, safely under the `0.5` gate) while keeping accuracy as high as
possible. The reported metrics are the **expectation** of the randomised policy:

```
accuracy = E[ 1(pred = y) ] = mean( p·y + (1-p)·(1-y) )
selection_rate, TPR, FPR   = mean( P(pred=1) ) over the relevant subgroup
```

Because this is an expectation, results are exact and deterministic — no Monte
Carlo.

## Reproducibility

- `equiforge/core/seed.set_all(seed)` seeds every RNG source from one integer.
- The data generator, model fit, and threshold search are all deterministic.
- `benchmark(..., seeds=[7,11,23])` aggregates `mean ± std`; two identical runs
  produce bit-identical `per_cell` metrics. The demo verifies this explicitly.

## Project layout

```
equiforge/
  core/        config, seed, errors, types, interfaces
  data/        synthetic fairness data generators
  models/      logistic-regression backend (sklearn + pure-NumPy IRLS fallback)
  fairness/    reweighing + EO/DP postprocessing + EquiFuse
  eval/        DPD / EOD / accuracy metrics (expectation-based)
  pipeline/    data -> method -> evaluate; benchmark + gate
  cli.py       command-line entrypoint
examples/      run_demo.py (benchmark + determinism + ablation + failure cases)
tests/         pytest suite (invariants, offline fallback, determinism, gate, CLI)
docs/          architecture.md, model_card.md
```

## License

MIT — see [LICENSE](LICENSE). Author: **晨星 (Morning Star)**.
