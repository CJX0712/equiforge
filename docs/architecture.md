# EquiForge — Architecture

This document describes the internal design of EquiForge and how the pieces fit
together. It is meant to be read alongside the source tree.

## 1. Design principles

1. **Determinism first.** Every source of randomness flows from a single integer
   seed (`core/seed.set_all`). Two runs with the same seed and config are
   *bit-identical* (verified in CI and the demo).
2. **Zero hard dependencies.** The only required runtime dependency is NumPy. A
   hand-written Newton–Raphson (IRLS) logistic-regression solver backs the
   toolkit when scikit-learn is unavailable.
3. **Expectation, not sampling.** The randomised-threshold postprocessor is
   evaluated in *expectation*, so fairness metrics are exact and reproducible —
   no Monte-Carlo noise.
4. **Group-aware inference, no leakage.** Postprocessing uses the protected
   attribute `s` *only at inference*; `s` is never a model feature.

## 2. Data flow

```
                 ┌──────────────┐
   seed ───────► │ split_dataset│  (train / val / test, seeded)
                 └──────┬───────┘
                        │
                 ┌──────▼───────┐
                 │  method.fit  │  Xtr,str,ytr, Xval,sval,yval
                 │              │  • train logistic regression
                 │              │  • (reweighing) sample weights
                 │              │  • (postproc) fit per-group policy on VAL
                 └──────┬───────┘
                        │
                 ┌──────▼───────┐
                 │ method.predict│  Xte, ste  →  (expected P(pred=1), scores)
                 └──────┬───────┘
                        │
                 ┌──────▼───────┐
                 │   evaluate   │  expectation-based DPD / EOD / accuracy
                 └──────────────┘
```

The pipeline is strictly acyclic (`run_one` → `fit` → `predict` → `evaluate`).
There are no back-edges, which is what makes global determinism tractable.

## 3. Modules

| Module | Responsibility |
| --- | --- |
| `core/seed.py` | Single-seed RNG seeding (legacy `np.random`, `random`, `PYTHONHASHSEED`, and a modern `Generator`). |
| `core/config.py` | `Config` dataclass with `EQUIFORGE_*` env overrides and schema validation. |
| `core/errors.py` | Typed error hierarchy (`ConfigError`, `DataError`, `ModelError`, `MethodError`, `EvalError`). |
| `core/types.py` | `FairnessDataset`, `GroupMetrics`, `FairnessResult` dataclasses. |
| `data/synthesizer.py` | Synthetic generators; controllable `group_gap`, `direct_effect`, `label_noise`; tuned presets. |
| `models/logreg.py` | `LogisticRegressionBackend` with sklearn (Tier-0) and pure-NumPy IRLS fallback. |
| `fairness/methods.py` | Reweighing weights, ROC-point construction (with randomised midpoints), adaptive threshold search, and the five method classes. |
| `eval/metrics.py` | Expectation-based DPD / EOD / accuracy. |
| `pipeline/pipeline.py` | `split_dataset`, `run_one`, `benchmark`, and the pre-registered gate `_evaluate_gate`. |
| `cli.py` | `methods` and `run` sub-commands. |

## 4. The postprocessing core

`fairness/methods.py` is the mathematical heart.

* **ROC points.** `_roc_points_for_groups` enumerates, per group, every unique
  threshold (capped at `max_pts`) and adds *randomised midpoints* between
  adjacent deterministic thresholds. Each point carries `tpr`, `fpr`, `sr`
  (selection rate), and `acc`.
* **Adaptive search.** `_search_thresholds` targets
  `disparity ≤ eo_tradeoff_c × baseline_disparity` (`eo_tradeoff_c = 0.45` by
  default, safely under the `0.5` gate). Among feasible operating-point pairs it
  maximises accuracy; if no pair is feasible it falls back to the minimum-disparity
  pair. Randomised thresholds make the feasible set essentially always non-empty.
* **Expectation.** `expected_pos_from_policy` returns `E[P(pred=1)]` under the
  randomised policy — the exact, sampling-free quantity reported everywhere.

## 5. Determinism guarantees

* Data generation uses `np.random.default_rng(seed)` with per-dataset seed offsets.
* Model fitting uses a fixed `random_state`.
* The threshold search is a deterministic `argmin` over a finite grid.
* The demo's determinism check reruns the full benchmark and asserts
  `max |metric difference| == 0.0`.

## 6. Offline robustness

`models/logreg.py` tries scikit-learn first and falls back to an IRLS solver:

```
backend="auto" → sklearn if importable else numpy
```

The numpy solver is L2-regularised and converges to the same decision boundary as
sklearn to within `≈ 0.05` accuracy / predictions (verified in tests). This means
EquiForge runs in minimal environments with **only NumPy installed**.

## 7. Testing strategy

`tests/` encodes independent, verifiable invariants:

* Metric correctness (expected accuracy formula; zero disparity for balanced
  assignments).
* Reweighing independence (`P_w(s,y) ≈ P_w(s)P_w(y)` up to `1e-9`).
* Offline backend agreement (NumPy vs sklearn).
* Bit-identical determinism across two runs.
* End-to-end gate passage and EOD reduction / accuracy floor.
* CLI smoke (method listing + report writing).
