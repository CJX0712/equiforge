# Changelog

All notable changes to EquiForge are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/) and this project adheres to
Semantic Versioning.

## [0.1.0] — 2026-10-09

### Added
- **EquiFuse** flagship = Reweighing preprocessing (Kamiran & Calders, 2012)
  + Equalized-Odds postprocessing (Hardt et al., 2016).
- Randomised per-group threshold policies with **expectation-based** metrics
  (deterministic, no sampling).
- Pure-NumPy Newton–Raphson (IRLS) logistic-regression backend as a zero-dependency
  offline fallback to scikit-learn.
- Synthetic fairness data generator with three tuned presets
  (`credit_like`, `adult_like`, `law_school_like`).
- Full benchmark harness with `mean ± std` aggregation over seeds and a
  pre-registered per-dataset fairness gate.
- CLI (`equiforge.cli`: `methods`, `run`) and an end-to-end demo
  (`examples/run_demo.py`) with determinism check, ablation, and failure cases.
- pytest suite: metric invariants, reweighing independence, offline backend
  agreement, determinism, gate, and CLI smoke tests.
- CI workflow (lint + format + tests + benchmark smoke + secret scan), Dockerfile,
  Makefile, packaging (`pyproject.toml`), and documentation (`architecture.md`,
  `model_card.md`).

### Gate (pre-registered)
- Per dataset, `equifuse` must satisfy
  `EOD ≤ 0.5 × Unconstrained EOD` **and** `ACC ≥ Unconstrained ACC − 0.04`.
- Verified passing on all three benchmark datasets with margin
  (aggregate EOD ratio ≈ 0.46).
- Two identical runs are bit-identical (determinism verified).

### Author
- **晨星 (Morning Star)** — MIT License.
