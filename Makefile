# EquiForge Makefile — local CI short-cuts.
# All commands use the active Python interpreter.

.PHONY: help install lint format test demo ci

help:
	@echo "EquiForge targets:"
	@echo "  install  install runtime + dev deps"
	@echo "  lint    ruff check"
	@echo "  format  ruff format (write)"
	@echo "  test    run the pytest suite"
	@echo "  demo    run the benchmark demo (writes benchmark.json)"
	@echo "  ci      run lint + format-check + test + demo (CI equivalent)"

install:
	pip install -r requirements.txt
	pip install pytest ruff

lint:
	python -m ruff check equiforge/ examples/ tests/

format:
	python -m ruff format equiforge/ examples/ tests/

test:
	python -m pytest tests/

demo:
	python examples/run_demo.py

ci: lint
	python -m ruff format --check equiforge/ examples/ tests/
	python -m pytest tests/
	python examples/run_demo.py
