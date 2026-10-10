.PHONY: install lint typecheck test format check rag-eval benchmark benchmark-baseline-check notebook-check release-check pre-commit run

install:
	poetry install --with dev

lint:
	poetry run ruff check .

typecheck:
	poetry run mypy src

test:
	poetry run pytest -q

format:
	poetry run ruff format .

check: lint typecheck test

rag-eval:
	poetry run python scripts/evaluate_rag.py

benchmark:
	poetry run rag-quality-engine benchmark

benchmark-baseline-check:
	@if [ -f benchmarks/baseline/benchmark-summary.json ] && [ -f benchmarks/baseline/cases.json ]; then \
		poetry run rag-quality-engine benchmark-compare \
			benchmarks/baseline/benchmark-summary.json \
			artifacts/evaluation/benchmark-summary.json \
			--baseline-cases benchmarks/baseline/cases.json \
			--candidate-cases artifacts/evaluation/cases.json; \
	else \
		echo "No benchmark baseline found; skipping comparison."; \
	fi

notebook-check:
	poetry run pytest --nbval-lax --no-cov notebooks -q

release-check: check pre-commit
	poetry check

pre-commit:
	poetry run pre-commit run --all-files

run:
	poetry run python -m ragops_lab.cli
