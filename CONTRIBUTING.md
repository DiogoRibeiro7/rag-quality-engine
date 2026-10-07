# Contributing

RAG Quality Engine is a portfolio-grade RAG quality and reliability project. Contributions should keep
the codebase deterministic, testable, and runnable without external model
credentials unless an integration is explicitly optional.

## Development Setup

```bash
poetry install --with dev
poetry run pre-commit install
```

## Quality Gates

Run the same checks used in CI before opening a pull request:

```bash
make check
```

For targeted iteration:

```bash
poetry run pytest tests/test_retrieval.py -q
poetry run ruff check src tests
poetry run mypy src
```

## Contribution Guidelines

- Keep package logic in `src/ragops_lab`; notebooks should demonstrate package
  behavior instead of defining notebook-only production logic.
- Add or update tests when changing retrieval, generation, evaluation, tracing,
  CLI, or API behavior.
- Keep offline defaults deterministic. Provider-backed LLM or embedding clients
  should sit behind existing abstractions and be optional at runtime.
- Persist generated benchmark outputs under ignored artifact directories unless
  the output is a deliberate fixture or documentation asset.
- Update `README.md`, `docs/architecture.md`, or `ROADMAP.md` when behavior,
  architecture, or project priorities change.
- Follow [`docs/release.md`](docs/release.md) for version bumps, tags, GitHub
  releases, and Zenodo metadata.

## Pull Request Checklist

- [ ] The change is covered by focused tests or the test gap is explained.
- [ ] `make lint` passes.
- [ ] `make typecheck` passes.
- [ ] `make test` passes.
- [ ] User-facing behavior is documented where appropriate.
