# Benchmark Baseline

This directory is the conventional location for an approved benchmark baseline.

The CI regression gate is intentionally optional:

- if both `benchmark-summary.json` and `cases.json` are present here, CI compares
  the current RAG evaluation against them;
- if either file is absent, the comparison step is skipped and CI behaves as it
  did before baseline support was added.

## Creating or Updating the Baseline

Generate a fresh deterministic evaluation:

```bash
make rag-eval
```

Inspect the results under `artifacts/evaluation/`, including:

- `benchmark-summary.json`
- `cases.json`
- `benchmark-summary.md`
- per-run artifacts

Only after reviewing the results should you promote them:

```bash
poetry run rag-quality-engine benchmark-promote
```

The command validates that the benchmark passed, the case count matches the
summary, and case queries are unique before writing the canonical baseline files.

Commit those files in a dedicated pull request.

Baseline promotion is a deliberate human decision. CI never rewrites or
auto-promotes the baseline.

## Comparison Semantics

The comparison uses:

```bash
make benchmark-baseline-check
```

The gate:

- requires matching benchmark provenance fingerprints;
- compares aggregate quality and performance metrics;
- compares per-query cases by query text;
- fails on quality regressions, increased p95 latency/token usage, missing
  baseline queries, or a pass-to-fail transition.

If the corpus, golden set, refusal set, or chunking configuration changes, the
fingerprint changes. In that situation, review the new benchmark and promote a
new baseline intentionally rather than bypassing the mismatch in CI.
