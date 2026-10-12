# Benchmark Baseline

This directory contains the approved benchmark baseline used by CI regression
comparison.

## Current approved baseline

The first reviewed baseline was promoted from the `rag-evaluation` artifact
produced by successful `main` CI for commit
`ad0934fa5f3092cf53b85c3dbd64ad27af530964`.

- cases: 19 total, 16 answerable, 3 unanswerable
- Recall@k: 0.875
- MRR: 0.6875
- faithfulness: 1.0
- citation support: 1.0
- refusal accuracy: 1.0
- p95 latency: 0.5943789999918181 ms
- p95 token estimate: 132
- evaluation strategy: lexical
- evaluation provider: builtin
- evaluation model: lexical-default
- fingerprint:
  `0bf436c0bfc927288cebe24a83894c6e3e96bf81e5225733315da0baa051597b`

Because `benchmark-summary.json`, `cases.json`, and `manifest.json` are now
committed, the CI baseline regression gate is active on pull requests.

## Creating or Updating the Baseline

Generate a fresh deterministic evaluation:

```bash
make rag-eval
```

Inspect the results under `artifacts/evaluation/`, including:

- `benchmark-summary.json`
- `cases.json`
- `benchmark-summary.md`
- promoted baselines also include `manifest.json`
- per-run artifacts

Only after reviewing the results should you promote them:

```bash
poetry run rag-quality-engine benchmark-promote
```

The command validates that the benchmark passed, the case count matches the
summary, and case queries are unique before writing the canonical baseline files.
It also writes `manifest.json` with the project version, Git commit SHA,
promotion timestamp, and benchmark fingerprint.

Commit those files in a dedicated pull request.

Baseline promotion is a deliberate human decision. CI never rewrites or
auto-promotes the baseline.

## Comparison Semantics

Validate the committed baseline directly with:

```bash
poetry run rag-quality-engine benchmark-baseline-validate
```

The command verifies that the summary, cases, and manifest agree on fingerprint,
case count, and query identity.

The CI comparison uses:

```bash
make benchmark-baseline-check
```

When `manifest.json` is present, the gate prints it before comparison so CI
logs identify the approved baseline version and commit.

The gate first validates baseline integrity, then:

- requires matching benchmark provenance fingerprints;
- compares aggregate quality and performance metrics;
- compares per-query cases by query text;
- fails on quality regressions, token-usage increases, material p95 latency
  regressions beyond the configured tolerance, missing baseline queries, or a
  pass-to-fail transition.

The default latency tolerance is `max(1 ms, 10% of baseline p95)`; this prevents
scheduler/runtime noise from making the deterministic benchmark gate flaky.

If the corpus, golden set, refusal set, or chunking configuration changes, the
fingerprint changes. In that situation, review the new benchmark and promote a
new baseline intentionally rather than bypassing the mismatch in CI.
