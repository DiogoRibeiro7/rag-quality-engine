# Release validation — v0.3.0

This document is the release-candidate validation receipt for RAG Quality Engine
v0.3.0.

It records the evidence that must remain true when the release is tagged. It does
not claim that a GitHub Release or Zenodo version has already been published.

## Release metadata

Expected software version:

- `pyproject.toml`: `0.3.0`
- `src/ragops_lab/__init__.py`: `0.3.0`
- `CITATION.cff`: software version `0.3.0`
- `CITATION.cff`: CFF schema version remains `1.2.0`
- `.zenodo.json`: `0.3.0`

Release date prepared in citation metadata: 2026-10-17.

The dependency graph is unchanged by the release-preparation work, so the release
prep does not deliberately rewrite dependency constraints or the lockfile.

## Active approved benchmark baseline

The active committed baseline remains the deliberately promoted reference in
`benchmarks/baseline/`.

Its provenance is intentionally historical and must not be rewritten merely
because the package release version changes.

Baseline manifest:

- project version at promotion: `0.2.0`
- promotion commit: `ad0934fa5f3092cf53b85c3dbd64ad27af530964`
- promoted at: `2026-10-12T21:12:25.825661Z`
- benchmark fingerprint:
  `0bf436c0bfc927288cebe24a83894c6e3e96bf81e5225733315da0baa051597b`

The same approved run is recorded in
`benchmarks/history/history.jsonl` as `baseline-0.2.0`.

## Approved benchmark snapshot

The committed history record reports:

- 19 cases
- 16 answerable cases
- 3 unanswerable cases
- Recall@k: `0.875`
- mean reciprocal rank: `0.6875`
- average faithfulness: `1.0`
- average citation support: `1.0`
- average refusal accuracy: `1.0`
- worst-run p95 latency: approximately `0.594 ms`
- worst-run p95 token estimate: `132`
- evaluation strategy: `lexical`
- evaluation provider: `builtin`
- evaluation model: `lexical-default`

These are baseline facts, not promises that every future candidate must reproduce
the latency value exactly. Candidate comparison uses the repository's configured
regression tolerances and provenance checks.

## Release capability contract

v0.3.0 consolidates the post-MVP work completed after v0.2.0:

- deterministic refusal benchmarking;
- benchmark provenance fingerprints;
- provenance-safe benchmark comparison;
- per-query regression diagnosis;
- approved baseline promotion and integrity validation;
- benchmark history and trend reporting;
- latency and token-budget gates;
- golden-set rebuilding from stable evidence phrases;
- sentence-aware and token-aware chunking;
- lexical, embedding-backed, and provider-backed LLM evaluation strategies;
- embedding-backed semantic reranking;
- dashboard KPI and quality-state improvements;
- API health/deployment guidance;
- request correlation IDs;
- structured request lifecycle and trace observability;
- complete Dependabot ecosystem coverage with shared Safe Dependabot validation.

## CI contract before tagging

The release candidate must pass the repository CI on the exact commit intended
for release.

Required jobs:

1. test matrix on Python 3.11
2. test matrix on Python 3.12
3. test matrix on Python 3.13
4. test matrix on Python 3.14
5. deterministic RAG evaluation regression gate
6. approved benchmark baseline integrity/comparison gate
7. notebook execution with `nbval`
8. shared Dependabot policy validation when relevant policy/manifests change

The RAG evaluation job uploads `artifacts/evaluation` for inspection.

## Tagging and publication checklist

Before publishing v0.3.0:

1. merge the release-preparation PR;
2. confirm CI succeeds on the exact resulting `main` commit;
3. tag that exact commit as `v0.3.0`;
4. create the GitHub Release from that tag;
5. verify the release notes match the 0.3.0 changelog;
6. verify Zenodo creates/updates the 0.3.0 version record;
7. preserve the all-versions DOI `10.5281/zenodo.21805398`;
8. do not rewrite the historical approved baseline manifest to 0.3.0.

## Scientific and operational boundary

The default evaluation path remains deterministic and offline. Embedding-backed
and provider-backed LLM judges are optional strategies and are provenance-tracked
so their results are not silently treated as comparable with lexical-default
baselines.

The repository remains local-first for traces, chunks, indexes, benchmark
artifacts, and approved baselines. Public deployment authentication, rate
limiting, and pluggable production storage remain future hardening work rather
than claims of the v0.3.0 release.
