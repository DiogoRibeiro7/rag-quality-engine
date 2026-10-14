# Benchmark History

This directory stores the local append-only benchmark history used for
longitudinal trend reporting.

The canonical history file is `history.jsonl`. Each line contains:

- recording timestamp;
- human-readable label;
- Git commit;
- the full persisted benchmark summary, including provenance.

The initial entry is the approved `baseline-0.2.0` benchmark.

Record a reviewed benchmark with:

```bash
poetry run rag-quality-engine benchmark-history-record \
  --summary artifacts/evaluation/benchmark-summary.json \
  --history benchmarks/history/history.jsonl \
  --label <label> \
  --git-commit <commit-sha>
```

Inspect the trend with:

```bash
poetry run rag-quality-engine benchmark-history-trend \
  --history benchmarks/history/history.jsonl
```

Trend reporting uses the latest provenance fingerprint by default. Runs with
different benchmark identities are therefore kept out of the same trend series
unless a specific fingerprint is selected explicitly.
