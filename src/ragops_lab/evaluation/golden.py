"""Golden-set maintenance utilities."""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, Field

from ragops_lab.ingestion import ChunkingConfig, ingest_directory


class GoldenSetExample(BaseModel):
    """Golden-set example with stable evidence anchors."""

    query: str = Field(min_length=1)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    evidence_phrases: list[str] = Field(default_factory=list)
    expected_unanswerable: bool = Field(default=False)


def _normalize_text(value: str) -> str:
    """Normalize whitespace and case for deterministic phrase matching."""
    return re.sub(r"\s+", " ", value).strip().casefold()


def rebuild_golden_set(
    *,
    source_dir: Path,
    input_path: Path,
    output_path: Path,
    chunks_path: Path,
    chunk_size: int = 400,
    overlap: int = 60,
) -> list[GoldenSetExample]:
    """Resolve evidence phrases against the current chunked corpus."""
    if not source_dir.exists():
        raise ValueError(f"Source directory not found: {source_dir}")
    if not source_dir.is_dir():
        raise ValueError(f"Source path is not a directory: {source_dir}")
    if not input_path.exists():
        raise ValueError(f"Golden set not found: {input_path}")

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    examples = [GoldenSetExample.model_validate(item) for item in payload]
    chunks = ingest_directory(
        source_dir,
        chunks_path,
        ChunkingConfig(chunk_size=chunk_size, overlap=overlap),
    )
    normalized_chunks = [
        (chunk.chunk_id, _normalize_text(chunk.text))
        for chunk in chunks
    ]

    rebuilt: list[GoldenSetExample] = []
    for example in examples:
        if example.expected_unanswerable:
            rebuilt.append(
                example.model_copy(update={"relevant_chunk_ids": []})
            )
            continue

        if not example.evidence_phrases:
            raise ValueError(
                f"Answerable golden example is missing evidence_phrases: {example.query}"
            )

        resolved_ids: list[str] = []
        for phrase in example.evidence_phrases:
            normalized_phrase = _normalize_text(phrase)
            if not normalized_phrase:
                raise ValueError(
                    f"Evidence phrase must not be empty: {example.query}"
                )
            matching_ids = [
                chunk_id
                for chunk_id, normalized_chunk in normalized_chunks
                if normalized_phrase in normalized_chunk
            ]
            if not matching_ids:
                raise ValueError(
                    f"Evidence phrase not found in current chunks for query "
                    f"{example.query!r}: {phrase!r}"
                )
            for chunk_id in matching_ids:
                if chunk_id not in resolved_ids:
                    resolved_ids.append(chunk_id)

        rebuilt.append(
            example.model_copy(update={"relevant_chunk_ids": resolved_ids})
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            [example.model_dump(mode="json") for example in rebuilt],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return rebuilt
