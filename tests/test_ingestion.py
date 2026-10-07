from __future__ import annotations

import csv
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragops_lab.cli import app
from ragops_lab.domain import Document
from ragops_lab.ingestion import (
    ChunkingConfig,
    chunk_document,
    discover_documents,
    ingest_directory,
    load_chunks_jsonl,
)


def test_ingestion_supports_text_markdown_and_csv(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "note.txt").write_text("hello world", encoding="utf-8")
    (raw_dir / "guide.md").write_text("# Title\nrag metrics", encoding="utf-8")
    with (raw_dir / "facts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["topic", "value"])
        writer.writeheader()
        writer.writerow({"topic": "mission", "value": "Apollo 11"})

    documents = discover_documents(raw_dir)
    output_path = tmp_path / "chunks.jsonl"
    chunks = ingest_directory(raw_dir, output_path, ChunkingConfig(chunk_size=30, overlap=5))

    assert len(documents) == 3
    assert output_path.exists()
    assert len(load_chunks_jsonl(output_path)) == len(chunks)


def test_cli_ingest_writes_chunks(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "doc.txt").write_text("Apollo 11 landed on the Moon.", encoding="utf-8")
    output_path = tmp_path / "processed" / "chunks.jsonl"

    result = CliRunner().invoke(app, ["ingest", str(raw_dir), "--out", str(output_path)])

    assert result.exit_code == 0
    assert output_path.exists()


def test_chunk_document_rejects_unsupported_strategy() -> None:
    document = Document(
        document_id="doc",
        title="Doc",
        text="A short document about RAG metrics.",
    )

    with pytest.raises(ValueError, match="Unsupported chunking strategy"):
        chunk_document(document, ChunkingConfig(strategy="tokens"))


def test_chunk_document_avoids_splitting_words_and_preserves_offsets() -> None:
    document = Document(
        document_id="doc",
        title="Doc",
        text="alpha beta gamma delta epsilon zeta eta theta",
    )

    chunks = chunk_document(document, ChunkingConfig(chunk_size=18, overlap=5))

    assert len(chunks) >= 2
    for chunk in chunks:
        assert document.text[chunk.start_offset : chunk.end_offset] == chunk.text
        if chunk.start_offset > 0:
            assert document.text[chunk.start_offset - 1].isspace()
        if chunk.end_offset < len(document.text):
            assert document.text[chunk.end_offset].isspace()


def test_chunk_document_progresses_on_long_unbroken_token() -> None:
    text = "x" * 80
    document = Document(document_id="doc", title="Doc", text=text)

    chunks = chunk_document(document, ChunkingConfig(chunk_size=20, overlap=5))

    assert len(chunks) >= 1
    assert chunks[-1].end_offset == len(text)
    for chunk in chunks:
        assert document.text[chunk.start_offset : chunk.end_offset] == chunk.text


def test_chunk_document_trims_offsets_with_content() -> None:
    document = Document(
        document_id="doc",
        title="Doc",
        text="   alpha beta gamma   ",
    )

    chunks = chunk_document(document, ChunkingConfig(chunk_size=50, overlap=5))

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.text == "alpha beta gamma"
    assert document.text[chunk.start_offset : chunk.end_offset] == chunk.text
