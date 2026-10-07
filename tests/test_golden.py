from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragops_lab.cli import app
from ragops_lab.evaluation import load_golden_examples, rebuild_golden_set


def _write_source(tmp_path: Path) -> Path:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "metrics.txt").write_text(
        "Context precision measures the share of retrieved chunks that are relevant. "
        "Faithfulness measures whether claims are supported by retrieved evidence.",
        encoding="utf-8",
    )
    return source_dir


def test_rebuild_golden_set_resolves_evidence_phrases(tmp_path: Path) -> None:
    source_dir = _write_source(tmp_path)
    input_path = tmp_path / "golden-input.json"
    output_path = tmp_path / "golden-output.json"
    chunks_path = tmp_path / "chunks.jsonl"
    input_path.write_text(
        json.dumps(
            [
                {
                    "query": "What does context precision measure?",
                    "evidence_phrases": [
                        (
                            "Context precision measures the share of retrieved chunks "
                            "that are relevant."
                        )
                    ],
                },
                {
                    "query": "What is the capital of France?",
                    "expected_unanswerable": True,
                },
            ]
        ),
        encoding="utf-8",
    )

    rebuilt = rebuild_golden_set(
        source_dir=source_dir,
        input_path=input_path,
        output_path=output_path,
        chunks_path=chunks_path,
        chunk_size=120,
        overlap=20,
    )

    assert rebuilt[0].relevant_chunk_ids == ["metrics:0"]
    assert rebuilt[1].relevant_chunk_ids == []
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload[0]["evidence_phrases"]
    assert payload[1]["expected_unanswerable"] is True

    loaded = load_golden_examples(output_path)
    assert loaded[0].relevant_chunk_ids == ["metrics:0"]
    assert loaded[1].expected_unanswerable is True


def test_rebuild_golden_set_is_deterministic(tmp_path: Path) -> None:
    source_dir = _write_source(tmp_path)
    input_path = tmp_path / "golden.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "query": "What does faithfulness measure?",
                    "evidence_phrases": [
                        "Faithfulness measures whether claims are supported by retrieved evidence."
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first = rebuild_golden_set(
        source_dir=source_dir,
        input_path=input_path,
        output_path=first_path,
        chunks_path=tmp_path / "first-chunks.jsonl",
        chunk_size=200,
        overlap=20,
    )
    second = rebuild_golden_set(
        source_dir=source_dir,
        input_path=input_path,
        output_path=second_path,
        chunks_path=tmp_path / "second-chunks.jsonl",
        chunk_size=200,
        overlap=20,
    )

    assert first == second
    assert first_path.read_text(encoding="utf-8") == second_path.read_text(encoding="utf-8")


def test_rebuild_golden_set_rejects_missing_evidence_phrase(tmp_path: Path) -> None:
    source_dir = _write_source(tmp_path)
    input_path = tmp_path / "golden.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "query": "What does context precision measure?",
                    "evidence_phrases": ["This phrase does not exist in the corpus."],
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Evidence phrase not found"):
        rebuild_golden_set(
            source_dir=source_dir,
            input_path=input_path,
            output_path=tmp_path / "out.json",
            chunks_path=tmp_path / "chunks.jsonl",
            chunk_size=120,
            overlap=20,
        )


def test_golden_rebuild_cli_writes_output(tmp_path: Path) -> None:
    source_dir = _write_source(tmp_path)
    input_path = tmp_path / "golden.json"
    output_path = tmp_path / "rebuilt.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "query": "What does context precision measure?",
                    "evidence_phrases": [
                        (
                            "Context precision measures the share of retrieved chunks "
                            "that are relevant."
                        )
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "golden-rebuild",
            "--source-dir",
            str(source_dir),
            "--input-path",
            str(input_path),
            "--out",
            str(output_path),
            "--chunks-path",
            str(tmp_path / "chunks.jsonl"),
            "--chunk-size",
            "120",
            "--overlap",
            "20",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert "examples_written" in result.output


def test_rebuild_golden_set_requires_evidence_phrases_for_answerable_cases(
    tmp_path: Path,
) -> None:
    source_dir = _write_source(tmp_path)
    input_path = tmp_path / "golden.json"
    input_path.write_text(
        json.dumps([{"query": "What does context precision measure?"}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing evidence_phrases"):
        rebuild_golden_set(
            source_dir=source_dir,
            input_path=input_path,
            output_path=tmp_path / "out.json",
            chunks_path=tmp_path / "chunks.jsonl",
            chunk_size=120,
            overlap=20,
        )
