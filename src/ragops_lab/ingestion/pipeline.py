"""Reusable ingestion pipeline."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ragops_lab.domain import Document, DocumentChunk

from ..retrieval.tokenizer import tokenize


class PdfExtractor(Protocol):
    """Optional PDF extraction dependency boundary."""

    def extract_text(self, path: Path) -> str:
        """Extract text from a PDF file."""


@dataclass(frozen=True)
class ChunkingConfig:
    """Chunking controls."""

    chunk_size: int = 500
    overlap: int = 50
    strategy: str = "chars"


SUPPORTED_CHUNKING_STRATEGIES = frozenset({"chars", "sentence"})


def slugify(value: str) -> str:
    """Create stable ASCII-ish identifiers from filenames."""
    normalized = "".join(character.lower() if character.isalnum() else "-" for character in value)
    collapsed = "-".join(part for part in normalized.split("-") if part)
    return collapsed or "document"


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_csv_file(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[str] = []
        for index, row in enumerate(reader, start=1):
            parts = [f"{key}: {value or ''}".strip() for key, value in row.items()]
            rows.append(f"Row {index}: " + " | ".join(parts))
    return "\n".join(rows)


def load_document(path: Path, pdf_extractor: PdfExtractor | None = None) -> Document:
    """Load a single supported document."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        text = _read_text_file(path)
    elif suffix == ".csv":
        text = _read_csv_file(path)
    elif suffix == ".pdf":
        if pdf_extractor is None:
            raise ValueError("PDF ingestion requires a PdfExtractor implementation.")
        text = pdf_extractor.extract_text(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    title = path.stem.replace("_", " ").replace("-", " ").strip().title()
    return Document(
        document_id=slugify(path.stem),
        title=title or path.stem,
        text=text,
        source_path=str(path),
        metadata={"suffix": suffix},
    )


def discover_documents(
    input_dir: Path, pdf_extractor: PdfExtractor | None = None
) -> list[Document]:
    """Load all supported documents from a directory tree."""
    documents: list[Document] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".txt", ".md", ".csv", ".pdf"}:
            continue
        documents.append(load_document(path, pdf_extractor=pdf_extractor))
    return documents


def _trimmed_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Return offsets for the non-whitespace content inside a candidate span."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _snap_end_to_boundary(text: str, start: int, candidate_end: int) -> int:
    """Prefer a whitespace boundary without preventing progress on long tokens."""
    if candidate_end >= len(text):
        return len(text)
    if text[candidate_end - 1].isspace() or text[candidate_end].isspace():
        return candidate_end

    boundary = candidate_end
    while boundary > start and not text[boundary - 1].isspace():
        boundary -= 1
    return boundary if boundary > start else candidate_end


def _snap_start_to_boundary(text: str, candidate_start: int, previous_start: int) -> int:
    """Move an overlapping start to the beginning of a word when possible."""
    if candidate_start <= 0 or candidate_start >= len(text):
        return candidate_start
    if text[candidate_start - 1].isspace():
        return candidate_start

    boundary = candidate_start
    while boundary > previous_start and not text[boundary - 1].isspace():
        boundary -= 1
    if boundary > previous_start:
        return boundary

    # A token longer than the chunk may span the whole overlap region. There is
    # no valid word boundary to snap to, so preserve coverage by falling back to
    # the raw overlap position. The caller guarantees candidate_start > start,
    # which still guarantees progress.
    return candidate_start


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return trimmed sentence spans while preserving source offsets."""
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r".+?(?:[.!?]+(?=\s|$)|$)", text, flags=re.DOTALL):
        start, end = _trimmed_span(text, match.start(), match.end())
        if start < end:
            spans.append((start, end))
    return spans


def _build_chunk(
    document: Document,
    *,
    index: int,
    start: int,
    end: int,
) -> DocumentChunk:
    """Create one chunk while preserving exact source offsets."""
    content = document.text[start:end]
    return DocumentChunk(
        chunk_id=f"{document.document_id}:{index}",
        document_id=document.document_id,
        text=content,
        start_offset=start,
        end_offset=end,
        token_count=max(1, len(tokenize(content))),
        source_path=document.source_path,
        metadata=document.metadata | {"title": document.title, "chunk_index": index},
    )


def _chunk_document_chars(document: Document, chunking: ChunkingConfig) -> list[DocumentChunk]:
    """Split a document into boundary-aware character chunks."""
    text = document.text
    chunks: list[DocumentChunk] = []
    index = 0
    start = 0

    while start < len(text):
        candidate_end = min(len(text), start + chunking.chunk_size)
        end = _snap_end_to_boundary(text, start, candidate_end)
        content_start, content_end = _trimmed_span(text, start, end)

        if content_start < content_end:
            chunks.append(
                _build_chunk(
                    document,
                    index=index,
                    start=content_start,
                    end=content_end,
                )
            )
            index += 1

        if end >= len(text):
            break

        candidate_start = max(start + 1, end - chunking.overlap)
        next_start = _snap_start_to_boundary(text, candidate_start, start)
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def _chunk_document_sentences(
    document: Document,
    chunking: ChunkingConfig,
) -> list[DocumentChunk]:
    """Group complete sentences into chunks up to the configured size."""
    text = document.text
    spans = _sentence_spans(text)
    if not spans:
        return []

    chunks: list[DocumentChunk] = []
    index = 0
    sentence_index = 0

    while sentence_index < len(spans):
        start, first_end = spans[sentence_index]

        if first_end - start > chunking.chunk_size:
            candidate_end = min(len(text), start + chunking.chunk_size)
            end = _snap_end_to_boundary(text, start, candidate_end)
            content_start, content_end = _trimmed_span(text, start, end)
            if content_start < content_end:
                chunks.append(
                    _build_chunk(
                        document,
                        index=index,
                        start=content_start,
                        end=content_end,
                    )
                )
                index += 1
            if end >= first_end:
                sentence_index += 1
            else:
                spans[sentence_index] = (end, first_end)
            continue

        end = first_end
        next_index = sentence_index + 1
        while next_index < len(spans):
            _, candidate_end = spans[next_index]
            if candidate_end - start > chunking.chunk_size:
                break
            end = candidate_end
            next_index += 1

        chunks.append(
            _build_chunk(
                document,
                index=index,
                start=start,
                end=end,
            )
        )
        index += 1

        if next_index >= len(spans):
            break

        if chunking.overlap == 0:
            sentence_index = next_index
            continue

        overlap_start = max(start + 1, end - chunking.overlap)
        overlapping_sentence_index = next_index
        for candidate_index in range(sentence_index, next_index):
            sentence_start, _ = spans[candidate_index]
            if sentence_start >= overlap_start:
                overlapping_sentence_index = candidate_index
                break
        sentence_index = max(sentence_index + 1, overlapping_sentence_index)

    return chunks


def chunk_document(document: Document, config: ChunkingConfig | None = None) -> list[DocumentChunk]:
    """Split a document using the configured chunking strategy."""
    chunking = config or ChunkingConfig()
    if chunking.chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunking.overlap < 0 or chunking.overlap >= chunking.chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")
    if chunking.strategy not in SUPPORTED_CHUNKING_STRATEGIES:
        supported = ", ".join(sorted(SUPPORTED_CHUNKING_STRATEGIES))
        raise ValueError(
            f"Unsupported chunking strategy: {chunking.strategy}. Supported: {supported}."
        )
    if chunking.strategy == "sentence":
        return _chunk_document_sentences(document, chunking)
    return _chunk_document_chars(document, chunking)


def ingest_directory(
    input_dir: Path,
    output_path: Path,
    config: ChunkingConfig | None = None,
    pdf_extractor: PdfExtractor | None = None,
) -> list[DocumentChunk]:
    """Ingest a directory and persist chunks as JSONL."""
    chunks: list[DocumentChunk] = []
    for document in discover_documents(input_dir, pdf_extractor=pdf_extractor):
        chunks.extend(chunk_document(document, config=config))
    save_chunks_jsonl(chunks, output_path)
    return chunks


def save_chunks_jsonl(chunks: list[DocumentChunk], output_path: Path) -> None:
    """Persist chunks to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(chunk.model_dump_json())
            handle.write("\n")


def load_chunks_jsonl(path: Path) -> list[DocumentChunk]:
    """Load chunks from JSONL."""
    with path.open("r", encoding="utf-8") as handle:
        return [DocumentChunk.model_validate(json.loads(line)) for line in handle if line.strip()]
