"""Corpus ingestion: raw file -> text -> chunks -> flat records."""

from pathlib import Path
from typing import Any

from regrag.chunking import Chunk, chunk_by_article, chunk_fixed, chunk_hybrid
from regrag.extraction import extract_text
from regrag.metadata import parse_source_filename

DEFAULT_SIZE = 1200
DEFAULT_OVERLAP = 200


def chunk_document(
    path: Path,
    strategy: str,
    size: int = DEFAULT_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    source = parse_source_filename(path.name)
    text = extract_text(path)

    if strategy == "article":
        return chunk_by_article(text, source)
    if strategy == "fixed":
        return chunk_fixed(text, source, size=size, overlap=overlap)
    if strategy == "hybrid":
        return chunk_hybrid(text, source, max_size=size, overlap=overlap)
    raise ValueError(f"Unknown strategy {strategy!r}")


def to_record(chunk: Chunk) -> dict[str, Any]:
    """Flatten a chunk into the row shape the vector store will index."""
    source = chunk.source
    document = "_".join(
        part
        for part in (source.country, source.agency, source.norm_id, source.variant)
        if part
    )
    return {
        "chunk_id": f"{document}#{chunk.strategy}#{chunk.index}",
        "text": chunk.text,
        "country": source.country,
        "agency": source.agency,
        "norm_id": source.norm_id,
        "variant": source.variant,
        "year": source.year,
        "article": chunk.article,
        "strategy": chunk.strategy,
        "index": chunk.index,
        "n_chars": len(chunk.text),
    }
