"""Chunking strategies for regulatory text.

Three strategies, so the choice can be settled by measurement rather than
intuition (see README):

- `article` — split on article boundaries. Semantically right, but some norms
  have articles that run tens of thousands of characters.
- `fixed`   — structure-blind overlapping windows. The baseline.
- `hybrid`  — article boundaries first, windows inside oversized articles,
  article label preserved on every piece so citations still resolve.
"""

import re
from dataclasses import dataclass

from regrag.metadata import SourceMetadata

# An article header must start a line and be followed by a separator. Both
# conditions matter: cross-references like "conforme al artículo 4° de la
# presente" appear constantly mid-sentence, and splitting on them shreds the
# article that actually contains the obligation.
_ARTICLE_HEADER = re.compile(
    r"^[ \t]*(?:ART[IÍ]CULO|ART\.)\s*(\d+)\s*[°º]?\s*[.\-–]",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class Chunk:
    text: str
    article: str | None
    source: SourceMetadata
    strategy: str
    index: int


def chunk_by_article(text: str, source: SourceMetadata) -> list[Chunk]:
    """Split on article boundaries, keeping each article whole."""
    return _assemble(_split_articles(text), source, strategy="article")


def chunk_fixed(
    text: str, source: SourceMetadata, size: int, overlap: int
) -> list[Chunk]:
    """Split into overlapping windows of `size` characters, snapped to words.

    The structure-blind baseline the other strategies are measured against.
    """
    _check_overlap(size, overlap)
    spans = [(None, body) for body in _windows(text, size, overlap)]
    return _assemble(spans, source, strategy="fixed")


def chunk_hybrid(
    text: str, source: SourceMetadata, max_size: int, overlap: int
) -> list[Chunk]:
    """Article boundaries first; window only the articles that overflow."""
    _check_overlap(max_size, overlap)

    spans: list[tuple[str | None, str]] = []
    for article, body in _split_articles(text):
        if len(body) <= max_size:
            spans.append((article, body))
            continue
        spans.extend((article, window) for window in _windows(body, max_size, overlap))

    return _assemble(spans, source, strategy="hybrid")


def _split_articles(text: str) -> list[tuple[str | None, str]]:
    """Text -> [(article number or None for the preamble, body)]."""
    spans: list[tuple[str | None, str]] = []
    matches = list(_ARTICLE_HEADER.finditer(text))

    preamble = text[: matches[0].start()] if matches else text
    if preamble.strip():
        spans.append((None, preamble.strip()))

    for current, following in zip(matches, matches[1:] + [None]):
        end = following.start() if following else len(text)
        spans.append((current.group(1), text[current.start() : end].strip()))

    return spans


def _windows(text: str, size: int, overlap: int) -> list[str]:
    """Overlapping character windows that never cut a word in half."""
    words = text.split()
    if not words:
        return []

    bodies: list[str] = []
    start = 0
    while start < len(words):
        window: list[str] = []
        length = 0
        end = start
        while end < len(words) and length + len(words[end]) + bool(window) <= size:
            length += len(words[end]) + bool(window)
            window.append(words[end])
            end += 1

        # A single word longer than the budget would otherwise loop forever.
        if not window:
            window, end = [words[start]], start + 1

        bodies.append(" ".join(window))
        if end >= len(words):
            break

        # Step back over the tail that fits in the overlap budget.
        back, tail = end, 0
        while back > start + 1 and tail + len(words[back - 1]) + 1 <= overlap:
            tail += len(words[back - 1]) + 1
            back -= 1
        start = back

    return bodies


def _assemble(
    spans: list[tuple[str | None, str]], source: SourceMetadata, strategy: str
) -> list[Chunk]:
    return [
        Chunk(
            text=body.strip(),
            article=article,
            source=source,
            strategy=strategy,
            index=index,
        )
        for index, (article, body) in enumerate(spans)
        if body.strip()
    ]


def _check_overlap(size: int, overlap: int) -> None:
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")
