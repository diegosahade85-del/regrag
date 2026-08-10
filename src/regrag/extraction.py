"""Turn raw corpus files (PDF, HTML) into plain text.

Line structure is preserved deliberately: the article chunker keys off headers
that start a line, so flattening everything into one blob would hide them.
"""

import re
from pathlib import Path

import pymupdf
from selectolax.parser import HTMLParser

_MARKUP_WHITESPACE = re.compile(r"\s+")
_INLINE_WHITESPACE = re.compile(r"[^\S\n]+")
_BLOCK_TAGS = ("script", "style", "noscript")

# Dot leaders mark a table-of-contents entry. Dropping those lines is not
# cosmetic: TOC entries look exactly like article headers, so left in place they
# capture an article boundary at the top of the document and the real article
# body downstream gets absorbed into whatever span precedes it. Four dots keeps
# a prose ellipsis ("el equipamiento... deberá") from being caught.
_DOT_LEADERS = re.compile(r"\.{4,}")


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _normalise(_from_pdf(path))
    if suffix in {".html", ".htm"}:
        return _normalise(_from_html(path))
    raise ValueError(f"Unsupported file type {suffix!r}: {path.name}")


def _from_pdf(path: Path) -> str:
    with pymupdf.open(path) as doc:
        return "\n".join(page.get_text() for page in doc)


def _from_html(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    # Whitespace is insignificant in HTML markup, so collapsing it up front
    # means a newline in the output can only have come from a block boundary.
    raw = _MARKUP_WHITESPACE.sub(" ", raw)

    tree = HTMLParser(raw)
    for tag in _BLOCK_TAGS:
        for node in tree.css(tag):
            node.decompose()

    root = tree.body or tree.root
    return root.text(separator="\n") if root else ""


def _normalise(text: str) -> str:
    text = text.replace("\xa0", " ")
    lines = (_INLINE_WHITESPACE.sub(" ", line).strip() for line in text.split("\n"))
    return "\n".join(
        line for line in lines if line and not _DOT_LEADERS.search(line)
    )
