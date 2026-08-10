"""Turn raw corpus files (PDF, HTML) into plain text.

Line structure is preserved deliberately: the article chunker keys off headers
that start a line, so flattening everything into one blob would hide them.
"""

import re
from pathlib import Path

import pymupdf
import trafilatura
from selectolax.parser import HTMLParser

_MARKUP_WHITESPACE = re.compile(r"\s+")
_INLINE_WHITESPACE = re.compile(r"[^\S\n]+")
# Script/style are never content. Nav, header, footer and aside are site
# furniture that government portals wrap the actual norm in — left in, the menu
# ("Presidencia de la Nación · Pasar al contenido principal · Buscar…") gets
# chunked and embedded like regulation, and it ranks above the articles on any
# query resembling the site's own name.
_BLOCK_TAGS = ("script", "style", "noscript", "nav", "header", "footer", "aside")

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
    """Main content first, whole-DOM text as a fallback.

    Government portals wrap each norm in navigation, breadcrumbs and search
    widgets held in anonymous <div>s — the Argentine site is a Drupal build with
    no <nav>, <header> or <footer> element at all — so there is no tag to filter
    on. Left in, that chrome gets chunked and embedded like regulation and
    outranks the actual articles on any query resembling the site's own name.
    A main-content extractor finds the article body by layout instead of markup.
    """
    raw = path.read_text(encoding="utf-8", errors="ignore")
    tree = _without_chrome_tags(raw)

    extracted = trafilatura.extract(
        tree.html or "",
        include_tables=True,
        include_comments=False,
        # These pages are mostly legal boilerplate by density; the default
        # precision setting discards annexes along with the chrome.
        favor_recall=True,
    )
    if extracted and extracted.strip():
        return extracted

    return _from_html_dom(tree)


def _without_chrome_tags(raw: str) -> HTMLParser:
    """Drop tags that are never article text, before content extraction runs.

    Semantic chrome is unambiguous, so removing it up front costs nothing and
    helps on the pages that do use it. It is not sufficient on its own — the
    portals in this corpus use anonymous <div>s — which is why trafilatura runs
    afterwards.
    """
    # Whitespace is insignificant in HTML markup, so collapsing it up front
    # means a newline in the output can only have come from a block boundary.
    tree = HTMLParser(_MARKUP_WHITESPACE.sub(" ", raw))
    for tag in _BLOCK_TAGS:
        for node in tree.css(tag):
            node.decompose()
    return tree


def _from_html_dom(tree: HTMLParser) -> str:
    """Every remaining text node — used when there is no main content to find."""
    root = tree.body or tree.root
    return root.text(separator="\n") if root else ""


def _normalise(text: str) -> str:
    text = text.replace("\xa0", " ")
    lines = (_INLINE_WHITESPACE.sub(" ", line).strip() for line in text.split("\n"))
    return "\n".join(
        line for line in lines if line and not _DOT_LEADERS.search(line)
    )
