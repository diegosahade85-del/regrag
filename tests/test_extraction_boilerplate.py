"""Boilerplate removal on realistically-shaped government portal HTML.

The pages in this corpus carry their chrome in anonymous <div>s — the Argentine
portal is a Drupal build with no <nav>, <header> or <footer> element anywhere —
so tag-based filtering cannot reach it. These tests are written against page
shapes big enough for a main-content extractor to work on, which is what the
fragments in test_extraction.py are not.
"""

from regrag.extraction import extract_text

ARTICLES = "".join(
    f"<p>ARTÍCULO {n}°.- Los productos comprendidos en el presente reglamento "
    f"deberán cumplir los requisitos esenciales de seguridad establecidos en el "
    f"Anexo I, acreditando su conformidad mediante los procedimientos que el "
    f"organismo competente determine para cada categoría de producto.</p>"
    for n in range(1, 9)
)

PORTAL_PAGE = f"""
<html><body class="page-normativa">
  <div id="skip-link"><a>Pasar al contenido principal</a></div>
  <div class="barra-institucional">Presidencia de la Nación</div>
  <div class="menu"><ul>
    <li><a>Inicio</a></li><li><a>Ministerio de Justicia</a></li>
    <li><a>Normativa</a></li><li><a>Ir a Mi Argentina</a></li>
  </ul></div>
  <div id="content"><div class="field-item">{ARTICLES}</div></div>
  <div class="pie"><ul>
    <li><a>Ediciones Anteriores</a></li><li><a>Búsqueda avanzada</a></li>
  </ul></div>
</body></html>
"""


def write(tmp_path, html, name="AR_SIC_res-16-2025_texto.html"):
    path = tmp_path / name
    path.write_text(html, encoding="utf-8")
    return path


def test_keeps_the_articles(tmp_path):
    text = extract_text(write(tmp_path, PORTAL_PAGE))

    assert "ARTÍCULO 1°.-" in text
    assert "ARTÍCULO 8°.-" in text
    assert text.count("ARTÍCULO") == 8


def test_drops_chrome_held_in_anonymous_divs(tmp_path):
    text = extract_text(write(tmp_path, PORTAL_PAGE))

    for chrome in (
        "Pasar al contenido principal",
        "Presidencia de la Nación",
        "Ir a Mi Argentina",
        "Ediciones Anteriores",
        "Búsqueda avanzada",
    ):
        assert chrome not in text, f"{chrome!r} survived extraction"


def test_article_headers_still_start_their_own_line(tmp_path):
    """The article chunker keys off line-initial headers — extraction must
    not run articles together onto one line."""
    text = extract_text(write(tmp_path, PORTAL_PAGE))

    starts = [line for line in text.splitlines() if line.startswith("ARTÍCULO")]
    assert len(starts) == 8
