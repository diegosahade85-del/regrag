import pymupdf
import pytest

from regrag.extraction import extract_text


def write_html(tmp_path, body: str):
    path = tmp_path / "AR_SIC_res-16-2025_texto.html"
    path.write_text(f"<html><body>{body}</body></html>", encoding="utf-8")
    return path


def test_extracts_visible_text_from_html(tmp_path):
    path = write_html(tmp_path, "<p>ARTÍCULO 1°.- Apruébase el Reglamento.</p>")

    assert "ARTÍCULO 1°.- Apruébase el Reglamento." in extract_text(path)


def test_drops_script_and_style_content(tmp_path):
    path = write_html(
        tmp_path,
        "<script>var oculto = 'no debe aparecer';</script>"
        "<style>.x { color: red; }</style>"
        "<p>Texto legal visible.</p>",
    )

    text = extract_text(path)

    assert "Texto legal visible." in text
    assert "no debe aparecer" not in text
    assert "color: red" not in text


def test_collapses_markup_whitespace_into_single_spaces(tmp_path):
    path = write_html(tmp_path, "<p>ARTÍCULO   1°.-\n\n   OBJETO.</p>")

    assert "ARTÍCULO 1°.- OBJETO." in extract_text(path)


def test_decodes_html_entities(tmp_path):
    path = write_html(tmp_path, "<p>Resoluci&oacute;n N&deg;&nbsp;16/2025</p>")

    assert "Resolución N° 16/2025" in extract_text(path)


def test_extracts_text_from_pdf(tmp_path):
    path = tmp_path / "AR_SIC_res-16-2025.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "ARTICULO 1 - Apruebase el Reglamento.")
    doc.save(path)
    doc.close()

    assert "ARTICULO 1 - Apruebase el Reglamento." in extract_text(path)


def test_keeps_page_text_in_order_for_multipage_pdf(tmp_path):
    path = tmp_path / "AR_SIC_res-16-2025.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Primera pagina.")
    doc.new_page().insert_text((72, 72), "Segunda pagina.")
    doc.save(path)
    doc.close()

    text = extract_text(path)

    assert text.index("Primera pagina.") < text.index("Segunda pagina.")


def test_rejects_unsupported_file_type(tmp_path):
    path = tmp_path / "AR_SIC_res-16-2025.docx"
    path.write_bytes(b"irrelevant")

    with pytest.raises(ValueError, match="Unsupported"):
        extract_text(path)


def test_drops_table_of_contents_lines_with_dot_leaders(tmp_path):
    path = write_html(
        tmp_path,
        '<p>ARTÍCULO 20º. REQUERIMIENTOS ....................... 83</p>'
        '<p>ARTÍCULO 20º. REQUERIMIENTOS. Los productos deberán cumplir.</p>',
    )

    text = extract_text(path)

    assert '83' not in text
    assert 'Los productos deberán cumplir.' in text
    assert text.count('ARTÍCULO 20º') == 1


def test_keeps_prose_containing_an_ellipsis(tmp_path):
    path = write_html(tmp_path, '<p>El equipamiento... deberá certificarse.</p>')

    assert 'El equipamiento... deberá certificarse.' in extract_text(path)
