import pytest

from regrag.chunking import chunk_by_article
from regrag.metadata import SourceMetadata

SOURCE = SourceMetadata(
    country="AR", agency="SIC", norm_id="res-16-2025", variant="texto", year=2025
)


def test_splits_one_chunk_per_article():
    text = (
        "ARTÍCULO 1°.- OBJETO. Apruébase el Reglamento Técnico.\n"
        "ARTÍCULO 2°.- ÁMBITO. Se aplica al equipamiento eléctrico.\n"
        "ARTÍCULO 3°.- Comuníquese.\n"
    )

    chunks = chunk_by_article(text, SOURCE)

    assert [c.article for c in chunks] == ["1", "2", "3"]
    assert "Apruébase el Reglamento" in chunks[0].text
    assert "equipamiento eléctrico" in chunks[1].text


def test_article_header_stays_inside_its_chunk():
    text = "ARTÍCULO 1°.- OBJETO. Apruébase el Reglamento Técnico.\n"

    (chunk,) = chunk_by_article(text, SOURCE)

    assert chunk.text.startswith("ARTÍCULO 1°.-")


def test_cross_reference_mid_sentence_does_not_start_a_new_chunk():
    text = (
        "ARTÍCULO 5°.- SANCIONES. Las infracciones previstas en el artículo 4° "
        "de la presente resolución serán sancionadas conforme al artículo 47 "
        "de la Ley N° 24.240.\n"
    )

    chunks = chunk_by_article(text, SOURCE)

    assert len(chunks) == 1
    assert chunks[0].article == "5"
    assert "Ley N° 24.240" in chunks[0].text


def test_recognises_accent_and_ordinal_spelling_variants():
    text = (
        "ARTICULO 1º - Primero.\n"
        "Artículo 2.- Segundo.\n"
        "Art. 3°.- Tercero.\n"
    )

    chunks = chunk_by_article(text, SOURCE)

    assert [c.article for c in chunks] == ["1", "2", "3"]


def test_text_before_the_first_article_is_kept_as_a_preamble_chunk():
    text = (
        "VISTO el Expediente N° 123, y CONSIDERANDO que corresponde actualizar.\n"
        "ARTÍCULO 1°.- OBJETO. Apruébase el Reglamento.\n"
    )

    chunks = chunk_by_article(text, SOURCE)

    assert chunks[0].article is None
    assert "CONSIDERANDO" in chunks[0].text
    assert chunks[1].article == "1"


def test_document_with_no_articles_yields_a_single_articleless_chunk():
    text = "Guía informativa sobre certificación de productos eléctricos.\n"

    (chunk,) = chunk_by_article(text, SOURCE)

    assert chunk.article is None
    assert "Guía informativa" in chunk.text


def test_chunks_carry_source_metadata_and_strategy():
    text = "ARTÍCULO 1°.- OBJETO. Apruébase el Reglamento.\n"

    (chunk,) = chunk_by_article(text, SOURCE)

    assert chunk.source == SOURCE
    assert chunk.strategy == "article"


def test_chunks_are_indexed_in_document_order():
    text = "ARTÍCULO 1°.- Uno.\nARTÍCULO 2°.- Dos.\nARTÍCULO 3°.- Tres.\n"

    chunks = chunk_by_article(text, SOURCE)

    assert [c.index for c in chunks] == [0, 1, 2]


def test_blank_document_yields_no_chunks():
    assert chunk_by_article("   \n\n  ", SOURCE) == []
