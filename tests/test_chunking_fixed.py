import pytest

from regrag.chunking import chunk_fixed
from regrag.metadata import SourceMetadata

SOURCE = SourceMetadata(
    country="AR", agency="SIC", norm_id="res-16-2025", variant="texto", year=2025
)

LONG_TEXT = " ".join(f"palabra{i:03d}" for i in range(200))


def test_short_text_fits_in_a_single_chunk():
    chunks = chunk_fixed("Texto breve.", SOURCE, size=100, overlap=20)

    assert len(chunks) == 1
    assert chunks[0].text == "Texto breve."


def test_long_text_is_split_into_several_chunks_within_the_size_budget():
    chunks = chunk_fixed(LONG_TEXT, SOURCE, size=200, overlap=50)

    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)


def test_consecutive_chunks_share_text_so_context_is_not_cut_at_the_seam():
    chunks = chunk_fixed(LONG_TEXT, SOURCE, size=200, overlap=50)

    first, second = set(chunks[0].text.split()), set(chunks[1].text.split())
    assert first & second


def test_no_chunk_ever_splits_a_word_in_half():
    original = set(LONG_TEXT.split())

    chunks = chunk_fixed(LONG_TEXT, SOURCE, size=200, overlap=50)

    for chunk in chunks:
        assert set(chunk.text.split()) <= original


def test_no_word_is_lost_between_chunks():
    chunks = chunk_fixed(LONG_TEXT, SOURCE, size=200, overlap=50)

    covered = set().union(*(set(c.text.split()) for c in chunks))
    assert covered == set(LONG_TEXT.split())


def test_chunks_carry_metadata_strategy_and_order():
    chunks = chunk_fixed(LONG_TEXT, SOURCE, size=200, overlap=50)

    assert all(c.source == SOURCE for c in chunks)
    assert all(c.strategy == "fixed" for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_fixed_chunks_carry_no_article_label():
    chunks = chunk_fixed(LONG_TEXT, SOURCE, size=200, overlap=50)

    assert all(c.article is None for c in chunks)


def test_blank_document_yields_no_chunks():
    assert chunk_fixed("   \n\n ", SOURCE, size=200, overlap=50) == []


def test_overlap_equal_to_or_larger_than_size_is_rejected():
    with pytest.raises(ValueError, match="overlap"):
        chunk_fixed(LONG_TEXT, SOURCE, size=200, overlap=200)
