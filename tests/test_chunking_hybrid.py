from regrag.chunking import chunk_hybrid
from regrag.metadata import SourceMetadata

SOURCE = SourceMetadata(
    country="CO", agency="MINENERGIA", norm_id="res-9703", variant=None, year=None
)

FILLER = " ".join(f"requisito{i:04d}" for i in range(500))


def test_articles_within_budget_are_left_whole():
    text = "ARTÍCULO 1°.- Breve.\nARTÍCULO 2°.- También breve.\n"

    chunks = chunk_hybrid(text, SOURCE, max_size=1000, overlap=100)

    assert [c.article for c in chunks] == ["1", "2"]
    assert chunks[0].text.startswith("ARTÍCULO 1°.-")


def test_oversized_article_is_split_into_several_chunks():
    text = f"ARTÍCULO 20.- TABLAS. {FILLER}\n"

    chunks = chunk_hybrid(text, SOURCE, max_size=1000, overlap=100)

    assert len(chunks) > 1


def test_every_piece_of_a_split_article_keeps_the_article_label():
    text = f"ARTÍCULO 20.- TABLAS. {FILLER}\n"

    chunks = chunk_hybrid(text, SOURCE, max_size=1000, overlap=100)

    assert all(c.article == "20" for c in chunks)


def test_no_chunk_exceeds_the_size_budget():
    text = f"ARTÍCULO 1°.- Breve.\nARTÍCULO 20.- TABLAS. {FILLER}\n"

    chunks = chunk_hybrid(text, SOURCE, max_size=1000, overlap=100)

    assert all(len(c.text) <= 1000 for c in chunks)


def test_splitting_one_article_does_not_disturb_its_neighbours():
    text = f"ARTÍCULO 1°.- Breve.\nARTÍCULO 20.- TABLAS. {FILLER}\nARTÍCULO 21.- Fin.\n"

    chunks = chunk_hybrid(text, SOURCE, max_size=1000, overlap=100)

    assert chunks[0].article == "1"
    assert chunks[0].text == "ARTÍCULO 1°.- Breve."
    assert chunks[-1].article == "21"
    assert chunks[-1].text == "ARTÍCULO 21.- Fin."


def test_oversized_preamble_is_split_and_stays_articleless():
    text = f"VISTO el Expediente. {FILLER}\nARTÍCULO 1°.- Breve.\n"

    chunks = chunk_hybrid(text, SOURCE, max_size=1000, overlap=100)

    preamble = [c for c in chunks if c.article is None]
    assert len(preamble) > 1
    assert all(len(c.text) <= 1000 for c in preamble)


def test_indices_stay_sequential_across_the_document():
    text = f"ARTÍCULO 1°.- Breve.\nARTÍCULO 20.- TABLAS. {FILLER}\n"

    chunks = chunk_hybrid(text, SOURCE, max_size=1000, overlap=100)

    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_strategy_is_reported_as_hybrid():
    text = "ARTÍCULO 1°.- Breve.\n"

    (chunk,) = chunk_hybrid(text, SOURCE, max_size=1000, overlap=100)

    assert chunk.strategy == "hybrid"


def test_no_text_is_lost_when_an_article_is_split():
    text = f"ARTÍCULO 20.- TABLAS. {FILLER}\n"

    chunks = chunk_hybrid(text, SOURCE, max_size=1000, overlap=100)

    covered = set().union(*(set(c.text.split()) for c in chunks))
    assert covered == set(text.split())
