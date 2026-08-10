import pytest

from regrag.store import create_schema, search_hybrid, upsert_chunks

DIM = 4
EAST = [1.0, 0.0, 0.0, 0.0]
NORTH = [0.0, 1.0, 0.0, 0.0]


def record(chunk_id, text, **overrides):
    base = {
        "chunk_id": chunk_id,
        "text": text,
        "country": "AR",
        "agency": "SIC",
        "norm_id": "res-16-2025",
        "variant": "texto",
        "year": 2025,
        "article": "3",
        "strategy": "hybrid",
        "index": 0,
        "n_chars": len(text),
    }
    return base | overrides


@pytest.fixture
def db(conn):
    create_schema(conn, dim=DIM)
    return conn


def test_surfaces_a_result_only_lexical_search_can_find(db):
    """The IEC 60364 case: the vector is pointed away from this chunk, so dense
    search will never reach it, and the standard number is the whole query."""
    upsert_chunks(
        db,
        [
            record("norma#hybrid#0", "Cumplirá la norma IEC 60364 en lo aplicable."),
            record("otro#hybrid#1", "Texto sobre certificación de productos."),
        ],
        [NORTH, EAST],
    )

    hits = search_hybrid(db, EAST, "IEC 60364", limit=5)

    assert "norma#hybrid#0" in {h.chunk_id for h in hits}


def test_surfaces_a_result_only_dense_search_can_find(db):
    """A chunk sharing no term with the query still reaches the results."""
    upsert_chunks(
        db,
        [
            record("cercano#hybrid#0", "Disposiciones sobre rotulado obligatorio."),
            record("lexico#hybrid#1", "El marcado deberá ser indeleble."),
        ],
        [EAST, NORTH],
    )

    hits = search_hybrid(db, EAST, "marcado", limit=5)

    assert "cercano#hybrid#0" in {h.chunk_id for h in hits}


def test_a_result_both_retrievers_rank_comes_first(db):
    upsert_chunks(
        db,
        [
            record("ambos#hybrid#0", "El marcado del producto deberá ser indeleble."),
            record("solo_denso#hybrid#1", "Disposiciones generales del reglamento."),
            record("solo_lexico#hybrid#2", "marcado", country="AR"),
        ],
        [EAST, EAST, NORTH],
    )

    hits = search_hybrid(db, EAST, "marcado producto indeleble", limit=3)

    assert hits[0].chunk_id == "ambos#hybrid#0"


def test_respects_the_limit(db):
    records = [record(f"c#hybrid#{i}", f"marcado obligatorio {i}") for i in range(8)]
    upsert_chunks(db, records, [EAST] * 8)

    assert len(search_hybrid(db, EAST, "marcado", limit=3)) == 3


def test_country_filter_applies_to_both_retrievers(db):
    upsert_chunks(
        db,
        [
            record("ar#hybrid#0", "requisitos de marcado", country="AR"),
            record("cl#hybrid#1", "requisitos de marcado", country="CL"),
        ],
        [EAST, EAST],
    )

    hits = search_hybrid(db, EAST, "marcado", limit=5, country="CL")

    assert [h.chunk_id for h in hits] == ["cl#hybrid#1"]


def test_results_carry_the_metadata_a_citation_needs(db):
    upsert_chunks(db, [record("a#hybrid#0", "ARTÍCULO 3°.- FABRICANTES.")], [EAST])

    (hit,) = search_hybrid(db, EAST, "fabricantes", limit=5)

    assert (hit.country, hit.norm_id, hit.article) == ("AR", "res-16-2025", "3")


def test_a_query_matching_no_terms_still_returns_dense_results(db):
    upsert_chunks(db, [record("a#hybrid#0", "Disposiciones generales.")], [EAST])

    assert search_hybrid(db, EAST, "transformadores trifásicos", limit=5)


def test_an_empty_query_text_degrades_to_dense_search(db):
    upsert_chunks(db, [record("a#hybrid#0", "Disposiciones generales.")], [EAST])

    assert [h.chunk_id for h in search_hybrid(db, EAST, "", limit=5)] == ["a#hybrid#0"]


def test_searching_an_empty_table_returns_nothing(db):
    assert search_hybrid(db, EAST, "marcado", limit=5) == []


def test_no_duplicate_chunks_when_both_retrievers_find_the_same_one(db):
    upsert_chunks(db, [record("a#hybrid#0", "El marcado deberá ser indeleble.")], [EAST])

    hits = search_hybrid(db, EAST, "marcado", limit=5)

    assert len(hits) == len({h.chunk_id for h in hits}) == 1


def test_the_candidate_pool_is_wider_than_the_requested_limit(db):
    """Fusing only the top-N of each ranker would discard the very results that
    win on combined support."""
    records = [record(f"denso#hybrid#{i}", f"documento {i}") for i in range(10)]
    records.append(record("lexico#hybrid#99", "marcado indeleble obligatorio"))
    upsert_chunks(db, records, [EAST] * 10 + [NORTH])

    hits = search_hybrid(db, EAST, "marcado", limit=3)

    assert "lexico#hybrid#99" in {h.chunk_id for h in hits}
