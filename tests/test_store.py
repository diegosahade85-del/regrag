import pytest

from regrag.store import create_schema, search_dense, upsert_chunks

DIM = 4

# Deterministic unit-ish vectors: cosine distance between them is predictable,
# so ranking assertions below don't depend on a real embedding model.
EAST = [1.0, 0.0, 0.0, 0.0]
NORTH = [0.0, 1.0, 0.0, 0.0]
NORTHEAST = [0.7071, 0.7071, 0.0, 0.0]


def record(chunk_id, text, **overrides):
    base = {
        "chunk_id": chunk_id,
        "text": text,
        "country": "AR",
        "agency": "SIC",
        "norm_id": "res-16-2025",
        "variant": "texto",
        "year": 2025,
        "article": "1",
        "strategy": "hybrid",
        "index": 0,
        "n_chars": len(text),
    }
    return base | overrides


@pytest.fixture
def db(conn):
    create_schema(conn, dim=DIM)
    return conn


def test_create_schema_is_idempotent(conn):
    create_schema(conn, dim=DIM)
    create_schema(conn, dim=DIM)  # must not raise


def test_upsert_stores_chunks(db):
    inserted = upsert_chunks(db, [record("a#hybrid#0", "texto uno")], [EAST])

    assert inserted == 1


def test_reingesting_the_same_chunk_id_updates_instead_of_duplicating(db):
    upsert_chunks(db, [record("a#hybrid#0", "versión vieja")], [EAST])
    upsert_chunks(db, [record("a#hybrid#0", "versión nueva")], [NORTH])

    hits = search_dense(db, NORTH, limit=10)
    assert len(hits) == 1
    assert hits[0].text == "versión nueva"


def test_search_ranks_by_vector_similarity(db):
    upsert_chunks(
        db,
        [record("east#hybrid#0", "este"), record("north#hybrid#0", "norte")],
        [EAST, NORTH],
    )

    hits = search_dense(db, NORTHEAST, limit=2)

    assert {h.chunk_id for h in hits} == {"east#hybrid#0", "north#hybrid#0"}
    assert hits[0].score == pytest.approx(hits[1].score, abs=1e-3)


def test_nearest_neighbour_comes_first(db):
    upsert_chunks(
        db,
        [record("east#hybrid#0", "este"), record("north#hybrid#0", "norte")],
        [EAST, NORTH],
    )

    (top, _) = search_dense(db, [0.9, 0.1, 0.0, 0.0], limit=2)

    assert top.chunk_id == "east#hybrid#0"


def test_search_respects_the_limit(db):
    records = [record(f"c#hybrid#{i}", f"texto {i}") for i in range(5)]
    upsert_chunks(db, records, [EAST] * 5)

    assert len(search_dense(db, EAST, limit=3)) == 3


def test_identical_vectors_score_one(db):
    upsert_chunks(db, [record("a#hybrid#0", "texto")], [EAST])

    (hit,) = search_dense(db, EAST, limit=1)

    assert hit.score == pytest.approx(1.0, abs=1e-6)


def test_results_carry_the_metadata_a_citation_needs(db):
    upsert_chunks(db, [record("a#hybrid#0", "ARTÍCULO 1°.- OBJETO.")], [EAST])

    (hit,) = search_dense(db, EAST, limit=1)

    assert hit.country == "AR"
    assert hit.norm_id == "res-16-2025"
    assert hit.article == "1"
    assert hit.text == "ARTÍCULO 1°.- OBJETO."


def test_country_filter_excludes_other_jurisdictions(db):
    upsert_chunks(
        db,
        [
            record("ar#hybrid#0", "norma argentina", country="AR"),
            record("cl#hybrid#0", "norma chilena", country="CL"),
        ],
        [EAST, EAST],
    )

    hits = search_dense(db, EAST, limit=10, country="CL")

    assert [h.chunk_id for h in hits] == ["cl#hybrid#0"]


def test_search_on_empty_table_returns_nothing(db):
    assert search_dense(db, EAST, limit=5) == []


def test_upsert_rejects_an_embedding_count_mismatch(db):
    with pytest.raises(ValueError, match="embeddings"):
        upsert_chunks(db, [record("a#hybrid#0", "uno")], [EAST, NORTH])


def test_existing_chunk_ids_is_empty_for_a_fresh_table(db):
    from regrag.store import existing_chunk_ids

    assert existing_chunk_ids(db) == set()


def test_existing_chunk_ids_reports_what_is_already_indexed(db):
    from regrag.store import existing_chunk_ids

    upsert_chunks(
        db,
        [record('a#hybrid#0', 'uno'), record('b#hybrid#1', 'dos')],
        [EAST, NORTH],
    )

    assert existing_chunk_ids(db) == {'a#hybrid#0', 'b#hybrid#1'}
