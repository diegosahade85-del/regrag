import pytest

from regrag.store import create_schema, search_lexical, upsert_chunks

DIM = 4
VEC = [1.0, 0.0, 0.0, 0.0]


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


def load(db, *pairs):
    records = [record(cid, text, **extra) for cid, text, extra in pairs]
    upsert_chunks(db, records, [VEC] * len(records))


def test_finds_the_chunk_containing_a_term(db):
    load(
        db,
        ("a#hybrid#0", "Los fabricantes deberán presentar una declaración.", {}),
        ("b#hybrid#1", "Ámbito de aplicación del presente reglamento.", {}),
    )

    hits = search_lexical(db, "declaración", limit=10)

    assert [h.chunk_id for h in hits] == ["a#hybrid#0"]


def test_finds_a_standard_number_that_embeddings_miss(db):
    """The reason lexical search is here at all."""
    load(
        db,
        ("a#hybrid#0", "Deberá cumplir con la norma IEC 60364 en lo aplicable.", {}),
        ("b#hybrid#1", "Deberá cumplir con la norma IEC 60335 en lo aplicable.", {}),
    )

    hits = search_lexical(db, "IEC 60364", limit=10)

    assert [h.chunk_id for h in hits] == ["a#hybrid#0"]


def test_distinguishes_standards_differing_by_one_digit(db):
    load(
        db,
        ("a#hybrid#0", "norma NTC 2050 aplicable a instalaciones", {}),
        ("b#hybrid#1", "norma NTC 4552 aplicable a instalaciones", {}),
    )

    assert [h.chunk_id for h in search_lexical(db, "NTC 4552", 10)] == ["b#hybrid#1"]


def test_matches_across_spanish_inflection(db):
    load(db, ("a#hybrid#0", "Los productos importados deberán certificarse.", {}))

    assert search_lexical(db, "producto importado", limit=10)


def test_ignores_stopwords_rather_than_requiring_them(db):
    load(db, ("a#hybrid#0", "Requisitos esenciales de seguridad eléctrica.", {}))

    assert search_lexical(db, "los requisitos de la seguridad", limit=10)


def test_ranks_denser_matches_first(db):
    load(
        db,
        ("many#hybrid#0", "marcado. El marcado del producto. Todo marcado visible.", {}),
        ("one#hybrid#1", "Un texto largo sobre certificación que menciona marcado.", {}),
    )

    hits = search_lexical(db, "marcado", limit=2)

    assert hits[0].chunk_id == "many#hybrid#0"


def test_results_carry_the_metadata_a_citation_needs(db):
    load(db, ("a#hybrid#0", "ARTÍCULO 3°.- FABRICANTES E IMPORTADORES.", {}))

    (hit,) = search_lexical(db, "fabricantes", limit=10)

    assert (hit.country, hit.norm_id, hit.article) == ("AR", "res-16-2025", "3")


def test_country_filter_excludes_other_jurisdictions(db):
    load(
        db,
        ("ar#hybrid#0", "requisitos de marcado", {"country": "AR"}),
        ("cl#hybrid#1", "requisitos de marcado", {"country": "CL"}),
    )

    hits = search_lexical(db, "marcado", limit=10, country="CL")

    assert [h.chunk_id for h in hits] == ["cl#hybrid#1"]


def test_respects_the_limit(db):
    load(db, *[(f"c#hybrid#{i}", "marcado obligatorio", {}) for i in range(5)])

    assert len(search_lexical(db, "marcado", limit=3)) == 3


def test_no_match_returns_nothing(db):
    load(db, ("a#hybrid#0", "requisitos de marcado", {}))

    assert search_lexical(db, "transformadores", limit=10) == []


def test_an_empty_query_returns_nothing_rather_than_everything(db):
    load(db, ("a#hybrid#0", "requisitos de marcado", {}))

    assert search_lexical(db, "   ", limit=10) == []


def test_punctuation_only_query_does_not_error(db):
    load(db, ("a#hybrid#0", "requisitos de marcado", {}))

    assert search_lexical(db, "?? -- ///", limit=10) == []
