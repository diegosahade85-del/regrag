"""Checking golden-set citations against the corpus they claim to come from.

A citation typed by hand — a transposed article number, a norm id that reads
right but is not what the file is called — makes an entry unscoreable. Nothing
about it looks wrong on the page, so it has to be checked mechanically.
"""

import pytest

from regrag.answering import Citation
from regrag.golden import GoldenQuestion, GoldenSet, unknown_citations
from regrag.store import create_schema, upsert_chunks

DIM = 4
VEC = [1.0, 0.0, 0.0, 0.0]


def chunk(norm_id="res-16-2025", article="3", country="AR"):
    return {
        "chunk_id": f"{country}_{norm_id}#hybrid#{article}",
        "text": "Texto del artículo.",
        "country": country,
        "agency": "SIC",
        "norm_id": norm_id,
        "variant": "texto",
        "year": 2025,
        "article": article,
        "strategy": "hybrid",
        "index": 0,
        "n_chars": 20,
    }


def question(citations, qid="q001"):
    return GoldenQuestion(
        id=qid,
        question="¿Qué exige la norma?",
        kind="factual",
        answerable=True,
        expected_answer="Una declaración jurada.",
        citations=citations,
        status="reviewed",
    )


@pytest.fixture
def db(conn):
    create_schema(conn, dim=DIM)
    upsert_chunks(conn, [chunk(article="3"), chunk(article="4")], [VEC, VEC])
    return conn


def test_a_citation_present_in_the_corpus_passes(db):
    golden = GoldenSet(
        questions=[question([Citation(country="AR", norm_id="res-16-2025", article="3")])]
    )

    assert unknown_citations(golden, db) == []


def test_a_citation_to_a_nonexistent_article_is_reported(db):
    golden = GoldenSet(
        questions=[question([Citation(country="AR", norm_id="res-16-2025", article="99")])]
    )

    [(qid, citation)] = unknown_citations(golden, db)
    assert qid == "q001"
    assert citation.article == "99"


def test_a_citation_to_a_nonexistent_norm_is_reported(db):
    golden = GoldenSet(
        questions=[question([Citation(country="AR", norm_id="res-00-0000", article="3")])]
    )

    assert len(unknown_citations(golden, db)) == 1


def test_the_country_must_match_too(db):
    golden = GoldenSet(
        questions=[question([Citation(country="CL", norm_id="res-16-2025", article="3")])]
    )

    assert len(unknown_citations(golden, db)) == 1


def test_reports_every_offending_question(db):
    golden = GoldenSet(
        questions=[
            question([Citation(country="AR", norm_id="res-16-2025", article="3")], "ok"),
            question([Citation(country="AR", norm_id="res-16-2025", article="98")], "malo1"),
            question([Citation(country="AR", norm_id="res-16-2025", article="99")], "malo2"),
        ]
    )

    assert {qid for qid, _ in unknown_citations(golden, db)} == {"malo1", "malo2"}


def test_traps_carry_no_citations_so_nothing_to_check(db):
    golden = GoldenSet(
        questions=[
            GoldenQuestion(
                id="t1",
                question="¿Cuál es el arancel en Brasil?",
                kind="trap",
                answerable=False,
                expected_answer="No está en el corpus.",
                citations=[],
                status="reviewed",
            )
        ]
    )

    assert unknown_citations(golden, db) == []
