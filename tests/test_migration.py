"""Schema evolution on a table that predates the current definition.

`CREATE TABLE IF NOT EXISTS` is a no-op against an existing table, so a column
added to the DDL never reaches a database created before it. Nothing errors:
lexical search just returns nothing on a corpus that is sitting right there.
"""

import pytest

from regrag.store import create_schema, search_lexical, upsert_chunks

DIM = 4
VEC = [1.0, 0.0, 0.0, 0.0]


def record(chunk_id, text):
    return {
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


@pytest.fixture
def legacy_db(conn):
    """A chunks table as it existed before full-text search was added."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE chunks (
                chunk_id  TEXT PRIMARY KEY,
                text      TEXT    NOT NULL,
                country   TEXT    NOT NULL,
                agency    TEXT    NOT NULL,
                norm_id   TEXT    NOT NULL,
                variant   TEXT,
                year      INTEGER,
                article   TEXT,
                strategy  TEXT    NOT NULL,
                idx       INTEGER NOT NULL,
                n_chars   INTEGER NOT NULL,
                embedding vector({DIM}) NOT NULL
            )
            """
        )
    return conn


def has_column(conn, name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'chunks' AND column_name = %s",
            (name,),
        )
        return cur.fetchone() is not None


def test_the_fixture_really_lacks_the_column(legacy_db):
    assert not has_column(legacy_db, "tsv")


def test_create_schema_adds_the_column_to_an_existing_table(legacy_db):
    create_schema(legacy_db, dim=DIM)

    assert has_column(legacy_db, "tsv")


def test_lexical_search_works_after_the_upgrade(legacy_db):
    upsert_chunks(legacy_db, [record("a#hybrid#0", "El marcado será indeleble.")], [VEC])

    create_schema(legacy_db, dim=DIM)

    assert [h.chunk_id for h in search_lexical(legacy_db, "marcado", 10)] == [
        "a#hybrid#0"
    ]
