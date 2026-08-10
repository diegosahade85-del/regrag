"""Postgres + pgvector storage and dense retrieval."""

from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.rows import class_row

# voyage-3 embedding width. Kept as a parameter rather than baked into the DDL
# so tests can use a small, hand-checkable dimension.
EMBEDDING_DIM = 1024


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    text: str
    country: str
    agency: str
    norm_id: str
    variant: str | None
    year: int | None
    article: str | None
    score: float


def create_schema(conn: psycopg.Connection, dim: int = EMBEDDING_DIM) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id  TEXT PRIMARY KEY,
                text      TEXT        NOT NULL,
                country   TEXT        NOT NULL,
                agency    TEXT        NOT NULL,
                norm_id   TEXT        NOT NULL,
                variant   TEXT,
                year      INTEGER,
                article   TEXT,
                strategy  TEXT        NOT NULL,
                idx       INTEGER     NOT NULL,
                n_chars   INTEGER     NOT NULL,
                embedding vector({dim}) NOT NULL
            )
            """
        )
        # Metadata filters are what make "how does AR differ from CL?" answerable,
        # so they get indexes alongside the vector.
        cur.execute("CREATE INDEX IF NOT EXISTS chunks_country ON chunks (country)")
        cur.execute("CREATE INDEX IF NOT EXISTS chunks_norm ON chunks (norm_id)")


# Below this many rows an exact scan is fast enough that an approximate index
# only trades correctness for latency it doesn't need to save. Measured on this
# corpus: 2,176 vectors scan exactly in ~17ms, while every IVFFlat setting that
# reached full recall took 16-20ms. See README.
ANN_MIN_ROWS = 50_000


def create_vector_index(conn: psycopg.Connection, lists: int | None = None) -> None:
    """(Re)build the IVFFlat index, sizing clusters to the current row count.

    Always drops first. An IVFFlat index stores centroids computed from whatever
    was in the table when it was built, so one built before a load — or left in
    place across a TRUNCATE — describes data that is no longer there. It raises
    no error and quietly returns the wrong neighbours; `CREATE INDEX IF NOT
    EXISTS` is therefore the wrong statement here.
    """
    with conn.cursor() as cur:
        if lists is None:
            cur.execute("SELECT count(*) FROM chunks")
            (rows,) = cur.fetchone()
            # pgvector's guidance for under a million rows.
            lists = max(1, rows // 1000)

        cur.execute("DROP INDEX IF EXISTS chunks_embedding")
        cur.execute(
            "CREATE INDEX chunks_embedding "
            "ON chunks USING ivfflat (embedding vector_cosine_ops) "
            f"WITH (lists = {lists})"
        )


def drop_vector_index(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("DROP INDEX IF EXISTS chunks_embedding")


def upsert_chunks(
    conn: psycopg.Connection,
    records: list[dict[str, Any]],
    embeddings: list[list[float]],
) -> int:
    """Insert or replace chunks, keyed by chunk_id so re-ingestion is safe."""
    if len(records) != len(embeddings):
        raise ValueError(
            f"got {len(records)} records but {len(embeddings)} embeddings"
        )
    if not records:
        return 0

    rows = [
        (
            r["chunk_id"],
            r["text"],
            r["country"],
            r["agency"],
            r["norm_id"],
            r["variant"],
            r["year"],
            r["article"],
            r["strategy"],
            r["index"],
            r["n_chars"],
            str(embedding),
        )
        for r, embedding in zip(records, embeddings)
    ]

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks (chunk_id, text, country, agency, norm_id, variant,
                                year, article, strategy, idx, n_chars, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO UPDATE SET
                text = EXCLUDED.text,
                country = EXCLUDED.country,
                agency = EXCLUDED.agency,
                norm_id = EXCLUDED.norm_id,
                variant = EXCLUDED.variant,
                year = EXCLUDED.year,
                article = EXCLUDED.article,
                strategy = EXCLUDED.strategy,
                idx = EXCLUDED.idx,
                n_chars = EXCLUDED.n_chars,
                embedding = EXCLUDED.embedding
            """,
            rows,
        )
    return len(rows)


def existing_chunk_ids(conn: psycopg.Connection) -> set[str]:
    """Chunk ids already indexed — lets a interrupted load resume where it stopped."""
    with conn.cursor() as cur:
        cur.execute("SELECT chunk_id FROM chunks")
        return {row[0] for row in cur.fetchall()}


def search_dense(
    conn: psycopg.Connection,
    query_embedding: list[float],
    limit: int = 10,
    country: str | None = None,
) -> list[SearchResult]:
    """Nearest neighbours by cosine similarity, most similar first."""
    with conn.cursor(row_factory=class_row(SearchResult)) as cur:
        cur.execute(
            """
            SELECT chunk_id, text, country, agency, norm_id, variant, year, article,
                   1 - (embedding <=> %(q)s::vector) AS score
            FROM chunks
            WHERE (%(country)s::text IS NULL OR country = %(country)s)
            ORDER BY embedding <=> %(q)s::vector
            LIMIT %(limit)s
            """,
            {"q": str(query_embedding), "country": country, "limit": limit},
        )
        return cur.fetchall()
