"""Postgres + pgvector storage and dense retrieval."""

from dataclasses import dataclass, replace
from typing import Any

import psycopg
from psycopg.rows import class_row

from regrag.fusion import DEFAULT_K, reciprocal_rank_fusion

# How many candidates each retriever contributes per requested result.
CANDIDATE_FACTOR = 5

# voyage-3 embedding width. Kept as a parameter rather than baked into the DDL
# so tests can use a small, hand-checkable dimension.
EMBEDDING_DIM = 1024

# Spanish stemming and stopwords: "producto importado" should match "los
# productos importados", and "de la" should not be a search term. Standard and
# document numbers survive stemming untouched, which is the point.
TEXT_SEARCH_CONFIG = "spanish"


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
                embedding vector({dim}) NOT NULL,
                -- Generated, not maintained: a tsvector written by application
                -- code drifts out of sync with the text the moment one write
                -- path forgets to update it, and the symptom is a chunk that
                -- silently stops being findable. Postgres keeps this current.
                tsv tsvector GENERATED ALWAYS AS
                    (to_tsvector('{TEXT_SEARCH_CONFIG}', text)) STORED
            )
            """
        )
        # CREATE TABLE IF NOT EXISTS is a no-op on a table that already exists,
        # so a column added to the DDL above never reaches a database created
        # before it — and the only symptom is lexical search quietly returning
        # nothing. Add it explicitly for databases that predate it.
        cur.execute(
            f"""
            ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tsv tsvector
            GENERATED ALWAYS AS (to_tsvector('{TEXT_SEARCH_CONFIG}', text)) STORED
            """
        )

        # Metadata filters are what make "how does AR differ from CL?" answerable,
        # so they get indexes alongside the vector.
        cur.execute("CREATE INDEX IF NOT EXISTS chunks_country ON chunks (country)")
        cur.execute("CREATE INDEX IF NOT EXISTS chunks_norm ON chunks (norm_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS chunks_tsv ON chunks USING GIN (tsv)")


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


def search_lexical(
    conn: psycopg.Connection,
    query: str,
    limit: int = 10,
    country: str | None = None,
) -> list[SearchResult]:
    """Full-text search, ranked by ts_rank.

    This is the half of retrieval that can tell IEC 60364 from IEC 60335. An
    embedding compresses meaning, and a standard number has no meaning to
    compress — the digits are the entire query.
    """
    if not query.strip():
        return []

    with conn.cursor(row_factory=class_row(SearchResult)) as cur:
        cur.execute(
            f"""
            SELECT chunk_id, text, country, agency, norm_id, variant, year, article,
                   ts_rank(tsv, q) AS score
            FROM chunks, websearch_to_tsquery('{TEXT_SEARCH_CONFIG}', %(q)s) AS q
            WHERE tsv @@ q
              AND (%(country)s::text IS NULL OR country = %(country)s)
            ORDER BY score DESC
            LIMIT %(limit)s
            """,
            {"q": query, "country": country, "limit": limit},
        )
        return cur.fetchall()


def search_hybrid(
    conn: psycopg.Connection,
    query_embedding: list[float],
    query_text: str,
    limit: int = 10,
    country: str | None = None,
    candidate_factor: int = CANDIDATE_FACTOR,
    k: int = DEFAULT_K,
) -> list[SearchResult]:
    """Dense and lexical retrieval, fused by rank.

    The two retrievers fail in different places, and neither failure is visible
    from inside the other: embeddings cannot tell IEC 60364 from IEC 60335, and
    full-text search cannot connect "rotulado" to "marcado". Fusing by position
    needs no shared score scale and no per-retriever weight to tune.
    """
    # Each retriever contributes more candidates than the caller asked for.
    # Fusing only the top-`limit` of each would throw away exactly the results
    # that win on combined support rather than on either ranking alone.
    pool = max(limit * candidate_factor, limit)
    dense = search_dense(conn, query_embedding, limit=pool, country=country)
    lexical = search_lexical(conn, query_text, limit=pool, country=country)

    by_id = {hit.chunk_id: hit for hit in [*dense, *lexical]}
    fused = reciprocal_rank_fusion(
        [[h.chunk_id for h in dense], [h.chunk_id for h in lexical]],
        k=k,
        limit=limit,
    )
    # The fused score replaces each retriever's own, which are not comparable.
    return [replace(by_id[cid], score=score) for cid, score in fused]


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
