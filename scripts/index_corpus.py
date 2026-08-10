"""Embed the chunked corpus and load it into Postgres.

    uv run python scripts/index_corpus.py [--strategy hybrid] [--tpm 10000]

Resumable and idempotent: chunks already in the table are skipped, and chunks
are keyed by chunk_id so a re-run replaces rather than duplicates. Batches are
sized by estimated tokens rather than a fixed count, because Voyage's free tier
limits tokens per minute, not requests.
"""

import argparse
import json
import os
import time
from pathlib import Path

import psycopg
import voyageai
from dotenv import load_dotenv

from regrag.embeddings import build_embedder
from regrag.store import (
    ANN_MIN_ROWS,
    create_schema,
    create_vector_index,
    drop_vector_index,
    existing_chunk_ids,
    upsert_chunks,
)

load_dotenv()

PROCESSED = Path("data/processed")
CHARS_PER_TOKEN = 3.5  # rough for Spanish; only used to pace requests
MAX_RETRIES = 6


def batches(records: list[dict], token_budget: int):
    """Group records so each batch stays under the per-request token budget."""
    batch: list[dict] = []
    tokens = 0
    for record in records:
        cost = record["n_chars"] / CHARS_PER_TOKEN
        if batch and tokens + cost > token_budget:
            yield batch, tokens
            batch, tokens = [], 0
        batch.append(record)
        tokens += cost
    if batch:
        yield batch, tokens


def embed_with_backoff(embedder, texts: list[str]) -> list[list[float]]:
    delay = 15.0
    for attempt in range(MAX_RETRIES):
        try:
            return embedder.embed_documents(texts)
        except voyageai.error.RateLimitError:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"    rate limited, waiting {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay *= 1.5
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="hybrid")
    parser.add_argument(
        "--tpm", type=int, default=10_000, help="tokens-per-minute ceiling"
    )
    args = parser.parse_args()

    path = PROCESSED / f"chunks_{args.strategy}.jsonl"
    records = [json.loads(line) for line in path.open(encoding="utf-8")]

    conn = psycopg.connect(os.environ["DATABASE_URL"])
    create_schema(conn)
    conn.commit()

    already = existing_chunk_ids(conn)
    pending = [r for r in records if r["chunk_id"] not in already]
    print(
        f"{len(records)} chunks in {path.name}; "
        f"{len(already)} already indexed, {len(pending)} to go"
    )
    # One batch per minute at the TPM ceiling keeps us inside both the token
    # and the request limits without needing to model them separately.
    budget = max(args.tpm - 500, 1000)
    started = time.monotonic()
    done = 0
    spent = 0.0
    embedder = build_embedder() if pending else None

    for batch, tokens in batches(pending, budget):
        window_start = time.monotonic()
        vectors = embed_with_backoff(embedder, [r["text"] for r in batch])
        upsert_chunks(conn, batch, vectors)
        conn.commit()

        done += len(batch)
        spent += tokens
        elapsed = time.monotonic() - started
        remaining = (len(pending) - done) * (elapsed / max(done, 1))
        print(
            f"  {done}/{len(pending)}  ~{spent:,.0f} tok  "
            f"eta {remaining / 60:.0f}m",
            flush=True,
        )

        # Pace to the TPM ceiling: a batch worth T tokens needs T/tpm minutes.
        pause = (tokens / args.tpm) * 60 - (time.monotonic() - window_start)
        if pause > 0 and done < len(pending):
            time.sleep(pause)

    with conn.cursor() as cur:
        cur.execute("SELECT count(*), count(DISTINCT norm_id) FROM chunks")
        rows, norms = cur.fetchone()

    # Approximate search is a trade, and at this size there is nothing to buy:
    # an exact scan is already fast. Leaving a stale index in place would be
    # strictly worse than having none, so drop it rather than skip past it.
    if rows >= ANN_MIN_ROWS:
        print("building vector index...")
        create_vector_index(conn)
    else:
        print(
            f"skipping ANN index: {rows:,} rows is under the {ANN_MIN_ROWS:,} "
            "threshold, exact search is faster than the recall it would cost"
        )
        drop_vector_index(conn)
    conn.commit()
    conn.close()

    print(
        f"\nindexed {rows} chunks across {norms} norms "
        f"in {(time.monotonic() - started) / 60:.1f}m"
    )


if __name__ == "__main__":
    main()
