"""Dense retrieval over the indexed corpus.

    uv run python scripts/search.py "¿qué exige el marcado de fuentes importadas?"
    uv run python scripts/search.py --country CL "certificación de enchufes"
"""

import argparse
import os
import textwrap

import psycopg
from dotenv import load_dotenv

from regrag.embeddings import build_embedder
from regrag.store import search_dense

load_dotenv()


def cite(hit) -> str:
    where = f"{hit.country} {hit.norm_id}"
    if hit.article:
        where += f" art. {hit.article}"
    return where


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("-k", "--limit", type=int, default=5)
    parser.add_argument("--country")
    parser.add_argument("--chars", type=int, default=200)
    args = parser.parse_args()

    embedder = build_embedder()
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        hits = search_dense(
            conn,
            embedder.embed_query(args.query),
            limit=args.limit,
            country=args.country,
        )

    print(f'query: "{args.query}"\n')
    if not hits:
        print("(sin resultados)")
        return

    for rank, hit in enumerate(hits, 1):
        snippet = " ".join(hit.text.split())[: args.chars]
        print(f"{rank}. [{hit.score:.3f}] {cite(hit)}")
        print(textwrap.indent(textwrap.fill(snippet, 88), "   "))
        print()


if __name__ == "__main__":
    main()
