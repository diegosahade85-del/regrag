"""Search the indexed corpus.

    uv run python scripts/search.py "¿qué exige el marcado de fuentes importadas?"
    uv run python scripts/search.py --mode dense "IEC 60364"
    uv run python scripts/search.py --country CL "certificación de enchufes"
"""

import argparse
import os
import textwrap

import psycopg
from dotenv import load_dotenv

from regrag.embeddings import build_embedder
from regrag.store import search_dense, search_hybrid, search_lexical

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
    parser.add_argument("--mode", choices=("hybrid", "dense", "lexical"), default="hybrid")
    parser.add_argument("--country")
    parser.add_argument("--chars", type=int, default=200)
    args = parser.parse_args()

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        if args.mode == "lexical":
            hits = search_lexical(conn, args.query, args.limit, args.country)
        else:
            vector = build_embedder().embed_query(args.query)
            hits = (
                search_dense(conn, vector, args.limit, args.country)
                if args.mode == "dense"
                else search_hybrid(conn, vector, args.query, args.limit, args.country)
            )

    print(f'[{args.mode}] "{args.query}"\n')
    if not hits:
        print("(sin resultados)")
        return

    for rank, hit in enumerate(hits, 1):
        snippet = " ".join(hit.text.split())[: args.chars]
        print(f"{rank}. [{hit.score:.4f}] {cite(hit)}")
        print(textwrap.indent(textwrap.fill(snippet, 88), "   "))
        print()


if __name__ == "__main__":
    main()
