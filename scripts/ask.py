"""Ask a compliance question against the indexed corpus.

    uv run python scripts/ask.py "¿qué obligaciones de marcado tienen los importadores?"
    uv run python scripts/ask.py --country CL -k 8 "¿quién certifica los enchufes?"
"""

import argparse
import os
import textwrap

import psycopg
from dotenv import load_dotenv

from regrag.answering import answer_question, build_client, unsupported_citations
from regrag.embeddings import build_embedder
from regrag.store import search_hybrid

load_dotenv()


def cite(country, norm_id, article) -> str:
    where = f"{country} {norm_id}"
    return f"{where} art. {article}" if article else where


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("-k", "--limit", type=int, default=8)
    parser.add_argument("--country")
    parser.add_argument("--show-context", action="store_true")
    args = parser.parse_args()

    vector = build_embedder().embed_query(args.question)
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        hits = search_hybrid(conn, vector, args.question, args.limit, args.country)

    if args.show_context:
        print("FRAGMENTOS RECUPERADOS")
        for rank, hit in enumerate(hits, 1):
            print(f"  {rank}. {cite(hit.country, hit.norm_id, hit.article)}")
        print()

    answer = answer_question(build_client(), args.question, hits)

    print(f"PREGUNTA  {args.question}\n")
    if not answer.answerable:
        print("NO RESPONDIBLE CON EL CORPUS ACTUAL")
        print(textwrap.indent(textwrap.fill(answer.answer, 84), "  "))
        return

    print("RESPUESTA")
    print(textwrap.indent(textwrap.fill(answer.answer, 84), "  "))

    # A citation the model invented is indistinguishable from a real one by
    # reading it, so it is checked against what was actually retrieved.
    unsupported = {c.key() for c in unsupported_citations(answer, hits)}
    print("\nCITAS")
    for citation in answer.citations:
        flag = "  <-- NO RESPALDADA POR EL CONTEXTO" if citation.key() in unsupported else ""
        print(f"  {cite(citation.country, citation.norm_id, citation.article)}{flag}")
    if not answer.citations:
        print("  (ninguna)")

    print(f"\nconfianza: {answer.confidence}   fragmentos recuperados: {len(hits)}")
    if unsupported:
        raise SystemExit(
            f"\nERROR: {len(unsupported)} cita(s) no corresponden a ningún "
            "fragmento recuperado."
        )


if __name__ == "__main__":
    main()
