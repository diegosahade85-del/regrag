"""Show the state of the golden set and validate it against the corpus.

    uv run python scripts/golden_status.py
    uv run python scripts/golden_status.py --drafts     # list what still needs review

Exits non-zero if any citation names something the corpus does not contain, so
this can gate CI.
"""

import argparse
import os
import textwrap

import psycopg
from dotenv import load_dotenv

from regrag.golden import DEFAULT_PATH, load_golden_set, reviewed_only, unknown_citations

load_dotenv()

TARGET = 50
KINDS = ("factual", "synthesis", "trap")


def bar(done: int, total: int, width: int = 28) -> str:
    filled = round(width * done / total) if total else 0
    return "█" * min(filled, width) + "·" * max(width - filled, 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=str(DEFAULT_PATH))
    parser.add_argument("--drafts", action="store_true", help="listar los sin revisar")
    args = parser.parse_args()

    golden = load_golden_set(args.path)
    reviewed = reviewed_only(golden.questions)
    kinds = golden.counts()
    statuses = golden.status_counts()

    print(f"{args.path}\n")
    print(f"  total          {len(golden.questions)}")
    print(f"  revisadas      {len(reviewed)}  {bar(len(reviewed), TARGET)}  "
          f"meta {TARGET}")
    print(f"  borradores     {statuses.get('draft', 0)}")
    print()
    for kind in KINDS:
        n_all = kinds.get(kind, 0)
        n_ok = sum(1 for q in reviewed if q.kind == kind)
        print(f"  {kind:<14} {n_ok:>3} revisadas / {n_all:>3} totales")

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        unknown = unknown_citations(golden, conn)

    if unknown:
        print(f"\n  {len(unknown)} cita(s) que no existen en el corpus:")
        for qid, citation in unknown:
            article = f" art. {citation.article}" if citation.article else ""
            print(f"    {qid}: {citation.country} {citation.norm_id}{article}")
    else:
        print("\n  todas las citas existen en el corpus")

    if args.drafts:
        print("\nPENDIENTES DE REVISIÓN")
        for entry in golden.questions:
            if entry.status != "draft":
                continue
            print(f"\n  [{entry.id}] ({entry.kind})")
            print(textwrap.indent(textwrap.fill(entry.question, 80), "    "))

    if unknown:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
