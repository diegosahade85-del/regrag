"""Measure where dense retrieval fails, using the corpus as its own ground truth.

    uv run python scripts/diagnose_dense.py

For each probe we find, by exact string match in SQL, which chunks actually
contain it — that is the answer set, no annotation required. Then we ask dense
retrieval the same thing and check whether any of those chunks come back.
This gives a recall@10 number for identifier-style queries versus conceptual
ones, which is the case for adding lexical search.
"""

import os

import psycopg
from dotenv import load_dotenv

from regrag.embeddings import build_embedder
from regrag.store import search_dense

load_dotenv()

K = 10

# Both lists are strings that appear verbatim somewhere in the corpus, so the
# question in both cases is the same: can dense retrieval find the chunk that
# actually contains what was asked for? Probes absent from the corpus are
# skipped rather than counted.
IDENTIFIERS = [
    "RESOL-2025-16-APN-SIYC#MEC",
    "IF-2025-18559087-APN-DNRT#MEC",
    "Resoluciones Nros. 731",
    "Ley N° 24.240",
    "NTC 2050",
    "Resolución 16/2025",
    "Decreto N° 274/19",
    "IRAM 2073",
    "NTC 4552",
    "IEC 60364",
    "Ley 1264",
    "Resolución 40117",
]
# Phrases a compliance officer would type, that also occur word-for-word in the
# articles that impose the obligation.
PHRASES = [
    "declaración jurada de conformidad",
    "organismo de certificación acreditado",
    "distribuidores y comercializadores",
    "requisitos esenciales de seguridad",
    "certificado de conformidad de producto",
    "evaluación de la conformidad",
    "grado de protección IP",
    "hilo incandescente",
]


def containing(conn, needle: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT chunk_id FROM chunks WHERE text ILIKE %s", (f"%{needle}%",))
        return {row[0] for row in cur.fetchall()}


def run(conn, embedder, probes: list[str], label: str) -> tuple[int, int]:
    print(f"\n{label}")
    print(f"  {'probe':<46} {'chunks':>7}  {'top-' + str(K):>6}")
    hits_total = 0
    scored = 0
    for probe in probes:
        truth = containing(conn, probe)
        if not truth:
            print(f"  {probe:<46} {'—':>7}  {'(ausente del corpus)':>6}")
            continue
        found = {h.chunk_id for h in search_dense(conn, embedder.embed_query(probe), K)}
        hit = bool(truth & found)
        hits_total += hit
        scored += 1
        print(f"  {probe:<46} {len(truth):>7}  {'SI' if hit else 'NO':>6}")
    return hits_total, scored


def main() -> None:
    embedder = build_embedder()
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        id_hits, id_n = run(conn, embedder, IDENTIFIERS, "IDENTIFICADORES")
        c_hits, c_n = run(conn, embedder, PHRASES, "FRASES LITERALES")

    print(f"\n{'':<46} recall@{K}")
    print(f"  {'identificadores':<46} {id_hits}/{id_n}")
    print(f"  {'frases literales':<46} {c_hits}/{c_n}")


if __name__ == "__main__":
    main()
