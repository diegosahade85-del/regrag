"""Compare dense, lexical and hybrid retrieval on the same probes.

    uv run python scripts/diagnose_retrieval.py

The corpus is its own ground truth: for each probe, SQL finds by exact string
match which chunks actually contain it, then each retriever is asked the same
thing. No annotation required, and re-running reproduces the numbers.
"""

import os

import psycopg
from dotenv import load_dotenv

from regrag.embeddings import build_embedder
from regrag.store import search_dense, search_hybrid, search_lexical

load_dotenv()

K = 10

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


# Queries whose wording appears NOWHERE in the corpus, paired with the term the
# corpus actually uses. Lexical search must fail on these by construction: there
# is no matching token to find. Without this category the comparison is rigged —
# every probe above is a verbatim string, which is precisely what full-text
# search is for, so it would show lexical matching hybrid and dense adding
# nothing.
PARAPHRASES = [
    ("penalidades por incumplimiento", "sanción"),
    ("entidad certificadora habilitada", "organismo de certificación"),
    ("vendedor minorista", "comercializador"),
    ("toma de tierra", "puesta a tierra"),
    ("puesta en el mercado de productos", "comercialización"),
]


def containing(conn, needle: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT chunk_id FROM chunks WHERE text ILIKE %s", (f"%{needle}%",))
        return {row[0] for row in cur.fetchall()}


def evaluate(conn, embedder, probes, label):
    """`probes` is either strings, or (query, ground-truth needle) pairs."""
    print(f"\n{label}")
    print(f"  {'probe':<40} {'chunks':>6} {'denso':>7} {'léxico':>7} {'híbrido':>8}")
    totals = {"dense": 0, "lexical": 0, "hybrid": 0}
    scored = 0

    for entry in probes:
        probe, needle = entry if isinstance(entry, tuple) else (entry, entry)
        truth = containing(conn, needle)
        if not truth:
            print(f"  {probe:<40} {'—':>6} {'(ausente del corpus)':>25}")
            continue

        vector = embedder.embed_query(probe)
        found = {
            "dense": {h.chunk_id for h in search_dense(conn, vector, K)},
            "lexical": {h.chunk_id for h in search_lexical(conn, probe, K)},
            "hybrid": {h.chunk_id for h in search_hybrid(conn, vector, probe, K)},
        }
        marks = []
        for name in ("dense", "lexical", "hybrid"):
            hit = bool(truth & found[name])
            totals[name] += hit
            marks.append("SI" if hit else "no")
        scored += 1
        print(f"  {probe:<40} {len(truth):>6} {marks[0]:>7} {marks[1]:>7} {marks[2]:>8}")

    return totals, scored


def main() -> None:
    embedder = build_embedder()
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        id_totals, id_n = evaluate(conn, embedder, IDENTIFIERS, "IDENTIFICADORES")
        ph_totals, ph_n = evaluate(conn, embedder, PHRASES, "FRASES LITERALES")
        pa_totals, pa_n = evaluate(
            conn, embedder, PARAPHRASES, "PARÁFRASIS (el término no está en el corpus)"
        )

    groups = [
        ("identificadores", id_totals, id_n),
        ("frases literales", ph_totals, ph_n),
        ("paráfrasis", pa_totals, pa_n),
    ]

    print(f"\n{'':<40} {'denso':>7} {'léxico':>7} {'híbrido':>8}")
    for label, totals, n in groups:
        cells = [f"{totals[r]}/{n}" for r in ("dense", "lexical", "hybrid")]
        print(f"  {label:<38} {cells[0]:>7} {cells[1]:>7} {cells[2]:>8}")

    total_n = sum(n for _, _, n in groups)
    cells = [
        f"{sum(t[r] for _, t, _ in groups)}/{total_n}"
        for r in ("dense", "lexical", "hybrid")
    ]
    print(f"  {'TOTAL recall@' + str(K):<38} {cells[0]:>7} {cells[1]:>7} {cells[2]:>8}")


if __name__ == "__main__":
    main()
