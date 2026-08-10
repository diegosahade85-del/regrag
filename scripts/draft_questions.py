"""Draft golden-set candidates from the corpus, for a human to review.

    uv run python scripts/draft_questions.py --n 30
    uv run python scripts/draft_questions.py --traps 10

Everything written here is `status: "draft"` and is excluded from evaluation
until a person changes it. A draft is a model's reading of a clause, which is
exactly the thing the eval is supposed to be checking — scoring against it would
produce a number that measures nothing.

Two biases to correct while reviewing:

- A question generated *from* a chunk shares that chunk's vocabulary, so
  retrieval finds it easily and recall comes out flattering. The prompt asks for
  a practitioner's phrasing rather than the article's, but rewriting the
  question in your own words is the real fix.
- The model drafts what it thinks the article says. Where it is subtly wrong —
  confusing "deberá" with "podrá", missing an exception — the draft will read
  fluently and be wrong, which is the failure mode hardest to catch by skimming.
"""

import argparse
import os
import random
from typing import Literal

import psycopg
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from regrag.answering import Citation, build_client
from regrag.golden import (
    DEFAULT_PATH,
    GoldenQuestion,
    GoldenSet,
    load_golden_set,
    save_golden_set,
)

load_dotenv()

MODEL = "claude-sonnet-5"

DRAFT_SYSTEM = """\
Sos un especialista en certificación de producto en América Latina que está \
armando un set de evaluación para un asistente de compliance.

A partir del artículo que se te entrega, escribí UNA pregunta que un \
profesional de compliance haría de verdad en su trabajo, junto con la respuesta \
que ese artículo permite dar.

Requisitos para la pregunta:
- Que sea la clase de pregunta que alguien hace antes de importar o certificar \
un producto, no una pregunta de examen sobre el texto.
- NO copies el vocabulario distintivo del artículo. Si el artículo dice \
"declaración jurada de conformidad", preguntá con otras palabras: "¿qué tiene \
que presentar el importador para acreditar que cumple?". El set se usa para \
medir un buscador, y una pregunta que repite las palabras del texto mide \
copiado, no búsqueda.
- Autocontenida: nombrá el país y el producto o el trámite, sin decir "según \
este artículo".

Requisitos para la respuesta:
- Solo lo que el artículo permite afirmar.
- Distinguí obligación de facultad: "deberá" y "podrá" no son lo mismo.
- Breve, dos o tres oraciones.\
"""

TRAP_SYSTEM = """\
Sos un especialista en certificación de producto en América Latina que está \
armando un set de evaluación para un asistente de compliance.

Escribí UNA pregunta que un profesional haría de verdad, sobre un tema \
plausiblemente cercano al corpus que se describe, pero cuya respuesta NO esté \
en él. El objetivo es comprobar que el asistente reconoce lo que no sabe en \
lugar de inventar.

La pregunta tiene que ser tentadora: mismo dominio, vocabulario parecido, un \
país o producto o trámite que el corpus no cubre. Una pregunta obviamente \
ajena (por ejemplo sobre cocina) no prueba nada.

En `expected_answer` explicá en una oración por qué el corpus no puede \
responderla.\
"""


class Draft(BaseModel):
    question: str = Field(description="La pregunta, en español")
    expected_answer: str = Field(description="La respuesta que el artículo permite dar")
    kind: Literal["factual", "synthesis"] = "factual"


SYNTHESIS_SYSTEM = """\
Sos un especialista en certificación de producto en América Latina que está \
armando un set de evaluación para un asistente de compliance.

Se te entregan dos artículos de países distintos que tratan un tema parecido. \
Escribí UNA pregunta comparativa que un profesional haría de verdad — la clase \
de pregunta que aparece cuando hay que colocar el mismo producto en los dos \
mercados — y la respuesta que surge de comparar ambos artículos.

Requisitos:
- La pregunta debe requerir los dos artículos. Si se responde con uno solo, no \
sirve.
- Nombrá los dos países.
- No copies el vocabulario distintivo de ninguno de los dos textos.
- En la respuesta, decí en qué se parecen y en qué se diferencian, y sé \
explícito sobre lo que los artículos NO permiten concluir.\
"""


class TrapDraft(BaseModel):
    question: str
    expected_answer: str = Field(description="Por qué el corpus no puede responderla")


class SynthesisDraft(BaseModel):
    question: str = Field(description="Pregunta comparativa entre los dos países")
    expected_answer: str = Field(description="Qué comparten y en qué difieren")


def sample_articles(conn, n: int, seed: int) -> list[dict]:
    """Longest article per (country, norm, article), spread across jurisdictions."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (country, norm_id, article)
                   country, norm_id, article, text
            FROM chunks
            WHERE article IS NOT NULL AND n_chars > 400
            ORDER BY country, norm_id, article, n_chars DESC
            """
        )
        rows = [
            {"country": c, "norm_id": nid, "article": a, "text": t}
            for c, nid, a, t in cur.fetchall()
        ]

    by_country: dict[str, list[dict]] = {}
    for row in rows:
        by_country.setdefault(row["country"], []).append(row)

    rng = random.Random(seed)
    for group in by_country.values():
        rng.shuffle(group)

    # Round-robin across countries so one large norm cannot dominate the set.
    picked, countries = [], sorted(by_country)
    while len(picked) < n and any(by_country.values()):
        for country in countries:
            if by_country[country] and len(picked) < n:
                picked.append(by_country[country].pop())
    return picked


def sample_cross_country_pairs(conn, n: int) -> list[tuple[dict, dict]]:
    """Article pairs from different countries that cover the same ground.

    Uses the embeddings already in the table: two articles whose vectors sit
    close together, in different jurisdictions, are two countries legislating
    the same topic — which is exactly what a comparative question needs. Picking
    pairs by hand would mean reading the corpus; this reads it for us.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH longest AS (
                SELECT DISTINCT ON (country, norm_id, article)
                       country, norm_id, article, text, embedding
                FROM chunks
                WHERE article IS NOT NULL AND n_chars > 600
                ORDER BY country, norm_id, article, n_chars DESC
            )
            SELECT a.country, a.norm_id, a.article, a.text,
                   b.country, b.norm_id, b.article, b.text
            FROM longest a JOIN longest b
              ON a.country < b.country
            ORDER BY a.embedding <=> b.embedding
            LIMIT %s
            """,
            (n * 4,),
        )
        rows = cur.fetchall()

    # One pair per country-pair-and-topic, so the batch is not ten variations of
    # the same comparison.
    picked, seen = [], set()
    for ac, an, aa, at, bc, bn, ba, bt in rows:
        key = (ac, an, bc, bn)
        if key in seen:
            continue
        seen.add(key)
        picked.append(
            (
                {"country": ac, "norm_id": an, "article": aa, "text": at},
                {"country": bc, "norm_id": bn, "article": ba, "text": bt},
            )
        )
        if len(picked) == n:
            break
    return picked


def corpus_summary(conn) -> str:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT country, norm_id, count(*) FROM chunks "
            "GROUP BY country, norm_id ORDER BY country, norm_id"
        )
        return "\n".join(f"- {c} {n} ({k} fragmentos)" for c, n, k in cur.fetchall())


def next_id(existing: list[GoldenQuestion], prefix: str) -> int:
    used = [int(q.id[len(prefix):]) for q in existing if q.id.startswith(prefix)
            and q.id[len(prefix):].isdigit()]
    return max(used, default=0) + 1


def attempt(label: str, fn):
    """Run one draft, reporting rather than aborting the batch on failure.

    A truncated response fails schema validation, and losing forty good drafts
    to the forty-first is not a trade worth making.
    """
    try:
        return fn()
    except Exception as exc:
        print(f"  {label}  OMITIDO ({type(exc).__name__})", flush=True)
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=0, help="borradores desde artículos")
    parser.add_argument("--synthesis", type=int, default=0,
                        help="borradores comparativos entre países")
    parser.add_argument("--traps", type=int, default=0, help="borradores de trampa")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=str(DEFAULT_PATH))
    args = parser.parse_args()

    if not (args.n or args.synthesis or args.traps):
        parser.error("indicá al menos uno de --n, --synthesis, --traps")

    try:
        golden = load_golden_set(args.out)
    except FileNotFoundError:
        golden = GoldenSet(questions=[])
    existing = list(golden.questions)

    client = build_client()
    conn = psycopg.connect(os.environ["DATABASE_URL"])

    counter = next_id(existing, "q")
    for article in sample_articles(conn, args.n, args.seed):
        response = client.messages.parse(
            model=MODEL,
            max_tokens=1000,
            system=DRAFT_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"País: {article['country']}\nNorma: {article['norm_id']}\n"
                        f"Artículo: {article['article']}\n\n{article['text']}"
                    ),
                }
            ],
            output_format=Draft,
        )
        draft = response.parsed_output
        existing.append(
            GoldenQuestion(
                id=f"q{counter:03d}",
                question=draft.question,
                kind=draft.kind,
                answerable=True,
                expected_answer=draft.expected_answer,
                citations=[
                    Citation(
                        country=article["country"],
                        norm_id=article["norm_id"],
                        article=article["article"],
                    )
                ],
                status="draft",
                notes="Generado automáticamente. Revisar y reescribir con tus palabras.",
            )
        )
        print(f"  q{counter:03d}  {article['country']} {article['norm_id']} "
              f"art {article['article']}", flush=True)
        counter += 1

    for left, right in sample_cross_country_pairs(conn, args.synthesis):
        label = f"q{counter:03d}  síntesis {left['country']}/{right['country']}"
        response = attempt(label, lambda: client.messages.parse(
            model=MODEL,
            max_tokens=3000,
            system=SYNTHESIS_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": "\n\n".join(
                        f"=== {a['country']} · {a['norm_id']} · artículo "
                        f"{a['article']} ===\n{a['text']}"
                        for a in (left, right)
                    ),
                }
            ],
            output_format=SynthesisDraft,
        ))
        if response is None:
            continue
        draft = response.parsed_output
        existing.append(
            GoldenQuestion(
                id=f"q{counter:03d}",
                question=draft.question,
                kind="synthesis",
                answerable=True,
                expected_answer=draft.expected_answer,
                citations=[
                    Citation(country=a["country"], norm_id=a["norm_id"],
                             article=a["article"])
                    for a in (left, right)
                ],
                status="draft",
                notes="Generado automáticamente. Verificá que de verdad haga falta "
                      "comparar ambos artículos para responder.",
            )
        )
        print(f"  {label}", flush=True)
        counter += 1

    summary = corpus_summary(conn) if args.traps else ""
    trap_counter = next_id(existing, "t")
    for i in range(args.traps):
        response = client.messages.parse(
            model=MODEL,
            max_tokens=1000,
            system=TRAP_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"El corpus contiene únicamente:\n{summary}\n\n"
                        f"Escribí la trampa número {i + 1}, distinta de las anteriores."
                    ),
                }
            ],
            output_format=TrapDraft,
        )
        trap = response.parsed_output
        existing.append(
            GoldenQuestion(
                id=f"t{trap_counter:03d}",
                question=trap.question,
                kind="trap",
                answerable=False,
                expected_answer=trap.expected_answer,
                citations=[],
                status="draft",
                notes="Generado automáticamente. Confirmá que de verdad no esté en el corpus.",
            )
        )
        print(f"  t{trap_counter:03d}  trampa", flush=True)
        trap_counter += 1

    conn.close()
    save_golden_set(GoldenSet(questions=existing), args.out)
    print(f"\n{len(existing)} preguntas en {args.out} "
          f"({len([q for q in existing if q.status == 'draft'])} sin revisar)")


if __name__ == "__main__":
    main()
