"""Answer compliance questions from retrieved context, with verifiable citations."""

import os
from typing import Literal

from pydantic import BaseModel, Field

from regrag.store import SearchResult

# The user's plan routes final answers to Sonnet; Haiku handles query
# reformulation from day 10.
MODEL = "claude-sonnet-5"
MAX_TOKENS = 2000

NO_CONTEXT = "(no se recuperó ningún fragmento del corpus)"

SYSTEM = """\
Sos un asistente de cumplimiento normativo sobre certificación de producto en \
América Latina. Respondés preguntas de profesionales de compliance usando \
EXCLUSIVAMENTE los fragmentos de normativa que se te entregan.

Reglas, en orden de prioridad:

1. Usá únicamente lo que dicen los fragmentos. No completes con conocimiento \
general de normativa, aunque estés seguro de que es correcto. Si el fragmento \
no lo dice, para vos no existe.

2. Si los fragmentos no alcanzan para responder, poné `answerable: false` y \
explicá qué falta. Una respuesta inventada le cuesta a un cliente una \
importación frenada en aduana; un "no está en el corpus" le cuesta una \
búsqueda más. No son errores equivalentes.

3. Citá siempre el país, la norma y el artículo de donde sale cada afirmación, \
copiando esos valores exactamente como aparecen en el encabezado del fragmento. \
No cites una norma que no esté entre los fragmentos entregados.

4. Si los fragmentos se contradicen, o si una norma deroga a otra, decilo en \
vez de elegir una en silencio.

5. Distinguí lo que la norma obliga de lo que permite o recomienda. En \
compliance "deberá" y "podrá" no son lo mismo.

Nivel de confianza:
- `alta`: los fragmentos responden la pregunta de forma directa y completa.
- `media`: responden parcialmente, o hay que inferir un paso.
- `baja`: los fragmentos son tangenciales, o hay contradicciones sin resolver.

Respondé en español, en prosa clara, sin repetir la pregunta.\
"""


class Citation(BaseModel):
    country: str = Field(description="Código de país del fragmento, ej. 'AR'")
    norm_id: str = Field(description="Identificador de la norma, ej. 'res-16-2025'")
    article: str | None = Field(
        default=None, description="Número de artículo, o null si el fragmento no tiene"
    )

    def key(self) -> tuple[str, str, str | None]:
        return (self.country, self.norm_id, self.article)


class Answer(BaseModel):
    answerable: bool = Field(
        description="false si los fragmentos no alcanzan para responder"
    )
    answer: str = Field(description="La respuesta, o qué información falta")
    citations: list[Citation] = Field(
        description="Fragmentos que respaldan la respuesta; vacío si answerable es false"
    )
    confidence: Literal["alta", "media", "baja"]


def format_context(hits: list[SearchResult]) -> str:
    """Render retrieved chunks with the citation fields the model must echo back."""
    if not hits:
        return NO_CONTEXT

    blocks = []
    for hit in hits:
        article = hit.article if hit.article else "sin artículo"
        blocks.append(
            f"<fragmento country=\"{hit.country}\" norm_id=\"{hit.norm_id}\" "
            f"article=\"{article}\">\n{hit.text}\n</fragmento>"
        )
    return "\n\n".join(blocks)


def unsupported_citations(answer: Answer, context: list[SearchResult]) -> list[Citation]:
    """Citations naming something the model was never shown.

    A hallucinated citation is indistinguishable from a real one by inspection —
    same shape, plausible norm, plausible article — so it has to be checked
    against the retrieved set rather than trusted. This is the last line before
    a wrong citation reaches someone who will act on it.
    """
    available = {(hit.country, hit.norm_id, hit.article) for hit in context}
    return [c for c in answer.citations if c.key() not in available]


def build_messages(question: str, hits: list[SearchResult]) -> list[dict]:
    return [
        {
            "role": "user",
            "content": (
                f"<fragmentos>\n{format_context(hits)}\n</fragmentos>\n\n"
                f"<pregunta>{question}</pregunta>"
            ),
        }
    ]


def build_client():
    import anthropic

    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def answer_question(client, question: str, hits: list[SearchResult]) -> Answer:
    """Ask the model, constrained to the retrieved context and to the schema."""
    response = client.messages.parse(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM,
        messages=build_messages(question, hits),
        output_format=Answer,
    )
    return response.parsed_output
