"""The golden set: questions with a verified answer and the clause behind it.

This file is the only part of the project whose correctness cannot be derived
from the corpus. Everything else is measured against it, so an entry nobody has
checked is worse than no entry — it yields a number that looks like a
measurement. Hence `status`, and hence `reviewed_only()` guarding the eval.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from regrag.answering import Citation

Kind = Literal["factual", "synthesis", "trap"]
Status = Literal["draft", "reviewed"]

DEFAULT_PATH = Path("evals/golden_set.json")


class GoldenQuestion(BaseModel):
    id: str
    question: str
    kind: Kind
    answerable: bool
    expected_answer: str
    citations: list[Citation] = Field(default_factory=list)
    status: Status = "draft"
    notes: str = ""

    @field_validator("question", "expected_answer")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("no puede estar vacío")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.kind == "trap" and self.answerable:
            raise ValueError("a trap must have answerable=false")
        if self.answerable and not self.citations:
            # Ground truth with no clause behind it cannot be checked against
            # the corpus, which is the whole reason for writing it down.
            raise ValueError("an answerable question needs at least one citation")
        if not self.answerable and self.citations:
            raise ValueError("an unanswerable question must not carry citations")
        return self


class GoldenSet(BaseModel):
    questions: list[GoldenQuestion]

    @model_validator(mode="after")
    def _unique_ids(self) -> Self:
        seen = Counter(q.id for q in self.questions)
        if repeated := [qid for qid, n in seen.items() if n > 1]:
            raise ValueError(f"ids duplicados: {', '.join(sorted(repeated))}")
        return self

    def counts(self) -> dict[str, int]:
        return dict(Counter(q.kind for q in self.questions))

    def status_counts(self) -> dict[str, int]:
        return dict(Counter(q.status for q in self.questions))


def reviewed_only(questions: list[GoldenQuestion]) -> list[GoldenQuestion]:
    """The subset an eval is allowed to score against."""
    return [q for q in questions if q.status == "reviewed"]


def unknown_citations(golden: GoldenSet, conn) -> list[tuple[str, Citation]]:
    """Golden-set citations that name nothing in the indexed corpus.

    Catches the hand-typed slip — a transposed article number, a norm id that
    reads right but is not what the file is called. The entry looks fine on the
    page and is simply unscoreable, so the check has to be mechanical.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT country, norm_id, article FROM chunks")
        available = set(cur.fetchall())

    return [
        (question.id, citation)
        for question in golden.questions
        for citation in question.citations
        if citation.key() not in available
    ]


def load_golden_set(path: Path = DEFAULT_PATH) -> GoldenSet:
    return GoldenSet(**json.loads(Path(path).read_text(encoding="utf-8")))


def save_golden_set(golden: GoldenSet, path: Path = DEFAULT_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(golden.model_dump(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
