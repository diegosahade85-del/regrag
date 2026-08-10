import json

import pytest
from pydantic import ValidationError

from regrag.golden import (
    GoldenQuestion,
    GoldenSet,
    load_golden_set,
    reviewed_only,
)

CITATION = {"country": "AR", "norm_id": "res-16-2025", "article": "3"}


def question(**overrides):
    base = {
        "id": "q001",
        "question": "¿Qué deben presentar los fabricantes e importadores?",
        "kind": "factual",
        "answerable": True,
        "expected_answer": "Una Declaración Jurada de Conformidad.",
        "citations": [CITATION],
        "status": "reviewed",
    }
    return base | overrides


class TestSchema:
    def test_accepts_a_well_formed_question(self):
        assert GoldenQuestion(**question()).id == "q001"

    def test_an_answerable_question_must_cite_something(self):
        """Ground truth with no clause behind it cannot be checked, which
        defeats the point of writing it down."""
        with pytest.raises(ValidationError, match="citation"):
            GoldenQuestion(**question(citations=[]))

    def test_a_trap_must_be_marked_unanswerable(self):
        with pytest.raises(ValidationError, match="trap"):
            GoldenQuestion(**question(kind="trap", answerable=True))

    def test_a_trap_carries_no_citations(self):
        with pytest.raises(ValidationError):
            GoldenQuestion(
                **question(kind="trap", answerable=False, citations=[CITATION])
            )

    def test_a_valid_trap_needs_no_citations(self):
        entry = GoldenQuestion(
            **question(
                kind="trap",
                answerable=False,
                citations=[],
                expected_answer="No está en el corpus.",
            )
        )

        assert entry.answerable is False

    def test_kind_is_restricted_to_the_three_types(self):
        with pytest.raises(ValidationError):
            GoldenQuestion(**question(kind="opinión"))

    def test_status_is_restricted_to_draft_or_reviewed(self):
        with pytest.raises(ValidationError):
            GoldenQuestion(**question(status="maybe"))

    def test_an_empty_question_is_rejected(self):
        with pytest.raises(ValidationError):
            GoldenQuestion(**question(question="   "))


class TestSet:
    def test_rejects_duplicate_ids(self):
        with pytest.raises(ValidationError, match="duplicad"):
            GoldenSet(questions=[GoldenQuestion(**question()),
                                 GoldenQuestion(**question())])

    def test_counts_by_kind(self):
        given = GoldenSet(
            questions=[
                GoldenQuestion(**question(id="q1")),
                GoldenQuestion(**question(id="q2", kind="synthesis")),
                GoldenQuestion(
                    **question(id="q3", kind="trap", answerable=False, citations=[])
                ),
            ]
        )

        assert given.counts() == {"factual": 1, "synthesis": 1, "trap": 1}


class TestReviewedOnly:
    def test_drafts_are_excluded(self):
        """A draft is a guess about the law. Evaluating against it produces a
        number that looks like a measurement and is not one."""
        entries = [
            GoldenQuestion(**question(id="q1", status="reviewed")),
            GoldenQuestion(**question(id="q2", status="draft")),
        ]

        assert [q.id for q in reviewed_only(entries)] == ["q1"]

    def test_a_set_of_only_drafts_yields_nothing(self):
        entries = [GoldenQuestion(**question(status="draft"))]

        assert reviewed_only(entries) == []


class TestLoading:
    def test_round_trips_through_json(self, tmp_path):
        path = tmp_path / "golden.json"
        path.write_text(
            json.dumps({"questions": [question()]}, ensure_ascii=False),
            encoding="utf-8",
        )

        loaded = load_golden_set(path)

        assert [q.id for q in loaded.questions] == ["q001"]

    def test_a_malformed_entry_fails_loudly_at_load_time(self, tmp_path):
        path = tmp_path / "golden.json"
        path.write_text(
            json.dumps({"questions": [question(citations=[])]}), encoding="utf-8"
        )

        with pytest.raises(ValidationError):
            load_golden_set(path)
