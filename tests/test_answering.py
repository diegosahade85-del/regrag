import pytest

from regrag.answering import (
    Answer,
    Citation,
    format_context,
    unsupported_citations,
)
from regrag.store import SearchResult


def hit(norm_id="res-16-2025", article="3", country="AR", text="Texto del artículo."):
    return SearchResult(
        chunk_id=f"{country}_{norm_id}#hybrid#0",
        text=text,
        country=country,
        agency="SIC",
        norm_id=norm_id,
        variant="texto",
        year=2025,
        article=article,
        score=0.5,
    )


def answer(citations, **overrides):
    base = {
        "answerable": True,
        "answer": "Los fabricantes deben presentar una declaración jurada.",
        "citations": citations,
        "confidence": "alta",
    }
    return Answer(**(base | overrides))


class TestFormatContext:
    def test_includes_the_text_of_every_chunk(self):
        rendered = format_context([hit(text="Primero."), hit(article="4", text="Segundo.")])

        assert "Primero." in rendered
        assert "Segundo." in rendered

    def test_labels_each_chunk_with_the_citation_the_model_must_return(self):
        rendered = format_context([hit(country="AR", norm_id="res-16-2025", article="3")])

        assert "AR" in rendered
        assert "res-16-2025" in rendered
        assert "3" in rendered

    def test_marks_a_chunk_that_has_no_article(self):
        rendered = format_context([hit(article=None, text="VISTO el Expediente.")])

        assert "VISTO el Expediente." in rendered

    def test_no_context_says_so_rather_than_rendering_empty(self):
        assert format_context([]).strip()


class TestUnsupportedCitations:
    def test_a_citation_matching_a_retrieved_chunk_is_supported(self):
        context = [hit(country="AR", norm_id="res-16-2025", article="3")]
        given = answer([Citation(country="AR", norm_id="res-16-2025", article="3")])

        assert unsupported_citations(given, context) == []

    def test_a_citation_for_a_norm_never_retrieved_is_flagged(self):
        """The failure this exists to catch: a well-formed citation to a norm the
        model was never shown. It looks exactly like a real one."""
        context = [hit(norm_id="res-16-2025")]
        given = answer([Citation(country="AR", norm_id="res-99-1999", article="3")])

        assert len(unsupported_citations(given, context)) == 1

    def test_a_citation_to_the_wrong_article_of_a_retrieved_norm_is_flagged(self):
        context = [hit(norm_id="res-16-2025", article="3")]
        given = answer([Citation(country="AR", norm_id="res-16-2025", article="9")])

        assert len(unsupported_citations(given, context)) == 1

    def test_a_citation_to_the_wrong_country_is_flagged(self):
        context = [hit(country="AR", norm_id="res-16-2025", article="3")]
        given = answer([Citation(country="CL", norm_id="res-16-2025", article="3")])

        assert len(unsupported_citations(given, context)) == 1

    def test_only_the_unsupported_ones_come_back(self):
        context = [hit(article="3"), hit(article="4")]
        given = answer(
            [
                Citation(country="AR", norm_id="res-16-2025", article="3"),
                Citation(country="AR", norm_id="res-16-2025", article="99"),
            ]
        )

        (bad,) = unsupported_citations(given, context)
        assert bad.article == "99"

    def test_an_articleless_citation_matches_an_articleless_chunk(self):
        context = [hit(article=None)]
        given = answer([Citation(country="AR", norm_id="res-16-2025", article=None)])

        assert unsupported_citations(given, context) == []

    def test_citing_nothing_is_not_an_unsupported_citation(self):
        assert unsupported_citations(answer([]), [hit()]) == []


class TestAnswerSchema:
    def test_an_unanswerable_question_carries_no_citations(self):
        """In compliance an invented answer is worse than none, so refusing must
        be a structural state the caller can branch on — not a sentence in the
        prose that has to be pattern-matched."""
        given = Answer(
            answerable=False,
            answer="El corpus no contiene información sobre este punto.",
            citations=[],
            confidence="baja",
        )

        assert given.answerable is False
        assert given.citations == []

    def test_confidence_is_an_ordinal_scale_not_a_float(self):
        with pytest.raises(ValueError):
            Answer(answerable=True, answer="x", citations=[], confidence="0.87")

    def test_confidence_accepts_the_three_levels(self):
        for level in ("alta", "media", "baja"):
            assert Answer(
                answerable=True, answer="x", citations=[], confidence=level
            ).confidence == level
