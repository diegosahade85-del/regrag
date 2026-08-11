import pytest

from regrag.golden import GoldenQuestion
from regrag.review import parse_edit_buffer, render_edit_buffer

from regrag.answering import Citation


def question(**overrides):
    base = {
        "id": "q001",
        "question": "¿Qué debe presentar el importador?",
        "kind": "factual",
        "answerable": True,
        "expected_answer": "Una declaración jurada.",
        "citations": [Citation(country="AR", norm_id="res-16-2025", article="3")],
        "status": "draft",
    }
    return GoldenQuestion(**(base | overrides))


class TestRender:
    def test_includes_the_current_question_and_answer(self):
        buffer = render_edit_buffer(question())

        assert "¿Qué debe presentar el importador?" in buffer
        assert "Una declaración jurada." in buffer

    def test_round_trips_unchanged(self):
        original = question()

        text, answer = parse_edit_buffer(render_edit_buffer(original))

        assert text == original.question
        assert answer == original.expected_answer


class TestParse:
    def test_reads_edited_text_back(self):
        buffer = (
            "# comentario\n"
            "PREGUNTA:\n¿Nueva pregunta?\n\n"
            "RESPUESTA:\nNueva respuesta.\n"
        )

        assert parse_edit_buffer(buffer) == ("¿Nueva pregunta?", "Nueva respuesta.")

    def test_ignores_comment_lines_anywhere(self):
        buffer = (
            "# instrucciones\n"
            "PREGUNTA:\n# no borres esta sección\n¿Pregunta?\n\n"
            "RESPUESTA:\nRespuesta.\n# fin\n"
        )

        assert parse_edit_buffer(buffer) == ("¿Pregunta?", "Respuesta.")

    def test_preserves_multi_line_answers(self):
        buffer = "PREGUNTA:\n¿P?\n\nRESPUESTA:\nLínea uno.\nLínea dos.\n"

        _, answer = parse_edit_buffer(buffer)

        assert answer == "Línea uno.\nLínea dos."

    def test_a_missing_section_is_an_error_not_a_silent_empty(self):
        """Losing the answer because a heading was deleted would overwrite good
        work with nothing."""
        with pytest.raises(ValueError, match="RESPUESTA"):
            parse_edit_buffer("PREGUNTA:\n¿Sólo pregunta?\n")

    def test_an_emptied_section_is_an_error(self):
        with pytest.raises(ValueError, match="vacía"):
            parse_edit_buffer("PREGUNTA:\n\nRESPUESTA:\nAlgo.\n")

    def test_surrounding_blank_lines_are_trimmed(self):
        buffer = "PREGUNTA:\n\n\n  ¿P?  \n\n\nRESPUESTA:\n\nR.\n\n"

        assert parse_edit_buffer(buffer) == ("¿P?", "R.")
