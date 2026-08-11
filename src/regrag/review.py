"""Editing a golden-set entry through a plain text buffer."""

from regrag.golden import GoldenQuestion

HEADER = """\
# Editá el texto de abajo, guardá y cerrá el editor.
#   nano: Ctrl+O, Enter, Ctrl+X
# Las líneas que empiezan con # se descartan.
# No borres los encabezados PREGUNTA: ni RESPUESTA:.
"""

QUESTION_HEADING = "PREGUNTA:"
ANSWER_HEADING = "RESPUESTA:"


def render_edit_buffer(entry: GoldenQuestion) -> str:
    return (
        f"{HEADER}\n"
        f"{QUESTION_HEADING}\n{entry.question}\n\n"
        f"{ANSWER_HEADING}\n{entry.expected_answer}\n"
    )


def parse_edit_buffer(text: str) -> tuple[str, str]:
    """Read the edited question and answer back.

    Missing or emptied sections raise rather than returning "", because
    overwriting a reviewed answer with nothing is not a recoverable mistake.
    """
    sections: dict[str, list[str]] = {}
    current: str | None = None

    for line in text.splitlines():
        if line.startswith("#"):
            continue
        stripped = line.strip()
        if stripped in (QUESTION_HEADING, ANSWER_HEADING):
            current = stripped
            sections[current] = []
        elif current is not None:
            sections[current].append(line)

    for heading in (QUESTION_HEADING, ANSWER_HEADING):
        if heading not in sections:
            raise ValueError(f"falta el encabezado {heading}")

    values = []
    for heading in (QUESTION_HEADING, ANSWER_HEADING):
        value = "\n".join(sections[heading]).strip()
        if not value:
            raise ValueError(f"la sección {heading} quedó vacía")
        values.append(value)

    return values[0], values[1]
