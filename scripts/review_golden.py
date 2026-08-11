"""Revisar el golden set de a un borrador por vez.

    uv run python scripts/review_golden.py

Muestra la pregunta, la respuesta propuesta y el texto del artículo citado, y
guarda después de cada decisión: podés cortar cuando quieras y retomar donde
ibas. Nunca editás JSON a mano.
"""

import argparse
import os
import subprocess
import tempfile
import textwrap
from pathlib import Path

import psycopg
from dotenv import load_dotenv

from regrag.golden import (
    DEFAULT_PATH,
    GoldenQuestion,
    GoldenSet,
    load_golden_set,
    save_golden_set,
)
from regrag.review import parse_edit_buffer, render_edit_buffer

load_dotenv()

WIDTH = 84
RULE = "─" * WIDTH

MENU = """\
  [a] aprobar        la pregunta y la respuesta están bien
  [e] editar         corregir el texto en el editor
  [d] descartar      borrar esta entrada del set
  [s] saltear        dejarla como borrador y seguir
  [q] salir          guarda y termina
"""


def wrap(text: str, indent: str = "  ") -> str:
    return "\n".join(
        textwrap.fill(line, WIDTH - len(indent), initial_indent=indent,
                      subsequent_indent=indent) or indent
        for line in text.splitlines()
    )


def article_text(conn, citation) -> str:
    """El texto del corpus detrás de una cita, para revisar contra la fuente."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT text FROM chunks
            WHERE country = %s AND norm_id = %s
              AND article IS NOT DISTINCT FROM %s
            ORDER BY idx LIMIT 3
            """,
            (citation.country, citation.norm_id, citation.article),
        )
        rows = [row[0] for row in cur.fetchall()]
    return "\n\n".join(rows) if rows else "(no encontrado en el corpus)"


def show(entry: GoldenQuestion, conn, position: str) -> None:
    print(f"\n{RULE}")
    print(f"  {position}   [{entry.id}]  tipo: {entry.kind}")
    print(RULE)
    print("\n  PREGUNTA")
    print(wrap(entry.question, "    "))
    print("\n  RESPUESTA PROPUESTA")
    print(wrap(entry.expected_answer, "    "))

    if entry.citations:
        print("\n  ARTÍCULO(S) CITADO(S) — verificá la respuesta contra esto")
        for citation in entry.citations:
            article = f" art. {citation.article}" if citation.article else ""
            print(f"\n    ── {citation.country} {citation.norm_id}{article} ──")
            print(wrap(article_text(conn, citation)[:1400], "      "))
    else:
        print("\n  SIN CITAS (es una trampa: confirmá que no esté en el corpus)")
        print(f"    uv run python scripts/search.py \"{entry.question[:60]}\"")
    print()


def edit(entry: GoldenQuestion) -> GoldenQuestion | None:
    editor = os.environ.get("EDITOR", "nano")
    with tempfile.NamedTemporaryFile(
        "w+", suffix=".txt", encoding="utf-8", delete=False
    ) as handle:
        handle.write(render_edit_buffer(entry))
        path = Path(handle.name)

    try:
        subprocess.run([editor, str(path)], check=True)
        question, answer = parse_edit_buffer(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        print(f"\n  no se pudo leer el texto: {exc}\n  la entrada queda sin cambios")
        return None
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"\n  no se pudo abrir el editor ({editor})")
        return None
    finally:
        path.unlink(missing_ok=True)

    return entry.model_copy(update={"question": question, "expected_answer": answer})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=str(DEFAULT_PATH))
    parser.add_argument("--kind", choices=("factual", "synthesis", "trap"))
    args = parser.parse_args()

    golden = load_golden_set(args.path)
    entries = list(golden.questions)
    pending = [
        e for e in entries
        if e.status == "draft" and (args.kind is None or e.kind == args.kind)
    ]

    if not pending:
        print("No quedan borradores pendientes.")
        return

    reviewed = sum(1 for e in entries if e.status == "reviewed")
    print(f"{len(pending)} borradores por revisar. Ya revisadas: {reviewed}.")
    print("Se guarda después de cada decisión; podés salir con [q] cuando quieras.")

    conn = psycopg.connect(os.environ["DATABASE_URL"])

    def persist() -> None:
        save_golden_set(GoldenSet(questions=entries), args.path)

    for number, entry in enumerate(pending, 1):
        while True:
            index = entries.index(entry)
            show(entry, conn, f"{number}/{len(pending)}")
            print(MENU)
            choice = input("  > ").strip().lower()[:1]

            if choice == "a":
                entries[index] = entry.model_copy(
                    update={"status": "reviewed", "notes": ""}
                )
                persist()
                print("  aprobada")
                break
            if choice == "e":
                if edited := edit(entry):
                    entry = edited
                    entries[index] = edited
                    persist()
                    continue  # volver a mostrarla ya editada
                continue
            if choice == "d":
                entries.pop(index)
                persist()
                print("  descartada")
                break
            if choice == "s":
                print("  salteada")
                break
            if choice == "q":
                conn.close()
                remaining = sum(1 for e in entries if e.status == "draft")
                print(f"\nGuardado. Quedan {remaining} borradores.")
                return
            print("  opción no reconocida")

    conn.close()
    remaining = sum(1 for e in entries if e.status == "draft")
    print(f"\nListo. Quedan {remaining} borradores.")


if __name__ == "__main__":
    main()
