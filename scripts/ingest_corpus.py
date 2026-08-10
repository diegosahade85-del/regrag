"""Chunk the whole raw corpus with both strategies and persist to JSONL.

    uv run python scripts/ingest_corpus.py

Writes data/processed/chunks_{strategy}.jsonl and prints a per-strategy
comparison so the choice recorded in the README is backed by numbers.
"""

import json
import statistics
from pathlib import Path

from regrag.ingest import chunk_document, deduplicate, to_record

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
STRATEGIES = ("article", "fixed", "hybrid")


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in RAW.iterdir() if p.suffix.lower() in {".pdf", ".html"})
    print(f"{len(files)} source documents in {RAW}/\n")

    for strategy in STRATEGIES:
        records, failures = [], []
        for path in files:
            try:
                records.extend(to_record(c) for c in chunk_document(path, strategy))
            except Exception as exc:  # keep going; report at the end
                failures.append((path.name, f"{type(exc).__name__}: {exc}"))

        before = len(records)
        records = deduplicate(records)
        dropped = before - len(records)

        out = PROCESSED / f"chunks_{strategy}.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        sizes = [r["n_chars"] for r in records] or [0]
        with_article = sum(1 for r in records if r["article"])
        print(f"[{strategy}] -> {out}")
        print(f"  chunks         {len(records)}  (dropped {dropped} duplicates)")
        print(f"  chars  median  {int(statistics.median(sizes))}")
        print(f"         p95     {int(sorted(sizes)[int(len(sizes) * 0.95) - 1])}")
        print(f"         max     {max(sizes)}")
        print(f"  with article   {with_article} ({with_article / max(len(records), 1):.0%})")
        for name, err in failures:
            print(f"  FAILED {name}: {err}")
        print()


if __name__ == "__main__":
    main()
