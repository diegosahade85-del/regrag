import json

from regrag.chunking import Chunk
from regrag.ingest import chunk_document, to_record
from regrag.metadata import SourceMetadata

SOURCE = SourceMetadata(
    country="AR", agency="SIC", norm_id="res-16-2025", variant="texto", year=2025
)


def write_html(tmp_path, body: str, name="AR_SIC_res-16-2025_texto.html"):
    path = tmp_path / name
    path.write_text(f"<html><body>{body}</body></html>", encoding="utf-8")
    return path


def test_chunk_document_reads_a_file_and_returns_chunks_with_its_metadata(tmp_path):
    path = write_html(
        tmp_path,
        "<p>ARTÍCULO 1°.- OBJETO. Apruébase.</p><p>ARTÍCULO 2°.- Vigencia.</p>",
    )

    chunks = chunk_document(path, strategy="article")

    assert [c.article for c in chunks] == ["1", "2"]
    assert chunks[0].source.country == "AR"
    assert chunks[0].source.norm_id == "res-16-2025"


def test_chunk_document_honours_the_fixed_strategy(tmp_path):
    path = write_html(tmp_path, "<p>" + " ".join(["palabra"] * 300) + "</p>")

    chunks = chunk_document(path, strategy="fixed", size=200, overlap=50)

    assert len(chunks) > 1
    assert all(c.strategy == "fixed" for c in chunks)


def test_record_flattens_metadata_for_downstream_indexing():
    chunk = Chunk(
        text="ARTÍCULO 1°.- OBJETO.",
        article="1",
        source=SOURCE,
        strategy="article",
        index=0,
    )

    record = to_record(chunk)

    assert record["text"] == "ARTÍCULO 1°.- OBJETO."
    assert record["country"] == "AR"
    assert record["agency"] == "SIC"
    assert record["norm_id"] == "res-16-2025"
    assert record["variant"] == "texto"
    assert record["year"] == 2025
    assert record["article"] == "1"
    assert record["strategy"] == "article"


def test_record_is_json_serialisable():
    chunk = Chunk(text="t", article=None, source=SOURCE, strategy="article", index=0)

    assert json.loads(json.dumps(to_record(chunk)))["text"] == "t"


def test_chunk_id_is_stable_and_unique_per_chunk():
    first = Chunk(text="a", article="1", source=SOURCE, strategy="article", index=0)
    second = Chunk(text="b", article="2", source=SOURCE, strategy="article", index=1)

    assert to_record(first)["chunk_id"] == to_record(first)["chunk_id"]
    assert to_record(first)["chunk_id"] != to_record(second)["chunk_id"]


def test_chunk_id_distinguishes_strategies_for_the_same_position():
    article = Chunk(text="a", article="1", source=SOURCE, strategy="article", index=0)
    fixed = Chunk(text="a", article=None, source=SOURCE, strategy="fixed", index=0)

    assert to_record(article)["chunk_id"] != to_record(fixed)["chunk_id"]


def test_chunk_document_honours_the_hybrid_strategy(tmp_path):
    body = '<p>ARTÍCULO 1°.- ' + ' '.join(['requisito'] * 400) + '</p>'
    path = write_html(tmp_path, body)

    chunks = chunk_document(path, strategy='hybrid', size=1000, overlap=100)

    assert len(chunks) > 1
    assert all(c.strategy == 'hybrid' for c in chunks)
    assert all(c.article == '1' for c in chunks)
    assert all(len(c.text) <= 1000 for c in chunks)
