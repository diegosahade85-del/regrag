from regrag.ingest import deduplicate


def rec(chunk_id, text, **overrides):
    base = {
        "chunk_id": chunk_id,
        "text": text,
        "country": "AR",
        "agency": "SIC",
        "norm_id": "res-16-2025",
        "variant": "texto",
        "year": 2025,
        "article": "3",
        "strategy": "hybrid",
        "index": 0,
        "n_chars": len(text),
    }
    return base | overrides


def test_identical_text_collapses_to_one_record():
    records = [
        rec("AR_SIC_res-16-2025_texto#hybrid#10", "ARTÍCULO 3°.- FABRICANTES."),
        rec(
            "AR_SIC_res-16-2025_boletin#hybrid#10",
            "ARTÍCULO 3°.- FABRICANTES.",
            variant="boletin",
        ),
    ]

    assert len(deduplicate(records)) == 1


def test_the_survivor_is_the_first_occurrence():
    records = [
        rec("a#hybrid#0", "mismo texto"),
        rec("b#hybrid#0", "mismo texto", variant="boletin"),
    ]

    (survivor,) = deduplicate(records)

    assert survivor["chunk_id"] == "a#hybrid#0"


def test_distinct_text_is_untouched():
    records = [
        rec("a#hybrid#0", "ARTÍCULO 3°.- FABRICANTES."),
        rec("b#hybrid#1", "ARTÍCULO 4°.- DISTRIBUIDORES."),
    ]

    assert len(deduplicate(records)) == 2


def test_whitespace_differences_do_not_defeat_deduplication():
    """Two renderings of one article differ in markup whitespace, not in law."""
    records = [
        rec("a#hybrid#0", "ARTÍCULO 3°.-  FABRICANTES\n  E IMPORTADORES."),
        rec("b#hybrid#0", "ARTÍCULO 3°.- FABRICANTES E IMPORTADORES."),
    ]

    assert len(deduplicate(records)) == 1


def test_the_survivor_records_how_many_renderings_carried_it():
    records = [
        rec("a#hybrid#0", "mismo texto"),
        rec("b#hybrid#0", "mismo texto", variant="boletin"),
    ]

    (survivor,) = deduplicate(records)

    assert survivor["n_sources"] == 2


def test_a_unique_chunk_reports_one_source():
    (survivor,) = deduplicate([rec("a#hybrid#0", "único")])

    assert survivor["n_sources"] == 1


def test_identical_text_in_different_norms_is_not_collapsed():
    """Boilerplate like 'Comuníquese y archívese' recurs across norms, and each
    occurrence is a real, separately-citable clause of its own norm."""
    records = [
        rec("a#hybrid#0", "Comuníquese, publíquese y archívese.", norm_id="res-16-2025"),
        rec("b#hybrid#0", "Comuníquese, publíquese y archívese.", norm_id="res-92-1998"),
    ]

    assert len(deduplicate(records)) == 2


def test_deduplicating_nothing_returns_nothing():
    assert deduplicate([]) == []
