import pytest

from regrag.fusion import reciprocal_rank_fusion as rrf


def ids(fused):
    return [chunk_id for chunk_id, _ in fused]


def test_a_single_ranking_keeps_its_order():
    assert ids(rrf([["a", "b", "c"]])) == ["a", "b", "c"]


def test_a_result_both_rankers_agree_on_beats_one_only_a_single_ranker_found():
    """The property the whole method exists for: agreement across independent
    rankers is evidence, and neither ranker has to be confident on its own."""
    dense = ["solo_denso", "acordado"]
    lexical = ["solo_lexico", "acordado"]

    assert ids(rrf([dense, lexical]))[0] == "acordado"


def test_ranks_are_one_indexed():
    ((_, score),) = rrf([["a"]], k=60)

    assert score == pytest.approx(1 / 61)


def test_score_is_the_sum_over_rankings():
    ((_, score),) = rrf([["a"], ["a"]], k=60)

    assert score == pytest.approx(2 / 61)


def test_a_first_place_in_one_ranking_can_lose_to_two_mid_places():
    """k damps the top of each list, so breadth of support outweighs one
    ranker's confidence. With k=60 the gap between rank 1 and rank 5 is small."""
    fused = rrf([["primero", "x", "y", "z", "ambos"], ["q", "r", "s", "t", "ambos"]])

    assert ids(fused)[0] == "ambos"


def test_a_smaller_k_lets_a_confident_first_place_win():
    """k is the knob for exactly that trade-off."""
    fused = rrf([["primero", "x", "y", "z", "ambos"], ["q", "r", "s", "t", "ambos"]], k=1)

    assert ids(fused)[0] == "primero"


def test_results_come_back_in_descending_score_order():
    fused = rrf([["a", "b", "c"], ["c", "b", "a"]])
    scores = [score for _, score in fused]

    assert scores == sorted(scores, reverse=True)


def test_ties_are_broken_deterministically():
    first = rrf([["a", "b"], ["b", "a"]])
    second = rrf([["a", "b"], ["b", "a"]])

    assert ids(first) == ids(second)


def test_an_empty_ranking_contributes_nothing():
    with_empty = rrf([["a", "b"], []])
    alone = rrf([["a", "b"]])

    assert ids(with_empty) == ids(alone)


def test_no_rankings_yields_nothing():
    assert rrf([]) == []
    assert rrf([[], []]) == []


def test_limit_truncates_after_fusing_not_before():
    """Truncating each input first would drop a result that only wins on the
    strength of its combined score."""
    dense = ["a", "b", "gana"]
    lexical = ["c", "d", "gana"]

    assert ids(rrf([dense, lexical], limit=1)) == ["gana"]


def test_a_duplicate_within_one_ranking_is_counted_once():
    ((_, score),) = rrf([["a", "a"]], k=60)

    assert score == pytest.approx(1 / 61)
