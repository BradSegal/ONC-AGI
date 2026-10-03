"""Matched chance recovery: a random list pushed through the same pipeline."""

from __future__ import annotations

import pytest
from arena_factories import fid, group, make_card, make_key, part
from onc_agi.services.scoring import chance_recovery, score_world

CARD = make_card(n_features=10)


def test_null_worlds_have_zero_chance() -> None:
    assert chance_recovery(make_key(CARD)) == (0.0, 0.0)


def test_uniform_chance_for_one_true_feature_among_ten_singletons_is_one_tenth() -> None:
    key = make_key(CARD, [group("g0", part(fid(0)))])
    raw, exact = chance_recovery(key, None, draws=4000)
    assert raw == pytest.approx(0.1, abs=0.015)
    assert exact == pytest.approx(raw)


def test_chance_is_deterministic_for_identical_inputs() -> None:
    key = make_key(CARD, [group("g0", part(fid(0)))], strata={fid(j): "a" for j in range(10)})
    ranking = [fid(3), fid(4)]
    assert chance_recovery(key, ranking) == chance_recovery(key, list(ranking))


def test_matched_chance_is_zero_for_an_empty_ranking() -> None:
    key = make_key(CARD, [group("g0", part(fid(0)))], strata={fid(j): "a" for j in range(10)})
    assert chance_recovery(key, []) == (0.0, 0.0)


def test_a_feature_alone_in_its_stratum_is_its_own_matched_replacement() -> None:
    """Picking the only member of a stratum is replicated exactly by chance, so it earns nothing."""
    strata = {fid(j): "a" for j in range(10)} | {fid(0): "b"}
    key = make_key(CARD, [group("g0", part(fid(0)))], strata=strata)
    raw, _ = chance_recovery(key, [fid(0)])
    assert raw == 1.0
    assert score_world([fid(0)], key).find_signed == 0.0


def test_matched_chance_tracks_list_length_within_a_stratum() -> None:
    """Listing k of n same-stratum features covers the single truth with probability k/n at depth 1.

    Only the first representative is scored at depth 1, so chance stays 1/n whatever k is.
    """
    key = make_key(CARD, [group("g0", part(fid(0)))], strata={fid(j): "a" for j in range(10)})
    one, _ = chance_recovery(key, [fid(5)], draws=4000)
    five, _ = chance_recovery(key, [fid(j) for j in range(5, 10)], draws=4000)
    assert one == pytest.approx(0.1, abs=0.015)
    assert five == pytest.approx(0.1, abs=0.015)


def test_listing_a_whole_stratum_that_contains_the_truth_beats_chance_only_by_order() -> None:
    """With depth 2 and two strata, a list of the full truth stratum is replicated by chance."""
    strata = {fid(j): ("t" if j < 3 else "o") for j in range(10)}
    key = make_key(CARD, [group("g0", part(fid(0))), group("g1", part(fid(1)))], strata=strata)
    raw, _ = chance_recovery(key, [fid(2), fid(1), fid(0)], draws=4000)
    # top-2 of a random ordering of {f00, f01, f02}: expected true count 2 * 2/3, over depth 2.
    assert raw == pytest.approx(2 / 3, abs=0.02)


def test_an_empty_list_has_zero_chance_even_without_strata() -> None:
    key = make_key(CARD, [group("g0", part(fid(0)))])
    assert chance_recovery(key, []) == (0.0, 0.0)
    assert score_world([], key).find_signed == 0.0
