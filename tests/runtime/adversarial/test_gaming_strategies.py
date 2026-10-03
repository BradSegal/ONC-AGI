"""Gaming strategies must not pay.

Two levels: list-construction attacks on hand-built answer keys (exact), and the
registered cheater agents on the packaged public-train fixture worlds (the smoke floor).
"""

from __future__ import annotations

from functools import cache
from importlib import resources
from pathlib import Path

import pytest
from arena_factories import fid, group, make_card, make_key, part
from onc_agi.adapters.agents import CHEATERS, make_agent
from onc_agi.core.schema import GroupLabel, Scorecard, Tier
from onc_agi.infra.bundles import FileWorldStore
from onc_agi.services.kit import evaluate
from onc_agi.services.scoring import score_world

CARD = make_card(n_features=12)
FLOOR = 0.02


# ---------------------------------------------------------------- list construction attacks


def test_padding_with_cluster_mates_of_a_wrong_feature_cannot_push_the_truth_down() -> None:
    """Near-duplicates collapse to one representative, so they occupy one slot, not many."""
    clusters = {fid(j): j for j in range(12)} | {fid(j): 5 for j in (5, 6, 7, 8)}
    key = make_key(CARD, [group("g0", part(fid(0))), group("g1", part(fid(1)))], clusters=clusters)
    score = score_world([fid(5), fid(6), fid(7), fid(8), fid(0)], key, chance=(0.0, 0.0))
    assert score.raw_recovery == 0.5


def test_listing_every_member_of_the_true_cluster_earns_no_extra_credit() -> None:
    clusters = {fid(j): j for j in range(12)} | {fid(1): 0, fid(2): 0}
    key = make_key(CARD, [group("g0", part(fid(0), fid(1), fid(2)))], clusters=clusters)
    assert score_world([fid(0), fid(1), fid(2)], key, chance=(0.0, 0.0)).raw_recovery == 1.0
    assert score_world([fid(0), fid(1), fid(2)], key, chance=(0.0, 0.0)).listed == 3


def test_hiding_a_leak_behind_the_true_answer_still_zeroes_the_world() -> None:
    key = make_key(CARD, [group("g0", part(fid(0)))], reject=(fid(11),))
    score = score_world([fid(0)] + [fid(j) for j in range(1, 11)] + [fid(11)], key)
    assert score.leaked and score.find == 0.0


def test_hiding_a_leak_behind_a_cluster_mate_still_zeroes_the_world() -> None:
    clusters = {fid(j): j for j in range(12)} | {fid(11): 3}
    key = make_key(CARD, [group("g0", part(fid(0)))], reject=(fid(11),), clusters=clusters)
    assert score_world([fid(0), fid(3), fid(11)], key).leaked


def test_neutral_padding_does_not_turn_a_claim_into_restraint_on_null_worlds() -> None:
    key = make_key(CARD, [group("n0", part(fid(4)), label=GroupLabel.NEUTRAL)])
    assert score_world([fid(4), fid(5)], key).restrained is False


def test_hedging_with_the_whole_feature_set_is_chance_on_signal_worlds() -> None:
    strata = {fid(j): "a" for j in range(12)}
    key = make_key(CARD, [group("g0", part(fid(3)))], strata=strata)
    everything = [fid(j) for j in range(12)]
    assert abs(score_world(everything, key).find_signed) < 0.15


# ---------------------------------------------------------------- registered cheaters on fixture worlds


@cache
def fixture_store() -> FileWorldStore:
    root = Path(str(resources.files("onc_agi") / "fixtures" / "store"))
    if not (root / "public_train").exists():
        pytest.skip("packaged fixture worlds are not built")
    return FileWorldStore(root)


@cache
def fixture_scorecard(name: str) -> Scorecard:
    store = fixture_store()
    card, _ = evaluate(
        make_agent(name, store), store, Tier.PUBLIC_TRAIN, with_alignment=False, bootstrap_draws=200
    )
    return card


def test_the_fixture_set_contains_both_world_types_and_both_modes() -> None:
    store = fixture_store()
    ids = store.world_ids(Tier.PUBLIC_TRAIN)
    keys = [store.answer_key(w) for w in ids]
    assert any(k.is_null for k in keys) and any(not k.is_null for k in keys)
    assert {store.card(w).mode.value for w in ids} == {"full_access", "sequential"}


def test_the_oracle_reference_scores_one_on_the_fixtures() -> None:
    card = fixture_scorecard("oracle")
    assert card.discovery_score == pytest.approx(1.0) and card.leak_rate == 0.0


@pytest.mark.parametrize("name", sorted(set(CHEATERS) | {"random"}))
def test_every_cheater_scores_at_the_floor_on_the_fixtures(name: str) -> None:
    card = fixture_scorecard(name)
    assert card.discovery_score is not None and card.discovery_score <= FLOOR
    assert card.discovery_score_unfloored is not None and card.discovery_score_unfloored <= FLOOR


@pytest.mark.parametrize("name", ["giant_list", "leak_exploiter"])
def test_leak_listing_cheaters_are_caught_on_every_world(name: str) -> None:
    assert fixture_scorecard(name).leak_rate == 1.0


def test_a_standard_baseline_sits_strictly_between_floor_and_oracle() -> None:
    card = fixture_scorecard("univariate_bh")
    assert card.discovery_score is not None and FLOOR < card.discovery_score < 1.0
