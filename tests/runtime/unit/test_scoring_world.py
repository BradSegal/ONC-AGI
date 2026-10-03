"""Per-world scoring known-answer tests.

Expected values are derived by hand from the scoring specification; chance is
passed explicitly so the arithmetic is exact.
"""

from __future__ import annotations

import pytest
from arena_factories import fid, group, make_card, make_key, part
from onc_agi.core.errors import ArenaError
from onc_agi.core.schema import CreditRule, ErrorCode, GroupLabel
from onc_agi.services import scoring
from onc_agi.services.scoring import score_world

CARD = make_card(n_features=10)
NO_CHANCE = (0.0, 0.0)


def key_with(*groups, **kw):
    return make_key(CARD, groups, **kw)


def test_true_feature_first_scores_full_credit() -> None:
    key = key_with(group("g0", part(fid(0))))
    score = score_world([fid(0)], key, chance=(0.1, 0.1))
    assert (score.raw_recovery, score.find, score.find_signed, score.find_exact) == (1.0, 1.0, 1.0, 1.0)
    assert not score.is_null and not score.abstained and not score.leaked


def test_chance_normalisation_is_affine_in_raw_recovery() -> None:
    key = key_with(group("g0", part(fid(0))), group("g1", part(fid(1))))
    score = score_world([fid(0), fid(5)], key, chance=(0.25, 0.25))
    assert score.raw_recovery == 0.5
    assert score.find == pytest.approx((0.5 - 0.25) / 0.75)
    assert score.find_signed == pytest.approx(score.find)


def test_below_chance_is_clipped_for_display_but_signed_for_statistics() -> None:
    key = key_with(group("g0", part(fid(0))))
    score = score_world([fid(5)], key, chance=(0.2, 0.2))
    assert score.find == 0.0
    assert score.find_signed == pytest.approx(-0.25)


def test_chance_of_one_gives_zero_rather_than_division_by_zero() -> None:
    key = key_with(group("g0", part(fid(0))))
    assert score_world([fid(0)], key, chance=(1.0, 1.0)).find_signed == 0.0


def test_only_the_top_r_representatives_are_credited() -> None:
    key = key_with(group("g0", part(fid(0))))
    assert score_world([fid(5), fid(0)], key, chance=NO_CHANCE).raw_recovery == 0.0


def test_cluster_mates_after_the_representative_neither_score_nor_use_depth() -> None:
    clusters = {fid(j): j for j in range(10)} | {fid(1): 0}
    key = key_with(group("g0", part(fid(0))), group("g1", part(fid(2))), clusters=clusters)
    assert score_world([fid(0), fid(1), fid(2)], key, chance=NO_CHANCE).raw_recovery == 1.0


def test_first_listed_cluster_member_is_the_representative() -> None:
    clusters = {fid(j): j for j in range(10)} | {fid(1): 0}
    key = key_with(group("g0", part(fid(0))), clusters=clusters)
    assert score_world([fid(1), fid(0)], key, chance=NO_CHANCE).raw_recovery == 0.0
    assert score_world([fid(0), fid(1)], key, chance=NO_CHANCE).raw_recovery == 1.0


def test_neutral_groups_are_removed_and_do_not_consume_depth() -> None:
    key = key_with(group("g0", part(fid(0))), group("g1", part(fid(3)), label=GroupLabel.NEUTRAL))
    assert key.depth == 1
    assert score_world([fid(3), fid(0)], key, chance=NO_CHANCE).raw_recovery == 1.0


def test_a_neutral_representative_shadows_its_cluster_mates() -> None:
    """Deduplication precedes neutral removal (proposal section 4, steps 1-2)."""
    clusters = {fid(j): j for j in range(10)} | {fid(4): 3}
    key = key_with(
        group("g0", part(fid(4))), group("g1", part(fid(3)), label=GroupLabel.NEUTRAL), clusters=clusters
    )
    assert score_world([fid(3), fid(4)], key, chance=NO_CHANCE).raw_recovery == 0.0


def test_a_feature_in_both_a_neutral_and_a_recoverable_set_is_not_neutral() -> None:
    key = key_with(group("g0", part(fid(0), fid(1))), group("g1", part(fid(1)), label=GroupLabel.NEUTRAL))
    assert score_world([fid(1)], key, chance=NO_CHANCE).raw_recovery == 1.0


def test_substitutes_in_the_equivalence_set_earn_group_credit_but_not_exact_credit() -> None:
    key = key_with(group("g0", part(fid(0), fid(6))))
    score = score_world([fid(6)], key, chance=NO_CHANCE)
    assert (score.raw_recovery, score.find_exact) == (1.0, 0.0)


def test_exact_credit_falls_back_to_group_credit_when_exact_is_unrecoverable() -> None:
    key = key_with(group("g0", part(fid(0), fid(6), exact=False)))
    assert score_world([fid(6)], key, chance=NO_CHANCE).find_exact == 1.0


def test_credit_is_one_to_one_by_maximum_matching_by_default() -> None:
    """D12 provisional default: the top R are matched to parts to maximise credit,
    so [f01, f00] and [f00, f01] both credit both groups."""
    assert scoring.CREDIT_ORDER == "max_matching"
    key = key_with(group("g0", part(fid(0), fid(1))), group("g1", part(fid(1))))
    assert score_world([fid(1), fid(0)], key, chance=NO_CHANCE).raw_recovery == 1.0
    assert score_world([fid(0), fid(1)], key, chance=NO_CHANCE).raw_recovery == 1.0


def test_greedy_credit_order_remains_selectable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The greedy alternative: each representative credits the first uncredited group containing it,
    so [f01, f00] credits only g0, while [f00, f01] credits both groups."""
    monkeypatch.setattr(scoring, "CREDIT_ORDER", "greedy")
    key = key_with(group("g0", part(fid(0), fid(1))), group("g1", part(fid(1))))
    assert score_world([fid(1), fid(0)], key, chance=NO_CHANCE).raw_recovery == 0.5
    assert score_world([fid(0), fid(1)], key, chance=NO_CHANCE).raw_recovery == 1.0


def test_matching_does_not_depend_on_hidden_group_order() -> None:
    """Greedy credit depended on the order of groups in the private key; matching does not."""
    a = key_with(group("g0", part(fid(0), fid(1))), group("g1", part(fid(1), fid(2))))
    b = key_with(group("g1", part(fid(1), fid(2))), group("g0", part(fid(0), fid(1))))
    for ranking in ([fid(1), fid(0)], [fid(1), fid(2)], [fid(2), fid(1)]):
        assert (
            score_world(ranking, a, chance=NO_CHANCE).raw_recovery
            == score_world(ranking, b, chance=NO_CHANCE).raw_recovery
            == 1.0
        )


def test_matching_prefers_exact_credit_when_raw_credit_ties() -> None:
    """A substitute and its true feature both in the top R: the true feature takes the part
    (greedy gave the slot to the substitute listed first, and Strict nothing)."""
    key = key_with(group("g0", part(fid(0), fid(1))), group("g1", part(fid(5))))
    score = score_world([fid(1), fid(0)], key, chance=NO_CHANCE)
    assert (score.raw_recovery, score.find_exact) == (0.5, 0.5)


def test_matching_respects_joint_groups() -> None:
    """f01 can stand in for g0's only part or complete the interaction; matching completes both."""
    key = key_with(
        group("g0", part(fid(0), fid(1))),
        group("i0", part(fid(1)), part(fid(2)), role="interaction", rule=CreditRule.JOINT),
    )
    assert score_world([fid(1), fid(2), fid(0)], key, chance=NO_CHANCE).raw_recovery == 1.0


def test_one_feature_cannot_credit_two_groups() -> None:
    key = key_with(group("g0", part(fid(0), fid(1))), group("g1", part(fid(1), fid(0))))
    assert score_world([fid(1)], key, chance=NO_CHANCE).raw_recovery == 0.5


@pytest.mark.parametrize(
    ("ranking", "expected"),
    [((0,), 0.0), ((0, 1), 1.0), ((1, 0), 1.0), ((0, 5, 1), 0.0)],
)
def test_interaction_counts_only_when_both_members_are_in_the_top_r(
    ranking: tuple[int, ...], expected: float
) -> None:
    key = key_with(group("i0", part(fid(0)), part(fid(1)), role="interaction", rule=CreditRule.JOINT))
    assert key.depth == 2
    assert score_world([fid(j) for j in ranking], key, chance=NO_CHANCE).raw_recovery == expected


MODULE = group(
    "m0",
    part(fid(0), weight=3.0),
    part(fid(1), weight=1.0),
    part(fid(2), weight=1.0),
    role="module",
    rule=CreditRule.WEIGHTED_COVERAGE,
)


@pytest.mark.parametrize(
    ("ranking", "expected"),
    [((0, 1, 2), 1.0), ((0,), 0.6), ((1,), 0.2), ((1, 2), 0.4), ((5, 6, 7), 0.0), ((0, 5, 6), 0.6)],
)
def test_module_credit_is_the_covered_share_of_absolute_weight(
    ranking: tuple[int, ...], expected: float
) -> None:
    """A module of three parts uses three depth slots; credit = covered weight / total weight."""
    key = key_with(MODULE)
    assert key.depth == 3
    assert score_world([fid(j) for j in ranking], key, chance=NO_CHANCE).raw_recovery == pytest.approx(
        expected
    )


def test_module_exact_credit_counts_only_exact_members() -> None:
    module = group(
        "m0", part(fid(0), fid(8), weight=1.0), part(fid(1), weight=1.0), rule=CreditRule.WEIGHTED_COVERAGE
    )
    score = score_world([fid(8), fid(1)], key_with(module), chance=NO_CHANCE)
    assert (score.raw_recovery, score.find_exact) == (1.0, 0.5)


def test_module_credit_combines_with_other_groups_by_slot() -> None:
    """Depth 4 = three module slots + one single group; covering g0 and the heaviest member gives (1 + 1.8) / 4."""
    key = key_with(MODULE, group("g0", part(fid(4))))
    assert key.depth == 4
    score = score_world([fid(4), fid(0), fid(5), fid(6)], key, chance=NO_CHANCE)
    assert score.raw_recovery == pytest.approx((1 + 3 * 3.0 / 5.0) / 4)


@pytest.mark.parametrize("position", [0, 1, 9])
def test_a_listed_leak_anywhere_zeroes_the_world(position: int) -> None:
    key = key_with(group("g0", part(fid(0))), reject=(fid(7),))
    rest = [fid(0), fid(1), fid(2), fid(3), fid(4), fid(5), fid(6), fid(8), fid(9)]
    ranking = [*rest[:position], fid(7), *rest[position:]]
    score = score_world(ranking, key, chance=(0.1, 0.1))
    assert score.leaked
    assert score.find == 0.0 and score.find_exact == 0.0
    assert score.find_signed == pytest.approx(-0.1 / 0.9)


def test_a_leak_on_a_null_world_is_not_restraint() -> None:
    key = key_with(reject=(fid(7),))
    score = score_world([fid(7)], key)
    assert score.is_null and score.leaked and not score.restrained and not score.abstained


def test_null_world_rewards_only_an_empty_list() -> None:
    key = key_with()
    empty, claim = score_world([], key), score_world([fid(2)], key)
    assert (empty.restrained, empty.abstained) == (True, True)
    assert (claim.restrained, claim.abstained) == (False, False)
    assert empty.find == claim.find == empty.find_signed == 0.0


def test_listing_only_neutral_features_counts_as_abstention() -> None:
    key = key_with(group("g1", part(fid(3)), label=GroupLabel.NEUTRAL))
    score = score_world([fid(3)], key)
    assert key.is_null and score.abstained and score.restrained


def test_unknown_features_are_rejected() -> None:
    with pytest.raises(ArenaError) as err:
        score_world(["nope"], key_with())
    assert err.value.code is ErrorCode.UNKNOWN_FEATURE


def test_repeated_features_are_rejected() -> None:
    with pytest.raises(ArenaError) as err:
        score_world([fid(0), fid(0)], key_with())
    assert err.value.code is ErrorCode.INVALID_PAYLOAD


@pytest.mark.parametrize(
    ("spent", "expected"), [(0.0, 1.0), (50.0, 1.0), (100.0, 1.0), (200.0, 0.5), (400.0, 0.25)]
)
def test_sequential_efficiency_is_reference_cost_over_spend_capped_at_one(
    spent: float, expected: float
) -> None:
    key = key_with(group("g0", part(fid(0))), reference_cost=100.0)
    assert score_world([fid(0)], key, spent=spent, sequential=True, chance=NO_CHANCE).efficiency == expected


def test_full_access_efficiency_ignores_spend() -> None:
    key = key_with(group("g0", part(fid(0))), reference_cost=100.0)
    assert score_world([fid(0)], key, spent=1e6, sequential=False, chance=NO_CHANCE).efficiency == 1.0


def test_listed_counts_the_raw_submission() -> None:
    key = key_with(group("g0", part(fid(0))))
    assert score_world([fid(0), fid(1), fid(2)], key, chance=NO_CHANCE).listed == 3
