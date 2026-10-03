"""Aggregation into Find, chance-corrected Restraint and the Discovery Score."""

from __future__ import annotations

import math

import pytest
from onc_agi.core.schema import Tier, WorldScore
from onc_agi.services.scoring import aggregate, bootstrap_interval, components
from pydantic import ValidationError


def ws(
    i: int,
    *,
    null: bool,
    find: float = 0.0,
    signed: float | None = None,
    exact: float | None = None,
    abstained: bool = False,
    leaked: bool = False,
    efficiency: float = 1.0,
    tier: int = 0,
    spent: float = 0.0,
) -> WorldScore:
    return WorldScore(
        world_id=f"w-{i:02d}",
        difficulty_tier=tier,
        is_null=null,
        find=0.0 if null else find,
        find_signed=0.0 if null else (find if signed is None else signed),
        find_exact=0.0 if null else (find if exact is None else exact),
        raw_recovery=0.0,
        chance_recovery=0.0,
        restrained=null and abstained and not leaked,
        abstained=abstained,
        leaked=leaked,
        spent=spent,
        efficiency=efficiency,
        listed=0 if abstained else 1,
    )


def test_components_follow_the_hand_computed_youden_form() -> None:
    scores = [
        ws(0, null=False, find=1.0),
        ws(1, null=False, find=0.5, abstained=True),
        ws(2, null=True, abstained=True),
        ws(3, null=True),
    ]
    c = components(scores)
    assert (c.find, c.restraint_null, c.abstention_signal) == (0.75, 0.5, 0.5)
    assert c.restraint == 0.0 and c.displayed == 0.0


@pytest.mark.parametrize("abstain_everywhere", [True, False])
def test_constant_policies_have_zero_restraint(abstain_everywhere: bool) -> None:
    scores = [ws(i, null=i % 2 == 0, abstained=abstain_everywhere, find=0.0) for i in range(10)]
    c = components(scores)
    assert c.restraint == 0.0
    assert c.displayed == 0.0 and c.unfloored == 0.0


def test_the_oracle_profile_scores_one() -> None:
    scores = [ws(0, null=False, find=1.0), ws(1, null=True, abstained=True)]
    c = components(scores)
    assert (c.find, c.restraint, c.displayed, c.unfloored) == (1.0, 1.0, 1.0, 1.0)


def test_display_floors_restraint_but_the_unfloored_estimand_keeps_its_sign() -> None:
    scores = [
        ws(0, null=False, find=0.8, abstained=False),
        ws(1, null=False, find=0.4, abstained=True),
        ws(2, null=True),
    ]
    c = components(scores)
    assert c.restraint == pytest.approx(-0.5)
    assert c.displayed == 0.0
    assert c.unfloored == pytest.approx(0.6 * -0.5)


def test_unfloored_estimand_uses_signed_find() -> None:
    scores = [ws(0, null=False, find=0.0, signed=-0.2), ws(1, null=True, abstained=True)]
    c = components(scores)
    assert c.find == 0.0 and c.unfloored == pytest.approx(-0.2)


def test_efficiency_scales_find_and_null_restraint_but_not_signal_abstention() -> None:
    scores = [
        ws(0, null=False, find=1.0, efficiency=0.5),
        ws(1, null=True, abstained=True, efficiency=0.5),
        ws(2, null=False, find=0.0, abstained=True, efficiency=0.25),
    ]
    c = components(scores)
    assert c.find == pytest.approx(0.25)
    assert c.restraint_null == pytest.approx(0.5)
    assert c.abstention_signal == pytest.approx(0.5)


def test_aggregating_nothing_fails_fast() -> None:
    with pytest.raises(ValueError):
        components([])


def test_without_null_worlds_restraint_and_the_headline_are_undefined() -> None:
    c = components([ws(0, null=False, find=1.0), ws(1, null=False, find=0.5)])
    assert c.find == 0.75
    assert math.isnan(c.restraint) and math.isnan(c.displayed) and math.isnan(c.unfloored)


def test_without_signal_worlds_find_and_the_headline_are_undefined() -> None:
    c = components([ws(0, null=True, abstained=True), ws(1, null=True)])
    assert c.restraint_null == 0.5
    assert math.isnan(c.find) and math.isnan(c.restraint) and math.isnan(c.displayed)


def test_single_type_scorecards_report_undefined_metrics_as_none() -> None:
    card = _card(Tier.PUBLIC_EVAL, [ws(0, null=False, find=1.0), ws(1, null=False)])
    assert card.discovery_score is None and card.restraint is None and card.discovery_score_unfloored is None
    assert (card.interval.low, card.interval.high) == (None, None)
    assert card.find == 0.5


def test_strict_score_is_undefined_when_restraint_is_undefined() -> None:
    assert _card(Tier.PUBLIC_EVAL, [ws(0, null=False, find=1.0)]).strict_discovery_score is None


def test_bootstrap_interval_is_degenerate_for_identical_worlds() -> None:
    scores = [ws(i, null=False, find=0.5) for i in range(5)] + [ws(9, null=True, abstained=True)]
    interval = bootstrap_interval(scores, draws=200)
    assert interval.low == pytest.approx(0.5) and interval.high == pytest.approx(0.5)


def test_bootstrap_interval_is_seeded_and_brackets_the_estimate() -> None:
    scores = [ws(i, null=False, find=i / 10) for i in range(10)] + [
        ws(10 + i, null=True, abstained=i % 3 != 0) for i in range(6)
    ]
    a, b = bootstrap_interval(scores, draws=300, seed=4), bootstrap_interval(scores, draws=300, seed=4)
    assert a == b
    estimate = components(scores).unfloored
    assert a.low <= estimate <= a.high


def _card(tier: Tier, scores: list[WorldScore], **kw: object):
    return aggregate(scores, scorecard_id="sc-1", agent="a", tier=tier, bootstrap_draws=50, **kw)  # type: ignore[arg-type]


SCORES = [
    ws(0, null=False, find=1.0, tier=0, spent=10.0),
    ws(1, null=True, abstained=True, tier=0, spent=0.0),
    ws(2, null=False, find=0.5, exact=0.25, tier=1, leaked=False, spent=20.0),
    ws(3, null=False, find=0.0, tier=1, leaked=True, spent=30.0),
]


def test_per_world_results_are_exposed_only_for_public_train() -> None:
    assert len(_card(Tier.PUBLIC_TRAIN, SCORES).worlds) == 4
    assert _card(Tier.PUBLIC_EVAL, SCORES).worlds == ()
    assert _card(Tier.PRIVATE, SCORES).worlds == ()


def test_scorecard_headline_fields_match_components() -> None:
    card = _card(Tier.PUBLIC_EVAL, SCORES)
    assert card.find == pytest.approx(0.5)
    assert card.restraint == pytest.approx(1.0)
    assert card.discovery_score == pytest.approx(0.5)
    assert card.strict_discovery_score == pytest.approx((1.0 + 0.25 + 0.0) / 3)
    assert card.leak_rate == pytest.approx(0.25)
    assert card.mean_data_cost == pytest.approx(15.0)
    assert card.n_worlds == 4


def test_tiers_without_both_world_types_report_undefined() -> None:
    per_tier = {t.difficulty_tier: t for t in _card(Tier.PUBLIC_TRAIN, SCORES).per_tier}
    assert per_tier[0].discovery_score == pytest.approx(1.0)
    assert (per_tier[1].n_signal, per_tier[1].n_null) == (2, 0)
    assert (per_tier[1].find, per_tier[1].restraint, per_tier[1].discovery_score) == (None, None, None)


def test_unknown_tracks_are_rejected_by_the_contract() -> None:
    with pytest.raises(ValidationError):
        _card(Tier.PUBLIC_TRAIN, SCORES, track="secret")


def test_the_unfloored_score_is_never_positive_when_neither_component_is() -> None:
    c = components(
        [
            ws(0, null=False, find=0.0, signed=-0.1, abstained=True),
            ws(1, null=True, abstained=True, efficiency=0.5),
        ]
    )
    assert c.find_signed < 0 and c.restraint < 0
    assert c.unfloored <= 0.0
