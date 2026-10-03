"""Interface contract v1: every boundary payload validates, and invalid ones fail loudly."""

from __future__ import annotations

import json

import pytest
from arena_factories import fid, group, make_card, make_key, part
from hypothesis import given
from hypothesis import strategies as st
from onc_agi.core.digest import canonical_sha256
from onc_agi.core.errors import ArenaError
from onc_agi.core.schema import (
    INTERFACE_VERSION,
    Action,
    ActionEnvelope,
    Assay,
    CreditRule,
    ErrorCode,
    FeatureMeta,
    GroupLabel,
    Observation,
    Recruit,
    Reset,
    RevealedData,
    Submit,
    TraceEvent,
    TrueGroup,
    TruthPart,
    WorldCard,
)
from pydantic import TypeAdapter, ValidationError

ACTION = TypeAdapter(Action)


def test_interface_version_is_major_minor() -> None:
    major, minor = INTERFACE_VERSION.split(".")
    assert major.isdigit() and minor.isdigit()


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "reset", "request_id": "r0", "world_id": "w-0"},
        {"kind": "recruit", "request_id": "r1", "count": 5},
        {"kind": "recruit", "request_id": "r1", "count": 5, "stratum": "LumA"},
        {"kind": "assay", "request_id": "r2", "feature_ids": ["f00", "f01"]},
        {"kind": "submit", "request_id": "r3", "ranking": ["f00"]},
        {"kind": "submit", "request_id": "r4", "ranking": []},
    ],
)
def test_every_action_kind_round_trips_through_the_discriminated_union(payload: dict[str, object]) -> None:
    action = ACTION.validate_python(payload)
    assert action.kind == payload["kind"]
    assert ACTION.validate_json(action.model_dump_json()) == action
    assert ActionEnvelope.model_validate({"action": payload}).action == action


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "teleport", "request_id": "r"},
        {"kind": "recruit", "request_id": "r", "count": 0},
        {"kind": "recruit", "request_id": "r", "count": 100_001},
        {"kind": "assay", "request_id": "r", "feature_ids": []},
        {"kind": "assay", "request_id": "r", "feature_ids": ["bad name!"]},
        {"kind": "submit", "request_id": "r", "ranking": ["f00", "f00"]},
        {"kind": "submit", "request_id": "", "ranking": []},
        {"kind": "submit", "request_id": "r", "ranking": [], "extra": 1},
        {"kind": "reset", "request_id": "r", "world_id": "Upper-Case"},
    ],
)
def test_invalid_actions_are_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ACTION.validate_python(payload)


@given(st.lists(st.sampled_from([fid(j) for j in range(12)]), max_size=20))
def test_submit_accepts_a_ranking_iff_it_has_no_repeats(ranking: list[str]) -> None:
    if len(set(ranking)) == len(ranking):
        assert Submit(request_id="r", ranking=tuple(ranking)).ranking == tuple(ranking)
    else:
        with pytest.raises(ValidationError):
            Submit(request_id="r", ranking=tuple(ranking))


def test_models_are_frozen() -> None:
    action = Recruit(request_id="r", count=1)
    with pytest.raises(ValidationError):
        action.count = 2  # type: ignore[misc]


def test_world_card_rejects_duplicate_features() -> None:
    card = make_card()
    with pytest.raises(ValidationError, match="unique"):
        WorldCard.model_validate(card.model_dump() | {"features": [card.features[0].model_dump()] * 2})


def test_world_card_carries_the_fixed_premise_and_round_trips() -> None:
    card = make_card()
    assert "ordered list" in card.premise and "empty list" in card.premise
    assert WorldCard.model_validate_json(card.model_dump_json()) == card


def test_negative_prices_are_rejected() -> None:
    with pytest.raises(ValidationError):
        FeatureMeta(feature_id="f00", data_type="expression", assay_price=-1.0)


def test_revealed_data_rejects_misaligned_columns() -> None:
    with pytest.raises(ValidationError, match="misaligned"):
        RevealedData(
            patient_ids=("p0", "p1"), outcome=(0, 1), stratum=("all", "all"), columns={"f00": (1.0,)}
        )
    with pytest.raises(ValidationError, match="misaligned"):
        RevealedData(patient_ids=("p0",), outcome=(0, 1), stratum=("all",), columns={})


def test_observation_with_missing_cells_round_trips() -> None:
    obs = Observation(
        world_id="w-0",
        mode="sequential",
        step=1,
        status="active",
        budget=10.0,
        spent=1.0,
        available_actions=("recruit", "assay", "submit"),
        revealed=RevealedData(patient_ids=("p0",), outcome=(1,), stratum=("all",), columns={"f00": (None,)}),
    )
    assert Observation.model_validate_json(obs.model_dump_json()) == obs


def test_truth_part_must_contain_its_true_feature() -> None:
    with pytest.raises(ValidationError, match="contain the true feature"):
        TruthPart(true_feature="f00", equivalence_set=("f01",), exact_recoverable=True)


def test_truth_part_weights_are_positive() -> None:
    with pytest.raises(ValidationError):
        TruthPart(true_feature="f00", equivalence_set=("f00",), exact_recoverable=True, weight=0.0)


@pytest.mark.parametrize(
    ("rule", "n_parts", "ok"),
    [
        (CreditRule.SINGLE, 1, True),
        (CreditRule.SINGLE, 2, False),
        (CreditRule.JOINT, 1, False),
        (CreditRule.JOINT, 2, True),
        (CreditRule.WEIGHTED_COVERAGE, 1, False),
        (CreditRule.WEIGHTED_COVERAGE, 12, True),
        (CreditRule.WEIGHTED_COVERAGE, 13, False),
    ],
)
def test_credit_rule_constrains_the_number_of_parts(rule: CreditRule, n_parts: int, ok: bool) -> None:
    parts = tuple(part(fid(j)) for j in range(n_parts))
    if ok:
        assert (
            len(
                TrueGroup(
                    group_id="g", role="r", label=GroupLabel.RECOVERABLE, credit_rule=rule, parts=parts
                ).parts
            )
            == n_parts
        )
    else:
        with pytest.raises(ValidationError):
            TrueGroup(group_id="g", role="r", label=GroupLabel.RECOVERABLE, credit_rule=rule, parts=parts)


def test_underdetermined_groups_cannot_be_recoverable() -> None:
    with pytest.raises(ValidationError, match="underdetermined"):
        TrueGroup(
            group_id="g",
            role="mixture",
            label=GroupLabel.RECOVERABLE,
            underdetermined=True,
            parts=(part(fid(0)),),
        )
    neutral = TrueGroup(
        group_id="g", role="mixture", label=GroupLabel.NEUTRAL, underdetermined=True, parts=(part(fid(0)),)
    )
    assert neutral.underdetermined


def test_answer_key_depth_counts_parts_of_recoverable_groups_only() -> None:
    card = make_card(n_features=10)
    key = make_key(
        card,
        [
            group("i0", part(fid(0)), part(fid(1))),
            group("g1", part(fid(2))),
            group("g2", part(fid(3)), label=GroupLabel.NEUTRAL),
        ],
    )
    assert key.depth == 3 and not key.is_null
    assert [g.group_id for g in key.recoverable] == ["i0", "g1"]
    assert make_key(card, [group("g2", part(fid(3)), label=GroupLabel.NEUTRAL)]).is_null


def test_trace_events_require_hex_digests() -> None:
    good = {"world_id": "w-0", "step": 0, "action_kind": "reset", "request_id": "r", "spent": 0.0}
    TraceEvent(**good, request_sha256="0" * 64, response_sha256="a" * 64)
    with pytest.raises(ValidationError):
        TraceEvent(**good, request_sha256="xyz", response_sha256="a" * 64)


def test_arena_errors_carry_a_stable_code_and_payload() -> None:
    err = ArenaError(ErrorCode.OVER_BUDGET, "too much")
    assert str(err) == "over_budget: too much"
    assert err.payload().model_dump(mode="json") == {"code": "over_budget", "message": "too much"}


def test_canonical_digest_ignores_key_order_but_not_values() -> None:
    assert canonical_sha256({"a": 1, "b": [1, 2]}) == canonical_sha256({"b": [1, 2], "a": 1})
    assert canonical_sha256({"a": 1}) != canonical_sha256({"a": 2})
    assert canonical_sha256(json.loads('{"x": 1.0}')) == canonical_sha256({"x": 1.0})


def test_action_digest_is_stable_across_serialisation() -> None:
    action = Assay(request_id="r", feature_ids=("f01", "f00"))
    again = ACTION.validate_json(action.model_dump_json())
    assert canonical_sha256(action.model_dump(mode="json")) == canonical_sha256(again.model_dump(mode="json"))
    assert Reset(request_id="r", world_id="w-0").kind == "reset"
