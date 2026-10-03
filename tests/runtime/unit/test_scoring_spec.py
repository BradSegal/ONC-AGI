"""The scoring specification executes: every worked example in docs/scoring.md holds."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pytest
from onc_agi.core.errors import ArenaError
from onc_agi.core.schema import AnswerKey, Tier, WorldScore
from onc_agi.services import scoring

DOC = Path(__file__).resolve().parents[3] / "docs" / "scoring.md"
_BLOCK = re.compile(r"```json (scoring-\w+)((?: \w+=[\w-]+)*)\n(.*?)```", re.S)


def _blocks(kind: str) -> list[tuple[dict[str, str], Any]]:
    out = []
    for found, attrs, body in _BLOCK.findall(DOC.read_text()):
        if found == kind:
            out.append((dict(a.split("=") for a in attrs.split()), json.loads(body)))
    return out


WORLDS = {attrs["name"]: AnswerKey.model_validate(body) for attrs, body in _blocks("scoring-world")}


def _label(case: tuple[dict[str, str], Any]) -> str:
    attrs, body = case
    return f"{attrs.get('world', 'agg')}:{','.join(body.get('ranking', [])) or 'empty'}"


def test_the_specification_carries_examples_of_every_kind() -> None:
    assert len(WORLDS) >= 4
    assert len(_blocks("scoring-example")) >= 10
    assert _blocks("scoring-chance") and _blocks("scoring-aggregate")
    assert _blocks("scoring-invariant") and _blocks("scoring-error")
    referenced = {
        attrs["world"]
        for kind in ("scoring-example", "scoring-chance", "scoring-invariant", "scoring-error")
        for attrs, _ in _blocks(kind)
    }
    assert referenced <= set(WORLDS)


@pytest.mark.parametrize("case", _blocks("scoring-example"), ids=_label)
def test_every_worked_example_matches_the_scorer(case: tuple[dict[str, str], Any]) -> None:
    attrs, body = case
    chance = tuple(body["chance"]) if "chance" in body else None
    score = scoring.score_world(
        tuple(body["ranking"]),
        WORLDS[attrs["world"]],
        spent=body.get("spent", 0.0),
        sequential=body.get("sequential", False),
        chance=chance,  # type: ignore[arg-type]
    )
    for field, expected in body["expect"].items():
        actual = getattr(score, field)
        if isinstance(expected, float):
            assert math.isclose(actual, expected, abs_tol=1e-12), (field, actual, expected)
        else:
            assert actual == expected, (field, actual, expected)


@pytest.mark.parametrize("case", _blocks("scoring-chance"), ids=_label)
def test_every_matched_chance_example_holds_within_its_tolerance(case: tuple[dict[str, str], Any]) -> None:
    attrs, body = case
    raw, _ = scoring.chance_recovery(WORLDS[attrs["world"]], tuple(body["ranking"]))
    assert abs(raw - body["expect_raw"]) <= body["tolerance"], raw


@pytest.mark.parametrize("case", _blocks("scoring-invariant"), ids=lambda c: c[0]["world"])
def test_every_invariance_example_gives_identical_values(case: tuple[dict[str, str], Any]) -> None:
    attrs, body = case
    values = {
        getattr(scoring.score_world(tuple(r), WORLDS[attrs["world"]]), body["field"])
        for r in body["rankings"]
    }
    assert len(values) == 1, values


@pytest.mark.parametrize("case", _blocks("scoring-error"), ids=_label)
def test_every_refused_submission_raises_its_typed_error(case: tuple[dict[str, str], Any]) -> None:
    attrs, body = case
    with pytest.raises(ArenaError) as refused:
        scoring.score_world(tuple(body["ranking"]), WORLDS[attrs["world"]])
    assert refused.value.code.value == body["code"]


def _world_score(i: int, spec: dict[str, Any]) -> WorldScore:
    null = spec["null"]
    return WorldScore(
        world_id=f"agg-{i}",
        difficulty_tier=1,
        is_null=null,
        find=spec.get("find", 0.0),
        find_signed=spec.get("find_signed", 0.0),
        find_exact=spec.get("find", 0.0),
        raw_recovery=0.0,
        chance_recovery=0.0,
        restrained=spec.get("restrained", False),
        abstained=spec.get("abstained", spec.get("restrained", False)),
        leaked=False,
        spent=0.0,
        efficiency=spec.get("efficiency", 1.0),
        listed=0,
    )


@pytest.mark.parametrize("case", _blocks("scoring-aggregate"), ids=lambda c: f"agg-{len(c[1]['worlds'])}")
def test_every_aggregate_example_matches_the_scorecard_formulae(case: tuple[dict[str, str], Any]) -> None:
    _, body = case
    scores = [_world_score(i, w) for i, w in enumerate(body["worlds"])]
    c = scoring.components(scores)
    card = scoring.aggregate(scores, scorecard_id="spec", agent="spec", tier=Tier.PUBLIC_TRAIN)
    observed = {
        "find": c.find,
        "restraint": c.restraint,
        "discovery_score": card.discovery_score,
        "unfloored": c.unfloored,
        "abstention_on_signal": c.abstention_signal,
        "restraint_on_null": c.restraint_null,
    }
    for field, expected in body["expect"].items():
        assert math.isclose(observed[field], expected, abs_tol=1e-12), (field, observed[field], expected)
