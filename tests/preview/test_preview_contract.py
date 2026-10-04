"""Published schemas and examples: every example validates; malformed payloads are refused."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[2]
INDEX = json.loads((ROOT / "examples" / "payloads" / "index.json").read_text())


def schema(name: str) -> dict[str, object]:
    loaded: dict[str, object] = json.loads((ROOT / "schemas" / name).read_text())
    return loaded


@pytest.mark.parametrize("example", sorted(INDEX))
def test_every_example_validates_against_its_schema(example: str) -> None:
    payload = json.loads((ROOT / "examples" / "payloads" / example).read_text())
    jsonschema.validate(payload, schema(INDEX[example]))


def test_every_error_code_in_examples_is_a_declared_code() -> None:
    codes = set(schema("ArenaErrorPayload.schema.json")["$defs"]["ErrorCode"]["enum"])  # type: ignore[index]
    seen = {
        json.loads((ROOT / "examples" / "payloads" / e).read_text())["code"]
        for e in INDEX
        if e.startswith("error_")
    }
    assert seen <= codes and len(seen) >= 6


@pytest.mark.parametrize(
    "bad",
    [
        {"action": {"kind": "submit", "request_id": "r", "ranking": ["a", "a"], "extra": 1}},
        {"action": {"kind": "recruit", "request_id": "r", "count": 0}},
        {"action": {"kind": "teleport", "request_id": "r"}},
    ],
)
def test_malformed_actions_fail_schema_validation(bad: dict[str, object]) -> None:
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema("ActionEnvelope.schema.json"))


def test_openapi_lists_the_eight_endpoints() -> None:
    paths = set(schema("openapi.json")["paths"])  # type: ignore[arg-type]
    assert paths == {
        "/v1/health",
        "/v1/worlds",  # rc3: public-train world listing (additive)
        "/v1/scorecards",
        "/v1/scorecards/{sid}",  # rc3: the closed scorecard on record (additive)
        "/v1/scorecards/{sid}/worlds/{wid}/actions",
        "/v1/scorecards/{sid}/worlds/{wid}",
        "/v1/scorecards/{sid}/close",
        "/v1/scorecards/{sid}/trace",  # rc4: a closed scorecard's server trace, for replay (additive)
    }
