"""The harness-author interface document cannot drift from the schemas: its examples execute."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from onc_agi.adapters import http
from onc_agi.core import schema
from onc_agi.core.schema import INTERFACE_VERSION, ErrorCode
from pydantic import BaseModel

pytestmark = pytest.mark.unit

DOC = Path(__file__).resolve().parents[3] / "docs" / "interface.md"
_BLOCK = re.compile(r"```json schema=(\w+)\n(.*?)```", re.S)


def _examples() -> list[tuple[str, str]]:
    return _BLOCK.findall(DOC.read_text())


def _model(name: str) -> type[BaseModel]:
    model = getattr(schema, name, None) or getattr(http, name)
    assert isinstance(model, type) and issubclass(model, BaseModel)
    return model


def test_the_document_carries_examples_for_every_wire_model() -> None:
    names = {name for name, _ in _examples()}
    assert {
        "WorldCard",
        "ActionEnvelope",
        "Observation",
        "ArenaErrorPayload",
        "OpenRequest",
        "Scorecard",
        "TraceEvent",
    } <= names
    kinds = {json.loads(body)["action"]["kind"] for name, body in _examples() if name == "ActionEnvelope"}
    assert kinds == {"reset", "recruit", "assay", "submit"}


@pytest.mark.parametrize(
    ("name", "body"), _examples(), ids=lambda v: v if isinstance(v, str) and len(v) < 30 else ""
)
def test_every_documented_example_validates_against_its_schema(name: str, body: str) -> None:
    payload = json.loads(body)
    validated = _model(name).model_validate(payload)
    # round trip: the example says nothing the model silently drops
    assert json.loads(validated.model_dump_json(exclude_unset=True)) == payload


def test_the_document_states_the_current_version_and_its_change_policy() -> None:
    text = DOC.read_text()
    assert f"contract v{INTERFACE_VERSION}" in text
    assert "MINOR" in text and "MAJOR" in text and "optional fields only" in text


def test_the_error_table_lists_every_error_code_with_its_http_status() -> None:
    rows = dict(re.findall(r"^\| `(\w+)` \| (\d{3}) \|", DOC.read_text(), re.M))
    assert {ErrorCode(code): int(status) for code, status in rows.items()} == http._STATUS
