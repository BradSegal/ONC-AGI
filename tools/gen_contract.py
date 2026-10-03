"""Generate the published contract artefacts from the runtime itself.

* ``schemas/*.schema.json`` - JSON Schema for every payload that crosses a boundary;
* ``schemas/openapi.json`` - the HTTP interface;
* ``examples/payloads/*.json`` - real requests and responses, captured by driving
  the HTTP server in-process over the toy fixture worlds, plus ``index.json``
  mapping each example to its schema.

The output is deterministic (random scorecard identifiers are normalised), so CI
regenerates it and fails on any difference. Run from the repository root::

    uv run python tools/gen_contract.py
"""

from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from onc_agi.adapters.http import OpenRequest, OpenResponse, create_app
from onc_agi.core.schema import (
    ActionEnvelope,
    AnswerKey,
    ArenaErrorPayload,
    Observation,
    Scorecard,
    TraceEvent,
    WorldCard,
    WorldScore,
)
from onc_agi.infra.bundles import FileWorldStore
from onc_agi.infra.ledger import JsonLedger
from onc_agi.infra.recorder import TraceRecorder
from onc_agi.services.scorecards import ScorecardService
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "src" / "onc_agi" / "fixtures" / "store"
SCHEMAS = ROOT / "schemas"
EXAMPLES = ROOT / "examples" / "payloads"
KEY = "example-key-0001"
SID = "sc-0000000000000000"

MODELS: dict[str, type[BaseModel]] = {
    "WorldCard": WorldCard,
    "ActionEnvelope": ActionEnvelope,
    "Observation": Observation,
    "ArenaErrorPayload": ArenaErrorPayload,
    "OpenRequest": OpenRequest,
    "OpenResponse": OpenResponse,
    "WorldScore": WorldScore,
    "Scorecard": Scorecard,
    "TraceEvent": TraceEvent,
    "AnswerKey": AnswerKey,
}


def stable(payload: Any) -> Any:
    """Floats to 10 significant digits, so reductions that differ in the last bits across CPUs agree."""
    if isinstance(payload, float):
        return float(f"{payload:.10g}")
    if isinstance(payload, dict):
        return {k: stable(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [stable(v) for v in payload]
    return payload


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stable(payload), indent=1, sort_keys=True) + "\n")


def normalise(payload: Any, sid: str) -> Any:
    return json.loads(json.dumps(payload).replace(sid, SID))


def write_schemas() -> None:
    for old in SCHEMAS.glob("*.json"):
        old.unlink()
    for name, model in MODELS.items():
        dump(SCHEMAS / f"{name}.schema.json", model.model_json_schema(mode="serialization"))


def capture() -> dict[str, tuple[str, Any]]:
    """Drive the real server over toy worlds; return ``{example: (schema, payload)}``."""
    examples: dict[str, tuple[str, Any]] = {}
    store = FileWorldStore(STORE)
    with tempfile.TemporaryDirectory() as tmp:
        recorders: dict[str, TraceRecorder] = {}

        def sink(sid: str) -> TraceRecorder:
            return recorders.setdefault(sid, TraceRecorder(Path(tmp) / f"{sid}.jsonl"))

        app = create_app(ScorecardService(store, JsonLedger(Path(tmp) / "ledger.json"), traces=sink))
        http = TestClient(app)
        headers = {"X-Arena-Key": KEY}
        dump(SCHEMAS / "openapi.json", app.openapi())

        def post(path: str, body: dict[str, Any] | None = None) -> Any:
            return http.post(path, json=body, headers=headers).json()

        # A small scorecard, to show the open request and response.
        small = {"agent": "my-agent", "tier": "public_train", "n_worlds": 2, "track": "open"}
        opened_small = post("/v1/scorecards", small)
        examples["open_request"] = ("OpenRequest", small)
        examples["open_response"] = ("OpenResponse", normalise(opened_small, opened_small["scorecard_id"]))

        # The full toy tier: play one sequential world step by step, then close.
        opened = post("/v1/scorecards", {"agent": "my-agent", "tier": "public_train", "n_worlds": 20})
        sid = opened["scorecard_id"]
        world = "toy-driver-seq"
        card = next(c for c in opened["cards"] if c["world_id"] == world)
        examples["world_card_sequential"] = ("WorldCard", card)
        fids = [f["feature_id"] for f in card["features"]]
        act = f"/v1/scorecards/{sid}/worlds/{world}/actions"

        def step(name: str, action: dict[str, Any], schema: str = "Observation") -> Any:
            envelope = {"action": action}
            examples[f"action_{name}"] = ("ActionEnvelope", envelope)
            response = post(act, envelope)
            examples[f"observation_after_{name}"] = (schema, normalise(response, sid))
            return response

        step("reset", {"kind": "reset", "request_id": "my-agent-0", "world_id": world})
        step("recruit", {"kind": "recruit", "request_id": "my-agent-1", "count": 6, "stratum": "all"})
        step("assay", {"kind": "assay", "request_id": "my-agent-2", "feature_ids": fids[:3]})
        key = store.answer_key(world)
        truth = key.groups[0].parts[0].true_feature
        other = next(f for f in fids if f != truth)
        step("submit", {"kind": "submit", "request_id": "my-agent-3", "ranking": [truth, other]})

        errors: dict[str, Callable[[], Any]] = {
            "episode_closed": lambda: post(
                act, {"action": {"kind": "assay", "request_id": "my-agent-4", "feature_ids": fids[:1]}}
            ),
            "request_conflict": lambda: post(
                act, {"action": {"kind": "recruit", "request_id": "my-agent-1", "count": 7, "stratum": "all"}}
            ),
            "unknown_feature": lambda: post(
                f"/v1/scorecards/{sid}/worlds/toy-null-a-seq/actions",
                {"action": {"kind": "assay", "request_id": "x-1", "feature_ids": ["not-a-feature"]}},
            ),
            "unknown_stratum": lambda: post(
                f"/v1/scorecards/{sid}/worlds/toy-null-a-seq/actions",
                {"action": {"kind": "recruit", "request_id": "x-2", "count": 5, "stratum": "nope"}},
            ),
            "action_not_available": lambda: post(
                f"/v1/scorecards/{sid}/worlds/toy-driver-full/actions",
                {"action": {"kind": "recruit", "request_id": "x-3", "count": 5, "stratum": "all"}},
            ),
            "invalid_payload": lambda: post(
                f"/v1/scorecards/{sid}/worlds/toy-null-a-seq/actions",
                {"action": {"kind": "submit", "request_id": "x-5", "ranking": [fids[0], fids[0]]}},
            ),
            "unknown_world": lambda: post(
                f"/v1/scorecards/{sid}/worlds/no-such-world/actions",
                {"action": {"kind": "reset", "request_id": "x-4", "world_id": "no-such-world"}},
            ),
        }
        for name, call in errors.items():
            examples[f"error_{name}"] = ("ArenaErrorPayload", normalise(call(), sid))

        closed = post(f"/v1/scorecards/{sid}/close")
        examples["scorecard_public_train"] = ("Scorecard", normalise(closed, sid))
        examples["world_score"] = (
            "WorldScore",
            next(w for w in closed["worlds"] if w["world_id"] == world),
        )
        examples["error_scorecard_closed"] = (
            "ArenaErrorPayload",
            normalise(post(f"/v1/scorecards/{sid}/close"), sid),
        )
        trace = [e.model_dump(mode="json") for e in recorders[sid].events()]
        examples["trace_event"] = ("TraceEvent", normalise(trace[-1], sid))
        examples["answer_key_public_train"] = (
            "AnswerKey",
            store.answer_key("toy-stand-in-full").model_dump(mode="json"),
        )
    return examples


def main() -> int:
    write_schemas()
    for old in EXAMPLES.glob("*.json"):
        old.unlink()
    examples = capture()
    index = {}
    for name, (schema, payload) in sorted(examples.items()):
        if re.search(r"/home/|/Users/", json.dumps(payload)):
            raise SystemExit(f"example {name} contains an absolute path")
        dump(EXAMPLES / f"{name}.json", payload)
        index[f"{name}.json"] = f"{schema}.schema.json"
    dump(EXAMPLES / "index.json", index)
    print(f"wrote {len(MODELS)} schemas, openapi.json and {len(examples)} examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
