"""Offline scripted tool-loop fixture for contract regression tests.

The supported model runner is ``onc-agi play --agent llm``. This fixture
exercises tool dispatch and scoring without contacting a model provider.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from onc_agi.adapters.cli import fixture_store
from onc_agi.core.errors import ArenaError
from onc_agi.core.schema import Assay, Recruit, Reset, Submit, WorldCard
from onc_agi.infra.bundles import FileWorldStore
from onc_agi.services import scoring
from onc_agi.services.engine import Episode, EpisodeView
from onc_agi.services.kit import analysis_input
from pipeline_agent import bonferroni_t  # type: ignore[import-not-found]

TOOLS: list[dict[str, Any]] = [
    {
        "name": "recruit",
        "description": "Recruit patients from a stratum's queue; reveals their outcome. Sequential mode only.",
        "input_schema": {
            "type": "object",
            "properties": {"count": {"type": "integer", "minimum": 1}, "stratum": {"type": "string"}},
            "required": ["count"],
        },
    },
    {
        "name": "assay",
        "description": "Measure features on every recruited patient not yet measured. Sequential mode only.",
        "input_schema": {
            "type": "object",
            "properties": {"feature_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1}},
            "required": ["feature_ids"],
        },
    },
    {
        "name": "analyse",
        "description": "Run a standard analysis on the revealed data and return a candidate ranking.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "submit",
        "description": "Submit an ordered list of driver feature ids, most likely first, or [] for none. Ends the world.",
        "input_schema": {
            "type": "object",
            "properties": {"ranking": {"type": "array", "items": {"type": "string"}}},
            "required": ["ranking"],
        },
    },
]


def task_card(card: WorldCard) -> str:
    """The premise and the world card as text (the standard track shows the same facts)."""
    lines = [card.premise, f"World {card.world_id}: {card.n_pool} patients, mode {card.mode.value}."]
    if card.mode.value == "sequential":
        lines.append(
            f"Budget {card.budget:.0f} USD; recruiting costs {card.prices.recruit_per_patient:.0f} per patient."
        )
    lines += [
        f"- {f.feature_id} ({f.data_type}, {f.timing.value}, {f.assay_price:.0f} USD)" for f in card.features
    ]
    lines.append("Listing any post_outcome feature scores the world 0.")
    return "\n".join(lines)


class ToolPolicy(Protocol):
    def next_call(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]: ...


class ScriptedPolicy:
    """Offline stand-in for a model: recruit everything, assay everything, analyse, submit."""

    def __init__(self) -> None:
        self.turn = 0

    def next_call(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        self.turn += 1
        if "mode full_access" in messages[0]["content"]:
            plan = ["analyse", "submit"]
        else:
            plan = ["recruit", "assay", "analyse", "submit"]
        name = plan[min(self.turn, len(plan)) - 1]
        if name == "recruit":
            return name, {"count": 100_000, "stratum": "all"}
        if name == "assay":
            return name, {"feature_ids": ["*"]}
        if name == "submit":
            return name, {"ranking": json.loads(messages[-1]["content"]).get("ranking", [])}
        return name, {}


def play(policy: ToolPolicy, episode: Episode, max_turns: int = 30) -> tuple[str, ...]:
    card = episode.world.card
    view: EpisodeView = episode.apply(Reset(request_id="llm-reset", world_id=card.world_id))
    messages = [{"role": "user", "content": task_card(card)}]
    for turn in range(max_turns):
        name, args = policy.next_call(messages)
        rid = f"llm-{turn}"
        try:
            if name == "recruit":
                for stratum in card.strata if args.get("stratum") in (None, "all") else [args["stratum"]]:
                    view = episode.apply(
                        Recruit(
                            request_id=f"{rid}-{stratum}",
                            count=min(args["count"], card.n_pool),
                            stratum=stratum,
                        )
                    )
                result: dict[str, Any] = {"revealed_patients": len(view.rows), "spent": view.spent}
            elif name == "assay":
                fids = card.feature_ids() if args["feature_ids"] == ["*"] else tuple(args["feature_ids"])
                view = episode.apply(Assay(request_id=rid, feature_ids=fids))
                result = {"measured": sum(view.measured), "spent": view.spent}
            elif name == "analyse":
                result = {"ranking": bonferroni_t(analysis_input(card, view))}
            elif name == "submit":
                episode.apply(Submit(request_id=rid, ranking=tuple(args["ranking"])))
                return tuple(args["ranking"])
            else:
                result = {"error": f"unknown tool {name}"}
        except ArenaError as exc:  # typed errors go back to the model as tool results
            result = {"error": exc.code.value, "message": exc.message}
        messages.append({"role": "tool", "content": json.dumps(result)})
    episode.apply(Submit(request_id="llm-timeout", ranking=()))
    return ()


def main() -> list[float]:
    store = FileWorldStore(fixture_store())
    finds = []
    for world_id in ("toy-driver-full", "toy-driver-seq", "toy-null-a-seq"):
        episode = Episode(store.world(world_id))
        ranking = play(ScriptedPolicy(), episode)
        score = scoring.score_world(
            ranking, store.answer_key(world_id), spent=episode.spent, sequential=world_id.endswith("-seq")
        )
        print(f"{world_id:18s} submitted={list(ranking)} find={score.find:.2f} restrained={score.restrained}")
        finds.append(score.find)
    return finds


if __name__ == "__main__":
    main()
