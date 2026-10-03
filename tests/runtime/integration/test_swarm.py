"""The swarm runner plays one scorecard concurrently, in-process or over HTTP, with the same results."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import ClassVar

import pytest
from arena_factories import InMemoryStore, standard_world
from fastapi.testclient import TestClient
from onc_agi.adapters.agents import make_agent
from onc_agi.adapters.client import ArenaClient
from onc_agi.adapters.http import create_app
from onc_agi.adapters.swarm import UNPLAYED, Arena, LocalArena, run_swarm
from onc_agi.core.schema import Action, Mode, Submit, Tier, WorldCard
from onc_agi.infra.ledger import JsonLedger
from onc_agi.infra.recordings import RecordingWriter, read_recording
from onc_agi.services.engine import EpisodeView
from onc_agi.services.kit import Agent, PipelineAgent, Usage
from onc_agi.services.scorecards import ScorecardService


@pytest.fixture
def mixed_store() -> InMemoryStore:
    """Eight public-train worlds: four full-access, four sequential (every third null)."""
    pairs = [standard_world(i) for i in range(4)]
    pairs += [standard_world(i, mode=Mode.SEQUENTIAL) for i in range(4, 8)]
    return InMemoryStore.of(*pairs)


def _http(store: InMemoryStore, tmp_path: Path) -> ArenaClient:
    app = create_app(ScorecardService(store, JsonLedger(tmp_path / "ledger.json")))
    return ArenaClient("http://testserver", "swarm-test-key", client=TestClient(app))


def _first(data) -> list[str]:  # type: ignore[no-untyped-def]
    return [data.feature_ids[0]]


@pytest.mark.parametrize("agent", ["univariate_bh", "seq_univariate_bh", "random"])
def test_local_and_http_give_the_same_per_world_scores(
    mixed_store: InMemoryStore, tmp_path: Path, agent: str
) -> None:
    results = []
    arenas: list[Arena] = [LocalArena.over_store(mixed_store), _http(mixed_store, tmp_path)]
    for arena in arenas:
        result = run_swarm(
            arena, lambda: make_agent(agent), agent_name=agent, tier=Tier.PUBLIC_TRAIN, n_worlds=8, workers=3
        )
        assert result.errors == {} and result.unplayed == () and not result.budget_reached
        assert all(r.submitted for r in result.runs)
        results.append(result)
    local, remote = (r.scorecard for r in results)
    assert local.n_worlds == remote.n_worlds == 8
    assert [w.model_dump() for w in local.worlds] == [w.model_dump() for w in remote.worlds]
    assert [r.ranking for r in results[0].runs] == [r.ranking for r in results[1].runs]
    assert local.find == remote.find and local.discovery_score == remote.discovery_score


def test_the_swarm_matches_the_kit_evaluator(store: InMemoryStore) -> None:
    from onc_agi.services.kit import evaluate

    reference, _ = evaluate(make_agent("univariate_bh"), store, Tier.PUBLIC_TRAIN)
    result = run_swarm(
        LocalArena.over_store(store),
        lambda: make_agent("univariate_bh"),
        agent_name="u",
        tier=Tier.PUBLIC_TRAIN,
        n_worlds=6,
    )
    assert [w.find for w in result.scorecard.worlds] == [w.find for w in reference.worlds]


def test_world_ids_select_the_scorecard_worlds(mixed_store: InMemoryStore, tmp_path: Path) -> None:
    chosen = ("w-05", "w-01")
    for arena in (LocalArena.over_store(mixed_store), _http(mixed_store, tmp_path)):
        result = run_swarm(
            arena,
            lambda: make_agent("univariate_bh"),
            agent_name="u",
            tier=Tier.PUBLIC_TRAIN,
            world_ids=chosen,
        )
        assert [r.world_id for r in result.runs] == list(chosen)
        assert {w.world_id for w in result.scorecard.worlds} == set(chosen)


def test_mode_restricts_the_draw(mixed_store: InMemoryStore) -> None:
    result = run_swarm(
        LocalArena.over_store(mixed_store),
        lambda: make_agent("random"),
        agent_name="r",
        tier=Tier.PUBLIC_TRAIN,
        n_worlds=4,
        mode=Mode.SEQUENTIAL,
    )
    assert {r.world_id for r in result.runs} == {"w-04", "w-05", "w-06", "w-07"}


class Fragile(PipelineAgent):
    """Raises on one world; records every close."""

    closed: ClassVar[list[str]] = []
    lock = threading.Lock()

    def __init__(self) -> None:
        super().__init__("fragile", _first)
        self.world: str | None = None

    def choose_action(self, card: WorldCard, view: EpisodeView) -> Action:
        self.world = card.world_id
        if card.world_id == "w-02":
            raise ValueError("analysis blew up")
        return super().choose_action(card, view)

    def close(self) -> None:
        with Fragile.lock:
            Fragile.closed.append(str(self.world))


def test_an_agent_exception_is_recorded_and_the_swarm_continues(store: InMemoryStore, tmp_path: Path) -> None:
    Fragile.closed = []
    with RecordingWriter(tmp_path / "rec.jsonl") as recorder:
        result = run_swarm(
            LocalArena.over_store(store),
            Fragile,
            agent_name="fragile",
            tier=Tier.PUBLIC_TRAIN,
            n_worlds=6,
            workers=2,
            recorder=recorder,
        )
    assert set(result.errors) == {"w-02"} and "ValueError: analysis blew up" in result.errors["w-02"]
    assert sum(r.submitted for r in result.runs) == 5
    assert sorted(Fragile.closed) == [f"w-{i:02d}" for i in range(6)]  # close() ran for every world
    failed = next(w for w in result.scorecard.worlds if w.world_id == "w-02")
    assert failed.listed == 0  # unsubmitted: scored as an empty submission
    recording = read_recording(tmp_path / "rec.jsonl")
    assert recording.header is not None and recording.header.scorecard_id == result.scorecard.scorecard_id
    notes = [e for e in recording.events if e.world_id == "w-02" and e.kind == "note"]
    assert notes and notes[0].error
    assert {r.world_id: r.error for r in recording.runs}["w-02"] == result.errors["w-02"]


class Unknown(Agent):
    name = "unknown"

    def choose_action(self, card: WorldCard, view: EpisodeView) -> Action:
        return Submit(request_id=self.request_id(), ranking=("not-a-feature",))


def test_a_refused_action_ends_the_world_by_default(store: InMemoryStore, tmp_path: Path) -> None:
    with RecordingWriter(tmp_path / "rec.jsonl") as recorder:
        result = run_swarm(
            LocalArena.over_store(store),
            Unknown,
            agent_name="u",
            tier=Tier.PUBLIC_TRAIN,
            n_worlds=2,
            recorder=recorder,
        )
    assert all("unknown_feature" in e for e in result.errors.values()) and len(result.errors) == 2
    refused = [e for e in read_recording(tmp_path / "rec.jsonl").events if e.kind == "action" and e.error]
    assert len(refused) == 2 and "refused: unknown_feature" in refused[0].content


class Paid(PipelineAgent):
    def __init__(self) -> None:
        super().__init__("paid", _first)

    def usage(self) -> Usage:
        return Usage(tokens=100, cost_usd=1.0, model="m-1")


def test_the_budget_stops_new_worlds_and_reports_the_rest(store: InMemoryStore) -> None:
    result = run_swarm(
        LocalArena.over_store(store),
        Paid,
        agent_name="paid",
        tier=Tier.PUBLIC_TRAIN,
        n_worlds=6,
        workers=1,
        budget_usd=1.5,
        model="m-1",
        harness="h-1",
    )
    assert result.budget_reached
    assert result.unplayed == ("w-02", "w-03", "w-04", "w-05")  # 2 USD spent > 1.5 after two worlds
    assert [r.submitted for r in result.runs] == [True, True, False, False, False, False]
    assert all(r.error == UNPLAYED for r in result.runs[2:]) and result.errors == {}
    card = result.scorecard
    assert card.n_worlds == 6 and card.cost_usd == pytest.approx(2.0) and card.tokens == 200
    assert card.model == "m-1" and card.harness == "h-1"
    assert all(w.listed == 0 for w in card.worlds[2:])


def test_over_http_the_labels_annotate_the_returned_scorecard(store: InMemoryStore, tmp_path: Path) -> None:
    result = run_swarm(
        _http(store, tmp_path),
        Paid,
        agent_name="paid",
        tier=Tier.PUBLIC_TRAIN,
        n_worlds=2,
        model="m",
        harness="h",
    )
    assert (result.scorecard.model, result.scorecard.harness, result.scorecard.tokens) == ("m", "h", 200)


def test_invalid_swarm_arguments_fail_fast(store: InMemoryStore) -> None:
    arena = LocalArena.over_store(store)
    with pytest.raises(ValueError, match="workers"):
        run_swarm(arena, Paid, agent_name="p", tier=Tier.PUBLIC_TRAIN, n_worlds=1, workers=0)
    with pytest.raises(ValueError, match="budget"):
        run_swarm(arena, Paid, agent_name="p", tier=Tier.PUBLIC_TRAIN, n_worlds=1, budget_usd=-1)
