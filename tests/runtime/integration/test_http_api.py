"""Versioned HTTP interface and client: the wire contract custom harnesses depend on."""

from __future__ import annotations

from pathlib import Path

import pytest
from arena_factories import InMemoryStore, fid, planted_world
from fastapi.testclient import TestClient
from onc_agi.adapters.agents import make_agent
from onc_agi.adapters.client import ArenaClient, view_from_observation
from onc_agi.adapters.conformance import run_conformance
from onc_agi.core.errors import ArenaError
from onc_agi.core.schema import ErrorCode, Mode, Recruit, Reset, Submit, Tier, TraceEvent
from onc_agi.infra.ledger import JsonLedger
from onc_agi.services.engine import Episode
from onc_agi.services.scorecards import ScorecardService

KEY = "test-key-0001"


class ListSink:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def record(self, event: TraceEvent) -> None:
        self.events.append(event)


def mixed_store() -> InMemoryStore:
    store = InMemoryStore()
    for tier in Tier:
        for i in range(8):
            mode = Mode.SEQUENTIAL if i % 2 else Mode.FULL_ACCESS
            wid = f"{tier.value.replace('_', '-')}-{i:02d}"
            store.add(*planted_world(wid, signal=i % 4 != 3, seed=i, n_pool=60, tier=tier, mode=mode))
    return store


@pytest.fixture
def served(tmp_path: Path) -> tuple[TestClient, InMemoryStore, dict[str, ListSink]]:
    store = mixed_store()
    sinks: dict[str, ListSink] = {}
    from onc_agi.adapters.http import create_app

    service = ScorecardService(
        store, JsonLedger(tmp_path / "ledger.json"), traces=lambda s: sinks.setdefault(s, ListSink())
    )
    return TestClient(create_app(service)), store, sinks


def open_card(http: TestClient, tier: str = "public_train", n: int = 2) -> dict:  # type: ignore[type-arg]
    r = http.post(
        "/v1/scorecards", json={"agent": "a", "tier": tier, "n_worlds": n}, headers={"X-Arena-Key": KEY}
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_health_reports_the_interface_version(served) -> None:  # type: ignore[no-untyped-def]
    http, _, _ = served
    assert http.get("/v1/health").json() == {"status": "ok", "interface_version": "1.0"}


def test_opening_requires_an_identifying_key(served) -> None:  # type: ignore[no-untyped-def]
    http, _, _ = served
    body = {"agent": "a", "tier": "public_train", "n_worlds": 1}
    for headers in ({}, {"X-Arena-Key": "short"}):
        r = http.post("/v1/scorecards", json=body, headers=headers)
        assert (r.status_code, r.json()["code"]) == (400, "invalid_payload")


def test_unknown_request_fields_are_refused(served) -> None:  # type: ignore[no-untyped-def]
    http, _, _ = served
    for body in (
        {"agent": "a", "tier": "public_train", "n_worlds": 1, "peek": True},
        {"agent": "a", "tier": "public_train", "n_worlds": 0},
    ):
        r = http.post("/v1/scorecards", json=body, headers={"X-Arena-Key": KEY})
        assert (r.status_code, r.json()["code"]) == (400, "invalid_payload")


def test_a_session_resets_acts_resumes_and_closes(served) -> None:  # type: ignore[no-untyped-def]
    http, _, sinks = served
    opened = open_card(http)
    sid, wid = opened["scorecard_id"], opened["cards"][0]["world_id"]
    base = f"/v1/scorecards/{sid}/worlds/{wid}"
    first = http.post(
        f"{base}/actions", json={"action": {"kind": "reset", "request_id": "r", "world_id": wid}}
    )
    assert first.status_code == 200
    assert http.get(base).json() == first.json()
    done = http.post(
        f"{base}/actions", json={"action": {"kind": "submit", "request_id": "s", "ranking": [fid(0)]}}
    )
    assert done.json()["status"] == "submitted"
    card = http.post(f"/v1/scorecards/{sid}/close").json()
    assert card["n_worlds"] == 2 and len(card["worlds"]) == 2
    assert [e.action_kind for e in sinks[sid].events] == ["reset", "submit"]


@pytest.mark.parametrize(
    ("action", "status", "code"),
    [
        ({"kind": "submit", "request_id": "s", "ranking": ["ghost"]}, 422, "unknown_feature"),
        ({"kind": "recruit", "request_id": "x", "count": 1}, 409, "action_not_available"),
        ({"kind": "reset", "request_id": "r", "world_id": "w-elsewhere"}, 404, "unknown_world"),
    ],
)
def test_domain_errors_map_to_stable_status_codes(served, action, status, code) -> None:  # type: ignore[no-untyped-def]
    http, _store, _ = served
    opened = open_card(http)
    sid = opened["scorecard_id"]
    wid = next(c["world_id"] for c in opened["cards"] if c["mode"] == "full_access")
    r = http.post(f"/v1/scorecards/{sid}/worlds/{wid}/actions", json={"action": action})
    assert (r.status_code, r.json()["code"]) == (status, code)


def test_malformed_actions_are_refused_before_reaching_the_engine(served) -> None:  # type: ignore[no-untyped-def]
    http, _, _ = served
    opened = open_card(http)
    sid, wid = opened["scorecard_id"], opened["cards"][0]["world_id"]
    r = http.post(
        f"/v1/scorecards/{sid}/worlds/{wid}/actions",
        json={"action": {"kind": "submit", "request_id": "s", "ranking": [fid(0), fid(0)]}},
    )
    assert (r.status_code, r.json()["code"]) == (400, "invalid_payload")
    client = ArenaClient("http://testserver", KEY, client=http)
    with pytest.raises(ArenaError) as err:
        client.act(sid, wid, Submit.model_construct(kind="submit", request_id="s", ranking=(fid(0), fid(0))))
    assert err.value.code is ErrorCode.INVALID_PAYLOAD


def test_unknown_and_closed_scorecards_are_refused(served) -> None:  # type: ignore[no-untyped-def]
    http, _, _ = served
    assert http.post("/v1/scorecards/sc-nope/close").json()["code"] == "invalid_payload"
    sid = open_card(http)["scorecard_id"]
    http.post(f"/v1/scorecards/{sid}/close")
    r = http.post(f"/v1/scorecards/{sid}/close")
    assert (r.status_code, r.json()["code"]) == (409, "scorecard_closed")


@pytest.mark.parametrize("tier", ["public_eval", "private"])
def test_eval_responses_never_carry_answers_or_per_world_results(served, tier: str) -> None:  # type: ignore[no-untyped-def]
    http, _store, _ = served
    opened = open_card(http, tier=tier, n=3)
    sid = opened["scorecard_id"]
    bodies = [str(opened)]
    for card in opened["cards"]:
        wid = card["world_id"]
        r = http.post(
            f"/v1/scorecards/{sid}/worlds/{wid}/actions",
            json={"action": {"kind": "submit", "request_id": "s", "ranking": [fid(0)]}},
        )
        bodies.append(r.text)
    closed = http.post(f"/v1/scorecards/{sid}/close")
    bodies.append(closed.text)
    assert closed.json()["worlds"] == []
    text = "\n".join(bodies)
    for forbidden in (
        "equivalence_set",
        "true_feature",
        "reject_set",
        "answer",
        "detection_threshold",
        "reference_cost",
    ):
        assert forbidden not in text


def test_the_client_plays_any_agent_and_matches_in_process_scoring(served) -> None:  # type: ignore[no-untyped-def]
    http, store, _ = served
    client = ArenaClient("http://testserver", KEY, client=http)
    sid, cards = client.open("client-test", Tier.PUBLIC_TRAIN, 4)
    for card in cards:
        client.play(make_agent("univariate_bh"), sid, card)
    remote = client.close(sid)
    from onc_agi.services import scoring
    from onc_agi.services.kit import run_episode

    for card, world_score in zip(cards, remote.worlds, strict=True):
        local = run_episode(make_agent("univariate_bh"), Episode(store.world(card.world_id)))
        expected = scoring.score_world(
            local.ranking,
            store.answer_key(card.world_id),
            spent=local.spent,
            sequential=card.mode is Mode.SEQUENTIAL,
        )
        assert world_score.find == pytest.approx(expected.find)
        assert world_score.restrained == expected.restrained


def test_the_client_raises_typed_errors(served) -> None:  # type: ignore[no-untyped-def]
    http, _, _ = served
    client = ArenaClient("http://testserver", KEY, client=http)
    sid, cards = client.open("client-test", Tier.PUBLIC_TRAIN, 2)
    card = next(c for c in cards if c.mode is Mode.SEQUENTIAL)
    client.act(sid, card.world_id, Recruit(request_id="a", count=3, stratum="all"))
    with pytest.raises(ArenaError) as err:
        client.act(sid, card.world_id, Recruit(request_id="a", count=4, stratum="all"))
    assert err.value.code is ErrorCode.REQUEST_CONFLICT


def test_client_views_reconstruct_the_engine_view(served) -> None:  # type: ignore[no-untyped-def]
    http, store, _ = served
    client = ArenaClient("http://testserver", KEY, client=http)
    sid, cards = client.open("client-test", Tier.PUBLIC_TRAIN, 2)
    card = next(c for c in cards if c.mode is Mode.FULL_ACCESS)
    obs = client.act(sid, card.world_id, Reset(request_id="r", world_id=card.world_id))
    view = view_from_observation(card, obs)
    world = store.world(card.world_id)
    assert view.x.shape == world.x.shape and all(view.measured)
    assert (view.outcome == world.y).all()
    final = client.act(sid, card.world_id, Submit(request_id="s", ranking=()))
    assert view_from_observation(card, final).available == ()


def test_the_conformance_suite_passes_against_the_reference_server(served) -> None:  # type: ignore[no-untyped-def]
    http, store, sinks = served
    client = ArenaClient("http://testserver", KEY, client=http)
    report = run_conformance(client, store, server_traces=lambda sid: sinks[sid].events)
    assert report.ok, report.details
    assert any(name.startswith("server_traces_replay_exactly") for name in report.checks)
    assert {
        "idempotent_retry_not_charged_twice",
        "conflicting_request_refused",
        "closed_episode_refuses_actions",
    } <= set(report.checks)
