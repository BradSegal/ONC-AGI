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
H = {"X-Arena-Key": KEY}


@pytest.mark.parametrize("tier", ["public_eval", "private"])
def test_hosted_default_minimum_has_no_http_caller_override(tmp_path: Path, tier: str) -> None:
    from onc_agi.adapters.http import create_app

    ledger = JsonLedger(tmp_path / "ledger.json")
    service = ScorecardService(mixed_store(), ledger)
    with TestClient(create_app(service), headers={"X-Arena-Key": KEY}) as http:
        body = {"agent": "a", "tier": tier, "n_worlds": 1}
        response = http.post("/v1/scorecards", json=body)
        assert response.status_code == 400 and response.json()["code"] == "invalid_payload"
        assert "40" in response.json()["message"]
        assert ledger.used(Tier(tier)) == set()
        response = http.post("/v1/scorecards", json=body | {"min_eval_worlds": 1})
        assert response.status_code == 400 and response.json()["code"] == "invalid_payload"
        response = http.post("/v1/scorecards", json=body | {"tier": "public_train"})
        assert response.status_code == 200


def test_http_lifespan_releases_archive_for_restart(tmp_path: Path) -> None:
    from onc_agi.adapters.http import create_app
    from onc_agi.infra.archive import FileScorecardArchive

    store = mixed_store()
    service = ScorecardService(
        store, JsonLedger(tmp_path / "ledger.json"), archive=FileScorecardArchive(tmp_path)
    )
    with TestClient(create_app(service), headers={"X-Arena-Key": KEY}) as http:
        sid = open_card(http, n=1)["scorecard_id"]
    with pytest.raises(RuntimeError, match="stopped"):
        service.close(sid)
    restored = ScorecardService(store, service.ledger, archive=FileScorecardArchive(tmp_path))
    assert restored.close(sid, api_key=KEY).n_worlds == 1
    restored.shutdown()


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
        store,
        JsonLedger(tmp_path / "ledger.json"),
        min_eval_worlds=1,
        traces=lambda s: sinks.setdefault(s, ListSink()),
    )
    # clients send their key on every call (as ArenaClient does); protected routes require it
    return TestClient(create_app(service), headers={"X-Arena-Key": KEY}), store, sinks


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
    served_client, _, _ = served
    http = TestClient(served_client.app)  # no default key: the request carries only what is given here
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
        f"{base}/actions", json={"action": {"kind": "reset", "request_id": "r", "world_id": wid}}, headers=H
    )
    assert first.status_code == 200
    assert http.get(base, headers=H).json() == first.json()
    done = http.post(
        f"{base}/actions",
        json={"action": {"kind": "submit", "request_id": "s", "ranking": [fid(0)]}},
        headers=H,
    )
    assert done.json()["status"] == "submitted"
    card = http.post(f"/v1/scorecards/{sid}/close", headers=H).json()
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
    r = http.post(f"/v1/scorecards/{sid}/worlds/{wid}/actions", json={"action": action}, headers=H)
    assert (r.status_code, r.json()["code"]) == (status, code)


def test_malformed_actions_are_refused_before_reaching_the_engine(served) -> None:  # type: ignore[no-untyped-def]
    http, _, _ = served
    opened = open_card(http)
    sid, wid = opened["scorecard_id"], opened["cards"][0]["world_id"]
    r = http.post(
        f"/v1/scorecards/{sid}/worlds/{wid}/actions",
        json={"action": {"kind": "submit", "request_id": "s", "ranking": [fid(0), fid(0)]}},
        headers=H,
    )
    assert (r.status_code, r.json()["code"]) == (400, "invalid_payload")
    client = ArenaClient("http://testserver", KEY, client=http)
    with pytest.raises(ArenaError) as err:
        client.act(sid, wid, Submit.model_construct(kind="submit", request_id="s", ranking=(fid(0), fid(0))))
    assert err.value.code is ErrorCode.INVALID_PAYLOAD


def test_unknown_and_closed_scorecards_are_refused(served) -> None:  # type: ignore[no-untyped-def]
    http, _, _ = served
    assert http.post("/v1/scorecards/sc-nope/close", headers=H).json()["code"] == "invalid_payload"
    sid = open_card(http)["scorecard_id"]
    http.post(f"/v1/scorecards/{sid}/close", headers=H)
    r = http.post(f"/v1/scorecards/{sid}/close", headers=H)
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
            headers=H,
        )
        bodies.append(r.text)
    closed = http.post(f"/v1/scorecards/{sid}/close", headers=H)
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


# ---------------------------------------------------------------- ownership, records, restart


def test_every_scorecard_endpoint_requires_a_key(served) -> None:  # type: ignore[no-untyped-def]
    http, _, _ = served
    bare = TestClient(http.app)  # sends no key at all (the fixture's client sends one on every call)
    opened = open_card(http)
    sid, wid = opened["scorecard_id"], opened["cards"][0]["world_id"]
    reset = {"action": {"kind": "reset", "request_id": "r", "world_id": wid}}
    calls = [
        lambda c, h: c.post(f"/v1/scorecards/{sid}/worlds/{wid}/actions", json=reset, headers=h),
        lambda c, h: c.get(f"/v1/scorecards/{sid}/worlds/{wid}", headers=h),
        lambda c, h: c.post(f"/v1/scorecards/{sid}/close", headers=h),
        lambda c, h: c.get(f"/v1/scorecards/{sid}", headers=h),
    ]
    for call in calls:
        for client, headers in ((bare, {}), (bare, {"X-Arena-Key": "short"})):
            r = call(client, headers)
            assert (r.status_code, r.json()["code"]) == (400, "invalid_payload")
    assert calls[0](http, H).status_code == 200  # nothing above touched the scorecard


def test_a_foreign_key_gets_the_unknown_scorecard_response(served) -> None:  # type: ignore[no-untyped-def]
    http, _, _ = served
    opened = open_card(http)
    sid, wid = opened["scorecard_id"], opened["cards"][0]["world_id"]
    thief = {"X-Arena-Key": "another-key-02"}
    reset = {"action": {"kind": "reset", "request_id": "r", "world_id": wid}}
    for path_of, method, body in (
        (lambda s: f"/v1/scorecards/{s}/worlds/{wid}/actions", "POST", reset),
        (lambda s: f"/v1/scorecards/{s}/worlds/{wid}", "GET", None),
        (lambda s: f"/v1/scorecards/{s}/close", "POST", None),
        (lambda s: f"/v1/scorecards/{s}", "GET", None),
    ):
        mine = http.request(method, path_of(sid), json=body, headers=thief)
        ghost = http.request(method, path_of("sc-ghost"), json=body, headers=thief)
        assert mine.status_code == ghost.status_code == 400
        assert mine.json()["message"].replace(sid, "?") == ghost.json()["message"].replace("sc-ghost", "?")
    assert http.post(f"/v1/scorecards/{sid}/close", headers=H).status_code == 200


def test_the_closed_scorecard_is_retrievable_by_its_owner(served) -> None:  # type: ignore[no-untyped-def]
    http, _, _ = served
    client = ArenaClient("http://testserver", KEY, client=http)
    sid, _cards = client.open("record", Tier.PUBLIC_EVAL, 3)
    with pytest.raises(ArenaError) as err:
        client.scorecard(sid)
    assert err.value.code is ErrorCode.ACTION_NOT_AVAILABLE
    assert http.get(f"/v1/scorecards/{sid}", headers=H).status_code == 409
    closed = client.close(sid)
    assert client.scorecard(sid) == closed and closed.worlds == ()


def test_scorecards_survive_a_server_restart(tmp_path: Path) -> None:
    from onc_agi.adapters.http import create_app
    from onc_agi.infra.archive import FileScorecardArchive

    store = mixed_store()

    services: list[ScorecardService] = []

    def boot() -> ArenaClient:
        if services:
            services[-1].shutdown()  # one live service owns an archive; the restart replaces it
        services.append(
            ScorecardService(
                store,
                JsonLedger(tmp_path / "ledger.json"),
                archive=FileScorecardArchive(tmp_path / "archive"),
            )
        )
        return ArenaClient("http://testserver", KEY, client=TestClient(create_app(services[-1])))

    first = boot()
    sid, cards = first.open("restart", Tier.PUBLIC_TRAIN, 2)
    seq = next(c for c in cards if c.mode is Mode.SEQUENTIAL)
    first.act(sid, seq.world_id, Reset(request_id="r", world_id=seq.world_id))
    before = first.act(sid, seq.world_id, Recruit(request_id="r1", count=4, stratum="all"))
    done, _ = first.open("done", Tier.PUBLIC_TRAIN, 1)
    closed = first.close(done)

    second = boot()
    assert second.state(sid, seq.world_id) == before
    assert second.scorecard(done) == closed
    second.act(sid, seq.world_id, Submit(request_id="s", ranking=()))
    assert second.close(sid).n_worlds == 2


def test_the_conformance_suite_checks_ownership_records_and_restart(tmp_path: Path) -> None:
    from onc_agi.adapters.http import create_app
    from onc_agi.infra.archive import FileScorecardArchive

    store = mixed_store()

    services: list[ScorecardService] = []

    def boot() -> tuple[ArenaClient, FileScorecardArchive]:
        if services:
            services[-1].shutdown()
        archive = FileScorecardArchive(tmp_path / "archive")
        services.append(ScorecardService(store, JsonLedger(tmp_path / "ledger.json"), archive=archive))
        return ArenaClient("http://testserver", KEY, client=TestClient(create_app(services[-1]))), archive

    client, archive = boot()
    report = run_conformance(client, store, server_traces=archive.events, restart=lambda: boot()[0])
    assert report.ok, (report.checks, report.details)
    assert {
        "foreign_key_refused_like_unknown_id[act]",
        "foreign_key_refused_like_unknown_id[state]",
        "foreign_key_refused_like_unknown_id[close]",
        "foreign_key_refused_like_unknown_id[scorecard]",
        "closed_scorecard_retrievable[resume]",
        "restart_restores_open_scorecard",
        "restart_scorecard_closes_and_is_retrievable",
    } <= set(report.checks)


def test_only_the_opening_key_can_act_read_or_close_a_scorecard(served) -> None:  # type: ignore[no-untyped-def]
    """any caller who learned a scorecard id could act on or close it."""
    http, _, _ = served
    opened = open_card(http)
    sid, wid = opened["scorecard_id"], opened["cards"][0]["world_id"]
    base = f"/v1/scorecards/{sid}/worlds/{wid}"
    reset = {"action": {"kind": "reset", "request_id": "r", "world_id": wid}}
    other = {"X-Arena-Key": "someone-else-key"}
    for response in (
        http.post(f"{base}/actions", json=reset, headers=other),
        http.get(base, headers=other),
        http.post(f"/v1/scorecards/{sid}/close", headers=other),
    ):
        assert response.status_code == 400
        assert response.json() == {"code": "invalid_payload", "message": f"unknown scorecard {sid!r}"}
    assert http.post(f"{base}/actions", json=reset).status_code == 200  # the owner still can
    assert http.post(f"/v1/scorecards/{sid}/close").status_code == 200


# ---------------------------------------------------------------- world listing and selection (ARC parity)


def test_public_train_worlds_are_listed_and_eval_tiers_never(served) -> None:  # type: ignore[no-untyped-def]
    http, store, _ = served
    r = http.get("/v1/worlds", params={"tier": "public_train"})
    assert r.status_code == 200
    assert [c["world_id"] for c in r.json()["cards"]] == list(store.world_ids(Tier.PUBLIC_TRAIN))
    for tier in ("public_eval", "private"):
        r = http.get("/v1/worlds", params={"tier": tier})
        assert (r.status_code, r.json()["code"]) == (400, "invalid_payload")


def test_named_public_train_worlds_open_in_the_given_order(served) -> None:  # type: ignore[no-untyped-def]
    http, _, _ = served
    wanted = ["public-train-05", "public-train-02"]
    r = http.post("/v1/scorecards", json={"agent": "a", "tier": "public_train", "world_ids": wanted})
    assert r.status_code == 200 and [c["world_id"] for c in r.json()["cards"]] == wanted


@pytest.mark.parametrize(
    "body",
    [
        {"agent": "a", "tier": "public_eval", "world_ids": ["public-eval-00"]},  # eval draws stay fresh
        {"agent": "a", "tier": "public_train", "world_ids": ["no-such-world"]},
        {"agent": "a", "tier": "public_train", "world_ids": ["public-train-01", "public-train-01"]},
        {"agent": "a", "tier": "public_train", "n_worlds": 2, "world_ids": ["public-train-01"]},
        {"agent": "a", "tier": "public_train"},
    ],
)
def test_world_selection_is_refused_outside_its_contract(served, body) -> None:  # type: ignore[no-untyped-def]
    http, _, _ = served
    r = http.post("/v1/scorecards", json=body)
    assert r.status_code in (400, 404) and r.json()["code"] in ("invalid_payload", "unknown_world")


def test_a_mode_filter_draws_only_that_mode(served) -> None:  # type: ignore[no-untyped-def]
    http, _, _ = served
    r = http.post(
        "/v1/scorecards", json={"agent": "a", "tier": "public_train", "n_worlds": 3, "mode": "sequential"}
    )
    assert {c["mode"] for c in r.json()["cards"]} == {"sequential"}


def test_large_responses_are_gzip_compressed(served) -> None:  # type: ignore[no-untyped-def]
    http, _, _ = served
    r = http.get("/v1/worlds", params={"tier": "public_train"}, headers={"Accept-Encoding": "gzip"})
    assert r.headers.get("content-encoding") == "gzip"
