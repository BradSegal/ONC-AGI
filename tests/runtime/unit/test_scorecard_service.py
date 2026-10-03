"""Scorecard service: fresh never-reused eval draws, caps, exposure and server-side traces."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from arena_factories import InMemoryStore, fid, planted_world
from onc_agi.core.errors import ArenaError
from onc_agi.core.schema import ErrorCode, Recruit, Reset, Submit, Tier, TraceEvent
from onc_agi.services.scorecards import PRIVATE_TOTAL_CAP, PUBLIC_EVAL_DAILY_CAP, ScorecardService


class DictLedger:
    def __init__(self) -> None:
        self.used_ids: dict[Tier, list[str]] = {}
        self.opened: dict[tuple[str, Tier], list[str]] = {}

    def used(self, tier: Tier) -> set[str]:
        return set(self.used_ids.get(tier, []))

    def mark_used(self, tier: Tier, world_ids: list[str]) -> None:
        self.used_ids.setdefault(tier, []).extend(world_ids)

    def openings(self, api_key: str, tier: Tier) -> list[str]:
        return list(self.opened.get((api_key, tier), []))

    def record_opening(self, api_key: str, tier: Tier, when: str) -> None:
        self.opened.setdefault((api_key, tier), []).append(when)


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 3, 12, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


class ListSink:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def record(self, event: TraceEvent) -> None:
        self.events.append(event)


def tiered_store(n: int = 10) -> InMemoryStore:
    store = InMemoryStore()
    for tier in Tier:
        for i in range(n):
            store.add(
                *planted_world(
                    f"{tier.value.replace('_', '-')}-{i:02d}", signal=i % 3 != 2, seed=i, n_pool=80, tier=tier
                )
            )
    return store


def service(store: InMemoryStore | None = None, **kw: object) -> tuple[ScorecardService, DictLedger, Clock]:
    ledger, clock = DictLedger(), Clock()
    return ScorecardService(store or tiered_store(), ledger, clock, **kw), ledger, clock  # type: ignore[arg-type]


def test_eval_scorecards_never_reuse_a_world() -> None:
    svc, ledger, _ = service()
    _, first = svc.open("key-aaaa", agent="a", track="open", tier=Tier.PUBLIC_EVAL, n_worlds=4)
    _, second = svc.open("key-bbbb", agent="a", track="open", tier=Tier.PUBLIC_EVAL, n_worlds=4)
    ids1, ids2 = {c.world_id for c in first}, {c.world_id for c in second}
    assert len(ids1) == 4 and len(ids2) == 4 and not ids1 & ids2
    assert ledger.used(Tier.PUBLIC_EVAL) == ids1 | ids2


def test_an_exhausted_pool_refuses_rather_than_reusing() -> None:
    svc, _, _ = service()
    svc.open("key-aaaa", agent="a", track="open", tier=Tier.PRIVATE, n_worlds=8)
    with pytest.raises(ArenaError) as err:
        svc.open("key-bbbb", agent="a", track="open", tier=Tier.PRIVATE, n_worlds=3)
    assert err.value.code is ErrorCode.CAP_EXCEEDED


def test_public_train_may_reuse_worlds_and_marks_nothing_used() -> None:
    svc, ledger, _ = service()
    _, a = svc.open("key-aaaa", agent="a", track="open", tier=Tier.PUBLIC_TRAIN, n_worlds=3)
    _, b = svc.open("key-aaaa", agent="a", track="open", tier=Tier.PUBLIC_TRAIN, n_worlds=3)
    assert [c.world_id for c in a] == [c.world_id for c in b]
    assert ledger.used(Tier.PUBLIC_TRAIN) == set()


def test_public_eval_is_capped_per_key_per_day() -> None:
    svc, _, clock = service(tiered_store(40))
    for _ in range(PUBLIC_EVAL_DAILY_CAP):
        svc.open("key-aaaa", agent="a", track="open", tier=Tier.PUBLIC_EVAL, n_worlds=1)
    with pytest.raises(ArenaError) as err:
        svc.open("key-aaaa", agent="a", track="open", tier=Tier.PUBLIC_EVAL, n_worlds=1)
    assert err.value.code is ErrorCode.CAP_EXCEEDED
    svc.open("key-bbbb", agent="a", track="open", tier=Tier.PUBLIC_EVAL, n_worlds=1)
    clock.now += timedelta(days=1)
    svc.open("key-aaaa", agent="a", track="open", tier=Tier.PUBLIC_EVAL, n_worlds=1)


def test_private_runs_are_capped_in_total_per_key() -> None:
    svc, _, clock = service(tiered_store(20))
    for _ in range(PRIVATE_TOTAL_CAP):
        svc.open("key-aaaa", agent="a", track="open", tier=Tier.PRIVATE, n_worlds=1)
        clock.now += timedelta(days=3)
    with pytest.raises(ArenaError) as err:
        svc.open("key-aaaa", agent="a", track="open", tier=Tier.PRIVATE, n_worlds=1)
    assert err.value.code is ErrorCode.CAP_EXCEEDED


def test_a_refused_opening_consumes_neither_cap_nor_worlds() -> None:
    svc, ledger, _ = service()
    with pytest.raises(ArenaError):
        svc.open("key-aaaa", agent="a", track="open", tier=Tier.PUBLIC_EVAL, n_worlds=999)
    assert ledger.used(Tier.PUBLIC_EVAL) == set() and ledger.openings("key-aaaa", Tier.PUBLIC_EVAL) == []


def test_unknown_tracks_are_refused() -> None:
    svc, _, _ = service()
    with pytest.raises(ArenaError) as err:
        svc.open("key-aaaa", agent="a", track="reference", tier=Tier.PUBLIC_TRAIN, n_worlds=1)
    assert err.value.code is ErrorCode.INVALID_PAYLOAD


def test_actions_are_confined_to_the_scorecard_worlds() -> None:
    svc, _, _ = service()
    sid, cards = svc.open("key-aaaa", agent="a", track="open", tier=Tier.PUBLIC_EVAL, n_worlds=2)
    outsider = next(w for w in svc.store.world_ids(Tier.PUBLIC_EVAL) if w not in {c.world_id for c in cards})
    with pytest.raises(ArenaError) as err:
        svc.act(sid, outsider, Reset(request_id="r", world_id=outsider))
    assert err.value.code is ErrorCode.UNKNOWN_WORLD
    with pytest.raises(ArenaError) as err:
        svc.act("sc-missing", cards[0].world_id, Reset(request_id="r", world_id=cards[0].world_id))
    assert err.value.code is ErrorCode.INVALID_PAYLOAD


def test_episodes_persist_within_a_scorecard_for_resumption() -> None:
    svc, _, _ = service()
    sid, cards = svc.open("key-aaaa", agent="a", track="open", tier=Tier.PUBLIC_TRAIN, n_worlds=1)
    wid = cards[0].world_id
    svc.act(sid, wid, Reset(request_id="r", world_id=wid))
    assert svc.episode(sid, wid).step == 1
    assert svc.episode(sid, wid) is svc.episode(sid, wid)


def test_close_scores_unplayed_worlds_as_empty_and_locks_the_scorecard() -> None:
    svc, _, _ = service()
    sid, cards = svc.open("key-aaaa", agent="a", track="open", tier=Tier.PUBLIC_TRAIN, n_worlds=6)
    played = cards[0].world_id
    svc.act(sid, played, Submit(request_id="s", ranking=(fid(0),)))
    card = svc.close(sid)
    by_world = {w.world_id: w for w in card.worlds}
    assert by_world[played].listed == 1
    assert all(w.abstained for wid, w in by_world.items() if wid != played)
    with pytest.raises(ArenaError) as err:
        svc.close(sid)
    assert err.value.code is ErrorCode.SCORECARD_CLOSED
    with pytest.raises(ArenaError) as err:
        svc.act(sid, played, Submit(request_id="s2", ranking=()))
    assert err.value.code is ErrorCode.SCORECARD_CLOSED


@pytest.mark.parametrize("tier", [Tier.PUBLIC_EVAL, Tier.PRIVATE])
def test_eval_scorecards_expose_aggregates_and_alignment_but_no_per_world_results(tier: Tier) -> None:
    svc, _, _ = service(pool_commitments={tier: "ab" * 32})
    sid, cards = svc.open("key-aaaa", agent="a", track="open", tier=tier, n_worlds=6)
    for card in cards:
        svc.act(sid, card.world_id, Submit(request_id="s", ranking=(fid(0),)))
    closed = svc.close(sid)
    assert closed.worlds == ()
    assert closed.alignment is not None
    assert closed.pool_commitment == "ab" * 32
    assert closed.versions.get("interface") == closed.interface_version


def test_every_applied_action_is_traced_server_side() -> None:
    sinks: dict[str, ListSink] = {}
    svc, _, _ = service(traces=lambda sid: sinks.setdefault(sid, ListSink()))
    sid, cards = svc.open("key-aaaa", agent="a", track="open", tier=Tier.PUBLIC_TRAIN, n_worlds=1)
    wid = cards[0].world_id
    svc.act(sid, wid, Reset(request_id="r", world_id=wid))
    svc.act(sid, wid, Submit(request_id="s", ranking=(fid(0),)))
    events = sinks[sid].events
    assert [e.action_kind for e in events] == ["reset", "submit"]
    assert all(e.action_json for e in events)


def test_refused_actions_are_not_traced() -> None:
    sinks: dict[str, ListSink] = {}
    svc, _, _ = service(traces=lambda sid: sinks.setdefault(sid, ListSink()))
    sid, cards = svc.open("key-aaaa", agent="a", track="open", tier=Tier.PUBLIC_TRAIN, n_worlds=1)
    with pytest.raises(ArenaError):
        svc.act(sid, cards[0].world_id, Recruit(request_id="x", count=1))
    assert sinks.get(sid) is None or sinks[sid].events == []


def test_eval_draws_keep_the_fixed_null_share() -> None:
    store = InMemoryStore()
    for i in range(50):
        store.add(*planted_world(f"e-{i:02d}", signal=i >= 10, seed=i, n_pool=60, tier=Tier.PUBLIC_EVAL))
    shares = []
    for k in range(5):
        svc, _, _ = service(store)
        _sid, cards = svc.open(f"key-{k:04d}", agent="a", track="open", tier=Tier.PUBLIC_EVAL, n_worlds=10)
        shares.append(sum(store.answer_key(c.world_id).is_null for c in cards))
    assert shares == [2] * 5
