"""Scorecard service: fresh never-reused eval draws, caps, exposure and server-side traces."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from arena_factories import InMemoryStore, fid, planted_world
from onc_agi.core.errors import ArenaError
from onc_agi.core.schema import ErrorCode, Mode, Recruit, Reset, Submit, Tier, TraceEvent
from onc_agi.services import sampling
from onc_agi.services.scorecards import (
    PRIVATE_TOTAL_CAP,
    PUBLIC_EVAL_DAILY_CAP,
    ScorecardService,
    key_digest,
)


@pytest.mark.parametrize("tier", [Tier.PUBLIC_EVAL, Tier.PRIVATE])
@pytest.mark.parametrize("n_worlds", [1, 2, 39])
def test_default_evaluation_minimum_refuses_small_draws_without_consumption(
    tier: Tier, n_worlds: int
) -> None:
    ledger = DictLedger()
    svc = ScorecardService(tiered_store(), ledger)
    with pytest.raises(ArenaError) as err:
        svc.open("owner-key", agent="a", track="open", tier=tier, n_worlds=n_worlds)
    assert err.value.code is ErrorCode.INVALID_PAYLOAD
    assert ledger.used_ids == {} and ledger.opened == {}


def test_evaluation_minimum_is_an_operator_setting_and_training_stays_useful() -> None:
    svc = ScorecardService(tiered_store(), DictLedger(), min_eval_worlds=5)
    _, cards = svc.open("owner-key", agent="a", track="open", tier=Tier.PUBLIC_EVAL, n_worlds=5)
    assert len(cards) == 5
    train = ScorecardService(tiered_store(), DictLedger())
    _, cards = train.open("owner-key", agent="a", track="open", tier=Tier.PUBLIC_TRAIN, n_worlds=1)
    assert len(cards) == 1


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
    kw.setdefault("min_eval_worlds", 1)  # synthetic fixtures deliberately exercise small draws
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


def test_public_train_draws_are_seeded_stratified_samples_not_the_first_ids() -> None:
    store = tiered_store(12)
    svc, _, _ = service(store)
    profiles = sampling.world_profiles(store, Tier.PUBLIC_TRAIN)
    for seed in (None, 0, 3):
        _, cards = svc.open(
            "key-aaaa", agent="a", track="open", tier=Tier.PUBLIC_TRAIN, n_worlds=4, seed=seed
        )
        expected = sampling.stratified_sample(profiles, 4, seed or 0).world_ids
        assert tuple(c.world_id for c in cards) == expected
        assert expected != store.world_ids(Tier.PUBLIC_TRAIN)[:4]
    named = ("public-train-07", "public-train-01")
    _, cards = svc.open("key-aaaa", agent="a", track="open", tier=Tier.PUBLIC_TRAIN, world_ids=named)
    assert tuple(c.world_id for c in cards) == named  # named worlds keep their order


@pytest.mark.parametrize(
    ("tier", "extra"),
    [
        (Tier.PUBLIC_EVAL, {"n_worlds": 2, "seed": 1}),
        (Tier.PUBLIC_TRAIN, {"world_ids": ("public-train-01",), "seed": 1}),
        (Tier.PUBLIC_TRAIN, {"n_worlds": 2, "seed": -1}),
    ],
)
def test_seeds_apply_to_drawn_public_train_worlds_only(tier: Tier, extra: dict[str, object]) -> None:
    svc, ledger, _ = service()
    with pytest.raises(ArenaError, match="seed") as err:
        svc.open("key-aaaa", agent="a", track="open", tier=tier, **extra)  # type: ignore[arg-type]
    assert err.value.code is ErrorCode.INVALID_PAYLOAD and ledger.used_ids == {}


def test_named_worlds_errors_name_only_the_offending_ids() -> None:
    svc, _, _ = service(tiered_store(12))
    named = tuple(f"public-train-{i:02d}" for i in range(10))
    with pytest.raises(ArenaError) as err:
        svc.open("key-aaaa", agent="a", track="open", tier=Tier.PUBLIC_TRAIN, world_ids=(*named, named[3]))
    assert err.value.code is ErrorCode.UNKNOWN_WORLD
    assert err.value.message == "listed more than once ['public-train-03']"
    with pytest.raises(ArenaError) as err:
        svc.open(
            "key-aaaa", agent="a", track="open", tier=Tier.PUBLIC_TRAIN, world_ids=(*named, "nope", "nope")
        )
    assert err.value.message == "unknown public_train worlds ['nope']; listed more than once ['nope']"


class CountingStore(InMemoryStore):
    """Counts card reads, to show which opens read the whole store."""

    def __init__(self) -> None:
        super().__init__()
        self.card_reads = 0

    def card(self, world_id: str):  # type: ignore[no-untyped-def]
        self.card_reads += 1
        return super().card(world_id)


def test_public_train_draws_read_the_store_once_per_world_set() -> None:
    store = CountingStore()
    for i in range(12):
        store.add(*planted_world(f"pt-{i:02d}", signal=i % 3 != 2, seed=i, n_pool=60))
    svc, _, _ = service(store)  # type: ignore[arg-type]
    _, first = svc.open("key-aaaa", agent="a", track="open", tier=Tier.PUBLIC_TRAIN, n_worlds=4, seed=1)
    reads = store.card_reads
    _, again = svc.open("key-aaaa", agent="a", track="open", tier=Tier.PUBLIC_TRAIN, n_worlds=4, seed=1)
    svc.open("key-aaaa", agent="a", track="open", tier=Tier.PUBLIC_TRAIN, n_worlds=3, mode=Mode.FULL_ACCESS)
    assert first == again and store.card_reads - reads == 4 + 3  # only the opened worlds' cards
    store.add(*planted_world("pt-12", signal=True, seed=12, n_pool=60))  # a new world: profiles re-read
    _, cards = svc.open("key-aaaa", agent="a", track="open", tier=Tier.PUBLIC_TRAIN, n_worlds=13)
    assert "pt-12" in {c.world_id for c in cards}


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


# ---------------------------------------------------------------- ownership and records


def _refusal(call) -> tuple[ErrorCode, str]:  # type: ignore[no-untyped-def]
    with pytest.raises(ArenaError) as err:
        call()
    return err.value.code, err.value.message


def test_a_foreign_key_is_told_exactly_what_an_unknown_id_is_told() -> None:
    svc, _, _ = service()
    sid, cards = svc.open("key-owner", agent="a", track="open", tier=Tier.PUBLIC_TRAIN, n_worlds=1)
    wid = cards[0].world_id
    reset = Reset(request_id="r", world_id=wid)
    for call in (
        lambda s: svc.act(s, wid, reset, api_key="key-thief"),
        lambda s: svc.episode(s, wid, api_key="key-thief"),
        lambda s: svc.view(s, wid, api_key="key-thief"),
        lambda s: svc.close(s, api_key="key-thief"),
        lambda s: svc.scorecard(s, api_key="key-thief"),
    ):
        code, message = _refusal(lambda c=call: c(sid))
        ghost_code, ghost_message = _refusal(lambda c=call: c("sc-ghost"))
        assert code is ghost_code is ErrorCode.INVALID_PAYLOAD
        assert message.replace(sid, "?") == ghost_message.replace("sc-ghost", "?")
    assert svc.act(sid, wid, reset, api_key="key-owner").step == 1  # untouched by the attempts
    assert svc.act(sid, wid, Submit(request_id="s", ranking=()), api_key=None).step == 2  # trusted path


def test_the_owner_digest_is_stored_never_the_raw_key() -> None:
    svc, _, _ = service()
    sid, _ = svc.open("key-secret-123", agent="a", track="open", tier=Tier.PUBLIC_TRAIN, n_worlds=1)
    owner = svc._open[sid].owner
    assert owner == key_digest("key-secret-123") and "key-secret-123" not in owner


def test_the_closed_scorecard_is_on_record_and_an_open_one_is_refused() -> None:
    svc, _, _ = service()
    sid, _ = svc.open("key-aaaa", agent="a", track="open", tier=Tier.PUBLIC_EVAL, n_worlds=2)
    with pytest.raises(ArenaError) as err:
        svc.scorecard(sid, api_key="key-aaaa")
    assert err.value.code is ErrorCode.ACTION_NOT_AVAILABLE
    closed = svc.close(sid, api_key="key-aaaa")
    assert svc.scorecard(sid, api_key="key-aaaa") == closed == svc.scorecard(sid)
    assert closed.worlds == ()  # the record keeps eval exposure


def test_close_passes_run_metadata_and_records_oracle_versions() -> None:
    svc, _, _ = service()
    sid, _ = svc.open("key-aaaa", agent="a", track="standard", tier=Tier.PUBLIC_TRAIN, n_worlds=2)
    closed = svc.close(sid, model="m-1", harness="inspect/0.1", tokens=1200, cost_usd=0.25)
    assert (closed.model, closed.harness, closed.tokens, closed.cost_usd) == (
        "m-1",
        "inspect/0.1",
        1200,
        0.25,
    )
    assert closed.versions["oracle"] == "test"


def test_open_can_restrict_the_draw_to_one_mode() -> None:
    store = InMemoryStore()
    for tier in (Tier.PUBLIC_TRAIN, Tier.PUBLIC_EVAL):
        for i in range(12):
            mode = Mode.SEQUENTIAL if i % 2 else Mode.FULL_ACCESS
            wid = f"{tier.value.replace('_', '-')}-{i:02d}"
            store.add(*planted_world(wid, signal=i % 4 > 1, seed=i, n_pool=60, tier=tier, mode=mode))
    svc, _, _ = service(store)
    for tier in (Tier.PUBLIC_TRAIN, Tier.PUBLIC_EVAL):
        for mode in Mode:
            _, cards = svc.open(
                f"key-{mode.value}", agent="a", track="open", tier=tier, n_worlds=3, mode=mode
            )
            assert len(cards) == 3 and {c.mode for c in cards} == {mode}
    _, cards = svc.open("key-any", agent="a", track="open", tier=Tier.PUBLIC_TRAIN, n_worlds=12)
    assert {c.mode for c in cards} == set(Mode)


def test_abandoned_scorecards_expire_after_the_ttl() -> None:
    svc, _, clock = service(ttl=timedelta(hours=2))
    stale, cards = svc.open("key-aaaa", agent="a", track="open", tier=Tier.PUBLIC_TRAIN, n_worlds=2)
    played = cards[0].world_id
    svc.act(stale, played, Submit(request_id="s", ranking=(fid(0),)))
    clock.now += timedelta(hours=1)
    fresh, _ = svc.open("key-aaaa", agent="a", track="open", tier=Tier.PUBLIC_TRAIN, n_worlds=1)
    clock.now += timedelta(hours=1, seconds=1)
    with pytest.raises(ArenaError) as err:  # the sweep runs first, so the stale scorecard is closed
        svc.act(stale, played, Reset(request_id="r", world_id=played))
    assert err.value.code is ErrorCode.SCORECARD_CLOSED and "expired" in err.value.message
    expired = svc.scorecard(stale)
    by_world = {w.world_id: w for w in expired.worlds}
    assert by_world[played].listed == 1 and by_world[cards[1].world_id].abstained
    with pytest.raises(ArenaError) as err:  # the younger scorecard is still within its TTL
        svc.scorecard(fresh)
    assert err.value.code is ErrorCode.ACTION_NOT_AVAILABLE


def test_ttl_must_be_positive() -> None:
    with pytest.raises(ValueError):
        service(ttl=timedelta(0))


def test_a_ledger_without_a_lock_is_still_accepted() -> None:
    svc, ledger, _ = service()
    assert not hasattr(ledger, "lock")
    svc.open("key-aaaa", agent="a", track="open", tier=Tier.PUBLIC_EVAL, n_worlds=2)
    assert len(ledger.used(Tier.PUBLIC_EVAL)) == 2


def test_a_fixed_eval_set_is_scored_whole_on_every_scorecard_and_marks_nothing_used() -> None:
    svc, ledger, _ = service(fixed_eval_sets=True)
    _, first = svc.open("key-aaaa", agent="a", track="open", tier=Tier.PUBLIC_EVAL)
    _, second = svc.open("key-bbbb", agent="b", track="open", tier=Tier.PUBLIC_EVAL, n_worlds=10)
    assert [c.world_id for c in first] == [c.world_id for c in second]
    assert len(first) == 10 and ledger.used(Tier.PUBLIC_EVAL) == set()


def test_a_fixed_eval_set_refuses_a_partial_draw_and_world_selection() -> None:
    svc, _, _ = service(fixed_eval_sets=True)
    with pytest.raises(ArenaError) as err:
        svc.open("key-aaaa", agent="a", track="open", tier=Tier.PUBLIC_EVAL, n_worlds=4)
    assert err.value.code is ErrorCode.INVALID_PAYLOAD
    with pytest.raises(ArenaError):
        svc.open("key-aaaa", agent="a", track="open", tier=Tier.PUBLIC_EVAL, world_ids=("public-eval-00",))


def test_a_fixed_eval_set_can_be_scored_one_mode_at_a_time() -> None:
    store = InMemoryStore()
    for i in range(6):
        mode = Mode.SEQUENTIAL if i % 2 else Mode.FULL_ACCESS
        store.add(*planted_world(f"ev-{i}", signal=True, seed=i, n_pool=80, tier=Tier.PUBLIC_EVAL, mode=mode))
    svc, _, _ = service(store, fixed_eval_sets=True)
    _, cards = svc.open("key-aaaa", agent="a", track="open", tier=Tier.PUBLIC_EVAL, mode=Mode.SEQUENTIAL)
    assert len(cards) == 3 and all(c.mode is Mode.SEQUENTIAL for c in cards)


def test_a_fixed_eval_set_keeps_per_world_results_hidden_and_the_daily_cap() -> None:
    svc, _, _ = service(fixed_eval_sets=True)
    for _ in range(PUBLIC_EVAL_DAILY_CAP):
        sid, _ = svc.open("key-aaaa", agent="a", track="open", tier=Tier.PUBLIC_EVAL)
    assert svc.close(sid).worlds == ()
    with pytest.raises(ArenaError) as err:
        svc.open("key-aaaa", agent="a", track="open", tier=Tier.PUBLIC_EVAL)
    assert err.value.code is ErrorCode.CAP_EXCEEDED


def test_caps_count_against_the_cap_identity_so_rotating_a_key_does_not_reset_them() -> None:
    account = {"key-old-1": "acct-1", "key-new-1": "acct-1"}
    svc, ledger, _ = service(tiered_store(40), cap_identity=account.__getitem__)
    for _ in range(PUBLIC_EVAL_DAILY_CAP):
        svc.open("key-old-1", agent="a", track="open", tier=Tier.PUBLIC_EVAL, n_worlds=1)
    with pytest.raises(ArenaError) as err:
        svc.open("key-new-1", agent="a", track="open", tier=Tier.PUBLIC_EVAL, n_worlds=1)
    assert err.value.code is ErrorCode.CAP_EXCEEDED
    assert set(k for k, _ in ledger.opened) == {"acct-1"}


def test_allowed_keys_may_be_a_live_container() -> None:
    class Issued:
        def __init__(self) -> None:
            self.keys: set[str] = set()

        def __contains__(self, key: object) -> bool:
            return key in self.keys

    issued = Issued()
    svc, _, _ = service(allowed_keys=issued)
    with pytest.raises(ArenaError):
        svc.open("key-late-1", agent="a", track="open", tier=Tier.PUBLIC_TRAIN, n_worlds=1)
    issued.keys.add("key-late-1")
    svc.open("key-late-1", agent="a", track="open", tier=Tier.PUBLIC_TRAIN, n_worlds=1)


def test_progress_counts_started_and_submitted_worlds_without_answer_data() -> None:
    svc, _, _ = service()
    sid, cards = svc.open("key-aaaa", agent="a", track="open", tier=Tier.PUBLIC_EVAL, n_worlds=3)
    svc.act(sid, cards[0].world_id, Submit(request_id="r1", ranking=()), api_key="key-aaaa")
    before = svc.progress(sid)
    assert (before.n_worlds, before.started, before.submitted, before.closed) == (3, 1, 1, None)
    svc.close(sid)
    after = svc.progress(sid)
    assert after.closed is not None and after.closed.worlds == ()
