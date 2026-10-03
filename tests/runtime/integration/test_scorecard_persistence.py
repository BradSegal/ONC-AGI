"""Durable scorecard state (B1, B3, B4): the JSON ledger's lock and atomic writes, the file
archive, restoring open scorecards by trace replay, TTL expiry and closed-scorecard records."""

from __future__ import annotations

import json
import multiprocessing as mp
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from arena_factories import InMemoryStore, fid, planted_world
from onc_agi.core.errors import ArenaError
from onc_agi.core.ports import ClosedRecord, OpenRecord
from onc_agi.core.schema import ErrorCode, Mode, Recruit, Reset, Submit, Tier
from onc_agi.infra.archive import FileScorecardArchive
from onc_agi.infra.ledger import JsonLedger
from onc_agi.services.kit import view_digest
from onc_agi.services.scorecards import RestoreError, ScorecardService, key_digest

KEY = "owner-key-01"


def _attempt_archive_claim(root: str, out) -> None:  # type: ignore[no-untyped-def]
    archive = FileScorecardArchive(Path(root))
    try:
        archive.acquire()
    except RuntimeError:
        out.put("refused")
    else:
        archive.release()
        out.put("accepted")


def test_archive_ownership_refuses_another_service_and_releases_on_shutdown(tmp_path: Path) -> None:
    store = mixed_store()
    svc = boot(store, tmp_path)
    with pytest.raises(RuntimeError, match="active service"):
        boot(store, tmp_path)
    assert svc.archive is not None
    with pytest.raises(RuntimeError, match="active service"):
        ScorecardService(store, svc.ledger, archive=svc.archive)
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    child = ctx.Process(target=_attempt_archive_claim, args=(str(tmp_path / "archive"), out))
    child.start()
    assert out.get(timeout=15) == "refused"
    child.join(15)
    assert child.exitcode == 0
    archive = svc.archive
    svc.shutdown()
    restored = ScorecardService(store, svc.ledger, archive=archive)
    svc.shutdown()  # release is idempotent; it cannot release the next owner's lock
    with pytest.raises(RuntimeError, match="active service"):
        boot(store, tmp_path)
    with pytest.raises(RuntimeError, match="stopped"):
        svc.open(KEY, agent="a", track="open", tier=Tier.PUBLIC_TRAIN, n_worlds=1)
    restored.shutdown()


def test_trace_write_failure_can_be_retried_and_restarted(tmp_path: Path) -> None:
    store = mixed_store()
    svc = boot(store, tmp_path)
    sid, cards = svc.open(KEY, agent="a", track="open", tier=Tier.PUBLIC_TRAIN, n_worlds=2)
    seq = next(c for c in cards if c.mode is Mode.SEQUENTIAL)
    assert isinstance(svc.archive, FileScorecardArchive)
    original = svc.archive.trace

    class FailingSink:
        def record(self, event) -> None:  # type: ignore[no-untyped-def]
            raise OSError("injected trace-write failure")

    svc.archive.trace = lambda _: FailingSink()  # type: ignore[method-assign,return-value]
    action = Recruit(request_id="retry", count=5, stratum=seq.strata[0])
    with pytest.raises(OSError, match="injected"):
        svc.act(sid, seq.world_id, action, api_key=KEY)
    svc.archive.trace = original  # type: ignore[method-assign]
    retry = svc.act(sid, seq.world_id, action, api_key=KEY)
    assert len(retry.rows) == 5
    svc.shutdown()
    restored = boot(store, tmp_path)
    assert view_digest(restored.view(sid, seq.world_id, api_key=KEY)) == view_digest(retry)
    assert view_digest(restored.act(sid, seq.world_id, action, api_key=KEY)) == view_digest(retry)
    restored.shutdown()


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 3, 12, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


def mixed_store() -> InMemoryStore:
    store = InMemoryStore()
    for tier in (Tier.PUBLIC_TRAIN, Tier.PUBLIC_EVAL):
        for i in range(10):
            mode = Mode.SEQUENTIAL if i % 2 else Mode.FULL_ACCESS
            wid = f"{tier.value.replace('_', '-')}-{i:02d}"
            store.add(*planted_world(wid, signal=i % 5 != 4, seed=i, n_pool=60, tier=tier, mode=mode))
    return store


def boot(
    store: InMemoryStore, root: Path, clock: Clock | None = None, ttl: timedelta | None = None
) -> ScorecardService:
    return ScorecardService(
        store,
        JsonLedger(root / "ledger.json"),
        clock,
        archive=FileScorecardArchive(root / "archive"),
        ttl=ttl,
        min_eval_worlds=5,  # bounded synthetic fixture; hosted default is 40
    )


# ---------------------------------------------------------------- ledger


def test_the_ledger_lock_is_reentrant_and_released(tmp_path: Path) -> None:
    ledger = JsonLedger(tmp_path / "ledger.json")
    with ledger.lock(), ledger.lock():
        ledger.mark_used(Tier.PRIVATE, ["a"])  # public methods take the lock internally
        assert ledger._depth == 2
    assert ledger._depth == 0 and ledger._fd is None
    assert ledger.lock_path.exists()
    other = JsonLedger(tmp_path / "ledger.json")  # a second holder can now take it
    with other.lock():
        assert other.used(Tier.PRIVATE) == {"a"}


def test_the_ledger_lock_excludes_a_second_instance(tmp_path: Path) -> None:
    """flock is per open file description, so two instances in one process exclude each other."""
    first, second = JsonLedger(tmp_path / "l.json"), JsonLedger(tmp_path / "l.json")
    entered = threading.Event()
    with first.lock():
        worker = threading.Thread(target=lambda: (second.used(Tier.PRIVATE), entered.set()))
        worker.start()
        assert not entered.wait(0.3)
    worker.join(5)
    assert entered.is_set()


def test_ledger_writes_leave_no_temp_files_and_valid_json(tmp_path: Path) -> None:
    ledger = JsonLedger(tmp_path / "ledger.json")
    threads = [
        threading.Thread(
            target=lambda k=k: [ledger.record_opening(f"k{k}", Tier.PUBLIC_TRAIN, "d") for _ in range(25)]
        )
        for k in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    data = json.loads((tmp_path / "ledger.json").read_text())
    assert all(len(data["openings"][f"k{k}|public_train"]) == 25 for k in range(8))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ledger.json", "ledger.json.lock"]


# ---------------------------------------------------------------- archive


def test_the_archive_round_trips_open_and_closed_records(tmp_path: Path) -> None:
    svc = boot(mixed_store(), tmp_path)
    sid, _ = svc.open(KEY, agent="a", track="open", tier=Tier.PUBLIC_EVAL, n_worlds=5, tags=("t",))
    archive = FileScorecardArchive(tmp_path / "archive")
    (record,) = archive.open_records()
    assert (record.scorecard_id, record.tier, record.tags) == (sid, Tier.PUBLIC_EVAL, ("t",))
    assert record.owner_sha256 == key_digest(KEY)
    assert KEY not in (tmp_path / "archive" / "open" / f"{sid}.json").read_text()
    closed = svc.close(sid, api_key=KEY)
    assert archive.open_records() == () and archive.closed_ids() == (sid,)
    stored = archive.load_closed(sid)
    assert stored is not None and stored.scorecard == closed and not stored.expired
    assert stored.scorecard.worlds == ()  # exposure as returned


@pytest.mark.parametrize("sid", ["../ledger", "a/b", "", ".hidden", "x" * 200])
def test_unsafe_ids_never_become_paths(tmp_path: Path, sid: str) -> None:
    archive = FileScorecardArchive(tmp_path)
    assert archive.load_closed(sid) is None
    with pytest.raises(ValueError):
        archive.trace(sid)


def test_a_closed_record_wins_over_a_stale_open_record(tmp_path: Path) -> None:
    """A crash between writing the closed record and removing the open one is harmless."""
    archive = FileScorecardArchive(tmp_path / "archive")
    svc = boot(mixed_store(), tmp_path)
    sid, _ = svc.open(KEY, agent="a", track="open", tier=Tier.PUBLIC_TRAIN, n_worlds=1)
    (opened,) = archive.open_records()
    svc.close(sid)
    archive.save_open(opened)  # simulate the crash window
    svc.shutdown()
    again = boot(mixed_store(), tmp_path)
    assert archive.open_records() == ()
    with pytest.raises(ArenaError) as err:
        again.close(sid)
    assert err.value.code is ErrorCode.SCORECARD_CLOSED


# ---------------------------------------------------------------- restore and records


def test_a_restarted_service_resumes_open_scorecards_mid_episode(tmp_path: Path) -> None:
    store = mixed_store()
    svc = boot(store, tmp_path)
    sid, cards = svc.open(KEY, agent="a", track="open", tier=Tier.PUBLIC_TRAIN, n_worlds=4)
    seq = next(c for c in cards if c.mode is Mode.SEQUENTIAL)
    full = next(c for c in cards if c.mode is Mode.FULL_ACCESS)
    svc.act(sid, seq.world_id, Reset(request_id="r", world_id=seq.world_id), api_key=KEY)
    recruit = Recruit(request_id="r1", count=7, stratum=seq.strata[0])
    svc.act(sid, seq.world_id, recruit, api_key=KEY)
    svc.act(sid, seq.world_id, recruit, api_key=KEY)  # idempotent retry is traced and replays
    svc.act(sid, full.world_id, Submit(request_id="s", ranking=(fid(0),)), api_key=KEY)
    before = {c.world_id: view_digest(svc.view(sid, c.world_id)) for c in cards}

    svc.shutdown()
    restored = boot(store, tmp_path)
    after = {c.world_id: view_digest(restored.view(sid, c.world_id, api_key=KEY)) for c in cards}
    assert after == before
    with pytest.raises(ArenaError) as err:  # ownership survives the restart
        restored.view(sid, seq.world_id, api_key="someone-else")
    assert err.value.code is ErrorCode.INVALID_PAYLOAD
    with pytest.raises(ArenaError) as err:  # submitted worlds stay submitted
        restored.act(sid, full.world_id, Submit(request_id="s2", ranking=()), api_key=KEY)
    assert err.value.code is ErrorCode.EPISODE_CLOSED
    final = Submit(request_id="s", ranking=())
    assert restored.act(sid, seq.world_id, final, api_key=KEY).step == 3
    closed = restored.close(sid, api_key=KEY)
    assert {w.world_id: w.listed for w in closed.worlds}[full.world_id] == 1

    restored.shutdown()
    third = boot(store, tmp_path)
    assert third.scorecard(sid, api_key=KEY) == closed
    with pytest.raises(ArenaError) as err:
        third.close(sid, api_key=KEY)
    assert err.value.code is ErrorCode.SCORECARD_CLOSED


def test_a_tampered_trace_fails_fast_on_restore(tmp_path: Path) -> None:
    store = mixed_store()
    svc = boot(store, tmp_path)
    sid, cards = svc.open(KEY, agent="a", track="open", tier=Tier.PUBLIC_TRAIN, n_worlds=2)
    seq = next(c for c in cards if c.mode is Mode.SEQUENTIAL)
    svc.act(sid, seq.world_id, Reset(request_id="r", world_id=seq.world_id))
    svc.act(sid, seq.world_id, Recruit(request_id="r1", count=5, stratum=seq.strata[0]))
    trace = tmp_path / "archive" / "traces" / f"{sid}.jsonl"
    lines = trace.read_text().splitlines()
    event = json.loads(lines[-1])
    event["action_json"] = json.dumps({"kind": "recruit", "request_id": "r1", "count": 6, "stratum": "all"})
    trace.write_text("\n".join([*lines[:-1], json.dumps(event)]) + "\n")
    svc.shutdown()
    with pytest.raises(RestoreError, match="differs from the recorded digest"):
        boot(store, tmp_path)


def test_a_trace_naming_a_foreign_world_fails_fast(tmp_path: Path) -> None:
    store = mixed_store()
    svc = boot(store, tmp_path)
    sid, cards = svc.open(KEY, agent="a", track="open", tier=Tier.PUBLIC_TRAIN, n_worlds=1)
    svc.act(sid, cards[0].world_id, Reset(request_id="r", world_id=cards[0].world_id))
    trace = tmp_path / "archive" / "traces" / f"{sid}.jsonl"
    trace.write_text(trace.read_text().replace(cards[0].world_id, "public-train-09"))
    svc.shutdown()
    with pytest.raises(RestoreError, match="does not belong"):
        boot(store, tmp_path)


@pytest.mark.parametrize("final_tail", ['{"torn":', ""])
def test_final_trace_is_repaired_before_restart_appends(tmp_path: Path, final_tail: str) -> None:
    store = mixed_store()
    svc = boot(store, tmp_path)
    sid, cards = svc.open(KEY, agent="a", track="open", tier=Tier.PUBLIC_TRAIN, n_worlds=2)
    seq = next(c for c in cards if c.mode is Mode.SEQUENTIAL)
    svc.act(sid, seq.world_id, Recruit(request_id="first", count=5, stratum=seq.strata[0]))
    svc.shutdown()
    trace = tmp_path / "archive" / "traces" / f"{sid}.jsonl"
    if final_tail:
        with trace.open("a") as handle:
            handle.write(final_tail)
    else:
        trace.write_text(trace.read_text().rstrip("\n"))
    restored = boot(store, tmp_path)
    after = restored.act(sid, seq.world_id, Recruit(request_id="second", count=5, stratum=seq.strata[0]))
    restored.shutdown()
    again = boot(store, tmp_path)
    assert view_digest(again.view(sid, seq.world_id)) == view_digest(after)
    again.shutdown()


def test_explicit_trace_sinks_still_receive_events_alongside_the_archive(tmp_path: Path) -> None:
    seen: list[str] = []

    class Sink:
        def record(self, event) -> None:  # type: ignore[no-untyped-def]
            seen.append(event.action_kind)

    svc = ScorecardService(
        mixed_store(),
        JsonLedger(tmp_path / "l.json"),
        traces=lambda _sid: Sink(),
        archive=FileScorecardArchive(tmp_path / "a"),
    )
    sid, cards = svc.open(KEY, agent="a", track="open", tier=Tier.PUBLIC_TRAIN, n_worlds=1)
    svc.act(sid, cards[0].world_id, Reset(request_id="r", world_id=cards[0].world_id))
    assert seen == ["reset"] and len(FileScorecardArchive(tmp_path / "a").events(sid)) == 1


def test_expiry_survives_a_restart_and_is_recorded(tmp_path: Path) -> None:
    store, clock = mixed_store(), Clock()
    svc = boot(store, tmp_path, clock, ttl=timedelta(hours=24))
    sid, _ = svc.open(KEY, agent="a", track="open", tier=Tier.PUBLIC_EVAL, n_worlds=5)
    clock.now += timedelta(hours=25)
    svc.shutdown()
    restored = boot(store, tmp_path, clock, ttl=timedelta(hours=24))
    restored.open("other-key-01", agent="a", track="open", tier=Tier.PUBLIC_TRAIN, n_worlds=1)  # sweeps
    record = FileScorecardArchive(tmp_path / "archive").load_closed(sid)
    assert isinstance(record, ClosedRecord) and record.expired
    assert record.scorecard.n_worlds == 5 and record.scorecard.worlds == ()
    assert restored.scorecard(sid, api_key=KEY) == record.scorecard


def test_open_records_validate_their_owner_digest() -> None:
    with pytest.raises(ValueError):
        OpenRecord(
            scorecard_id="sc-1",
            agent="a",
            track="open",
            tier=Tier.PUBLIC_TRAIN,
            world_ids=("w",),
            owner_sha256="raw-key-not-a-digest",
            opened_at=datetime(2026, 10, 3, tzinfo=UTC),
        )
