"""B1: concurrent opens can neither bypass a cap nor reuse an eval world, within one server
(threads) or across server processes sharing one ledger file.

Regression for the strict-xfail A1 pinned in ``test_arena_attack_scorecard.py``: before the
fix, 64 concurrent opens gave HTTP 500s (a shared ``ledger.tmp``), lost ledger updates and
re-served eval worlds.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import threading
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from arena_factories import planted_world
from onc_agi.core.errors import ArenaError
from onc_agi.core.schema import Tier
from onc_agi.infra.bundles import FileWorldStore, write_world
from onc_agi.infra.ledger import JsonLedger
from onc_agi.services.scorecards import PUBLIC_EVAL_DAILY_CAP, ScorecardService, key_digest

THREADS = 16
N_WORLDS = 5


def _clock() -> datetime:
    return datetime(2026, 10, 3, 12, tzinfo=UTC)


@pytest.fixture(scope="module")
def eval_store(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """120 eval worlds on disk, every fifth null, three difficulty tiers."""
    root = tmp_path_factory.mktemp("eval")
    for i in range(120):
        world, key = planted_world(
            f"public-eval-{i:04d}",
            signal=i % 5 != 0,
            seed=7000 + i,
            n_pool=60,
            tier=Tier.PUBLIC_EVAL,
            difficulty_tier=i % 3,
        )
        write_world(root / "store", world, key, keys_dir=root / "keys")
    return root / "store", root / "keys"


def _open_once(svc: ScorecardService, key: str) -> tuple[str, ...] | str:
    try:
        _, cards = svc.open(key, agent="p", track="open", tier=Tier.PUBLIC_EVAL, n_worlds=N_WORLDS)
    except ArenaError as exc:
        return exc.code.value
    return tuple(c.world_id for c in cards)


def _race(n: int, target: Callable[[int], Any]) -> list[Any]:
    """Run ``target(i)`` on ``n`` threads released together; unexpected exceptions are results."""
    barrier = threading.Barrier(n)
    results: list[Any] = [None] * n

    def go(i: int) -> None:
        barrier.wait()
        try:
            results[i] = target(i)
        except Exception as exc:  # noqa: BLE001 - a crash is the failure being characterised
            results[i] = f"crash:{type(exc).__name__}:{exc}"

    threads = [threading.Thread(target=go, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


@pytest.mark.parametrize("trial", range(6))
def test_concurrent_opens_on_one_key_respect_the_cap_and_freshness(
    eval_store: tuple[Path, Path], tmp_path: Path, trial: int
) -> None:
    store = FileWorldStore(*eval_store)
    svc = ScorecardService(
        store, JsonLedger(tmp_path / "ledger.json"), clock=_clock, min_eval_worlds=N_WORLDS
    )
    results = _race(THREADS, lambda _i: _open_once(svc, "one-key"))
    assert not [r for r in results if isinstance(r, str) and r.startswith("crash")], results
    ok = [r for r in results if isinstance(r, tuple)]
    drawn = [w for ids in ok for w in ids]
    assert len(ok) == PUBLIC_EVAL_DAILY_CAP, "cap bypassed (or refused too early) under concurrency"
    assert set(results) - set(ok) == {"cap_exceeded"}
    assert len(drawn) == len(set(drawn)), "worlds reused across scorecards under concurrency"
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert sorted(ledger["used"]["public_eval"]) == sorted(drawn)
    assert (
        ledger["openings"][f"{key_digest('one-key')}|public_eval"] == ["2026-10-03"] * PUBLIC_EVAL_DAILY_CAP
    )
    assert "one-key" not in json.dumps(ledger), "raw API keys must never rest in the ledger"


def test_concurrent_opens_on_distinct_keys_lose_no_update(
    eval_store: tuple[Path, Path], tmp_path: Path
) -> None:
    store = FileWorldStore(*eval_store)
    svc = ScorecardService(
        store, JsonLedger(tmp_path / "ledger.json"), clock=_clock, min_eval_worlds=N_WORLDS
    )
    results = _race(THREADS, lambda i: _open_once(svc, f"key-{i:04d}"))
    assert all(isinstance(r, tuple) for r in results), results
    drawn = [w for ids in results for w in ids]
    assert len(drawn) == THREADS * N_WORLDS == len(set(drawn))
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert len(ledger["used"]["public_eval"]) == THREADS * N_WORLDS
    assert len(ledger["openings"]) == THREADS


# ---------------------------------------------------------------- across processes


def _process_opens(
    store_dir: str, keys_dir: str, ledger: str, key: str, attempts: int, barrier: Any, out: Any
) -> None:
    """Child process: a separate service (its own locks) on the shared ledger file."""
    svc = ScorecardService(
        FileWorldStore(Path(store_dir), Path(keys_dir)),
        JsonLedger(Path(ledger)),
        clock=_clock,
        min_eval_worlds=N_WORLDS,
    )
    barrier.wait()
    results: list[Any] = []
    for _ in range(attempts):
        try:
            results.append(_open_once(svc, key))
        except Exception as exc:  # noqa: BLE001
            results.append(f"crash:{type(exc).__name__}:{exc}")
    out.put(results)


def _process_hammer(ledger: str, key: str, n: int, barrier: Any, out: Any) -> None:
    target = JsonLedger(Path(ledger))
    barrier.wait()
    try:
        for _ in range(n):
            target.record_opening(key, Tier.PUBLIC_TRAIN, "2026-10-03")
    except Exception as exc:  # noqa: BLE001 - report, so the parent fails instead of waiting
        out.put(f"crash:{type(exc).__name__}:{exc}")
        return
    out.put(key)


def _spawn(target: Callable[..., None], argsets: list[tuple[Any, ...]]) -> list[Any]:
    ctx = mp.get_context("spawn")
    barrier, out = ctx.Barrier(len(argsets)), ctx.Queue()
    procs = [ctx.Process(target=target, args=(*args, barrier, out)) for args in argsets]
    for p in procs:
        p.start()
    results = [out.get(timeout=180) for _ in procs]
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    return results


def test_processes_sharing_a_ledger_respect_the_cap_and_freshness(
    eval_store: tuple[Path, Path], tmp_path: Path
) -> None:
    ledger = tmp_path / "ledger.json"
    JsonLedger(ledger)
    store_dir, keys_dir = (str(p) for p in eval_store)
    per_process = _spawn(
        _process_opens, [(store_dir, keys_dir, str(ledger), "shared-key", 4) for _ in range(3)]
    )
    flat = [r for results in per_process for r in results]
    assert not [r for r in flat if isinstance(r, str) and r.startswith("crash")], flat
    ok = [r for r in flat if isinstance(r, tuple)]
    drawn = [w for ids in ok for w in ids]
    assert len(ok) == PUBLIC_EVAL_DAILY_CAP and Counter(flat)["cap_exceeded"] == 12 - PUBLIC_EVAL_DAILY_CAP
    assert len(drawn) == len(set(drawn))
    data = json.loads(ledger.read_text())
    assert sorted(data["used"]["public_eval"]) == sorted(drawn)


def test_processes_sharing_a_ledger_lose_no_update(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    JsonLedger(ledger)
    reported = _spawn(_process_hammer, [(str(ledger), f"k{p}", 60) for p in range(3)])
    assert sorted(reported) == ["k0", "k1", "k2"], reported
    data = json.loads(ledger.read_text())
    assert {k: len(v) for k, v in data["openings"].items()} == {f"k{p}|public_train": 60 for p in range(3)}
