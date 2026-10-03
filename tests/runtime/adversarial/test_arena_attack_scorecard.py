"""Adaptive scorecard probing against the exposure rules.

These tests never import ``private_generator``: the public scorecard service must defend its
exposure rules on its own. Worlds come from the public ``arena_factories`` helpers.

They establish that, used as designed (sequentially, under the caps), fresh never-reused
draws make each world's result unobservable and recovery stays at chance; that per-world
results are exposed only for public train. They exercise operator-configured small
fixtures rather than the hosted minimum. Optional issued keys remain a development
exposure; the former opening race is covered by an ordinary regression.
"""

from __future__ import annotations

import json
import threading
from collections import Counter
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from arena_factories import planted_world
from onc_agi.core.errors import ArenaError
from onc_agi.core.schema import Submit, Tier
from onc_agi.infra.bundles import FileWorldStore, write_world
from onc_agi.infra.ledger import JsonLedger
from onc_agi.services.scorecards import (
    PRIVATE_TOTAL_CAP,
    PUBLIC_EVAL_DAILY_CAP,
    ScorecardService,
)

NULL_RATE = 0.2


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 3, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, days: int) -> None:
        self.now = self.now + timedelta(days=days)


def _pool(root, n, seed, tier):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    store_root, keys = root / "store", root / "keys"
    for i in range(n):
        is_null = bool(rng.random() < NULL_RATE)
        world, key = planted_world(
            f"{tier.value}-{i:04d}",
            signal=not is_null,
            seed=seed * 9973 + i,
            n_pool=int(rng.integers(60, 140)),
            tier=tier,
            difficulty_tier=int(rng.integers(0, 3)),
        )
        write_world(store_root, world, key, keys_dir=keys)
    return FileWorldStore(store_root, keys)


@pytest.fixture(scope="module")
def eval_pool(tmp_path_factory):  # type: ignore[no-untyped-def]
    return _pool(tmp_path_factory.mktemp("eval"), 120, 202, Tier.PUBLIC_EVAL)


@pytest.fixture(scope="module")
def private_pool(tmp_path_factory):  # type: ignore[no-untyped-def]
    return _pool(tmp_path_factory.mktemp("priv"), 40, 203, Tier.PRIVATE)


def _service(store, tmp_path, clock, allowed=None):  # type: ignore[no-untyped-def]
    return ScorecardService(
        store, JsonLedger(tmp_path / "ledger.json"), clock=clock, allowed_keys=allowed, min_eval_worlds=1
    )


def test_public_eval_draws_are_fresh_and_never_reused(eval_pool, tmp_path) -> None:
    """Across the daily cap over many days, no world is drawn twice."""
    clock = _Clock()
    svc = _service(eval_pool, tmp_path, clock)
    drawn: list[str] = []
    singleton_scorecards = 0
    for _ in range(40):
        for _ in range(PUBLIC_EVAL_DAILY_CAP):
            try:
                _, cards = svc.open("k", agent="p", track="open", tier=Tier.PUBLIC_EVAL, n_worlds=5)
            except ArenaError:
                break
            drawn.extend(c.world_id for c in cards)
            singleton_scorecards += len(cards) == 1
        clock.advance(1)
    counts = Counter(drawn)
    assert max(counts.values()) == 1, "a world was reused across scorecards"
    # with multi-world draws no scorecard aggregate equals a single world's own result
    assert singleton_scorecards == 0


def test_daily_cap_and_private_total_cap_hold_sequentially(eval_pool, private_pool, tmp_path) -> None:
    clock = _Clock()
    svc = _service(eval_pool, tmp_path, clock)
    opened = 0
    for _ in range(PUBLIC_EVAL_DAILY_CAP + 3):
        try:
            svc.open("k", agent="p", track="open", tier=Tier.PUBLIC_EVAL, n_worlds=5)
            opened += 1
        except ArenaError:
            break
    assert opened == PUBLIC_EVAL_DAILY_CAP
    priv = _service(private_pool, tmp_path / "p", _Clock())
    popened = 0
    for _ in range(PRIVATE_TOTAL_CAP + 3):
        try:
            priv.open("k", agent="p", track="open", tier=Tier.PRIVATE, n_worlds=5)
            popened += 1
        except ArenaError:
            break
    assert popened == PRIVATE_TOTAL_CAP


def test_small_draws_never_contain_a_null_and_never_expose_per_world_results(eval_pool, tmp_path) -> None:
    """At draws below 1/NULL_RATE the fixed composition yields only signal worlds, and eval
    scorecards still expose no per-world ``worlds`` list, so a specific world's null status
    cannot be isolated from a small draw."""
    clock = _Clock()
    svc = _service(eval_pool, tmp_path, clock)
    for draw in (1, 2):
        scorecard_id, cards = svc.open("k", agent="p", track="open", tier=Tier.PUBLIC_EVAL, n_worlds=draw)
        for c in cards:
            assert not eval_pool.answer_key(c.world_id).is_null
            svc.act(scorecard_id, c.world_id, Submit(request_id=f"s-{c.world_id}", ranking=()))
        closed = svc.close(scorecard_id)
        assert closed.worlds == ()  # eval: no per-world results
        clock.advance(1)


def test_recovery_of_eval_null_status_from_card_knobs_stays_at_chance(eval_pool, tmp_path) -> None:
    """A classifier trained on public-train cards+labels cannot beat balanced-accuracy chance
    on eval worlds: card-visible knobs carry no null-status signal, and fresh draws block any
    aggregate channel."""
    from sklearn.linear_model import LogisticRegression

    train = _pool(tmp_path / "train", 120, 777, Tier.PUBLIC_TRAIN)

    def feats(store, tier):  # type: ignore[no-untyped-def]
        ids = list(store.world_ids(tier))
        x = np.array(
            [[store.card(w).n_pool, len(store.card(w).features), store.card(w).budget] for w in ids], float
        )
        y = np.array([float(store.answer_key(w).is_null) for w in ids])
        return x, y

    xtr, ytr = feats(train, Tier.PUBLIC_TRAIN)
    xev, yev = feats(eval_pool, Tier.PUBLIC_EVAL)
    mu, sd = xtr.mean(0), np.where(xtr.std(0) == 0, 1.0, xtr.std(0))
    pred = LogisticRegression(max_iter=1000).fit((xtr - mu) / sd, ytr).predict((xev - mu) / sd)
    accs = [float((pred[yev == c] == c).mean()) for c in (0.0, 1.0) if (yev == c).any()]
    balanced = float(np.mean(accs))
    assert abs(balanced - 0.5) <= 0.1, f"balanced accuracy {balanced:.3f} beats chance"


def test_self_asserted_keys_mint_fresh_caps_known_open_decision_d16(eval_pool, tmp_path) -> None:
    """D16 (reported, not fixed): with no issued-key allow-list, rotating invented keys each get
    their own daily cap, so the per-key cap does not bound total scorecards. An allow-list closes
    it, but making issued keys mandatory is the programme lead's decision."""
    clock = _Clock()
    no_allowlist = _service(eval_pool, tmp_path / "a", clock)
    minted = 0
    for k in range(6):
        try:
            no_allowlist.open(f"invented-{k}", agent="p", track="open", tier=Tier.PUBLIC_EVAL, n_worlds=5)
            minted += 1
        except ArenaError:
            break
    assert minted > PUBLIC_EVAL_DAILY_CAP  # more than one key's daily cap, same day

    allow = _service(eval_pool, tmp_path / "b", _Clock(), allowed=frozenset({"issued"}))
    with pytest.raises(ArenaError):
        allow.open("not-issued", agent="p", track="open", tier=Tier.PUBLIC_EVAL, n_worlds=5)


@pytest.mark.slow
def test_concurrent_opens_respect_cap_and_freshness(eval_pool, tmp_path) -> None:
    """The daily cap holds with actual successful draws, no world is reused, and the
    ledger stays valid JSON. Regression for the former check-then-mark race."""
    threads = 16
    for trial in range(12):
        sub = tmp_path / f"c{trial}"
        svc = _service(eval_pool, sub, _Clock())
        results: list[object] = []
        barrier = threading.Barrier(threads)

        def go(svc=svc, results=results, barrier=barrier) -> None:  # type: ignore[no-untyped-def]
            barrier.wait()
            try:
                _, cards = svc.open("one-key", agent="p", track="open", tier=Tier.PUBLIC_EVAL, n_worlds=5)
                results.append(tuple(c.world_id for c in cards))
            except ArenaError as exc:
                results.append(exc.code.value)
            except Exception as exc:  # noqa: BLE001 - characterising the crash is the point
                results.append(type(exc).__name__)

        ts = [threading.Thread(target=go) for _ in range(threads)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        ok = [r for r in results if isinstance(r, tuple)]
        drawn = [w for ids in ok for w in ids]
        ledger_valid = True
        try:
            json.loads((sub / "ledger.json").read_text())
        except Exception:  # noqa: BLE001
            ledger_valid = False
        assert len(ok) == PUBLIC_EVAL_DAILY_CAP, "cap bypassed or successful opens not exercised"
        assert set(r for r in results if not isinstance(r, tuple)) == {"cap_exceeded"}
        assert len(drawn) == len(set(drawn)), "worlds reused across scorecards under concurrency"
        assert ledger_valid, "ledger corrupted under concurrency"
