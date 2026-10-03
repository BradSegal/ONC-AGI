"""Online scorecards: fresh never-reused draws, caps and exposure.

Every public-eval and private scorecard draws worlds that no earlier scorecard
has used, so comparing scorecards reveals nothing about any world. Caps: public
eval at most 5 scorecards per day per key; private at most 3 in total per key;
public train is unlimited and may reuse worlds. Per-world results appear only
for public train. Unsubmitted worlds score as empty submissions.

Hardening (B1-B4):

* ``open`` holds the service lock and ``ledger.lock()`` across check -> draw ->
  mark -> record, so concurrent opens can neither bypass a cap nor reuse a world.
  A ledger without ``lock()`` is accepted but is then safe for one process only.
* A scorecard belongs to the key that opened it (SHA-256 digest; raw keys are never
  stored). ``api_key=None`` is the trusted in-process path; any other key gets
  exactly the error an unknown scorecard id gets, so existence is not revealed.
* With an archive, open records, traces and closed scorecards survive restarts:
  construction replays each open scorecard's trace and checks every response digest.
  Episodes driven directly through :meth:`ScorecardService.episode` are not traced,
  so only actions applied through :meth:`ScorecardService.act` are restored.
* With a TTL, abandoned scorecards are closed by a sweep on every open/act/close.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import secrets
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Protocol

from pydantic import TypeAdapter

from onc_agi.core.digest import canonical_sha256
from onc_agi.core.errors import ArenaError
from onc_agi.core.ports import ClosedRecord, OpenRecord, ScorecardArchive, TraceSink, WorldStore
from onc_agi.core.schema import (
    Action,
    ErrorCode,
    Mode,
    Scorecard,
    Tier,
    TraceEvent,
    WorldCard,
    WorldScore,
)
from onc_agi.services import alignment, scoring
from onc_agi.services.engine import Episode, EpisodeView
from onc_agi.services.kit import view_digest

PUBLIC_EVAL_DAILY_CAP = 5
NULL_SHARE = 0.2
PRIVATE_TOTAL_CAP = 3
MIN_EVAL_WORLDS = 40  # provisional hosted bound: published interface example

_ACTION: TypeAdapter[Action] = TypeAdapter(Action)


class Ledger(Protocol):
    """Durable record of used worlds and opened scorecards."""

    def used(self, tier: Tier) -> set[str]: ...

    def mark_used(self, tier: Tier, world_ids: list[str]) -> None: ...

    def openings(self, api_key: str, tier: Tier) -> list[str]: ...  # ISO dates of openings

    def record_opening(self, api_key: str, tier: Tier, when: str) -> None: ...

    def lock(self) -> AbstractContextManager[None]:
        """Exclusive, reentrant hold over the ledger (optional: see the module docstring)."""
        ...


class RestoreError(RuntimeError):
    """An archived scorecard cannot be rebuilt faithfully from its trace (fail fast)."""


def key_digest(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


@dataclass
class _Open:
    scorecard_id: str
    agent: str
    track: str
    tier: Tier
    world_ids: tuple[str, ...]
    tags: tuple[str, ...]
    owner: str  # SHA-256 of the opening key
    opened_at: datetime
    episodes: dict[str, Episode] = field(default_factory=dict)
    closed: Scorecard | None = None
    expired: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock)


class ScorecardService:
    def __init__(
        self,
        store: WorldStore,
        ledger: Ledger,
        clock: Callable[[], datetime] | None = None,
        *,
        traces: Callable[[str], TraceSink] | None = None,
        pool_commitments: dict[Tier, str] | None = None,
        allowed_keys: frozenset[str] | None = None,
        archive: ScorecardArchive | None = None,
        ttl: timedelta | None = None,
        min_eval_worlds: int = MIN_EVAL_WORLDS,
    ) -> None:
        if ttl is not None and ttl <= timedelta(0):
            raise ValueError("ttl must be positive (use None to disable expiry)")
        if not 1 <= min_eval_worlds <= 10_000:
            raise ValueError("min_eval_worlds must be between 1 and 10000")
        self.store = store
        self.ledger = ledger
        self.clock = clock or (lambda: datetime.now(UTC))
        self.traces = traces  # scorecard id -> sink; every applied action is recorded server-side
        self.pool_commitments = pool_commitments or {}
        self.allowed_keys = allowed_keys  # issued keys (ARC-style); None accepts any key (development only)
        self.archive = archive
        self.ttl = ttl
        self.min_eval_worlds = min_eval_worlds
        self._lock = threading.RLock()  # guards _open and serialises openings in this process
        self._open: dict[str, _Open] = {}
        self._stopped = False
        if archive is not None:
            archive.acquire()
            try:
                self._restore(archive)
            except BaseException:
                archive.release()
                raise

    def shutdown(self) -> None:
        """Stop this service and release the archive after in-flight work finishes."""
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            for sc in self._open.values():
                with sc.lock:
                    pass
            if self.archive is not None:
                self.archive.release()

    def _ensure_active(self) -> None:
        if self._stopped:
            raise RuntimeError("scorecard service has stopped")

    # ------------------------------------------------------------------ open

    def open(
        self,
        api_key: str,
        *,
        agent: str,
        track: str,
        tier: Tier,
        n_worlds: int | None = None,
        tags: tuple[str, ...] = (),
        mode: Mode | None = None,
        world_ids: tuple[str, ...] | None = None,
    ) -> tuple[str, tuple[WorldCard, ...]]:
        """Open a scorecard; ``mode`` restricts the draw to worlds of that mode.

        ``world_ids`` names the worlds outright. Exactly one of ``n_worlds`` and ``world_ids`` is given.
        """
        if (n_worlds is None) == (world_ids is None):
            raise ArenaError(ErrorCode.INVALID_PAYLOAD, "give exactly one of n_worlds and world_ids")
        if world_ids is not None:
            if tier is not Tier.PUBLIC_TRAIN:
                raise ArenaError(ErrorCode.INVALID_PAYLOAD, "world selection is public-train only")
            known = set(self.store.world_ids(tier))
            missing = [w for w in world_ids if w not in known]
            if missing or not world_ids or len(set(world_ids)) != len(world_ids):
                raise ArenaError(
                    ErrorCode.UNKNOWN_WORLD, f"not distinct public-train worlds: {missing or world_ids}"
                )
        if self.allowed_keys is not None and api_key not in self.allowed_keys:
            raise ArenaError(
                ErrorCode.INVALID_PAYLOAD, "unknown API key; keys are issued by the arena operators"
            )
        if track not in ("standard", "open"):
            raise ArenaError(ErrorCode.INVALID_PAYLOAD, "track must be 'standard' or 'open'")
        if n_worlds is not None and not 1 <= n_worlds <= 10_000:
            raise ArenaError(ErrorCode.INVALID_PAYLOAD, "n_worlds must be between 1 and 10000")
        if tier is not Tier.PUBLIC_TRAIN and (n_worlds or 0) < self.min_eval_worlds:
            raise ArenaError(
                ErrorCode.INVALID_PAYLOAD, f"eval scorecards require at least {self.min_eval_worlds} worlds"
            )
        self._sweep()
        with self._lock, self._ledger_lock():
            self._ensure_active()
            now = self.clock()
            today = now.date().isoformat()
            history = self.ledger.openings(key_digest(api_key), tier)  # keys never rest on disk
            if tier is Tier.PUBLIC_EVAL and sum(1 for d in history if d == today) >= PUBLIC_EVAL_DAILY_CAP:
                raise ArenaError(
                    ErrorCode.CAP_EXCEEDED, f"at most {PUBLIC_EVAL_DAILY_CAP} public-eval scorecards per day"
                )
            if tier is Tier.PRIVATE and len(history) >= PRIVATE_TOTAL_CAP:
                raise ArenaError(
                    ErrorCode.CAP_EXCEEDED, f"at most {PRIVATE_TOTAL_CAP} private scorecards in total"
                )
            available = world_ids if world_ids is not None else self.store.world_ids(tier)
            n_worlds = len(available) if n_worlds is None else n_worlds
            if mode is not None:
                available = tuple(w for w in available if self.store.card(w).mode is mode)
            if tier is not Tier.PUBLIC_TRAIN:
                used = self.ledger.used(tier)
                available = tuple(w for w in available if w not in used)
            if len(available) < n_worlds:
                raise ArenaError(
                    ErrorCode.CAP_EXCEEDED, f"only {len(available)} unused worlds remain in {tier.value}"
                )
            if tier is Tier.PUBLIC_TRAIN:
                chosen = available[:n_worlds]
            else:
                chosen = self._stratified_draw(available, n_worlds)
                self.ledger.mark_used(tier, list(chosen))
            self.ledger.record_opening(key_digest(api_key), tier, today)
            scorecard_id = f"sc-{secrets.token_hex(8)}"
            sc = _Open(
                scorecard_id, agent, track, tier, chosen, tuple(tags), key_digest(api_key), opened_at=now
            )
            if self.archive is not None:
                self.archive.save_open(
                    OpenRecord(
                        scorecard_id=scorecard_id,
                        agent=agent,
                        track=track,
                        tier=tier,
                        world_ids=chosen,
                        tags=sc.tags,
                        owner_sha256=sc.owner,
                        opened_at=now,
                    )
                )
            self._open[scorecard_id] = sc
        return scorecard_id, tuple(self.store.card(w) for w in chosen)

    def list_worlds(self, tier: Tier) -> tuple[WorldCard, ...]:
        """Cards of every public-train world (the counterpart of ARC's game list). Eval tiers are
        never listed: their worlds are drawn fresh and stay unknown until drawn."""
        if tier is not Tier.PUBLIC_TRAIN:
            raise ArenaError(ErrorCode.INVALID_PAYLOAD, "only public-train worlds are listed")
        return tuple(self.store.card(w) for w in self.store.world_ids(tier))

    def _ledger_lock(self) -> AbstractContextManager[None]:
        lock = getattr(self.ledger, "lock", None)  # third-party ledgers may lack it: single-process only
        return lock() if callable(lock) else contextlib.nullcontext()

    def _stratified_draw(self, available: tuple[str, ...], n_worlds: int) -> tuple[str, ...]:
        """Fresh draw with the fixed composition: round(20% x n) null worlds and
        the rest spread as evenly as possible across difficulty tiers."""
        rng = secrets.SystemRandom()
        keys = {w: self.store.answer_key(w) for w in available}
        nulls = [w for w in available if keys[w].is_null]
        by_tier: dict[int, list[str]] = {}
        for w in available:
            if not keys[w].is_null:
                by_tier.setdefault(keys[w].difficulty_tier, []).append(w)
        n_null = round(NULL_SHARE * n_worlds)
        if len(nulls) < n_null:
            raise ArenaError(
                ErrorCode.CAP_EXCEEDED, "not enough unused null worlds for the fixed composition"
            )
        chosen = rng.sample(nulls, n_null)
        pools = {t: rng.sample(ws, len(ws)) for t, ws in by_tier.items()}
        tiers = sorted(pools)
        need = n_worlds - n_null
        i = 0
        while need > 0 and any(pools.values()):
            tier_pool = pools[tiers[i % len(tiers)]]
            if tier_pool:
                chosen.append(tier_pool.pop())
                need -= 1
            i += 1
        if need > 0:
            raise ArenaError(ErrorCode.CAP_EXCEEDED, "not enough unused signal worlds for the requested draw")
        return tuple(sorted(chosen))

    # ------------------------------------------------------------------ lookup and ownership

    def _get(self, scorecard_id: str, api_key: str | None) -> _Open:
        with self._lock:
            self._ensure_active()
            sc = self._open.get(scorecard_id)
        if sc is None and self.archive is not None:
            record = self.archive.load_closed(scorecard_id)
            if record is not None:
                sc = _Open(
                    scorecard_id,
                    record.scorecard.agent,
                    record.scorecard.track,
                    record.scorecard.tier,
                    (),
                    record.scorecard.tags,
                    record.owner_sha256,
                    opened_at=record.closed_at,
                    closed=record.scorecard,
                    expired=record.expired,
                )
        if sc is None or (api_key is not None and not hmac.compare_digest(sc.owner, key_digest(api_key))):
            # a foreign key sees exactly what an unknown id sees
            raise ArenaError(ErrorCode.INVALID_PAYLOAD, f"unknown scorecard {scorecard_id!r}")
        return sc

    @staticmethod
    def _refuse_closed(sc: _Open, message: str) -> None:
        if sc.closed is not None:
            suffix = " (expired: the server closed it after its time limit)" if sc.expired else ""
            raise ArenaError(ErrorCode.SCORECARD_CLOSED, message + suffix)

    def _episode(self, sc: _Open, world_id: str) -> Episode:
        self._refuse_closed(sc, "scorecard is closed")
        if world_id not in sc.world_ids:
            raise ArenaError(ErrorCode.UNKNOWN_WORLD, f"{world_id!r} is not part of this scorecard")
        if world_id not in sc.episodes:
            sc.episodes[world_id] = Episode(self.store.world(world_id))
        return sc.episodes[world_id]

    # ------------------------------------------------------------------ play

    def episode(self, scorecard_id: str, world_id: str, *, api_key: str | None = None) -> Episode:
        sc = self._get(scorecard_id, api_key)
        with sc.lock:
            self._ensure_active()
            return self._episode(sc, world_id)

    def view(self, scorecard_id: str, world_id: str, *, api_key: str | None = None) -> EpisodeView:
        """Current state of one world (resume), read under the scorecard's lock."""
        sc = self._get(scorecard_id, api_key)
        with sc.lock:
            self._ensure_active()
            return self._episode(sc, world_id).view()

    def act(
        self, scorecard_id: str, world_id: str, action: Action, *, api_key: str | None = None
    ) -> EpisodeView:
        self._sweep()
        sc = self._get(scorecard_id, api_key)
        with sc.lock:  # apply and record in one step, so the trace order is the apply order
            self._ensure_active()
            view = self._episode(sc, world_id).apply(action)
            event = TraceEvent(
                scorecard_id=scorecard_id,
                world_id=world_id,
                step=view.step,
                action_kind=action.kind,
                request_id=action.request_id,
                request_sha256=canonical_sha256(action.model_dump(mode="json")),
                response_sha256=view_digest(view),
                spent=view.spent,
                action_json=action.model_dump_json(),
            )
            if self.archive is not None:
                self.archive.trace(scorecard_id).record(event)
            if self.traces is not None:
                self.traces(scorecard_id).record(event)
        return view

    # ------------------------------------------------------------------ close and results

    def close(
        self,
        scorecard_id: str,
        *,
        api_key: str | None = None,
        model: str | None = None,
        harness: str | None = None,
        tokens: int | None = None,
        cost_usd: float | None = None,
    ) -> Scorecard:
        self._sweep()
        sc = self._get(scorecard_id, api_key)
        return self._close(sc, model=model, harness=harness, tokens=tokens, cost_usd=cost_usd)

    def _close(
        self,
        sc: _Open,
        *,
        expired: bool = False,
        model: str | None = None,
        harness: str | None = None,
        tokens: int | None = None,
        cost_usd: float | None = None,
    ) -> Scorecard:
        with sc.lock:
            self._ensure_active()
            self._refuse_closed(sc, "scorecard is already closed")
            scores: list[WorldScore] = []
            pairs: list[tuple[float, float]] = []
            oracle_versions: set[str] = set()
            for world_id in sc.world_ids:
                world = self.store.world(world_id)
                episode = sc.episodes.get(world_id) or Episode(world)
                key = self.store.answer_key(world_id)
                oracle_versions.add(key.oracle_version)
                ranking = episode.submission if episode.submission is not None else ()
                score = scoring.score_world(
                    ranking, key, spent=episode.spent, sequential=world.card.mode is Mode.SEQUENTIAL
                )
                scores.append(score)
                pairs.append(alignment.world_alignment(world, key, episode.view(), score))
            closed = scoring.aggregate(
                scores,
                scorecard_id=sc.scorecard_id,
                agent=sc.agent,
                tier=sc.tier,
                track=sc.track,
                tags=sc.tags,
                alignment=alignment.summarise(pairs),
                pool_commitment=self.pool_commitments.get(sc.tier),
                model=model,
                harness=harness,
                tokens=tokens,
                cost_usd=cost_usd,
                extra_versions={"oracle": ",".join(sorted(oracle_versions))},
            )
            if self.archive is not None:
                self.archive.save_closed(
                    ClosedRecord(
                        scorecard=closed, owner_sha256=sc.owner, closed_at=self.clock(), expired=expired
                    )
                )
            sc.closed, sc.expired = closed, expired
            sc.episodes = {}  # unreachable once closed
            return closed

    def scorecard(self, scorecard_id: str, *, api_key: str | None = None) -> Scorecard:
        """The closed scorecard on record: exactly what ``close`` returned.

        A scorecard that is still open is refused with ``action_not_available``.
        """
        sc = self._get(scorecard_id, api_key)
        if sc.closed is None:
            raise ArenaError(ErrorCode.ACTION_NOT_AVAILABLE, "scorecard is still open; close it first")
        return sc.closed

    # ------------------------------------------------------------------ expiry and restore

    def _sweep(self) -> None:
        """Close every open scorecard older than the TTL (unsubmitted worlds score as empty)."""
        if self.ttl is None:
            return
        deadline = self.clock() - self.ttl
        with self._lock:
            stale = [sc for sc in self._open.values() if sc.closed is None and sc.opened_at <= deadline]
        for sc in stale:
            with contextlib.suppress(ArenaError):  # closed concurrently by its owner
                self._close(sc, expired=True)

    def _restore(self, archive: ScorecardArchive) -> None:
        for record in archive.open_records():
            closed = archive.load_closed(record.scorecard_id)
            if closed is not None:  # crashed between writing the closed record and retiring the open one
                archive.save_closed(closed)
                continue
            sc = _Open(
                record.scorecard_id,
                record.agent,
                record.track,
                record.tier,
                record.world_ids,
                record.tags,
                record.owner_sha256,
                opened_at=record.opened_at,
            )
            for event in archive.events(record.scorecard_id):
                where = f"scorecard {record.scorecard_id} world {event.world_id} step {event.step}"
                if event.world_id not in sc.world_ids or event.action_json is None:
                    raise RestoreError(
                        f"{where}: trace event does not belong to this scorecard or has no action"
                    )
                if event.world_id not in sc.episodes:
                    sc.episodes[event.world_id] = Episode(self.store.world(event.world_id))
                try:
                    view = sc.episodes[event.world_id].apply(_ACTION.validate_json(event.action_json))
                except ArenaError as exc:
                    raise RestoreError(f"{where}: replayed action was refused ({exc})") from exc
                if view_digest(view) != event.response_sha256:
                    raise RestoreError(f"{where}: replayed response differs from the recorded digest")
            self._open[record.scorecard_id] = sc
