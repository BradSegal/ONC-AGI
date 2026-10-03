"""Online scorecards: fresh never-reused draws, caps and exposure.

Every public-eval and private scorecard draws worlds that no earlier scorecard
has used, so comparing scorecards reveals nothing about any world. Caps: public
eval at most 5 scorecards per day per key; private at most 3 in total per key;
public train is unlimited and may reuse worlds. Per-world results appear only
for public train. Unsubmitted worlds score as empty submissions.
"""

from __future__ import annotations

import hashlib
import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from onc_agi.core.digest import canonical_sha256
from onc_agi.core.errors import ArenaError
from onc_agi.core.ports import TraceSink, WorldStore
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


class Ledger(Protocol):
    """Durable record of used worlds and opened scorecards."""

    def used(self, tier: Tier) -> set[str]: ...

    def mark_used(self, tier: Tier, world_ids: list[str]) -> None: ...

    def openings(self, api_key: str, tier: Tier) -> list[str]: ...  # ISO dates of openings

    def record_opening(self, api_key: str, tier: Tier, when: str) -> None: ...


@dataclass
class _Open:
    scorecard_id: str
    agent: str
    track: str
    tier: Tier
    world_ids: tuple[str, ...]
    tags: tuple[str, ...]
    episodes: dict[str, Episode] = field(default_factory=dict)
    closed: Scorecard | None = None
    owner: str = ""  # sha256 of the opening key; only that key may act, read or close


def _owner(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


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
    ) -> None:
        self.store = store
        self.ledger = ledger
        self.clock = clock or (lambda: datetime.now(UTC))
        self.traces = traces  # scorecard id -> sink; every applied action is recorded server-side
        self.pool_commitments = pool_commitments or {}
        self.allowed_keys = allowed_keys  # issued keys (ARC-style); None accepts any key (development only)
        self._open: dict[str, _Open] = {}
        # one server process owns a ledger (arena serve runs a single worker); within it, openings
        # are serialised so concurrent requests cannot both pass a cap or draw the same world
        self._opening = threading.Lock()

    def open(
        self, api_key: str, *, agent: str, track: str, tier: Tier, n_worlds: int, tags: tuple[str, ...] = ()
    ) -> tuple[str, tuple[WorldCard, ...]]:
        if self.allowed_keys is not None and api_key not in self.allowed_keys:
            raise ArenaError(
                ErrorCode.INVALID_PAYLOAD, "unknown API key; keys are issued by the arena operators"
            )
        if track not in ("standard", "open"):
            raise ArenaError(ErrorCode.INVALID_PAYLOAD, "track must be 'standard' or 'open'")
        with self._opening:  # check caps, draw, mark used and record as one critical section
            return self._open_locked(
                api_key, agent=agent, track=track, tier=tier, n_worlds=n_worlds, tags=tags
            )

    def _open_locked(
        self, api_key: str, *, agent: str, track: str, tier: Tier, n_worlds: int, tags: tuple[str, ...]
    ) -> tuple[str, tuple[WorldCard, ...]]:
        today = self.clock().date().isoformat()
        history = self.ledger.openings(api_key, tier)
        if tier is Tier.PUBLIC_EVAL and sum(1 for d in history if d == today) >= PUBLIC_EVAL_DAILY_CAP:
            raise ArenaError(
                ErrorCode.CAP_EXCEEDED, f"at most {PUBLIC_EVAL_DAILY_CAP} public-eval scorecards per day"
            )
        if tier is Tier.PRIVATE and len(history) >= PRIVATE_TOTAL_CAP:
            raise ArenaError(
                ErrorCode.CAP_EXCEEDED, f"at most {PRIVATE_TOTAL_CAP} private scorecards in total"
            )
        available = self.store.world_ids(tier)
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
        self.ledger.record_opening(api_key, tier, today)
        scorecard_id = f"sc-{secrets.token_hex(8)}"
        self._open[scorecard_id] = _Open(
            scorecard_id, agent, track, tier, chosen, tuple(tags), owner=_owner(api_key)
        )
        return scorecard_id, tuple(self.store.card(w) for w in chosen)

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

    def _get(self, scorecard_id: str, api_key: str | None = None) -> _Open:
        """The scorecard, if it exists and (when a key is given) belongs to that key.

        A scorecard opened by another key is reported exactly like an unknown one, so a
        guessed or leaked id reveals nothing and cannot be acted on, read or closed.
        """
        sc = self._open.get(scorecard_id)
        if sc is None or (api_key is not None and sc.owner != _owner(api_key)):
            raise ArenaError(ErrorCode.INVALID_PAYLOAD, f"unknown scorecard {scorecard_id!r}")
        return sc

    def episode(self, scorecard_id: str, world_id: str, *, api_key: str | None = None) -> Episode:
        sc = self._get(scorecard_id, api_key)
        if sc.closed is not None:
            raise ArenaError(ErrorCode.SCORECARD_CLOSED, "scorecard is closed")
        if world_id not in sc.world_ids:
            raise ArenaError(ErrorCode.UNKNOWN_WORLD, f"{world_id!r} is not part of this scorecard")
        if world_id not in sc.episodes:
            sc.episodes[world_id] = Episode(self.store.world(world_id))
        return sc.episodes[world_id]

    def act(
        self, scorecard_id: str, world_id: str, action: Action, *, api_key: str | None = None
    ) -> EpisodeView:
        view = self.episode(scorecard_id, world_id, api_key=api_key).apply(action)
        if self.traces is not None:
            self.traces(scorecard_id).record(
                TraceEvent(
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
            )
        return view

    def close(self, scorecard_id: str, *, api_key: str | None = None) -> Scorecard:
        sc = self._get(scorecard_id, api_key)
        if sc.closed is not None:
            raise ArenaError(ErrorCode.SCORECARD_CLOSED, "scorecard is already closed")
        scores: list[WorldScore] = []
        pairs: list[tuple[float, float]] = []
        for world_id in sc.world_ids:
            world = self.store.world(world_id)
            episode = sc.episodes.get(world_id) or Episode(world)
            key = self.store.answer_key(world_id)
            ranking = episode.submission if episode.submission is not None else ()
            score = scoring.score_world(
                ranking, key, spent=episode.spent, sequential=world.card.mode is Mode.SEQUENTIAL
            )
            scores.append(score)
            pairs.append(alignment.world_alignment(world, key, episode.view(), score))
        sc.closed = scoring.aggregate(
            scores,
            scorecard_id=scorecard_id,
            agent=sc.agent,
            tier=sc.tier,
            track=sc.track,
            tags=sc.tags,
            alignment=alignment.summarise(pairs),
            pool_commitment=self.pool_commitments.get(sc.tier),
        )
        return sc.closed
