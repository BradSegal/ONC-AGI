"""Swarm: play one scorecard's worlds concurrently, in-process or over HTTP (the ARC ``swarm.py``).

:func:`run_swarm` opens ONE scorecard, plays each world with a fresh agent from the
factory (``agent.close()`` always runs), records every action, closes the scorecard and
returns it with one :class:`RunRecord` per world. The arena is anything with
:class:`ArenaClient`'s ``open/act/state/close/worlds``: the HTTP client itself, or
:class:`LocalArena` over a :class:`ScorecardService`, which returns the same
:class:`Observation` objects the HTTP interface returns, so an agent sees identical
data either way.

A refused action (``ArenaError``) leaves the episode unchanged and goes back to the
agent through :meth:`Agent.on_refused` (which re-raises by default); an agent that
raises ends its world with the error recorded, and the other worlds play on. With
``budget_usd``, once the agents' reported cost exceeds it no new world starts;
unstarted worlds stay unsubmitted and score as empty submissions, as abandoned worlds
always do.
"""

from __future__ import annotations

import tempfile
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from onc_agi.adapters.client import view_from_observation
from onc_agi.core.errors import ArenaError
from onc_agi.core.ports import RecordingHeader, RecordingSink, RunRecord, WorldStore
from onc_agi.core.schema import (
    Action,
    EpisodeStatus,
    Mode,
    Observation,
    Reset,
    Scorecard,
    Submit,
    Tier,
    WorldCard,
)
from onc_agi.infra.ledger import JsonLedger
from onc_agi.services.kit import MAX_STEPS, Agent, EndEpisode, Usage
from onc_agi.services.scorecards import ScorecardService

AgentFactory = Callable[[], Agent]
UNPLAYED = "unplayed: budget_usd reached before this world started"


class Arena(Protocol):
    """The scorecard surface :class:`~onc_agi.adapters.client.ArenaClient` exposes."""

    def open(
        self,
        agent: str,
        tier: Tier,
        n_worlds: int | None = None,
        *,
        track: str = "open",
        mode: Mode | None = None,
        world_ids: Sequence[str] | None = None,
        tags: Sequence[str] = (),
    ) -> tuple[str, list[WorldCard]]: ...

    def act(self, sid: str, wid: str, action: Action) -> Observation: ...

    def state(self, sid: str, wid: str) -> Observation: ...

    def close(self, sid: str) -> Scorecard: ...

    def worlds(self, tier: Tier = Tier.PUBLIC_TRAIN) -> list[WorldCard]: ...


class SwarmRecorder(RecordingSink, Protocol):
    """A recording sink that also takes the scorecard header and per-world run records."""

    def header(self, record: RecordingHeader) -> None: ...

    def run(self, record: RunRecord) -> None: ...


class LocalArena:
    """In-process :class:`Arena` over a :class:`ScorecardService` (no server, same observations)."""

    def __init__(self, service: ScorecardService, api_key: str = "local-arena") -> None:
        self.service = service
        self.api_key = api_key
        self._tmp: tempfile.TemporaryDirectory[str] | None = None

    @classmethod
    def over_store(cls, store: WorldStore) -> LocalArena:
        """A private service with a throwaway ledger (public-train play needs no shared state)."""
        tmp = tempfile.TemporaryDirectory(prefix="arena-local-")
        arena = cls(ScorecardService(store, JsonLedger(Path(tmp.name) / "ledger.json")))
        arena._tmp = tmp  # removed with the arena
        return arena

    def open(
        self,
        agent: str,
        tier: Tier,
        n_worlds: int | None = None,
        *,
        track: str = "open",
        mode: Mode | None = None,
        world_ids: Sequence[str] | None = None,
        tags: Sequence[str] = (),
    ) -> tuple[str, list[WorldCard]]:
        sid, cards = self.service.open(
            self.api_key,
            agent=agent,
            track=track,
            tier=tier,
            n_worlds=n_worlds,
            tags=tuple(tags),
            mode=mode,
            world_ids=tuple(world_ids) if world_ids is not None else None,
        )
        return sid, list(cards)

    def act(self, sid: str, wid: str, action: Action) -> Observation:
        view = self.service.act(sid, wid, action, api_key=self.api_key)
        return view.to_observation(self.service.store.world(wid).patient_ids)

    def state(self, sid: str, wid: str) -> Observation:
        view = self.service.view(sid, wid, api_key=self.api_key)
        return view.to_observation(self.service.store.world(wid).patient_ids)

    def close(
        self,
        sid: str,
        *,
        model: str | None = None,
        harness: str | None = None,
        tokens: int | None = None,
        cost_usd: float | None = None,
    ) -> Scorecard:
        return self.service.close(
            sid, api_key=self.api_key, model=model, harness=harness, tokens=tokens, cost_usd=cost_usd
        )

    def worlds(self, tier: Tier = Tier.PUBLIC_TRAIN) -> list[WorldCard]:
        return list(self.service.list_worlds(tier))


@dataclass(frozen=True)
class SwarmResult:
    scorecard: Scorecard
    runs: tuple[RunRecord, ...]  # one per world, in scorecard order
    unplayed: tuple[str, ...]  # worlds never started because the budget was reached
    budget_reached: bool

    @property
    def errors(self) -> dict[str, str]:
        return {
            r.world_id: r.error for r in self.runs if r.error is not None and r.world_id not in self.unplayed
        }


class _Budget:
    """Cumulative reported cost across worlds; thread-safe."""

    def __init__(self, limit: float | None) -> None:
        self.limit = limit
        self.spent = 0.0
        self._lock = threading.Lock()

    def add(self, cost: float | None) -> None:
        with self._lock:
            self.spent += cost or 0.0

    @property
    def reached(self) -> bool:
        with self._lock:
            return self.limit is not None and self.spent > self.limit


def _summed(usages: Sequence[Usage | None]) -> tuple[int | None, float | None]:
    """Total tokens (None if no world reported any) and cost (None unless every reporting world priced it)."""
    reported = [u for u in usages if u is not None]
    if not reported:
        return None, None
    costs = [u.cost_usd for u in reported]
    return sum(u.tokens for u in reported), None if None in costs else float(sum(c or 0.0 for c in costs))


def play_world(
    arena: Arena,
    sid: str,
    card: WorldCard,
    agent: Agent,
    *,
    max_steps: int = MAX_STEPS,
) -> tuple[RunRecord, Usage | None]:
    """Play one world to the end with ``agent``; never raises for the agent's own failures."""
    wid = card.world_id
    ranking: tuple[str, ...] = ()
    status = EpisodeStatus.ACTIVE
    spent = 0.0
    error: str | None = None

    def apply(action: Action) -> Observation:
        try:
            obs = arena.act(sid, wid, action)
        except ArenaError as exc:
            agent.record(wid, "action", f"{action.model_dump_json()}\nrefused: {exc}", error=True)
            raise
        agent.record(wid, "action", action.model_dump_json())
        return obs

    try:
        obs = apply(Reset(request_id=f"{agent.name}-reset", world_id=wid))
        status, spent = obs.status, obs.spent
        view = view_from_observation(card, obs)
        for _ in range(max_steps):
            if agent.is_done(view):
                break
            try:
                action = agent.choose_action(card, view)
            except EndEpisode as exc:
                agent.record(wid, "note", f"agent ended the world without submitting: {exc}")
                break
            try:
                obs = apply(action)
            except ArenaError as exc:
                agent.on_refused(action, exc)  # the episode is unchanged; the agent may try again
                continue
            status, spent = obs.status, obs.spent
            if isinstance(action, Submit) and status is EpisodeStatus.SUBMITTED:
                ranking = action.ranking
            view = view_from_observation(card, obs)
        else:
            if not agent.is_done(view):
                raise RuntimeError(f"agent {agent.name} did not submit within {max_steps} steps")
    except Exception as exc:  # noqa: BLE001 - one world's failure is recorded; the swarm plays on
        error = f"{type(exc).__name__}: {exc}"
        agent.record(wid, "note", f"world ended by an error: {error}", error=True)
    usage = agent.usage()
    run = RunRecord(
        world_id=wid,
        ranking=ranking,
        submitted=status is EpisodeStatus.SUBMITTED,
        spent=spent,
        tokens=usage.tokens if usage else None,
        cost_usd=usage.cost_usd if usage else None,
        model=usage.model if usage else None,
        error=error,
    )
    return run, usage


def run_swarm(
    arena: Arena,
    agent_factory: AgentFactory,
    *,
    agent_name: str,
    tier: Tier,
    n_worlds: int | None = None,
    world_ids: Sequence[str] | None = None,
    mode: Mode | None = None,
    tags: Sequence[str] = (),
    track: str = "open",
    workers: int = 4,
    recorder: SwarmRecorder | None = None,
    max_steps: int = MAX_STEPS,
    budget_usd: float | None = None,
    model: str | None = None,
    harness: str | None = None,
) -> SwarmResult:
    """Open one scorecard, play its worlds on ``workers`` threads, close it and return the result.

    ``model`` and ``harness`` label the scorecard; tokens and cost are the agents' own reports.
    In-process they are recorded by the service; over HTTP the server's record carries none of
    them (its close takes no client claims), so they annotate the returned copy only.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if budget_usd is not None and budget_usd < 0:
        raise ValueError("budget_usd must be non-negative")
    sid, cards = arena.open(
        agent_name, tier, n_worlds, track=track, mode=mode, world_ids=world_ids, tags=tags
    )
    if recorder is not None:
        recorder.header(
            RecordingHeader(
                scorecard_id=sid,
                agent=agent_name,
                tier=tier,
                track=track,
                world_ids=tuple(c.world_id for c in cards),
                tags=tuple(tags),
                model=model,
                harness=harness,
                opened_at=datetime.now(UTC),
            )
        )
    budget = _Budget(budget_usd)

    def play(card: WorldCard) -> tuple[RunRecord, Usage | None, bool]:
        played, usage = False, None
        if budget.reached:
            run = RunRecord(world_id=card.world_id, error=UNPLAYED)
        else:
            played = True
            try:
                agent = agent_factory()
            except Exception as exc:  # noqa: BLE001 - recorded like any other world failure
                run = RunRecord(world_id=card.world_id, error=f"agent not built: {type(exc).__name__}: {exc}")
            else:
                agent.recorder = recorder
                try:
                    run, usage = play_world(arena, sid, card, agent, max_steps=max_steps)
                finally:
                    agent.close()
                budget.add(usage.cost_usd if usage else None)
        if recorder is not None:
            recorder.run(run)  # as each world ends, so a crash keeps the finished worlds
        return run, usage, played

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="arena-swarm") as pool:
        outcomes = list(pool.map(play, cards))
    runs = [run for run, _, _ in outcomes]
    tokens, cost = _summed([usage for _, usage, _ in outcomes])
    if isinstance(arena, LocalArena):  # the service records the labels with the scorecard
        scorecard = arena.close(sid, model=model, harness=harness, tokens=tokens, cost_usd=cost)
    else:
        labels = {"model": model, "harness": harness, "tokens": tokens, "cost_usd": cost}
        scorecard = arena.close(sid).model_copy(update={k: v for k, v in labels.items() if v is not None})
    return SwarmResult(
        scorecard=scorecard,
        runs=tuple(runs),
        unplayed=tuple(run.world_id for run, _, played in outcomes if not played),
        budget_reached=budget.reached,
    )
