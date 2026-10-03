"""Agent kit: the one execution path for every agent, baseline, cheater and reference.

Mirrors the ARC-AGI-3 pattern (``choose_action`` / ``is_done``) without copying it.
``evaluate`` runs agents over worlds from a :class:`WorldStore` and scores them;
it is used offline (public-train), by the gates (private, with keys) and by the
server when a scorecard closes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from onc_agi.core.digest import canonical_sha256
from onc_agi.core.errors import ArenaError
from onc_agi.core.ports import RecordingKind, RecordingSink, TraceSink, WorldStore
from onc_agi.core.schema import (
    Action,
    Assay,
    EpisodeStatus,
    Mode,
    Recruit,
    Reset,
    Scorecard,
    Submit,
    Tier,
    TraceEvent,
    WorldCard,
    WorldScore,
)
from onc_agi.services import alignment as alignment_service
from onc_agi.services import scoring
from onc_agi.services.engine import Episode, EpisodeView

MAX_STEPS = 10_000


class EndEpisode(Exception):
    """Raised by :meth:`Agent.choose_action` to stop a world without submitting (it scores as empty)."""


@dataclass(frozen=True)
class Usage:
    """What an agent reports having spent on its current world (LLM agents)."""

    tokens: int
    cost_usd: float | None = None
    model: str | None = None


class Agent(ABC):
    """Base agent. Subclasses implement :meth:`choose_action`.

    Optional hooks used by the swarm runner (``adapters/swarm.py``); the defaults keep
    :func:`run_episode` and :func:`evaluate` unchanged:

    * ``recorder`` receives the agent's own transcript through :meth:`record`;
    * :meth:`on_refused` is told when the arena refuses an action (the default re-raises);
    * :meth:`usage` reports tokens and cost; :meth:`close` releases per-world resources.
    """

    name: str = "agent"
    recorder: RecordingSink | None = None

    def __init__(self) -> None:
        self._ids = itertools.count()

    def request_id(self) -> str:
        return f"{self.name}-{next(self._ids)}"

    @abstractmethod
    def choose_action(self, card: WorldCard, view: EpisodeView) -> Action: ...

    def is_done(self, view: EpisodeView) -> bool:
        return view.status is EpisodeStatus.SUBMITTED

    def record(
        self,
        world_id: str,
        kind: RecordingKind,
        content: str,
        *,
        tool: str | None = None,
        error: bool = False,
    ) -> None:
        """Emit one recording event, if a recorder is attached."""
        if self.recorder is not None:
            self.recorder.event(world_id, kind, content, tool=tool, error=error)

    def on_refused(self, action: Action, error: ArenaError) -> None:
        """The arena refused ``action``; the episode is unchanged. Re-raise to end the world."""
        raise error

    def usage(self) -> Usage | None:
        return None

    def close(self) -> None:  # noqa: B027 - an optional hook: most agents hold nothing per world
        """Release per-world resources (sandboxes, connections)."""


@dataclass(frozen=True)
class AnalysisInput:
    """What an analysis function receives: revealed rows and measured columns only."""

    card: WorldCard
    feature_ids: tuple[str, ...]
    x: NDArray[np.float64]
    y: NDArray[np.int64]  # survival worlds: the event indicator
    stratum: tuple[str, ...]
    time: NDArray[np.float64] | None = None  # survival worlds: follow-up in days


Analyst = Callable[[AnalysisInput], Sequence[str]]


def analysis_input(card: WorldCard, view: EpisodeView) -> AnalysisInput:
    cols = [j for j, m in enumerate(view.measured) if m]
    return AnalysisInput(
        card=card,
        feature_ids=tuple(view.feature_ids[j] for j in cols),
        x=view.x[:, cols],
        y=view.outcome,
        stratum=view.stratum,
        time=view.time,
    )


class PipelineAgent(Agent):
    """Wraps ``analyst(AnalysisInput) -> ranking``.

    In sequential mode it uses a fixed design: recruit ``n_fraction`` of every
    stratum's pool, assay every feature, then analyse (the events-per-variable
    style baseline). Budget is truth-blind, so the full pool is always affordable.
    """

    def __init__(self, name: str, analyst: Analyst, *, n_fraction: float = 1.0) -> None:
        super().__init__()
        self.name = name
        self.analyst = analyst
        self.n_fraction = n_fraction
        self._phase = 0

    def choose_action(self, card: WorldCard, view: EpisodeView) -> Action:
        if view.mode is Mode.SEQUENTIAL and not view.rows:
            self._phase = 0  # a new episode: the same agent instance plays many worlds
        if view.mode is Mode.SEQUENTIAL and self._phase < len(card.strata):
            stratum = card.strata[self._phase]
            self._phase += 1
            per_stratum = max(1, round(self.n_fraction * card.n_pool / len(card.strata)))
            return Recruit(request_id=self.request_id(), count=per_stratum, stratum=stratum)
        if view.mode is Mode.SEQUENTIAL and not all(view.measured):
            return Assay(request_id=self.request_id(), feature_ids=card.feature_ids())
        ranking = list(dict.fromkeys(self.analyst(analysis_input(card, view))))
        return Submit(request_id=self.request_id(), ranking=tuple(ranking))


@dataclass(frozen=True)
class EpisodeResult:
    world_id: str
    ranking: tuple[str, ...]
    spent: float
    final_view: EpisodeView
    score: WorldScore | None = None


def view_digest(view: EpisodeView) -> str:
    """Canonical digest of what an action revealed (for traces and replay)."""
    h = hashlib.sha256()
    h.update(
        f"{view.world_id}|{view.step}|{view.status.value}|{view.spent:.6f}|{view.rows}|{view.measured}".encode()
    )
    h.update(np.ascontiguousarray(np.nan_to_num(view.x, nan=-9.87654321e300)).tobytes())
    h.update(np.ascontiguousarray(view.outcome).tobytes())
    if view.time is not None:  # binary digests are unchanged
        h.update(np.ascontiguousarray(view.time).tobytes())
    return h.hexdigest()


def _apply(episode: Episode, action: Action, recorder: TraceSink | None) -> EpisodeView:
    view = episode.apply(action)
    if recorder is not None:
        recorder.record(
            TraceEvent(
                world_id=view.world_id,
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


def run_episode(
    agent: Agent, episode: Episode, *, max_steps: int = MAX_STEPS, recorder: TraceSink | None = None
) -> EpisodeResult:
    """Run one agent to submission. Raises ``RuntimeError`` if it never submits."""
    card = episode.world.card
    view = _apply(episode, Reset(request_id=f"{agent.name}-reset", world_id=card.world_id), recorder)
    for _ in range(max_steps):
        if agent.is_done(view):
            break
        view = _apply(episode, agent.choose_action(card, view), recorder)
    if episode.submission is None:
        raise RuntimeError(f"agent {agent.name} did not submit within {max_steps} steps")
    return EpisodeResult(card.world_id, episode.submission, episode.spent, view)


def evaluate(
    agent: Agent,
    store: WorldStore,
    tier: Tier,
    *,
    world_ids: Sequence[str] | None = None,
    scorecard_id: str | None = None,
    track: str = "open",
    with_alignment: bool = True,
    bootstrap_draws: int = scoring.BOOTSTRAP_DRAWS,
    recorder: TraceSink | None = None,
) -> tuple[Scorecard, list[EpisodeResult]]:
    """Run ``agent`` on worlds and return its scorecard and episode results."""
    ids = tuple(world_ids) if world_ids is not None else store.world_ids(tier)
    results: list[EpisodeResult] = []
    scores: list[WorldScore] = []
    regrets: list[tuple[float, float]] = []
    for world_id in ids:
        world = store.world(world_id)
        key = store.answer_key(world_id)
        result = run_episode(agent, Episode(world), recorder=recorder)
        score = scoring.score_world(
            result.ranking, key, spent=result.spent, sequential=world.card.mode is Mode.SEQUENTIAL
        )
        results.append(dataclasses.replace(result, score=score))
        scores.append(score)
        if with_alignment:
            regrets.append(alignment_service.world_alignment(world, key, result.final_view, score))
    diagnostics = alignment_service.summarise(regrets) if with_alignment else None
    card = scoring.aggregate(
        scores,
        scorecard_id=scorecard_id or f"{agent.name}-{tier.value}-{len(ids)}",
        agent=agent.name,
        tier=tier,
        track=track,
        alignment=diagnostics,
        bootstrap_draws=bootstrap_draws,
    )
    return card, results
