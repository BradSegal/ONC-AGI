"""Replay verification: recompute an episode and its score from a trace.

A trace records each action's exact payload with request and response digests.
Replaying the actions against the world bundle must reproduce every digest and
the final score; any mismatch is reported, never silently ignored.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import TypeAdapter

from onc_agi.core.digest import canonical_sha256
from onc_agi.core.schema import Action, AnswerKey, Mode, TraceEvent, WorldScore
from onc_agi.core.world import WorldData
from onc_agi.services import scoring
from onc_agi.services.engine import Episode
from onc_agi.services.kit import view_digest

_ACTION: TypeAdapter[Action] = TypeAdapter(Action)


@dataclass(frozen=True)
class ReplayReport:
    world_id: str
    steps: int
    mismatches: tuple[str, ...]
    score: WorldScore | None

    @property
    def ok(self) -> bool:
        return not self.mismatches


def replay(events: Sequence[TraceEvent], world: WorldData, key: AnswerKey) -> ReplayReport:
    mismatches: list[str] = []
    episode = Episode(world)
    for event in events:
        if event.action_json is None:
            mismatches.append(f"step {event.step}: trace has no action payload")
            break
        action = _ACTION.validate_json(event.action_json)
        if canonical_sha256(action.model_dump(mode="json")) != event.request_sha256:
            mismatches.append(f"step {event.step}: request digest differs")
        view = episode.apply(action)
        if view_digest(view) != event.response_sha256:
            mismatches.append(f"step {event.step}: response digest differs")
        if abs(view.spent - event.spent) > 1e-6:
            mismatches.append(f"step {event.step}: spend {view.spent} != recorded {event.spent}")
    score = None
    if episode.submission is not None:
        score = scoring.score_world(
            episode.submission, key, spent=episode.spent, sequential=world.card.mode is Mode.SEQUENTIAL
        )
    else:
        mismatches.append("episode was never submitted")
    return ReplayReport(world.card.world_id, len(events), tuple(mismatches), score)
