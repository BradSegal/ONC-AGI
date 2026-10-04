"""Replay verification: recompute an episode and its score from a trace.

A trace records each action's exact payload with request and response digests.
Replaying the actions against the world bundle must reproduce every digest and
the final score; any mismatch is reported, never silently ignored.

:func:`verify_run` checks a whole run the same way for both tracks. It fails on anything
missing (no recording header, no runs, no scorecard) and on any disagreement:

* the server trace replays exactly against the world bundles;
* the run's worlds are the opened set: the recording header, its runs, the scorecard's count
  (and, on public train, its worlds) and the traced worlds all agree;
* the whole scorecard (every aggregate, interval, per-tier row, alignment and version, and the
  per-world scores on public train) is recomputed from the replay by the scorer itself;
* the recording agrees with the trace: submissions, spend and, where it lists them, actions.

Given the server's own copies of the trace and scorecard, the local files must equal them and
the replay uses the server's trace: the run is then *verified against the server*. Without them
the run can only be *consistent*: its files agree with each other and with the world bundles.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import TypeAdapter, ValidationError

from onc_agi.core.digest import canonical_sha256
from onc_agi.core.errors import ArenaError
from onc_agi.core.ports import RecordingEvent, RecordingHeader, RunRecord, WorldStore
from onc_agi.core.schema import Action, AnswerKey, Mode, Scorecard, TraceEvent, WorldScore
from onc_agi.core.world import WorldData
from onc_agi.services import alignment, scoring
from onc_agi.services.engine import Episode, EpisodeView
from onc_agi.services.kit import view_digest

_ACTION: TypeAdapter[Action] = TypeAdapter(Action)
CLAIMS = ("model", "harness", "tokens", "cost_usd")  # client labels a server's close does not record


@dataclass(frozen=True)
class ReplayReport:
    world_id: str
    steps: int
    mismatches: tuple[str, ...]
    score: WorldScore | None
    submission: tuple[str, ...] | None = None
    spent: float = 0.0
    final_view: EpisodeView | None = None

    @property
    def ok(self) -> bool:
        return not self.mismatches


def replay(
    events: Sequence[TraceEvent], world: WorldData, key: AnswerKey, *, require_submission: bool = True
) -> ReplayReport:
    """Replay one world's trace. Without ``require_submission`` an unsubmitted world scores as an
    empty submission at its spend, as the server scores it at close. A malformed or illegal traced
    action is a mismatch, and the replay stops there."""
    mismatches: list[str] = []
    episode = Episode(world)
    for event in events:
        if event.action_json is None:
            mismatches.append(f"step {event.step}: trace has no action payload")
            break
        try:
            action = _ACTION.validate_json(event.action_json)
        except ValidationError as exc:
            mismatches.append(f"step {event.step}: unreadable action ({exc.error_count()} errors)")
            break
        if canonical_sha256(action.model_dump(mode="json")) != event.request_sha256:
            mismatches.append(f"step {event.step}: request digest differs")
        try:
            view = episode.apply(action)
        except ArenaError as exc:
            mismatches.append(f"step {event.step}: the arena refuses this action ({exc})")
            break
        if view_digest(view) != event.response_sha256:
            mismatches.append(f"step {event.step}: response digest differs")
        if abs(view.spent - event.spent) > 1e-6:
            mismatches.append(f"step {event.step}: spend {view.spent} != recorded {event.spent}")
    sequential = world.card.mode is Mode.SEQUENTIAL
    score = None
    if episode.submission is not None:
        score = scoring.score_world(episode.submission, key, spent=episode.spent, sequential=sequential)
    elif require_submission:
        mismatches.append("episode was never submitted")
    else:
        score = scoring.score_world((), key, spent=episode.spent, sequential=sequential)
    return ReplayReport(
        world.card.world_id,
        len(events),
        tuple(mismatches),
        score,
        episode.submission,
        episode.spent,
        episode.view(),
    )


def trace_digest(trace: Sequence[TraceEvent]) -> str:
    """SHA-256 of a trace's canonical JSON, to compare a local copy with the server's."""
    return canonical_sha256([e.model_dump(mode="json") for e in trace])


@dataclass(frozen=True)
class RunVerification:
    """Per-world replays of a run's server trace, every disagreement found, and what it was checked against."""

    reports: tuple[ReplayReport, ...]
    mismatches: tuple[str, ...]
    authority: Literal["server", "local"] = "local"

    @property
    def ok(self) -> bool:
        return not self.mismatches

    def summary(self) -> str:
        steps = sum(r.steps for r in self.reports)
        if not self.ok:
            return f"NOT VERIFIED: {len(self.mismatches)} mismatch(es): " + "; ".join(self.mismatches)
        if self.authority == "server":
            return f"verified against the server: {len(self.reports)} worlds, {steps} traced actions replayed"
        return (
            f"consistent: {len(self.reports)} worlds, {steps} traced actions replayed"
            " (local files only; not checked against the server)"
        )


def _recorded_actions(events: Sequence[RecordingEvent], mismatches: list[str]) -> dict[str, list[str]]:
    """Request digests of the actions a recording says the arena applied, per world, in order."""
    out: dict[str, list[str]] = {}
    for event in events:
        if event.kind == "action" and not event.error:
            try:
                action = _ACTION.validate_python(json.loads(event.content.split("\n", 1)[0]))
            except (ValueError, ValidationError):
                mismatches.append(f"{event.world_id}: recorded action at turn {event.turn} is not an action")
                continue
            out.setdefault(event.world_id, []).append(canonical_sha256(action.model_dump(mode="json")))
    return out


def _differing(a: Scorecard, b: Scorecard, *, ignore: Sequence[str] = ()) -> list[str]:
    da, db = a.model_dump(mode="json"), b.model_dump(mode="json")
    return sorted(k for k in da.keys() | db.keys() if k not in ignore and da.get(k) != db.get(k))


def verify_run(
    trace: Sequence[TraceEvent],
    store: WorldStore,
    *,
    scorecard: Scorecard | None,
    header: RecordingHeader | None,
    runs: Sequence[RunRecord] = (),
    events: Sequence[RecordingEvent] = (),
    server_trace: Sequence[TraceEvent] | None = None,
    server_scorecard: Scorecard | None = None,
) -> RunVerification:
    """Verify a run (see the module docstring). ``store`` needs every world's answer key."""
    mismatches: list[str] = []
    authority: Literal["server", "local"] = "local"
    if server_trace is not None:
        authority = "server"
        if trace_digest(trace) != trace_digest(server_trace):
            mismatches.append("the local trace is not the server's (digests differ)")
        trace = server_trace  # the server's record is the one replayed
    if server_scorecard is not None and scorecard is not None:
        claims = [c for c in CLAIMS if getattr(server_scorecard, c) is None]  # an HTTP close records none
        fields = _differing(scorecard, server_scorecard, ignore=claims)
        if fields:
            mismatches.append(f"the local scorecard is not the server's: {', '.join(fields)} differ")
    if header is None:
        mismatches.append("the recording has no scorecard header, so the run's worlds are unknown")
    if not runs:
        mismatches.append("the recording lists no runs")
    if scorecard is None:
        mismatches.append("the run has no scorecard")
    if header is None or scorecard is None:
        return RunVerification((), tuple(mismatches), authority)

    world_ids = list(header.world_ids)
    sids: set[str | None] = {scorecard.scorecard_id, header.scorecard_id}
    sids |= {e.scorecard_id for e in trace}
    sids |= {r.scorecard_id for r in runs if r.scorecard_id is not None}
    sids |= {e.scorecard_id for e in events if e.scorecard_id is not None}
    if len(sids) > 1:
        mismatches.append(
            f"the trace, scorecard and recording name different scorecards: {sorted(map(str, sids))}"
        )
    if len(set(world_ids)) != len(world_ids):
        mismatches.append("the recording header lists a world twice")
    if scorecard.n_worlds != len(world_ids):
        mismatches.append(f"the scorecard counts {scorecard.n_worlds} worlds, the recording {len(world_ids)}")
    if scorecard.worlds and [w.world_id for w in scorecard.worlds] != world_ids:
        mismatches.append("the scorecard lists other worlds than the recording header")
    recorded = [r.world_id for r in runs]
    if sorted(recorded) != sorted(world_ids):
        missing, extra = set(world_ids) - set(recorded), set(recorded) - set(world_ids)
        mismatches.append(
            f"the recorded runs are not the run's worlds (missing {sorted(missing)}, extra {sorted(extra)})"
        )
    traced = {e.world_id for e in trace}
    if traced - set(world_ids):
        mismatches.append(f"the trace holds worlds outside the run: {sorted(traced - set(world_ids))}")

    reports: list[ReplayReport] = []
    for wid in world_ids:
        try:
            world, key = store.world(wid), store.answer_key(wid)
        except ArenaError as exc:
            mismatches.append(f"{wid}: cannot replay ({exc})")
            continue
        reports.append(replay([e for e in trace if e.world_id == wid], world, key, require_submission=False))
    mismatches += [f"{r.world_id}: {m}" for r in reports for m in r.mismatches]
    by_world = {r.world_id: r for r in reports}

    if len(reports) == len(world_ids) and all(r.score is not None for r in reports):
        recomputed = scoring.aggregate(
            [r.score for r in reports if r.score is not None],
            scorecard_id=scorecard.scorecard_id,
            agent=scorecard.agent,
            tier=scorecard.tier,
            track=scorecard.track,
            tags=scorecard.tags,
            alignment=alignment.summarise(
                [
                    alignment.world_alignment(
                        store.world(r.world_id), store.answer_key(r.world_id), r.final_view, r.score
                    )
                    for r in reports
                    if r.final_view is not None and r.score is not None
                ]
            ),
            pool_commitment=scorecard.pool_commitment,
            model=scorecard.model,
            harness=scorecard.harness,
            tokens=scorecard.tokens,
            cost_usd=scorecard.cost_usd,
            extra_versions={
                "oracle": ",".join(sorted({store.answer_key(w).oracle_version for w in world_ids}))
            },
        )
        fields = _differing(scorecard, recomputed)
        if fields:
            mismatches.append(f"the scorecard is not the replayed one: {', '.join(fields)} differ")

    for run in runs:
        report = by_world.get(run.world_id)
        if report is None:
            continue
        if run.submitted != (report.submission is not None):
            mismatches.append(f"{run.world_id}: recorded submitted={run.submitted}, the trace disagrees")
        elif run.submitted and tuple(run.ranking) != report.submission:
            mismatches.append(f"{run.world_id}: the recorded ranking is not the traced submission")
        if abs(run.spent - report.spent) > 1e-6:
            mismatches.append(f"{run.world_id}: recorded spend {run.spent} != traced {report.spent}")
    traced_requests: dict[str, list[str]] = {}
    for e in trace:
        traced_requests.setdefault(e.world_id, []).append(e.request_sha256)
    for wid, digests in _recorded_actions(events, mismatches).items():
        if digests != traced_requests.get(wid, []):
            mismatches.append(f"{wid}: the recorded actions are not the traced ones")
    return RunVerification(tuple(reports), tuple(mismatches), authority)
