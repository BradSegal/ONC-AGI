"""Conformance suite for custom harnesses.

Runs reference clients over HTTP and checks that (a) remote scorecards equal
in-process scores for the same worlds, (b) a dropped connection can resume from
state, (c) retried requests are not charged twice and conflicting reuse of a
request id is refused, and (d) traces replay exactly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from onc_agi.adapters.agents import make_agent
from onc_agi.adapters.client import ArenaClient
from onc_agi.core.errors import ArenaError
from onc_agi.core.ports import WorldStore
from onc_agi.core.schema import ErrorCode, Mode, Recruit, Reset, Submit, Tier, TraceEvent
from onc_agi.services import scoring
from onc_agi.services.engine import Episode
from onc_agi.services.kit import run_episode
from onc_agi.services.replay import replay

REFERENCE_AGENTS = ("random", "univariate_bh", "giant_list")


@dataclass
class ConformanceReport:
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, str] = field(default_factory=dict)

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks[name] = ok
        if detail:
            self.details[name] = detail

    @property
    def ok(self) -> bool:
        return all(self.checks.values())


def run_conformance(
    client: ArenaClient,
    store: WorldStore,
    *,
    n_worlds: int = 4,
    server_traces: Callable[[str], Sequence[TraceEvent]] | None = None,
) -> ConformanceReport:
    """Run the suite. ``server_traces`` (scorecard id -> recorded events) enables check (d)."""
    report = ConformanceReport()
    for name in REFERENCE_AGENTS:
        sid, cards = client.open(f"conformance-{name}", Tier.PUBLIC_TRAIN, n_worlds)
        local = []
        for card in cards:
            client.play(make_agent(name, store), sid, card)
            result = run_episode(make_agent(name, store), Episode(store.world(card.world_id)))
            key = store.answer_key(card.world_id)
            local.append(
                scoring.score_world(
                    result.ranking, key, spent=result.spent, sequential=card.mode is Mode.SEQUENTIAL
                )
            )
        remote = client.close(sid)
        if server_traces is not None:
            events = list(server_traces(sid))
            replays = [
                replay(
                    [e for e in events if e.world_id == card.world_id],
                    store.world(card.world_id),
                    store.answer_key(card.world_id),
                )
                for card in cards
            ]
            report.check(
                f"server_traces_replay_exactly[{name}]",
                all(r.ok for r in replays)
                and all(
                    r.score is not None and abs(r.score.find - w.find) < 1e-12
                    for r, w in zip(replays, remote.worlds, strict=True)
                ),
                "; ".join(m for r in replays for m in r.mismatches),
            )
        same = all(
            abs(a.find - b.find) < 1e-12 and a.restrained == b.restrained and a.leaked == b.leaked
            for a, b in zip(remote.worlds, local, strict=True)
        )
        report.check(f"remote_equals_local[{name}]", same)

    sid, cards = client.open("conformance-resume", Tier.PUBLIC_TRAIN, len(store.world_ids(Tier.PUBLIC_TRAIN)))
    seq = next((c for c in cards if c.mode is Mode.SEQUENTIAL), None)
    card = seq or cards[0]
    first = client.act(sid, card.world_id, Reset(request_id="r0", world_id=card.world_id))
    resumed = client.state(sid, card.world_id)
    report.check("resume_returns_same_state", resumed == first)
    if seq is not None:
        action = Recruit(request_id="r1", count=5, stratum=seq.strata[0])
        once = client.act(sid, seq.world_id, action)
        twice = client.act(sid, seq.world_id, action)
        report.check("idempotent_retry_not_charged_twice", once.spent == twice.spent and once == twice)
        try:
            client.act(sid, seq.world_id, Recruit(request_id="r1", count=6, stratum=seq.strata[0]))
            report.check("conflicting_request_refused", False)
        except ArenaError as exc:
            report.check("conflicting_request_refused", exc.code is ErrorCode.REQUEST_CONFLICT)
    client.act(sid, card.world_id, Submit(request_id="s", ranking=()))
    try:
        client.act(sid, card.world_id, Submit(request_id="s2", ranking=()))
        report.check("closed_episode_refuses_actions", False)
    except ArenaError as exc:
        report.check("closed_episode_refuses_actions", exc.code is ErrorCode.EPISODE_CLOSED)
    client.close(sid)
    return report
