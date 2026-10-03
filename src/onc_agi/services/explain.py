"""Explain each world's result from what the agent did (recordings and Inspect logs).

Every harness is reduced to one event stream per world (:class:`RecordingEvent`) plus a
:class:`RunRecord`; :func:`explain` joins them with the world's answer key and
classifies the outcome by *where* the agent went wrong:

* ``found`` / ``partial``: the scorer credited all / some of the world's depth R (the
  credit is the arena scorer's own, recomputed here, so explanations never disagree
  with the scorecard);
* ``too_deep``: credited features were listed, but after the first R cluster
  representatives, where the scorer stops reading;
* ``dropped``: a credited feature surfaced in the agent's own analysis output or
  reasoning but was not submitted (a selection failure);
* ``never_surfaced``: no credited feature ever appeared (a search failure);
* ``leak_listed``: a post-outcome feature was submitted, which zeroes the world;
* ``abstained_on_signal``, ``correct_abstain``, ``false_claim`` (null worlds);
* ``no_submission``: the world ended without a submission (it scores as empty).

Answer keys are read here, so explanations are operator material. Per-world results
may only be shown for public-train worlds; other tiers need ``operator=True``
and the report is marked operator-only.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence

from pydantic import Field

from onc_agi.core.ports import RecordingEvent, RunRecord, WorldStore
from onc_agi.core.schema import Frozen, GroupLabel, Mode, Tier
from onc_agi.services import scoring

LISTING = 10  # an output naming more features than this is a listing, not a finding
SALIENT = 5  # in a listing, only the first features named count as surfaced (a ranked table's top)

HINTS = {
    "interaction": "the effect exists only jointly, so marginal screens rarely show it;"
    " credit needs the pair",
    "module": "several co-regulated features share one effect; any one member earns the group's credit",
    "effect_modifier": "the effect differs by subgroup, so pooled tests dilute it",
    "mediator": "the effect runs through an intermediate feature",
    "confounder": "a shared cause links a decoy to the outcome; adjusting for it separates driver from decoy",
    "hidden_cause": "an unmeasured cause produces correlated stand-ins",
    "collider": "selection on a common effect creates a spurious association",
    "mixture": "different subgroups carry contradictory mechanisms",
    "shift": "the mechanism differs across cohorts",
    "wrong_type": "the signal sits in a different data type than the obvious one",
    "stand_in": "a correlated stand-in shares credit with the true driver",
}

_SPEECH = ("assistant", "reasoning")  # the agent's own words


class Explanation(Frozen):
    world_id: str
    tier: Tier
    is_null: bool
    outcome: str
    summary: str
    ranking: tuple[str, ...]
    mechanism: str
    depth: int
    raw_recovery: float
    chance_recovery: float
    find: float | None = Field(description="The scorer's Find for this world (None on null worlds).")
    truth_first_surfaced_turn: int | None
    truth_first_reasoned_turn: int | None
    surfaced_not_submitted: tuple[str, ...]
    leaks_listed: tuple[str, ...]
    leaks_mentioned: tuple[str, ...]
    turns: int
    tool_calls: dict[str, int]
    actions: dict[str, int]
    tool_errors: int
    tokens: int | None
    cost_usd: float | None
    error: str | None
    operator_only: bool
    quotes: tuple[str, ...] = ()


def _positions(text: str, ids: Iterable[str]) -> dict[str, int]:
    """First position of each feature id named in ``text`` (whole-token matches only)."""
    found: dict[str, int] = {}
    for f in ids:
        m = re.search(rf"(?<![A-Za-z0-9_]){re.escape(f)}(?![A-Za-z0-9_])", text)
        if m:
            found[f] = m.start()
    return found


def mentions(text: str, ids: Iterable[str]) -> set[str]:
    return set(_positions(text, ids))


def salient(text: str, ids: Iterable[str]) -> set[str]:
    """Features an output actually brings forward. Printing every column (or every feature's
    statistic in arbitrary order) does not surface the truth; heading a ranked table does."""
    pos = _positions(text, ids)
    if len(pos) <= LISTING:
        return set(pos)
    return {f for f, _ in sorted(pos.items(), key=lambda kv: kv[1])[:SALIENT]}


def _quote(events: Sequence[RecordingEvent], features: set[str], limit: int = 2) -> tuple[str, ...]:
    """Sentences of the agent's own words that name the given features."""
    out: list[str] = []
    for event in events:
        if event.kind not in _SPEECH:
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", event.content):
            if mentions(sentence, features):
                out.append(f"turn {event.turn}: {sentence.strip()[:240]}")
                if len(out) >= limit:
                    return tuple(out)
    return tuple(out)


def explain(
    events: Sequence[RecordingEvent], run: RunRecord, store: WorldStore, *, operator: bool = False
) -> Explanation:
    """Classify one world's outcome. Raises ``PermissionError`` for a non-public world unless ``operator``."""
    card = store.card(run.world_id)
    if card.tier is not Tier.PUBLIC_TRAIN and not operator:
        raise PermissionError(
            f"{run.world_id} is a {card.tier.value} world; per-world explanations are public-train only"
        )
    key = store.answer_key(run.world_id)
    ids = card.feature_ids()
    leaks = set(key.reject_set)
    parts = [p for g in key.groups if g.label is GroupLabel.RECOVERABLE for p in g.parts]
    truth = {f for p in parts for f in p.equivalence_set}
    ranking = run.ranking if run.submitted else ()  # an unsubmitted world scores as empty
    listed = set(ranking)
    score = scoring.score_world(ranking, key, spent=run.spent, sequential=card.mode is Mode.SEQUENTIAL)

    surfaced_turn: int | None = None
    reasoned_turn: int | None = None
    surfaced: set[str] = set()
    mentioned_leaks: set[str] = set()
    for e in events:
        mentioned_leaks |= mentions(e.content, leaks)
        if e.kind == "tool_result":
            hits = salient(e.content, ids) & truth
            if hits:
                surfaced |= hits
                surfaced_turn = e.turn if surfaced_turn is None else surfaced_turn
        elif e.kind in _SPEECH:
            hits = mentions(e.content, ids) & truth
            if hits:
                surfaced |= hits
                reasoned_turn = e.turn if reasoned_turn is None else reasoned_turn
    dropped = tuple(sorted(surfaced - listed))
    leaks_listed = tuple(sorted(listed & leaks))

    raw, chance = score.raw_recovery, score.chance_recovery
    if not run.submitted:
        outcome = "no_submission"
    elif key.is_null:
        outcome = "correct_abstain" if score.restrained else "false_claim"
    elif leaks_listed:
        outcome = "leak_listed"
    elif score.abstained:
        outcome = "abstained_on_signal"
    elif raw >= 1 - 1e-9:
        outcome = "found"
    elif raw > 0:
        outcome = "partial"
    elif listed & truth:
        outcome = "too_deep"
    elif dropped:
        outcome = "dropped"
    else:
        outcome = "never_surfaced"

    efficiency = (
        f", at efficiency {score.efficiency:.2f} of the reference cost."
        if card.mode is Mode.SEQUENTIAL
        else "."
    )
    summary = {
        "found": f"Full credit within depth {key.depth} (chance {chance:.2f}, Find {score.find:.2f}).",
        "partial": f"Credit {raw:.2f} of depth {key.depth} (chance {chance:.2f}, Find {score.find:.2f}).",
        "too_deep": f"Listed {', '.join(sorted(listed & truth))}, but after the first {key.depth} cluster"
        " representatives, where credit stops.",
        "dropped": f"Its own analysis surfaced {', '.join(dropped)} but the list omitted them:"
        " a selection failure.",
        "never_surfaced": "No credited feature appeared in any analysis output or reasoning:"
        " a search failure.",
        "leak_listed": f"Listed post-outcome feature(s) {', '.join(leaks_listed)}, which zeroes the world.",
        "abstained_on_signal": "Submitted nothing on a world with recoverable signal."
        + (" It had surfaced " + ", ".join(dropped) + "." if dropped else ""),
        "correct_abstain": "Null world: correctly submitted nothing" + efficiency,
        "false_claim": f"Null world: claimed {len(listed)} feature(s) where nothing is recoverable.",
        "no_submission": "The world ended without a submission"
        + (f" ({run.error})." if run.error else " (turn, step or time limit)."),
    }[outcome]
    roles = sorted(
        {f"{g.role} ({g.credit_rule.value} credit)" for g in key.groups if g.label is GroupLabel.RECOVERABLE}
    )
    mechanism = ", ".join(roles) or ("null" if key.is_null else "no recoverable group")
    missed = outcome in ("never_surfaced", "abstained_on_signal", "too_deep", "partial", "dropped")
    hints = sorted(
        {HINTS[g.role] for g in key.groups if g.label is GroupLabel.RECOVERABLE and g.role in HINTS}
    )
    if missed and hints:
        summary += " Mechanism: " + mechanism + ": " + "; ".join(hints) + "."
    quote_targets = set(dropped) | set(leaks_listed) | (listed if outcome == "false_claim" else set())
    return Explanation(
        world_id=run.world_id,
        tier=card.tier,
        is_null=key.is_null,
        outcome=outcome,
        summary=summary,
        ranking=ranking,
        mechanism=mechanism,
        depth=key.depth,
        raw_recovery=raw,
        chance_recovery=chance,
        find=None if key.is_null else score.find,
        truth_first_surfaced_turn=surfaced_turn,
        truth_first_reasoned_turn=reasoned_turn,
        surfaced_not_submitted=dropped,
        leaks_listed=leaks_listed,
        leaks_mentioned=tuple(sorted(mentioned_leaks)),
        turns=max((e.turn for e in events), default=-1) + 1,
        tool_calls=dict(Counter(e.tool or "?" for e in events if e.kind == "tool_call")),
        actions=dict(Counter(_action_kind(e.content) for e in events if e.kind == "action" and not e.error)),
        tool_errors=sum(e.error for e in events if e.kind in ("tool_result", "action")),
        tokens=run.tokens,
        cost_usd=run.cost_usd,
        error=run.error,
        operator_only=card.tier is not Tier.PUBLIC_TRAIN,
        quotes=_quote(events, quote_targets),
    )


def _action_kind(content: str) -> str:
    try:
        return str(json.loads(content.split("\n", 1)[0]).get("kind", "?"))
    except (ValueError, AttributeError):
        return "?"


def explain_run(
    events: Sequence[RecordingEvent], runs: Sequence[RunRecord], store: WorldStore, *, operator: bool = False
) -> list[Explanation]:
    by_world: dict[str, list[RecordingEvent]] = {}
    for e in events:
        by_world.setdefault(e.world_id, []).append(e)
    return [explain(by_world.get(r.world_id, []), r, store, operator=operator) for r in runs]


def report_markdown(explanations: Sequence[Explanation], *, title: str) -> str:
    counts = Counter(x.outcome for x in explanations)
    out = [f"# {title}", ""]
    if any(x.operator_only for x in explanations):
        out += ["> **OPERATOR ONLY.** Contains per-world results for non-public worlds. Do not share.", ""]
    out += ["| Outcome | Worlds |", "|---|---|"] + [f"| {k} | {v} |" for k, v in counts.most_common()]
    out += [
        "",
        "| World | Mechanism | Outcome | Find | Surfaced at turn | Turns | Tools | Explanation |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for x in explanations:
        tools = ", ".join(f"{k}×{v}" for k, v in sorted(x.tool_calls.items()))
        find = "" if x.find is None else format(x.find, ".2f")
        surfaced = "" if x.truth_first_surfaced_turn is None else x.truth_first_surfaced_turn
        out.append(
            f"| {x.world_id} | {x.mechanism} | {x.outcome} | {find} | {surfaced} | {x.turns} |"
            f" {tools} | {x.summary} |"
        )
    quoted = [x for x in explanations if x.quotes]
    if quoted:
        out += ["", "## In the agent's words", ""]
        for x in quoted:
            out.append(f"**{x.world_id}** ({x.outcome})")
            out += [f"> {q}" for q in x.quotes]
            out.append("")
    return "\n".join(out) + "\n"


def report_json(explanations: Sequence[Explanation]) -> str:
    return json.dumps([x.model_dump(mode="json") for x in explanations], indent=1)
