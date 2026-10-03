"""Explanations classify each world's outcome with the scorer's own credit (fixture worlds)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from arena_factories import InMemoryStore, planted_world
from onc_agi.adapters.agents import make_agent
from onc_agi.adapters.cli import fixture_store
from onc_agi.core.ports import RecordingEvent, RecordingKind, RunRecord
from onc_agi.core.schema import GroupLabel, Tier
from onc_agi.infra.bundles import FileWorldStore
from onc_agi.services import explain as ex
from onc_agi.services.kit import evaluate

STORE = FileWorldStore(fixture_store())
IDS = STORE.world_ids(Tier.PUBLIC_TRAIN)


def event(
    wid: str, kind: RecordingKind, content: str, turn: int = 0, tool: str | None = None
) -> RecordingEvent:
    return RecordingEvent(
        world_id=wid, turn=turn, kind=kind, content=content, tool=tool, ts=datetime.now(UTC)
    )


def _signal_world() -> tuple[str, tuple[str, ...], str]:
    for wid in IDS:
        key = STORE.answer_key(wid)
        parts = [p for g in key.groups if g.label is GroupLabel.RECOVERABLE for p in g.parts]
        if not key.is_null and parts and STORE.card(wid).mode.value == "full_access":
            truths = tuple(dict.fromkeys(p.true_feature for p in parts))
            planted = {f for g in key.groups for p in g.parts for f in p.equivalence_set} | set(
                key.reject_set
            )
            planted |= {f for f, c in key.clusters.items() if any(key.clusters.get(t) == c for t in planted)}
            decoy = next(f for f in STORE.card(wid).feature_ids() if f not in planted)
            return wid, truths, decoy
    raise AssertionError("fixtures need a signal world")


def _explain(wid: str, said: str, listed: tuple[str, ...], *, submitted: bool = True) -> ex.Explanation:
    events = [event(wid, "tool_result", said, tool="python")]
    (x,) = ex.explain_run(events, [RunRecord(world_id=wid, ranking=listed, submitted=submitted)], STORE)
    return x


def test_found_dropped_never_surfaced_and_abstained() -> None:
    wid, truths, decoy = _signal_world()
    truth = truths[0]
    assert _explain(wid, f"top: {truth}", truths).outcome == "found"
    dropped = _explain(wid, f"top: {truth} z=4", (decoy,))
    assert dropped.outcome == "dropped" and truth in dropped.surfaced_not_submitted
    assert dropped.truth_first_surfaced_turn == 0 and "selection failure" in dropped.summary
    assert _explain(wid, "nothing notable", (decoy,)).outcome == "never_surfaced"
    assert _explain(wid, "nothing", ()).outcome == "abstained_on_signal"
    unsubmitted = _explain(wid, f"top: {truth}", truths, submitted=False)
    assert unsubmitted.outcome == "no_submission" and unsubmitted.ranking == ()


def test_null_worlds_and_leaks() -> None:
    null = next(w for w in IDS if STORE.answer_key(w).is_null)
    assert _explain(null, "flat", ()).outcome == "correct_abstain"
    claim = STORE.card(null).feature_ids()[0]
    assert _explain(null, "flat", (claim,)).outcome == "false_claim"
    leaky = next(w for w in IDS if STORE.answer_key(w).reject_set and not STORE.answer_key(w).is_null)
    leak = STORE.answer_key(leaky).reject_set[0]
    x = _explain(leaky, f"{leak} predicts best", (leak,))
    assert x.outcome == "leak_listed" and x.leaks_listed == (leak,) and leak in x.leaks_mentioned


def test_a_full_listing_does_not_count_as_surfacing() -> None:
    ids = [f"F{i}" for i in range(30)]
    listing = "\n".join(f"{f} 0.0{i}" for i, f in enumerate(ids))
    assert ex.salient(listing, ids) == {"F0", "F1", "F2", "F3", "F4"}
    assert ex.salient("top: F17 z=5.2, F3 z=4.1", ids) == {"F17", "F3"}
    wid, _, decoy = _signal_world()
    key = STORE.answer_key(wid)
    credited = {f for g in key.groups for part in g.parts for f in part.equivalence_set}
    others = [f for f in STORE.card(wid).feature_ids() if f not in credited][: ex.SALIENT]
    table = "\n".join(f"{f} 0.1" for f in [*others, *STORE.card(wid).feature_ids()])  # every column printed
    assert _explain(wid, table, (decoy,)).outcome == "never_surfaced"


def test_reasoning_names_count_but_the_agents_own_code_and_actions_do_not() -> None:
    wid, truths, decoy = _signal_world()
    truth = truths[0]
    events = [
        event(wid, "tool_call", f"df['{truth}']", 0, "python"),
        event(wid, "action", f'{{"kind":"submit","ranking":["{truth}"]}}', 1),
    ]
    run = RunRecord(world_id=wid, ranking=(decoy,), submitted=True)
    assert ex.explain(events, run, STORE).outcome == "never_surfaced"
    events.append(event(wid, "reasoning", f"I suspect {truth} drives it. Moving on.", 2))
    x = ex.explain(events, run, STORE)
    assert x.outcome == "dropped" and x.truth_first_reasoned_turn == 2
    assert x.quotes == (f"turn 2: I suspect {truth} drives it.",)
    assert x.tool_calls == {"python": 1} and x.actions == {"submit": 1}


def test_explanations_find_equals_the_scorers() -> None:
    _, results = evaluate(make_agent("univariate_bh", STORE), STORE, Tier.PUBLIC_TRAIN)
    for result in results:
        run = RunRecord(world_id=result.world_id, ranking=result.ranking, submitted=True, spent=result.spent)
        x = ex.explain([], run, STORE)
        assert result.score is not None
        if not result.score.is_null:
            assert x.find == pytest.approx(result.score.find)
            assert x.raw_recovery == pytest.approx(result.score.raw_recovery)


def test_non_public_worlds_need_the_operator_and_are_marked() -> None:
    store = InMemoryStore.of(planted_world("e-00", signal=True, seed=1, tier=Tier.PUBLIC_EVAL))
    run = RunRecord(world_id="e-00", ranking=("f00",), submitted=True)
    with pytest.raises(PermissionError, match="public-train only"):
        ex.explain([], run, store)
    x = ex.explain([], run, store, operator=True)
    assert x.operator_only and x.outcome == "found"
    assert "OPERATOR ONLY" in ex.report_markdown([x], title="t")


def test_reports_render_markdown_and_json() -> None:
    wid, truths, decoy = _signal_world()
    xs = [_explain(wid, f"top: {truths[0]}", truths), _explain(wid, "nothing", (decoy,))]
    md = ex.report_markdown(xs, title="Run")
    assert md.startswith("# Run") and "| found | 1 |" in md and "OPERATOR" not in md
    import json

    assert [d["outcome"] for d in json.loads(ex.report_json(xs))] == ["found", "never_surfaced"]
