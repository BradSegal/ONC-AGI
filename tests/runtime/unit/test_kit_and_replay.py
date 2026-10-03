"""Agent kit, trace digests and replay verification."""

from __future__ import annotations

import numpy as np
import pytest
from arena_factories import InMemoryStore, fid, make_card, make_world, planted_world
from onc_agi.core.schema import Action, Assay, Mode, Recruit, Submit, TraceEvent, WorldCard
from onc_agi.services.engine import Episode, EpisodeView
from onc_agi.services.kit import (
    Agent,
    PipelineAgent,
    analysis_input,
    evaluate,
    run_episode,
    view_digest,
)
from onc_agi.services.replay import replay


class ListSink:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def record(self, event: TraceEvent) -> None:
        self.events.append(event)


def first_feature(data) -> list[str]:  # type: ignore[no-untyped-def]
    return [data.feature_ids[0]] if data.feature_ids else []


class NeverSubmits(Agent):
    name = "stuck"

    def choose_action(self, card: WorldCard, view: EpisodeView) -> Action:
        return Assay(request_id=self.request_id(), feature_ids=(card.feature_ids()[0],))


def test_full_access_pipeline_agent_submits_its_analysis_immediately() -> None:
    world, _ = planted_world("w-00", signal=True, seed=1)
    result = run_episode(PipelineAgent("first", first_feature), Episode(world))
    assert result.ranking == (fid(0),)
    assert result.spent == 0.0
    assert result.final_view.step == 2  # reset + submit


def test_sequential_pipeline_agent_recruits_assays_then_submits() -> None:
    world, _ = planted_world("w-00", signal=True, seed=1, mode=Mode.SEQUENTIAL, n_pool=40)
    sink = ListSink()
    result = run_episode(PipelineAgent("first", first_feature, n_fraction=0.5), Episode(world), recorder=sink)
    kinds = [e.action_kind for e in sink.events]
    assert kinds == ["reset", "recruit", "assay", "submit"]
    assert len(result.final_view.rows) == 20
    assert result.spent == pytest.approx(20 * 1.0 + 20 * 0.5 * 8)


def test_the_same_agent_instance_restarts_its_design_on_each_sequential_world() -> None:
    agent = PipelineAgent("first", first_feature, n_fraction=0.25)
    for seed in (1, 2):
        world, _ = planted_world(f"w-{seed:02d}", signal=True, seed=seed, mode=Mode.SEQUENTIAL, n_pool=40)
        assert len(run_episode(agent, Episode(world)).final_view.rows) == 10


def test_an_agent_that_never_submits_fails_loudly() -> None:
    world = make_world(make_card(mode=Mode.SEQUENTIAL, n_pool=10))
    with pytest.raises(RuntimeError, match="did not submit"):
        run_episode(NeverSubmits(), Episode(world), max_steps=5)


def test_analysis_input_contains_only_measured_columns() -> None:
    world = make_world(make_card(mode=Mode.SEQUENTIAL, n_pool=10, n_features=4))
    ep = Episode(world)
    ep.apply(Recruit(request_id="a", count=5))
    view = ep.apply(Assay(request_id="b", feature_ids=(fid(2),)))
    data = analysis_input(world.card, view)
    assert data.feature_ids == (fid(2),) and data.x.shape == (5, 1)
    np.testing.assert_array_equal(data.y, world.y[list(view.rows)])


def test_view_digest_changes_with_any_revealed_value() -> None:
    world = make_world(make_card(n_pool=10))
    view = Episode(world).view()
    altered = make_world(make_card(n_pool=10), x=world.x.copy(), y=world.y)
    altered.x[3, 2] += 1e-9
    assert view_digest(view) == view_digest(Episode(make_world(make_card(n_pool=10))).view())
    assert view_digest(view) != view_digest(Episode(altered).view())


def test_evaluate_scores_every_world_through_the_same_path(store: InMemoryStore) -> None:
    from onc_agi.core.schema import Tier

    card, results = evaluate(
        PipelineAgent("first", first_feature), store, Tier.PUBLIC_TRAIN, bootstrap_draws=50
    )
    assert card.n_worlds == len(results) == 6
    assert [r.world_id for r in results] == list(store.world_ids(Tier.PUBLIC_TRAIN))
    assert all(r.score is not None for r in results)
    assert card.alignment is not None


def _traced_episode(mode: Mode) -> tuple[list[TraceEvent], InMemoryStore, str]:
    world, key = planted_world("w-00", signal=True, seed=1, mode=mode, n_pool=40)
    store = InMemoryStore.of((world, key))
    sink = ListSink()
    run_episode(PipelineAgent("first", first_feature, n_fraction=0.5), Episode(world), recorder=sink)
    return sink.events, store, "w-00"


@pytest.mark.parametrize("mode", [Mode.FULL_ACCESS, Mode.SEQUENTIAL])
def test_a_recorded_episode_replays_exactly(mode: Mode) -> None:
    events, store, wid = _traced_episode(mode)
    report = replay(events, store.world(wid), store.answer_key(wid))
    assert report.ok, report.mismatches
    assert report.steps == len(events) and report.score is not None
    assert report.score.raw_recovery == 1.0


def test_replay_detects_a_tampered_action() -> None:
    events, store, wid = _traced_episode(Mode.SEQUENTIAL)
    forged = Submit(request_id=events[-1].request_id, ranking=(fid(3),)).model_dump_json()
    events[-1] = events[-1].model_copy(update={"action_json": forged})
    report = replay(events, store.world(wid), store.answer_key(wid))
    assert not report.ok and any("request digest" in m for m in report.mismatches)


def test_replay_detects_a_tampered_spend() -> None:
    events, store, wid = _traced_episode(Mode.SEQUENTIAL)
    events[1] = events[1].model_copy(update={"spent": events[1].spent + 1})
    report = replay(events, store.world(wid), store.answer_key(wid))
    assert any("spend" in m for m in report.mismatches)


def test_replay_against_different_world_data_reports_response_mismatches() -> None:
    events, _store, _wid = _traced_episode(Mode.SEQUENTIAL)
    other, key = planted_world("w-00", signal=True, seed=2, mode=Mode.SEQUENTIAL, n_pool=40)
    report = replay(events, other, key)
    assert any("response digest" in m for m in report.mismatches)


def test_replay_requires_payloads_and_a_submission() -> None:
    events, store, wid = _traced_episode(Mode.SEQUENTIAL)
    stripped = [events[0].model_copy(update={"action_json": None})]
    assert "no action payload" in replay(stripped, store.world(wid), store.answer_key(wid)).mismatches[0]
    unfinished = replay(events[:-1], store.world(wid), store.answer_key(wid))
    assert "never submitted" in unfinished.mismatches[-1] and unfinished.score is None
