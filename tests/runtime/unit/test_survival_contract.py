"""Survival outcomes and missing cells in the public runtime.

Binary payloads must serialise exactly as interface v1.0 did; survival payloads add only
``horizon_days`` (card) and ``time`` (revealed data). The Cox score statistic is checked
against statsmodels, and the Cox-aware baselines against planted survival worlds.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from arena_factories import HORIZON, InMemoryStore, fid, make_card, planted_world, survival_world
from onc_agi.adapters.agents import GroupSequentialAgent, baselines, make_agent
from onc_agi.core.schema import Mode, Observation, Reset, RevealedData, Tier, WorldCard
from onc_agi.core.world import WorldData
from onc_agi.services import score_test
from onc_agi.services.engine import Episode
from onc_agi.services.kit import analysis_input, evaluate, run_episode, view_digest
from pydantic import ValidationError

V1_CARD_KEYS = {
    "interface_version",
    "world_id",
    "tier",
    "mode",
    "outcome_type",
    "n_pool",
    "features",
    "strata",
    "stratum_sizes",
    "prices",
    "budget",
    "name_visibility",
    "premise",
}
V1_REVEALED_KEYS = {"patient_ids", "outcome", "stratum", "columns"}


# ---------------------------------------------------------------- contract


def test_binary_cards_and_observations_serialise_with_exactly_the_v1_fields() -> None:
    world, _ = planted_world("w-bin", signal=True, seed=1)
    assert set(json.loads(world.card.model_dump_json())) == V1_CARD_KEYS
    obs = Episode(world).apply(Reset(request_id="r", world_id="w-bin")).to_observation(world.patient_ids)
    assert set(json.loads(obs.model_dump_json())["revealed"]) == V1_REVEALED_KEYS
    assert set(obs.revealed.model_dump()) == V1_REVEALED_KEYS


def test_survival_payloads_add_only_the_horizon_and_follow_up_time() -> None:
    world, _ = survival_world("w-surv", signal=True, seed=2)
    assert set(json.loads(world.card.model_dump_json())) == V1_CARD_KEYS | {"horizon_days"}
    obs = Episode(world).apply(Reset(request_id="r", world_id="w-surv")).to_observation(world.patient_ids)
    revealed = json.loads(obs.model_dump_json())["revealed"]
    assert set(revealed) == V1_REVEALED_KEYS | {"time"}
    assert Observation.model_validate_json(obs.model_dump_json()) == obs
    assert WorldCard.model_validate_json(world.card.model_dump_json()) == world.card


def test_the_horizon_and_the_outcome_type_must_agree() -> None:
    card = make_card("w-x")
    with pytest.raises(ValidationError, match="follow-up horizon"):
        WorldCard.model_validate(card.model_dump() | {"outcome_type": "survival"})
    with pytest.raises(ValidationError, match="follow-up horizon"):
        WorldCard.model_validate(card.model_dump() | {"horizon_days": 100.0})
    with pytest.raises(ValidationError):
        WorldCard.model_validate(card.model_dump() | {"outcome_type": "competing_risks"})


def test_revealed_time_must_align_with_the_rows() -> None:
    with pytest.raises(ValidationError, match="misaligned"):
        RevealedData(patient_ids=("a", "b"), outcome=(1, 0), stratum=("all", "all"), columns={}, time=(1.0,))


def test_world_data_carries_time_exactly_when_the_card_is_survival() -> None:
    world, _ = survival_world("w-s", signal=True, seed=3)
    binary, _ = planted_world("w-b", signal=True, seed=3)
    with pytest.raises(ValueError, match="follow-up time"):
        WorldData(world.card, world.patient_ids, world.x, world.y, world.stratum, world.queues)
    with pytest.raises(ValueError, match="follow-up time"):
        WorldData(
            binary.card,
            binary.patient_ids,
            binary.x,
            binary.y,
            binary.stratum,
            binary.queues,
            time=np.ones(200),
        )


# ---------------------------------------------------------------- engine and kit


def test_sequential_recruitment_reveals_time_with_the_outcome() -> None:
    world, _ = survival_world("w-seq", signal=True, seed=4, mode=Mode.SEQUENTIAL)
    from onc_agi.core.schema import Recruit

    episode = Episode(world)
    episode.apply(Reset(request_id="r", world_id="w-seq"))
    view = episode.apply(Recruit(request_id="a", count=10, stratum="all"))
    assert view.time is not None and world.time is not None
    np.testing.assert_array_equal(view.time, world.time[list(view.rows)])
    assert analysis_input(world.card, view).time is not None


def test_binary_view_digests_ignore_the_absent_time_and_survival_digests_cover_it() -> None:
    world, _ = survival_world("w-d", signal=True, seed=5)
    view = Episode(world).apply(Reset(request_id="r", world_id="w-d"))
    import dataclasses

    assert view_digest(view) != view_digest(dataclasses.replace(view, time=view.time + 1.0))  # type: ignore[operator]
    assert view_digest(dataclasses.replace(view, time=None)) != view_digest(view)


def test_group_sequential_agents_finish_when_cells_are_missing() -> None:
    world, _ = survival_world("w-gap", signal=True, seed=6, mode=Mode.SEQUENTIAL, missing=0.2)
    assert np.isnan(world.x).any()
    agent = GroupSequentialAgent("seq_univariate_bh", baselines.univariate_bh)
    result = run_episode(agent, Episode(world), max_steps=40)
    assert result.final_view.status.value == "submitted"
    assert fid(0) in result.ranking


# ---------------------------------------------------------------- Cox score statistic


def _statsmodels_score_z(x: np.ndarray, time: np.ndarray, event: np.ndarray) -> np.ndarray:
    from statsmodels.duration.hazard_regression import PHReg

    xs = (x - x.mean(axis=0)) / x.std(axis=0)
    k = xs.shape[1]
    out = []
    for j in range(k):
        others = [i for i in range(k) if i != j]
        beta = np.zeros(k)
        if others:
            beta[others] = PHReg(time, xs[:, others], status=event.astype(float), ties="breslow").fit().params
        model = PHReg(time, xs, status=event.astype(float), ties="breslow")
        u, info = model.score(beta), -model.hessian(beta)
        var = info[j, j] - (
            info[j, others] @ np.linalg.solve(info[np.ix_(others, others)], info[others, j])
            if others
            else 0.0
        )
        out.append(u[j] / np.sqrt(var))
    return np.array(out)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_cox_score_z_matches_statsmodels_with_tied_times(k: int) -> None:
    world, _ = survival_world("w-cox", signal=True, seed=7)
    assert world.time is not None
    time = np.round(world.time, -1)  # force ties
    x = world.x[:, :k]
    z = score_test.cox_score_z([x[:, j] for j in range(k)], time, world.y.astype(bool))
    assert z is not None
    np.testing.assert_allclose(z, _statsmodels_score_z(x, time, world.y), rtol=1e-6, atol=1e-8)


def test_candidate_scores_equal_the_score_given_the_selected_columns() -> None:
    world, _ = survival_world("w-cand", signal=True, seed=8)
    assert world.time is not None
    x = world.x[:, :5]
    xs = (x - x.mean(axis=0)) / x.std(axis=0)
    event = world.y.astype(bool)
    cand = score_test.candidate_score_z(xs, [0], world.y, world.time)
    assert cand is not None and cand[0] == 0.0
    for j in range(1, 5):
        z = score_test.cox_score_z([xs[:, 0], xs[:, j]], world.time, event)
        assert z is not None
        assert cand[j] == pytest.approx(z[1], rel=1e-6)
    binary, _ = planted_world("w-bc", signal=True, seed=8)
    xb = (binary.x - binary.x.mean(axis=0)) / binary.x.std(axis=0)
    cb = score_test.candidate_score_z(xb, [1], binary.y)
    zb = score_test.logistic_score_z([xb[:, 1], xb[:, 3]], binary.y)
    assert cb is not None and zb is not None and cb[3] == pytest.approx(zb[1], rel=1e-6)


def test_cox_score_z_is_undefined_without_events_or_with_a_constant_column() -> None:
    t = np.arange(1.0, 21.0)
    assert score_test.cox_score_z([np.ones(20)], t, np.ones(20, dtype=bool)) is None
    assert score_test.cox_score_z([t], t, np.zeros(20, dtype=bool)) is None


# ---------------------------------------------------------------- Cox-aware baselines


def test_cox_baselines_find_a_planted_survival_driver() -> None:
    world, _ = survival_world("w-base", signal=True, seed=9)
    data = analysis_input(world.card, Episode(world).apply(Reset(request_id="r", world_id="w-base")))
    assert baselines.univariate_bh(data)[:1] == [fid(0)]
    assert baselines.forward_score(data)[:1] == [fid(0)]


def test_cox_baselines_rarely_claim_on_null_survival_worlds() -> None:
    claims = 0
    for seed in range(10):
        world, _ = survival_world("w-null", signal=False, seed=seed)
        data = analysis_input(world.card, Episode(world).apply(Reset(request_id="r", world_id="w-null")))
        claims += bool(baselines.univariate_bh(data)) + bool(baselines.forward_score(data))
    assert claims <= 3


def test_survival_scorecards_place_baselines_between_chance_and_the_oracle() -> None:
    store = InMemoryStore.of(
        *(
            survival_world(f"s-{i:02d}", signal=i % 4 != 3, seed=20 + i, log_hazard_ratio=0.5)
            for i in range(12)
        )
    )
    oracle, _ = evaluate(make_agent("oracle", store), store, Tier.PUBLIC_TRAIN, with_alignment=False)
    bh, _ = evaluate(make_agent("univariate_bh", store), store, Tier.PUBLIC_TRAIN, with_alignment=False)
    rand, _ = evaluate(make_agent("random", store), store, Tier.PUBLIC_TRAIN, with_alignment=False)
    assert oracle.find == pytest.approx(1.0)
    assert rand.find is not None and bh.find is not None and rand.find < bh.find <= 1.0


def test_horizon_constant_is_the_default_five_years() -> None:
    assert pytest.approx(5 * 365.25) == HORIZON
