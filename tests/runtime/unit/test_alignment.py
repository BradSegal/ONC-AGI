"""Optimal-task alignment diagnostics: analysis regret and acquisition gap."""

from __future__ import annotations

import numpy as np
import pytest
from arena_factories import fid, group, make_card, make_key, make_world, part, planted_world
from onc_agi.core.schema import CreditRule, GroupLabel, Mode, Recruit
from onc_agi.services.alignment import oracle_analyst_ranking, summarise, world_alignment
from onc_agi.services.engine import Episode
from onc_agi.services.scoring import score_world


def test_oracle_analyst_finds_a_strong_true_term() -> None:
    world, key = planted_world("w-00", signal=True, seed=1)
    assert oracle_analyst_ranking(key, world.card.feature_ids(), world.x, world.y) == (fid(0),)


def test_oracle_analyst_returns_nothing_on_a_null_key() -> None:
    world, key = planted_world("w-00", signal=False, seed=1)
    assert oracle_analyst_ranking(key, world.card.feature_ids(), world.x, world.y) == ()


@pytest.mark.parametrize("rows", [0, 5, 9])
def test_oracle_analyst_needs_at_least_ten_rows(rows: int) -> None:
    world, key = planted_world("w-00", signal=True, seed=1)
    assert oracle_analyst_ranking(key, world.card.feature_ids(), world.x[:rows], world.y[:rows]) == ()


def test_oracle_analyst_cannot_use_unmeasured_true_features() -> None:
    world, key = planted_world("w-00", signal=True, seed=1)
    ids = world.card.feature_ids()[1:]
    assert oracle_analyst_ranking(key, ids, world.x[:, 1:], world.y) == ()


def test_oracle_analyst_fits_a_joint_group_as_a_product_term() -> None:
    card = make_card(n_pool=400, n_features=4)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 4))
    y = (rng.random(400) < 1 / (1 + np.exp(-3.0 * x[:, 0] * x[:, 1]))).astype(np.int64)
    make_world(card, x=x, y=y)  # validates the planted data
    key = make_key(card, [group("i0", part(fid(0)), part(fid(1)), rule=CreditRule.JOINT)])
    ranking = oracle_analyst_ranking(key, card.feature_ids(), x, y)
    assert set(ranking) == {fid(0), fid(1)}
    assert score_world(ranking, key, chance=(0.0, 0.0)).raw_recovery == 1.0


def test_an_agent_matching_the_oracle_on_full_data_has_no_regret_or_gap() -> None:
    world, key = planted_world("w-00", signal=True, seed=1)
    view = Episode(world).view()
    agent = score_world([fid(0)], key)
    assert world_alignment(world, key, view, agent) == (0.0, 0.0)


def test_an_agent_that_abstains_despite_sufficient_data_carries_analysis_regret() -> None:
    world, key = planted_world("w-00", signal=True, seed=1)
    regret, gap = world_alignment(world, key, Episode(world).view(), score_world([], key))
    assert regret == pytest.approx(1.0) and gap == 0.0


def test_an_agent_that_acquires_nothing_carries_the_whole_acquisition_gap() -> None:
    world, key = planted_world("w-00", signal=True, seed=1, mode=Mode.SEQUENTIAL)
    view = Episode(world).view()
    regret, gap = world_alignment(world, key, view, score_world([], key))
    assert (regret, gap) == (0.0, pytest.approx(1.0))


def test_partial_acquisition_splits_regret_from_gap() -> None:
    world, key = planted_world("w-00", signal=True, seed=1, mode=Mode.SEQUENTIAL)
    ep = Episode(world)
    ep.apply(Recruit(request_id="a", count=5))
    regret, gap = world_alignment(world, key, ep.view(), score_world([], key))
    assert regret == 0.0 and gap == pytest.approx(1.0)


def test_null_worlds_reward_the_analyst_for_abstaining() -> None:
    world, key = planted_world("w-00", signal=False, seed=1)
    regret, gap = world_alignment(world, key, Episode(world).view(), score_world([fid(1)], key))
    assert (regret, gap) == (1.0, 0.0)


def test_summary_averages_pairs_and_is_absent_without_worlds() -> None:
    assert summarise([]) is None
    diag = summarise([(1.0, 0.0), (0.0, 0.5)])
    assert diag is not None and (diag.analysis_regret, diag.acquisition_gap) == (0.5, 0.25)


def test_binary_measurements_are_centred_in_an_interaction() -> None:
    card = make_card(n_pool=40, n_features=2)
    a = np.repeat([0.0, 0.0, 1.0, 1.0], 10)
    b = np.repeat([0.0, 1.0, 0.0, 1.0], 10)
    x = np.column_stack([a, b])
    y = (a == b).astype(np.int64)
    key = make_key(card, [group("i0", part(fid(0)), part(fid(1)), role="interaction")], threshold=4.0)
    assert set(oracle_analyst_ranking(key, card.feature_ids(), x, y)) == {fid(0), fid(1)}


def test_alignment_conditions_on_a_known_neutral_nuisance() -> None:
    card = make_card(n_pool=400, n_features=2)
    rng = np.random.default_rng(0)
    cohort = rng.binomial(1, 0.5, 400).astype(float)
    gene = cohort + 0.4 * rng.normal(size=400)
    y = rng.binomial(1, 1 / (1 + np.exp(-(-1.5 + 3 * cohort + 0.15 * gene))))
    key = make_key(
        card,
        [
            group("g0", part(fid(0))),
            group("c0", part(fid(1)), label=GroupLabel.NEUTRAL, role="cohort"),
        ],
    )
    # Marginal association is strong, but the true conditional score in this outcome draw is < 3.
    # A recoverability label describes repeated draws, not guaranteed detection in this draw.
    assert oracle_analyst_ranking(key, card.feature_ids(), np.column_stack([gene, cohort]), y) == ()


@pytest.mark.parametrize("role", ["effect_modifier", "mixture"])
def test_binary_measurement_and_clinical_indicator_have_distinct_product_rules(role: str) -> None:
    card = make_card(n_pool=40, n_features=2)
    gene = np.repeat([0.0, 0.0, 1.0, 1.0], 10)
    clinical = np.repeat([0.0, 1.0, 0.0, 1.0], 10)
    if role == "mixture":
        y = (gene == clinical).astype(np.int64)
        threshold = 5.0
    else:
        y = np.where(clinical == 1, gene, np.tile([0, 1], 20)).astype(np.int64)
        threshold = 3.5
    key = make_key(card, [group("i0", part(fid(0)), part(fid(1)), role=role)], threshold=threshold)
    assert set(oracle_analyst_ranking(key, card.feature_ids(), np.column_stack([gene, clinical]), y)) == {
        fid(0),
        fid(1),
    }


def test_a_constant_neutral_term_does_not_block_an_informative_revealed_term() -> None:
    card = make_card(n_pool=40, n_features=2)
    gene = np.arange(40, dtype=float)
    x = np.column_stack([gene, np.zeros(40)])
    y = (gene > 19).astype(np.int64)
    key = make_key(
        card,
        [group("g0", part(fid(0))), group("c0", part(fid(1)), label=GroupLabel.NEUTRAL, role="cohort")],
    )
    assert oracle_analyst_ranking(key, card.feature_ids(), x, y) == (fid(0),)
