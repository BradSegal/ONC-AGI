"""Optimal-task alignment diagnostics: analysis regret and acquisition gap."""

from __future__ import annotations

import numpy as np
import pytest
from arena_factories import fid, group, make_card, make_key, make_world, part, planted_world
from onc_agi.core.schema import CreditRule, Mode, Recruit
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
