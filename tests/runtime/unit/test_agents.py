"""Registered reference, baseline and cheater agents.

The central property for outcome-blind cheaters is checked directly: permuting the
outcome must not change their list, so any credit they earn is chance by construction.
"""

from __future__ import annotations

import numpy as np
import pytest
from arena_factories import LEAK, InMemoryStore, fid, planted_world
from onc_agi.adapters.agents import (
    AGENT_NAMES,
    BASELINES,
    CHANCE_LEVEL,
    CHEATERS,
    OracleAgent,
    cheaters,
    make_agent,
)
from onc_agi.core.schema import Mode, Tier
from onc_agi.services.engine import Episode
from onc_agi.services.kit import AnalysisInput, analysis_input, run_episode

OUTCOME_BLIND = ("metadata_only", "variance_ranker", "hub_ranker", "cluster_size_ranker", "synthetic_ranker")


def data_for(signal: bool = True, seed: int = 1, *, permute_outcome: bool = False) -> AnalysisInput:
    world, _ = planted_world("w-00", signal=signal, seed=seed)
    data = analysis_input(world.card, Episode(world).view())
    if permute_outcome:
        y = np.random.default_rng(99).permutation(data.y)
        data = AnalysisInput(
            card=data.card, feature_ids=data.feature_ids, x=data.x, y=y, stratum=data.stratum
        )
    return data


def test_registry_lists_every_agent_and_rejects_unknown_names() -> None:
    assert {"oracle", "random", *BASELINES, *CHEATERS} == set(AGENT_NAMES)
    with pytest.raises(KeyError, match="unknown agent"):
        make_agent("nobody")
    with pytest.raises(ValueError, match="answer keys"):
        make_agent("oracle")


@pytest.mark.parametrize("name", sorted(CHEATERS))
def test_cheaters_return_valid_unique_feature_lists(name: str) -> None:
    data = data_for()
    ranking = CHEATERS[name](data)
    assert len(set(ranking)) == len(ranking)
    assert set(ranking) <= set(data.card.feature_ids())


@pytest.mark.parametrize("name", OUTCOME_BLIND)
def test_outcome_blind_cheaters_never_read_the_outcome(name: str) -> None:
    assert name in CHANCE_LEVEL
    assert CHEATERS[name](data_for()) == CHEATERS[name](data_for(permute_outcome=True))


def test_metadata_only_never_reads_values() -> None:
    data = data_for()
    blank = AnalysisInput(
        card=data.card,
        feature_ids=data.feature_ids,
        x=np.zeros_like(data.x),
        y=data.y * 0,
        stratum=data.stratum,
    )
    assert cheaters.metadata_only(data) == cheaters.metadata_only(blank)


def test_giant_list_lists_everything_including_leaks() -> None:
    data = data_for()
    assert cheaters.giant_list(data) == list(data.card.feature_ids()) and LEAK in cheaters.giant_list(data)


def test_leak_exploiter_puts_post_outcome_features_first() -> None:
    assert cheaters.leak_exploiter(data_for())[0] == LEAK


def test_auc_maximiser_ranks_the_leak_first() -> None:
    assert cheaters.auc_maximiser(data_for())[0] == LEAK


def test_always_empty_is_empty_and_random_rankings_are_seeded_per_world() -> None:
    assert cheaters.always_empty(data_for()) == []
    assert cheaters.random_ranking(data_for()) == cheaters.random_ranking(data_for())


def test_famous_genes_fall_back_to_a_guess_when_names_are_hidden() -> None:
    ranking = cheaters.famous_genes(data_for())
    assert len(ranking) == 5 and not set(ranking) & set(cheaters.FAMOUS)


def test_learned_ranker_must_be_fitted_on_training_worlds_with_both_labels() -> None:
    with pytest.raises(RuntimeError, match="fitted"):
        cheaters.LearnedRanker()(data_for())
    null_store = InMemoryStore.of(planted_world("w-00", signal=False, seed=1))
    with pytest.raises(ValueError, match="both"):
        cheaters.LearnedRanker().fit(null_store, ["w-00"])


def test_learned_ranker_is_outcome_blind_once_fitted(store: InMemoryStore) -> None:
    ranker = cheaters.LearnedRanker().fit(store, store.world_ids(Tier.PUBLIC_TRAIN))
    assert ranker(data_for(seed=7)) == ranker(data_for(seed=7, permute_outcome=True))


@pytest.mark.parametrize("name", sorted(BASELINES))
def test_baselines_list_only_baseline_features(name: str) -> None:
    if name == "knockoffs":
        pytest.importorskip("knockpy")
    ranking = BASELINES[name](data_for())
    assert len(set(ranking)) == len(ranking)
    assert LEAK not in ranking


@pytest.mark.parametrize("name", ["univariate_bh", "lasso", "stability"])
def test_standard_baselines_find_a_strong_planted_driver_first(name: str) -> None:
    assert BASELINES[name](data_for())[:1] == [fid(0)]


def test_oracle_agent_submits_one_true_feature_per_recoverable_part() -> None:
    world, key = planted_world("w-00", signal=True, seed=1, mode=Mode.SEQUENTIAL)
    store = InMemoryStore.of((world, key))
    result = run_episode(OracleAgent(store), Episode(world))
    assert result.ranking == (fid(0),) and result.spent == 0.0


def test_group_sequential_agent_stops_once_its_selection_is_stable() -> None:
    world, _ = planted_world("w-00", signal=True, seed=1, mode=Mode.SEQUENTIAL, n_pool=200)
    result = run_episode(make_agent("seq_univariate_bh"), Episode(world))
    assert result.ranking[:1] == (fid(0),)
    assert len(result.final_view.rows) < world.card.n_pool


def test_group_sequential_agents_also_play_full_access_worlds() -> None:
    world, _ = planted_world("w-00", signal=True, seed=1, mode=Mode.FULL_ACCESS, n_pool=200)
    result = run_episode(make_agent("seq_univariate_bh"), Episode(world))
    assert result.ranking[:1] == (fid(0),)


def test_group_sequential_agents_reset_between_sequential_worlds() -> None:
    agent = make_agent("seq_univariate_bh")
    for seed in (1, 2):
        world, _ = planted_world(f"w-{seed:02d}", signal=True, seed=seed, mode=Mode.SEQUENTIAL, n_pool=200)
        result = run_episode(agent, Episode(world))
        assert result.ranking[:1] == (fid(0),)


@pytest.mark.parametrize("name", ["random_forest", "knockoffs", "elastic_net", "lasso"])
def test_selection_baselines_find_a_strong_driver_and_abstain_on_null_worlds(name: str) -> None:
    if name == "knockoffs":
        pytest.importorskip("knockpy")
    found = BASELINES[name](data_for(signal=True, seed=11))
    assert fid(0) in found[:2]
    assert BASELINES[name](data_for(signal=True, seed=11)) == found  # seeded per world


@pytest.mark.parametrize("name", sorted(BASELINES))
def test_baselines_abstain_when_the_data_cannot_support_analysis(name: str) -> None:
    if name == "knockoffs":
        pytest.importorskip("knockpy")
    data = data_for()
    tiny = AnalysisInput(
        card=data.card, feature_ids=data.feature_ids, x=data.x[:5], y=data.y[:5], stratum=data.stratum[:5]
    )
    one_class = AnalysisInput(
        card=data.card, feature_ids=data.feature_ids, x=data.x, y=data.y * 0, stratum=data.stratum
    )
    assert BASELINES[name](tiny) == [] and BASELINES[name](one_class) == []


@pytest.mark.parametrize("name", ["univariate_bh", "random_forest"])
def test_error_controlled_baselines_usually_abstain_on_null_worlds(name: str) -> None:
    """Knockoffs is excluded: its offset-0 threshold claims on roughly half of small null worlds by design."""
    claims = sum(bool(BASELINES[name](data_for(signal=False, seed=s))) for s in range(10))
    assert claims <= 3


def test_nothing_learns_from_eval_or_private_worlds() -> None:
    """Hackathon rule 3: the only cross-world learner refuses any world outside public-train."""
    from arena_factories import InMemoryStore, make_card, make_key, make_world
    from onc_agi.adapters.agents.cheaters import LearnedRanker
    from onc_agi.core.schema import Tier

    store = InMemoryStore()
    for tier in (Tier.PUBLIC_EVAL, Tier.PRIVATE):
        card = make_card(f"w-{tier.value.replace('_', '-')}", tier=tier)
        store.add(make_world(card), make_key(card))
        with pytest.raises(ValueError, match="public-train only"):
            LearnedRanker().fit(store, [card.world_id])


@pytest.mark.parametrize("strata", [("s-a", "s-b"), ("s-a", "s-b", "s-c")])
def test_group_sequential_agents_recruit_every_stratum_without_overdrawing(strata: tuple[str, ...]) -> None:
    """recruiting only from strata[0] exhausted it and aborted play."""
    from collections import Counter

    from arena_factories import make_card, make_world
    from onc_agi.adapters.agents import make_agent
    from onc_agi.core.schema import Mode, Reset
    from onc_agi.services.engine import Episode

    card = make_card("w-strata", mode=Mode.SEQUENTIAL, strata=strata, n_pool=90)
    world = make_world(card)
    agent = make_agent("seq_univariate_bh")
    episode = Episode(world)
    view = episode.apply(Reset(request_id="r", world_id=card.world_id))
    for _ in range(100):
        if episode.submission is not None:
            break
        view = episode.apply(agent.choose_action(card, view))  # any refusal would raise here
    assert episode.submission is not None
    recruited = Counter(view.stratum)
    assert set(recruited) == set(strata)
    assert max(recruited.values()) - min(recruited.values()) <= 1  # proportional stages


def test_group_sequential_agents_terminate_on_genuinely_missing_data() -> None:
    """NaN from source missingness looked like an unassayed cell, so the agent
    re-assayed forever. It must submit, recruit every stage, and never re-buy acquired cells."""
    import numpy as np
    from arena_factories import make_card, make_world
    from onc_agi.adapters.agents import make_agent
    from onc_agi.core.schema import Assay, Mode, Reset
    from onc_agi.services.engine import Episode

    card = make_card("w-missing", mode=Mode.SEQUENTIAL, strata=("s-a", "s-b"), n_pool=80)
    world = make_world(card)
    x = world.x.copy()
    x[::3, 0] = np.nan  # genuine missingness in the source
    world = type(world)(**{**world.__dict__, "x": x})
    agent = make_agent("seq_univariate_bh")
    episode = Episode(world)
    view = episode.apply(Reset(request_id="r", world_id=card.world_id))
    assays = 0
    for _ in range(60):
        if episode.submission is not None:
            break
        action = agent.choose_action(card, view)
        assays += isinstance(action, Assay)
        view = episode.apply(action)
    assert episode.submission is not None
    assert assays <= len(agent.stages)  # one assay per stage at most
