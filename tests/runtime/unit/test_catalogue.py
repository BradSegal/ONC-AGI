"""Properties of the methods catalogue: each method does what its reference says on planted worlds.

The worlds here are small and fully specified, so each property has one cause: a confounded
correlate, a pure interaction, an environment shift, a survival hazard, an acquisition budget.
"""

from __future__ import annotations

import numpy as np
import pytest
from arena_factories import LEAK, fid, make_card, make_world, planted_world, survival_world
from onc_agi.adapters.agents import TwoPhaseAgent, baselines, catalogue, make_agent
from onc_agi.core.schema import Assay, FeatureMeta, Mode, Reset, WorldCard
from onc_agi.core.world import WorldData
from onc_agi.services.engine import Episode
from onc_agi.services.kit import AnalysisInput, PipelineAgent, analysis_input, run_episode
from onc_agi.services.score_test import cox_terms


def data_of(world: WorldData) -> AnalysisInput:
    return analysis_input(world.card, Episode(world).view())


def with_clinical(card: WorldCard, clinical: tuple[str, ...]) -> WorldCard:
    features = [
        (
            FeatureMeta.model_validate(f.model_dump() | {"data_type": "clinical"})
            if f.feature_id in clinical
            else f
        )
        for f in card.features
    ]
    return WorldCard.model_validate(card.model_dump() | {"features": features})


def logistic_outcome(rng: np.random.Generator, logit: np.ndarray) -> np.ndarray:
    return (rng.random(len(logit)) < 1 / (1 + np.exp(-logit))).astype(np.int64)


# ---------------------------------------------------------------- penalised paths


def test_the_cox_loss_and_gradient_are_the_breslow_partial_likelihood_and_score() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 5))
    time = np.round(rng.exponential(10.0, size=120))  # rounded: many tied event times
    event = (rng.random(120) < 0.6).astype(np.int64)
    beta = 0.3 * rng.normal(size=5)
    value, grad = catalogue._problem(x, event, time).loss(beta)
    loglik, score, _ = cox_terms(x, beta, time, event.astype(bool))
    assert value == pytest.approx(-loglik / event.sum())
    np.testing.assert_allclose(grad, -score / event.sum(), rtol=1e-9, atol=1e-12)


def test_the_logistic_gradient_matches_finite_differences() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(80, 4))
    y = (rng.random(80) < 0.4).astype(np.int64)
    problem = catalogue._problem(x, y, None)
    beta = np.r_[0.2, 0.5 * rng.normal(size=4)]
    _, grad = problem.loss(beta)
    step = 1e-6 * np.eye(5)
    numeric = [(problem.loss(beta + s)[0] - problem.loss(beta - s)[0]) / 2e-6 for s in step]
    np.testing.assert_allclose(grad, numeric, atol=1e-6)


def test_the_l1_path_enters_the_planted_driver_first() -> None:
    data = data_of(planted_world("w-path", signal=True, seed=3)[0])
    _, x = baselines.prepared(data)
    assert catalogue.entry_order(x, data.y, None, 1) == [0]


# ---------------------------------------------------------------- stability selection


@pytest.mark.parametrize("p", [1, 2, 7, 37, 53, 250, 400])
def test_stability_selection_holds_the_meinshausen_buhlmann_bound_at_one(p: int) -> None:
    q = catalogue.stability_q(p)
    assert catalogue.pfer_bound(q, p) <= catalogue.STABILITY_PFER
    assert catalogue.pfer_bound(q + 1, p) > catalogue.STABILITY_PFER  # the largest q the bound allows


def test_stability_selection_abstains_when_the_bound_allows_no_selection() -> None:
    world, _ = planted_world("w-one", signal=True, seed=2, n_features=1)  # one feature: q = 0
    assert catalogue.stability_q(1) == 0 and catalogue.stability_pfer(data_of(world)) == []


def test_stability_selection_finds_a_strong_driver_and_rarely_claims_on_nulls() -> None:
    assert catalogue.stability_pfer(data_of(planted_world("w-s", signal=True, seed=4)[0])) == [fid(0)]
    claims = sum(
        bool(catalogue.stability_pfer(data_of(planted_world("w-n", signal=False, seed=s)[0])))
        for s in range(8)
    )
    assert claims <= 2


# ---------------------------------------------------------------- adjusted analysis


def confounded_world(seed: int) -> WorldData:
    """``f00`` is a clinical indicator causing both the outcome and ``f01``; ``f01`` has no effect."""
    card = with_clinical(make_card("w-conf", n_pool=600, n_features=6), (fid(0),))
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(600, 6))
    x[:, 0] = (rng.random(600) < 0.5).astype(float)
    x[:, 1] = 1.5 * x[:, 0] + 0.5 * rng.normal(size=600)
    return make_world(card, x=x, y=logistic_outcome(rng, 2.0 * x[:, 0] - 1.0))


def test_adjustment_for_the_clinical_indicators_drops_a_confounded_correlate() -> None:
    data = data_of(confounded_world(5))
    assert fid(1) in baselines.univariate_bh(data)  # the marginal analysis is fooled
    assert catalogue.adjusted_analysis(data) == [fid(0)]


def test_pairwise_interaction_screening_recovers_a_pure_interaction() -> None:
    card = make_card("w-int", n_pool=800, n_features=6)
    rng = np.random.default_rng(6)
    x = rng.normal(size=(800, 6))
    world = make_world(card, x=x, y=logistic_outcome(rng, 1.5 * x[:, 0] * x[:, 1]))
    data = data_of(world)
    assert baselines.univariate_bh(data) == []  # no main effect to find
    assert set(catalogue.adjusted_analysis(data)) == {fid(0), fid(1)}


def test_the_adjusted_analysis_uses_cox_tests_on_survival_worlds() -> None:
    assert catalogue.adjusted_analysis(data_of(survival_world("w-sv", signal=True, seed=7)[0]))[:1] == [
        fid(0)
    ]


# ---------------------------------------------------------------- Boruta


@pytest.mark.parametrize("importance", ["impurity", "permutation"])
def test_boruta_confirms_a_strong_driver_with_either_importance(
    importance: catalogue.BorutaImportance,
) -> None:
    data = data_of(planted_world("w-b", signal=True, seed=8)[0])
    found = catalogue.boruta(data, importance=importance, trees=60, max_runs=25)
    assert found[:1] == [fid(0)] and LEAK not in found
    assert catalogue.boruta(data, importance=importance, trees=60, max_runs=25) == found  # seeded per world


# ---------------------------------------------------------------- penalised Cox


def test_penalised_cox_lists_the_planted_hazard_first_and_abstains_on_a_null_world() -> None:
    assert catalogue.penalised_cox(data_of(survival_world("w-pc", signal=True, seed=9)[0]))[:1] == [fid(0)]
    assert catalogue.penalised_cox(data_of(survival_world("w-pn", signal=False, seed=10)[0])) == []


def test_penalised_cox_fits_the_logistic_lasso_on_binary_worlds() -> None:
    assert catalogue.penalised_cox(data_of(planted_world("w-pb", signal=True, seed=11)[0]))[:1] == [fid(0)]


# ---------------------------------------------------------------- invariant causal prediction


def environment_world(seed: int, *, survival: bool = False) -> WorldData:
    """Two strata; the second shifts the cause ``f00``. ``f01`` is a child of the outcome, ``f02`` noise."""
    card = make_card("w-icp", n_pool=800, n_features=4, strata=("s-a", "s-b"))
    rng = np.random.default_rng(seed)
    shift = np.array([i % 2 for i in range(800)], dtype=float)  # make_world assigns strata alternately
    x = rng.normal(size=(800, 4))
    x[:, 0] += 1.5 * shift
    if not survival:
        y = logistic_outcome(rng, 1.5 * x[:, 0] - 1.0)
        x[:, 1] = y + 0.8 * rng.normal(size=800)
        return make_world(card, x=x, y=y)
    time = rng.exponential(1500.0 / np.exp(0.8 * x[:, 0]))
    y = (time < 1500.0).astype(np.int64)
    x[:, 1] = y + 0.8 * rng.normal(size=800)
    survival_card = WorldCard.model_validate(
        card.model_dump() | {"outcome_type": "survival", "horizon_days": 1500.0}
    )
    plain = make_world(card, x=x, y=y)
    return WorldData(
        card=survival_card,
        patient_ids=plain.patient_ids,
        x=plain.x,
        y=plain.y,
        stratum=plain.stratum,
        queues=plain.queues,
        time=np.minimum(time, 1500.0),
    )


@pytest.mark.parametrize("survival", [False, True], ids=["binary", "survival"])
def test_icp_selects_the_cause_and_not_the_outcomes_child_across_environments(survival: bool) -> None:
    data = data_of(environment_world(12, survival=survival))
    assert fid(1) in baselines.univariate_bh(data)  # the child is strongly associated
    assert catalogue.invariant_prediction(data) == [fid(0)]


def test_icp_abstains_with_a_single_environment() -> None:
    assert catalogue.invariant_prediction(data_of(planted_world("w-1env", signal=True, seed=13)[0])) == []


def test_icp_uses_a_binary_clinical_indicator_as_the_environment() -> None:
    world = environment_world(14)
    card = with_clinical(make_card("w-icp", n_pool=800, n_features=5), (fid(4),))
    x = np.column_stack([world.x, np.array([i % 2 for i in range(800)], dtype=float)])
    data = data_of(make_world(card, x=x, y=world.y))  # one stratum; the shift is carried by f04
    assert catalogue.invariant_prediction(data) == [fid(0)]


# ---------------------------------------------------------------- two-phase acquisition


def test_two_phase_buys_less_than_the_full_design_and_never_buys_post_outcome_assays() -> None:
    world, _ = planted_world("w-2p", signal=True, seed=15, mode=Mode.SEQUENTIAL, n_pool=400)
    agent = make_agent("two_phase")
    full = run_episode(PipelineAgent("full", baselines.univariate_bh), Episode(world))
    result = run_episode(agent, Episode(world))
    assert isinstance(agent, TwoPhaseAgent)
    assert result.ranking[:1] == (fid(0),)
    assert len(result.final_view.rows) == world.card.n_pool and result.spent < full.spent
    assert not result.final_view.measured[world.card.feature_ids().index(LEAK)]


def test_two_phase_assays_only_screen_survivors_on_later_patients() -> None:
    world, _ = planted_world("w-2q", signal=True, seed=16, mode=Mode.SEQUENTIAL, n_pool=400)
    episode = Episode(world)
    agent = TwoPhaseAgent()
    view = episode.apply(Reset(request_id="reset", world_id=world.card.world_id))
    assays: list[tuple[str, ...]] = []
    while episode.submission is None:
        action = agent.choose_action(world.card, view)
        if isinstance(action, Assay):
            assays.append(action.feature_ids)
        view = episode.apply(action)
    assert len(assays) == 2 and fid(0) in assays[1] and len(assays[1]) < len(assays[0])
    later = np.isnan(view.x[200:, :]).all(axis=0)  # columns never bought for the phase-2 patients
    assert {world.card.feature_ids()[j] for j in np.flatnonzero(~later)} == set(assays[1])


def test_two_phase_abstains_without_buying_more_when_nothing_survives_the_screen() -> None:
    card = make_card("w-2n", n_pool=200, n_features=3, mode=Mode.SEQUENTIAL)
    rng = np.random.default_rng(18)  # a null draw in which no feature passes the screen
    world = make_world(card, x=rng.normal(size=(200, 3)), y=(rng.random(200) < 0.5).astype(np.int64))
    result = run_episode(TwoPhaseAgent(), Episode(world))
    assert result.ranking == () and len(result.final_view.rows) == 100
    assert result.spent == pytest.approx(
        100 * (1.0 + 3 * 0.5)
    )  # phase 1 only: half the patients, every assay


def test_two_phase_analyses_everything_in_full_access_worlds() -> None:
    world, _ = planted_world("w-2f", signal=True, seed=18)
    result = run_episode(make_agent("two_phase"), Episode(world))
    assert result.ranking[:1] == (fid(0),) and result.spent == 0.0


# ---------------------------------------------------------------- review fixes: score tests, error budget, solver


def test_a_separating_adjustment_covariate_is_refitted_under_the_weak_prior() -> None:
    """The shared score test returns None when the nuisance separates; the catalogue's refits it."""
    from onc_agi.services.score_test import candidate_score_z

    rng = np.random.default_rng(19)
    x = rng.normal(size=(300, 4))
    y = (x[:, 0] > 0).astype(np.int64)  # column 0 separates the outcome perfectly
    x[:, 1] += 1.5 * y
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    assert candidate_score_z(x, [0], y, None) is None
    z = catalogue.catalogue_score_z(x, [0], y, None)
    assert z is not None and z[0] == 0.0 and abs(z[1]) > 2 > max(abs(z[2]), abs(z[3]))


def test_a_separating_clinical_indicator_no_longer_voids_the_adjusted_analysis() -> None:
    card = with_clinical(make_card("w-sep", n_pool=400, n_features=6), (fid(0),))
    rng = np.random.default_rng(20)
    x = rng.normal(size=(400, 6))
    x[:, 0] = (rng.random(400) < 0.3).astype(float)
    y = np.where(x[:, 0] == 1, 1, logistic_outcome(rng, 1.5 * x[:, 1] - 1.0))  # the indicator separates
    assert catalogue.adjusted_analysis(data_of(make_world(card, x=x, y=y)))[:1] in ([fid(0)], [fid(1)])


def test_the_robust_cox_score_is_calibrated_for_a_one_carrier_feature() -> None:
    """Review finding: with the model-based variance a single carrier inflates |z| (SD about 2.5)."""
    n = 600
    carrier = np.zeros(n)
    carrier[0] = 1.0
    carrier = (carrier - carrier.mean()) / carrier.std()
    z = []
    for r in range(200):
        g = np.random.default_rng(r)
        x = np.column_stack([g.normal(size=n), carrier])
        z.append(
            catalogue.catalogue_score_z(x, [0], (g.random(n) < 0.5).astype(np.int64), g.exponential(10.0, n))[
                1
            ]
        )
    assert np.mean(np.abs(z) > 1.96) <= 0.05 and np.std(z) < 1.2


def test_the_robust_cox_score_agrees_with_the_model_based_score_for_regular_columns() -> None:
    from onc_agi.services.score_test import candidate_score_z

    rng = np.random.default_rng(21)
    x = rng.normal(size=(3000, 5))
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    time = np.round(rng.exponential(10.0, 3000))
    event = (rng.random(3000) < 0.6).astype(np.int64)
    robust = catalogue.catalogue_score_z(x, [0, 1], event, time)
    model = candidate_score_z(x, [0, 1], event, time)
    assert robust is not None and model is not None
    np.testing.assert_allclose(robust, model, atol=0.05)


def test_the_adjusted_analysis_spends_one_error_budget_on_pure_null_worlds() -> None:
    """Main effects and the 45 screened interactions share one BH at 5%: P(any claim) stays near 5%."""
    card = with_clinical(make_card("w-null", n_pool=300, n_features=12), (fid(0), fid(1)))
    claims = 0
    for seed in range(60):
        rng = np.random.default_rng(1000 + seed)
        x = rng.normal(size=(300, 12))
        x[:, :2] = rng.random((300, 2)) < 0.4
        claims += bool(
            catalogue.adjusted_analysis(
                data_of(make_world(card, x=x, y=logistic_outcome(rng, np.zeros(300))))
            )
        )
    assert claims <= 7  # 5% of 60 is 3; 7 is beyond the 99th percentile of Binomial(60, 0.05)


@pytest.mark.parametrize("survival", [False, True], ids=["logistic", "cox"])
def test_the_l1_solver_meets_the_lasso_optimality_conditions(survival: bool) -> None:
    """KKT: grad_j = -lam * sign(beta_j) on the active set and |grad_j| <= lam off it."""
    world = (
        survival_world("w-kkt", signal=True, seed=22)[0]
        if survival
        else planted_world("w-kkt", signal=True, seed=22)[0]
    )
    data = data_of(world)
    _, x = baselines.prepared(data)
    problem = catalogue._problem(x, data.y, data.time)
    lambdas = catalogue._grid(problem)
    offset = 1 if problem.intercept else 0
    for lam, beta in zip(lambdas[::6], catalogue._l1_path(problem, lambdas)[::6], strict=True):
        _, grad = problem.loss(beta)
        if offset:
            assert abs(grad[0]) < 1e-3  # unpenalised intercept
        g, b = grad[offset:], beta[offset:]
        active = b != 0
        np.testing.assert_allclose(g[active], -lam * np.sign(b[active]), atol=2e-3)
        assert np.all(np.abs(g[~active]) <= lam + 2e-3)
