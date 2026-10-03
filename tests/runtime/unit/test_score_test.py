"""Finite-input and fit-validity contracts of the shared logistic score statistic."""

from __future__ import annotations

import numpy as np
import pytest
from onc_agi.services.score_test import logistic_fit, logistic_score_z


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("entire_column", [False, True])
def test_nonfinite_design_is_refused_instead_of_returning_zero(value: float, entire_column: bool) -> None:
    column = np.linspace(-1.0, 1.0, 60)
    column[:] = value if entire_column else column
    column[0] = value
    y = (np.arange(60) % 2).astype(np.int64)
    assert logistic_score_z([column], y) is None
    assert logistic_fit(np.column_stack([np.ones(60), column]), y.astype(float)) is None


def test_a_finite_score_matches_the_intercept_only_score_formula() -> None:
    x = np.linspace(-2.0, 2.0, 60)
    y = (x > 0).astype(np.int64)
    z = logistic_score_z([x], y)
    assert z is not None
    standardized = (x - x.mean()) / x.std()
    expected = standardized @ (y - y.mean()) / np.sqrt(len(y) * y.mean() * (1 - y.mean()))
    assert z[0] == pytest.approx(expected)


def test_unfinished_fit_is_refused() -> None:
    x = np.linspace(-2.0, 2.0, 60)
    y = (x > 0).astype(float)
    assert logistic_fit(np.column_stack([np.ones(60), x]), y, iterations=0) is None


@pytest.mark.parametrize("bad_y", [np.full(60, np.nan), np.full(60, np.inf), np.arange(60)])
def test_invalid_outcomes_are_refused(bad_y: np.ndarray) -> None:
    assert logistic_score_z([np.linspace(-1.0, 1.0, 60)], bad_y) is None


def test_degenerate_and_empty_designs_are_refused() -> None:
    y = (np.arange(60) % 2).astype(np.int64)
    assert logistic_score_z([], y) is None
    assert logistic_score_z([np.ones(60)], y) is None
    assert logistic_score_z([np.arange(59.0)], y) is None


def test_a_term_aliased_with_its_nuisance_column_is_refused() -> None:
    x = np.linspace(-2.0, 2.0, 60)
    y = (np.arange(60) % 2).astype(np.int64)
    assert logistic_score_z([x, x.copy()], y) is None


def test_a_separating_nuisance_model_is_refitted_under_the_weak_prior() -> None:
    rng = np.random.default_rng(4)
    driver, noise = rng.normal(size=300), rng.normal(size=300)
    y = (driver > 0.3).astype(np.int64)
    # The driver score's nuisance model is finite; the noise score's nuisance (the driver) separates
    score = logistic_score_z([driver, noise], y, n_test=1)
    assert score is not None and score.shape == (1,) and score[0] > 8
    both = logistic_score_z([driver, noise], y)
    assert both is not None and both[0] == score[0] and abs(both[1]) < 3
    beside = logistic_score_z([noise, driver], y, n_test=1)
    assert beside is not None and abs(beside[0]) < 3  # noise stays undetected beside a perfect predictor
