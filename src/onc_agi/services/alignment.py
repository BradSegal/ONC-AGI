"""Optimal-task alignment diagnostics.

An *oracle analyst* knows the generating terms (not their weights). Given some
revealed rows and measured features it fits the true-term logistic model and
lists the parts whose |z| clears the world's Monte Carlo detection threshold,
strongest first. Comparing three utilities per world separates analysis from
acquisition:

* ``u_agent`` - the agent's own result (Find on signal worlds, restraint on null);
* ``u_data`` - the oracle analyst on exactly the agent's revealed data;
* ``u_pool`` - the oracle analyst on the whole revealable pool (the reference).

``analysis_regret = u_data - u_agent`` and ``acquisition_gap = u_pool - u_data``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from onc_agi.core.schema import AlignmentDiagnostics, AnswerKey, CreditRule, WorldScore
from onc_agi.core.world import WorldData
from onc_agi.services import scoring
from onc_agi.services.engine import EpisodeView
from onc_agi.services.score_test import logistic_score_z

log = logging.getLogger(__name__)


def oracle_analyst_ranking(
    key: AnswerKey,
    feature_ids: Sequence[str],
    x: NDArray[np.float64],
    y: NDArray[np.int64],
) -> tuple[str, ...]:
    """Ranking the oracle analyst submits from the given data (empty when nothing clears the threshold)."""
    index = {f: j for j, f in enumerate(feature_ids)}
    terms: list[tuple[str, NDArray[np.float64]]] = []
    for group in key.recoverable:
        feats = [p.true_feature for p in group.parts]
        if group.credit_rule is CreditRule.JOINT:
            if all(f in index for f in feats):
                factors = [_factor(x[:, index[f]], group.role) for f in feats]
                terms.append(("|".join(feats), np.prod(np.column_stack(factors), axis=1)))
            continue
        terms.extend((f, x[:, index[f]]) for f in feats if f in index)
    if not terms or len(y) < 10:
        return ()
    design = np.column_stack([t for _, t in terms])
    ok = ~np.isnan(design).any(axis=1)
    if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
        return ()
    z = _logit_z(design[ok], y[ok])
    if z is None:
        return ()
    hits = sorted(
        (
            (abs(float(zi)), name)
            for (name, _), zi in zip(terms, z, strict=True)
            if abs(zi) > key.detection_threshold
        ),
        reverse=True,
    )
    ranking: list[str] = []
    for _, name in hits:
        ranking.extend(f for f in name.split("|") if f not in ranking)
    return tuple(ranking)


def _factor(column: NDArray[np.float64], role: str) -> NDArray[np.float64]:
    """One factor of a product term, built as the generator builds it.

    Measurements enter standardised; a binary indicator enters raw (effect modifier: the effect
    exists in one group) or as -1/+1 (mixture: the effect reverses). Raw products of positive-mean
    measurements are dominated by main effects and miss the planted term.
    """
    values = column[~np.isnan(column)]
    if values.size and set(np.unique(values)) <= {0.0, 1.0}:
        return 2 * column - 1 if role == "mixture" else column
    sd = float(np.nanstd(column))
    return (column - np.nanmean(column)) / (sd if sd > 0 else 1.0)


def _logit_z(design: NDArray[np.float64], y: NDArray[np.int64]) -> NDArray[np.float64] | None:
    """The arena's detection statistic: signed score z of each term given the others."""
    return logistic_score_z(list(design.T), y)


def _utility(score: WorldScore) -> float:
    return float(score.restrained) if score.is_null else score.find


def world_alignment(
    world: WorldData, key: AnswerKey, final_view: EpisodeView, agent_score: WorldScore
) -> tuple[float, float]:
    """(analysis regret, acquisition gap) for one world."""
    cols = [j for j, m in enumerate(final_view.measured) if m]
    data_rank = oracle_analyst_ranking(
        key,
        [final_view.feature_ids[j] for j in cols],
        final_view.x[:, cols],
        final_view.outcome,
    )
    pool_rank = oracle_analyst_ranking(key, world.card.feature_ids(), world.x, world.y)
    u_data = _utility(scoring.score_world(data_rank, key))
    u_pool = _utility(scoring.score_world(pool_rank, key))
    return u_data - _utility(agent_score), u_pool - u_data


def summarise(pairs: Sequence[tuple[float, float]]) -> AlignmentDiagnostics | None:
    if not pairs:
        return None
    arr = np.asarray(pairs, dtype=float)
    return AlignmentDiagnostics(
        analysis_regret=float(arr[:, 0].mean()), acquisition_gap=float(arr[:, 1].mean())
    )
