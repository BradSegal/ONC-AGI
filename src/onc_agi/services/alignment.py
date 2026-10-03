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
    # Score recoverable terms first; known neutral terms still belong in their nuisance models.
    recoverable = key.recoverable
    groups = list(recoverable) + [g for g in key.groups if g not in recoverable]
    n_test = 0
    for group in groups:
        feats = [p.true_feature for p in group.parts]
        start = len(terms)
        if group.credit_rule is CreditRule.JOINT:
            if all(f in index for f in feats):
                factors = [_factor(x[:, index[f]], group.role, factor_index=i) for i, f in enumerate(feats)]
                terms.append(("|".join(feats), np.prod(np.column_stack(factors), axis=1)))
        else:
            terms.extend((f, x[:, index[f]]) for f in feats if f in index)
        if group in recoverable:
            n_test += len(terms) - start
    if not n_test or len(y) < 10:
        return ()
    design = np.column_stack([t for _, t in terms])
    ok = ~np.isnan(design).any(axis=1)
    if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
        return ()
    # Finite constant terms carry no information in this revealed subset; they are nondetections.
    active = [i for i, column in enumerate(design[ok].T) if np.any(column != column[0])]
    tested = [i for i in active if i < n_test]
    if not tested:
        return ()
    z = _logit_z(design[ok][:, active], y[ok], n_test=len(tested))
    if z is None:
        return ()
    hits = sorted(
        (
            (abs(float(zi)), terms[i][0])
            for i, zi in zip(tested, z, strict=True)
            if abs(zi) > key.detection_threshold
        ),
        reverse=True,
    )
    ranking: list[str] = []
    for _, name in hits:
        ranking.extend(f for f in name.split("|") if f not in ranking)
    return tuple(ranking)


def _factor(column: NDArray[np.float64], role: str, *, factor_index: int = 0) -> NDArray[np.float64]:
    """One factor of a product term, built as the generator builds it.

    All measurements, including binary mutations, enter standardised. In effect-modifier and
    mixture keys the second part is the clinical indicator, which enters raw or as -1/+1.
    Generator answer order distinguishes that indicator from binary molecular measurements.
    """
    if factor_index == 1 and role in {"effect_modifier", "mixture"}:
        return 2 * column - 1 if role == "mixture" else column
    sd = float(np.nanstd(column))
    return (column - np.nanmean(column)) / (sd if sd > 0 else 1.0)


def _logit_z(
    design: NDArray[np.float64], y: NDArray[np.int64], *, n_test: int | None = None
) -> NDArray[np.float64] | None:
    """The arena's detection statistic: signed score z of each term given the others."""
    return logistic_score_z(list(design.T), y, n_test=n_test)


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
