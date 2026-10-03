"""Signed Rao score z for logistic models: the arena's one detection statistic.

Used by the private oracle (detection, tuning, equivalence) and by the public alignment
analyst, so both judge a term with the same monotone statistic. With a single column it is the
Pearson score z on which the oracle's null max threshold is calibrated.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

Array = NDArray[np.float64]


def logistic_fit(exog: Array, y: Array, iterations: int = 25) -> Array | None:
    """Fitted probabilities of a small logistic model by Newton-Raphson with step halving."""
    beta = np.zeros(exog.shape[1])
    beta[0] = np.log(y.mean() / (1 - y.mean()))

    def loglik(b: Array) -> float:
        eta = exog @ b
        return float(np.sum(y * eta - np.logaddexp(0.0, eta)))

    current = loglik(beta)
    for _ in range(iterations):
        p = expit(exog @ beta)
        w = p * (1 - p)
        hessian = exog.T @ (exog * w[:, None])
        try:
            step = np.linalg.solve(hessian + 1e-9 * np.eye(len(beta)), exog.T @ (y - p))
        except np.linalg.LinAlgError:
            return None
        scale = 1.0
        while scale > 1e-4:
            candidate = beta + scale * step
            value = loglik(candidate)
            if value >= current - 1e-12:
                break
            scale /= 2
        beta, previous, current = candidate, current, value
        if abs(current - previous) < 1e-10:
            break
    return np.asarray(expit(exog @ beta), dtype=float)


def logistic_score_z(columns: Sequence[Array], y: NDArray[np.int64]) -> Array | None:
    """Signed Rao score z of each column given an intercept and the other columns.

    The score test is monotone in effect size and defined under (quasi-)separation, unlike
    the Wald z, which shrinks for very strong effects (Hauck-Donner) and is undefined under
    separation. With a single
    column it is exactly the Pearson score z on which the null max threshold is calibrated.
    """
    design = np.column_stack(columns)
    sd = design.std(axis=0)
    if np.any(sd == 0) or len(np.unique(y)) < 2:
        return None
    x = (design - design.mean(axis=0)) / sd
    yf = y.astype(float)
    k = x.shape[1]
    z = np.empty(k)
    for j in range(k):
        others = np.column_stack([np.ones(len(yf)), np.delete(x, j, axis=1)])
        p = np.full(len(yf), yf.mean()) if others.shape[1] == 1 else logistic_fit(others, yf)
        if p is None:
            return None
        w = p * (1 - p)
        u = float(x[:, j] @ (yf - p))
        weighted = others * w[:, None]
        try:
            projection = np.linalg.solve(
                others.T @ weighted + 1e-9 * np.eye(others.shape[1]), weighted.T @ x[:, j]
            )
        except np.linalg.LinAlgError:
            return None
        variance = float(x[:, j] @ (w * x[:, j]) - (weighted.T @ x[:, j]) @ projection)
        z[j] = u / np.sqrt(variance) if variance > 1e-12 else 0.0
    return z
