"""Signed Rao score z for logistic and Cox models: the arena's one detection statistic.

Used by the private oracle (detection, tuning, equivalence) and by the public alignment
analyst, so both judge a term with the same monotone statistic. With a single column it is the
Pearson score z (logistic) or the Cox score statistic (survival) on which the oracle's null max
threshold is calibrated. :func:`candidate_score_z` scores every candidate column against a
fitted model at once, which the forward-selection baseline uses on both outcome types.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

Array = NDArray[np.float64]
# A nuisance model that separates (a strong leak or nuisance among the other columns) has no
# finite MLE. It is refitted under a weakly informative Gaussian prior, SD 2.5 per standardised
# coefficient (intercept unpenalised), so the score of the tested term stays defined.
SEPARATION_RIDGE = 1.0 / 2.5**2


def logistic_fit(exog: Array, y: Array, iterations: int = 25, ridge: float = 0.0) -> Array | None:
    """Newton-Raphson probabilities, or None for invalid input or an unfinished/failed fit.

    ``ridge`` adds ``ridge * beta_j**2 / 2`` to the negative log likelihood for every column
    but the first (the intercept).
    """
    if (
        exog.ndim != 2
        or y.ndim != 1
        or exog.shape[0] != len(y)
        or not exog.size
        or not np.isfinite(exog).all()
        or not np.isfinite(y).all()
        or not np.isin(y, [0, 1]).all()
        or len(np.unique(y)) != 2
    ):
        return None
    beta = np.zeros(exog.shape[1])
    beta[0] = np.log(y.mean() / (1 - y.mean()))
    penalty = np.full(len(beta), ridge)
    penalty[0] = 0.0

    def loglik(b: Array) -> float:
        eta = exog @ b
        return float(np.sum(y * eta - np.logaddexp(0.0, eta)) - 0.5 * np.sum(penalty * b * b))

    current = loglik(beta)
    for _ in range(iterations):
        p = expit(exog @ beta)
        w = p * (1 - p)
        hessian = exog.T @ (exog * w[:, None]) + np.diag(penalty)
        try:
            step = np.linalg.solve(hessian + 1e-9 * np.eye(len(beta)), exog.T @ (y - p) - penalty * beta)
        except np.linalg.LinAlgError:
            return None
        if not np.isfinite(step).all():
            return None
        scale = 1.0
        while scale > 1e-4:
            candidate = beta + scale * step
            value = loglik(candidate)
            if np.isfinite(value) and value >= current - 1e-12:
                break
            scale /= 2
        else:
            return None
        beta, previous, current = candidate, current, value
        if abs(current - previous) < 1e-10:
            fitted = np.asarray(expit(exog @ beta), dtype=float)
            return fitted if np.isfinite(fitted).all() else None
    return None


def logistic_score_z(
    columns: Sequence[Array], y: NDArray[np.int64], *, n_test: int | None = None
) -> Array | None:
    """Signed Rao score z of each column given an intercept and the other columns.

    A separating tested term can retain an informative score when its null nuisance
    model admits a finite fit. This avoids the Wald shrinkage seen for strong binary
    effects (Hauck-Donner). A separating nuisance model is refitted under
    :data:`SEPARATION_RIDGE`, so a strong leak beside the tested term does not void it.
    With a single column it is the Pearson score z used for null calibration.
    Invalid input, failed nuisance fits and negligible conditional information
    return None; none of these is evidence of a zero score.
    ``n_test`` scores only the first requested columns, while all other columns
    remain in their nuisance models. An undefined reverse score for a nuisance
    column must not invalidate a valid requested statistic.
    """
    if (
        not columns
        or y.ndim != 1
        or not np.isfinite(y).all()
        or not np.isin(y, [0, 1]).all()
        or len(np.unique(y)) != 2
        or any(c.ndim != 1 or len(c) != len(y) or not np.isfinite(c).all() for c in columns)
    ):
        return None
    design = np.column_stack(columns)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        sd = design.std(axis=0)
        x = (design - design.mean(axis=0)) / sd
    if np.any(sd == 0) or not np.isfinite(x).all():
        return None
    yf = y.astype(float)
    k = x.shape[1]
    n_test = k if n_test is None else n_test
    if not 1 <= n_test <= k:
        raise ValueError("n_test must select between one and all columns")
    z = np.empty(n_test)
    for j in range(n_test):
        others = np.column_stack([np.ones(len(yf)), np.delete(x, j, axis=1)])
        p = np.full(len(yf), yf.mean()) if others.shape[1] == 1 else logistic_fit(others, yf)
        if p is None and others.shape[1] > 1:
            p = logistic_fit(others, yf, ridge=SEPARATION_RIDGE)
        if p is None or not np.isfinite(p).all():
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
        information = float(x[:, j] @ (w * x[:, j]))
        variance = float(information - (weighted.T @ x[:, j]) @ projection)
        # Ridge stabilisation must not manufacture information for an aliased term.
        if not np.isfinite(variance) or variance <= max(1e-12, 1e-10 * information) or not np.isfinite(u):
            return None
        z[j] = u / np.sqrt(variance)
    return z if np.isfinite(z).all() else None


# --------------------------------------------------------------------------- Cox (survival) score tests


def _suffix(values: Array) -> Array:
    """Reverse cumulative sums with a trailing zero row (risk-set sums over time-sorted rows)."""
    return np.concatenate([np.cumsum(values[::-1], axis=0)[::-1], np.zeros((1, *values.shape[1:]))])


def _sorted_events(
    time: Array, event: NDArray[np.bool_]
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Time order, event positions in that order, and each event's risk-set start (Breslow ties)."""
    order = np.argsort(time, kind="stable")
    t = time[order]
    ev = np.flatnonzero(event[order])
    start = np.searchsorted(t, t[ev], side="left")
    return order, ev, start


def cox_terms(x: Array, beta: Array, time: Array, event: NDArray[np.bool_]) -> tuple[float, Array, Array]:
    """Breslow log partial likelihood (up to a constant), score vector and information matrix at ``beta``.

    Parameters
    ----------
    x
        ``(n, k)`` covariates.
    beta
        ``(k,)`` coefficients.
    time, event
        Follow-up and event indicator per row; the risk set of an event at ``t`` is every
        row with follow-up at least ``t``, so tied events share one risk set.
    """
    order, ev, start = _sorted_events(time, event)
    xs = x[order]
    eta = xs @ beta
    top = float(eta.max()) if eta.size else 0.0
    w = np.exp(eta - top)
    s0 = _suffix(w)[start]
    s1 = _suffix(w[:, None] * xs)[start]
    s2 = _suffix(w[:, None, None] * xs[:, :, None] * xs[:, None, :])[start]
    mean = s1 / s0[:, None]
    loglik = float(np.sum(eta[ev] - top - np.log(s0)))
    score = (xs[ev] - mean).sum(axis=0)
    info = (s2 / s0[:, None, None] - mean[:, :, None] * mean[:, None, :]).sum(axis=0)
    return loglik, score, info


def cox_fit(x: Array, time: Array, event: NDArray[np.bool_], iterations: int = 25) -> Array | None:
    """Cox coefficients by Newton-Raphson with step halving (``None`` if the information is singular)."""
    beta = np.zeros(x.shape[1])
    current, score, info = cox_terms(x, beta, time, event)
    for _ in range(iterations):
        try:
            step = np.linalg.solve(info + 1e-9 * np.eye(len(beta)), score)
        except np.linalg.LinAlgError:
            return None
        scale = 1.0
        while True:
            candidate = beta + scale * step
            value, cand_score, cand_info = cox_terms(x, candidate, time, event)
            if value >= current - 1e-12 or scale < 1e-4:
                break
            scale /= 2
        beta, previous = candidate, current
        current, score, info = value, cand_score, cand_info
        if abs(current - previous) < 1e-10:
            break
    return beta if np.all(np.isfinite(beta)) else None


def cox_score_z(columns: Sequence[Array], time: Array, event: NDArray[np.bool_]) -> Array | None:
    """Signed Rao score z of each column given the others in a Cox model (Breslow ties).

    The survival counterpart of :func:`logistic_score_z`: for column ``j`` the other columns
    are fitted, and ``U_j / sqrt(I_jj - I_jo I_oo^-1 I_oj)`` is evaluated at
    ``(beta_o_hat, beta_j = 0)``. With one column it is the Cox score (log-rank-type)
    statistic of that column. ``None`` when a column is constant, there are fewer than two
    events, or a nuisance fit is singular.
    """
    design = np.column_stack(columns)
    sd = design.std(axis=0)
    if np.any(sd == 0) or int(np.sum(event)) < 2:
        return None
    x = (design - design.mean(axis=0)) / sd
    k = x.shape[1]
    z = np.empty(k)
    for j in range(k):
        beta = np.zeros(k)
        others = np.delete(np.arange(k), j)
        if k > 1:
            fitted = cox_fit(x[:, others], time, event)
            if fitted is None:
                return None
            beta[others] = fitted
        _, score, info = cox_terms(x, beta, time, event)
        information = variance = float(info[j, j])
        if k > 1:
            try:
                variance -= float(
                    info[j, others]
                    @ np.linalg.solve(info[np.ix_(others, others)] + 1e-9 * np.eye(k - 1), info[others, j])
                )
            except np.linalg.LinAlgError:
                return None
        # as for the logistic test: an aliased or degenerate term has no information, not z = 0
        if (
            not np.isfinite(variance)
            or variance <= max(1e-12, 1e-10 * information)
            or not np.isfinite(score[j])
        ):
            return None
        z[j] = score[j] / np.sqrt(variance)
    return z if np.isfinite(z).all() else None


def candidate_score_z(
    x: Array,
    selected: Sequence[int],
    y: NDArray[np.int64],
    time: Array | None = None,
) -> Array | None:
    """Score z of adding each column of ``x`` to a model holding the ``selected`` columns.

    Logistic (with an intercept) when ``time`` is ``None``, Cox (Breslow ties) otherwise; in
    both cases the efficient score ``U_j / sqrt(I_jj - I_jS I_SS^-1 I_Sj)`` at the fitted
    nuisance, vectorised over candidates. ``x`` should be standardised. Selected columns
    get z = 0. ``None`` when the nuisance fit fails.
    """
    sel = list(selected)
    if time is None:
        yf = y.astype(float)
        nuisance = np.column_stack([np.ones(len(yf)), x[:, sel]]) if sel else np.ones((len(yf), 1))
        p = np.full(len(yf), yf.mean()) if not sel else logistic_fit(nuisance, yf)
        if p is None:
            return None
        w = p * (1 - p)
        u = x.T @ (yf - p)
        cross = nuisance.T @ (x * w[:, None])  # (s+1, p)
        info_ss = nuisance.T @ (nuisance * w[:, None])
        info_jj = (w[:, None] * x**2).sum(axis=0)
    else:
        event = y.astype(bool)
        beta = np.zeros(len(sel))
        if sel:
            fitted = cox_fit(x[:, sel], time, event)
            if fitted is None:
                return None
            beta = fitted
        order, ev, start = _sorted_events(time, event)
        xs = x[order]
        eta = xs[:, sel] @ beta if sel else np.zeros(len(xs))
        wt = np.exp(eta - (eta.max() if eta.size else 0.0))
        s0 = _suffix(wt)[start]
        mean = _suffix(wt[:, None] * xs)[start] / s0[:, None]  # (events, p)
        u = (xs[ev] - mean).sum(axis=0)
        info_jj = (_suffix(wt[:, None] * xs**2)[start] / s0[:, None] - mean**2).sum(axis=0)
        if sel:
            ms = mean[:, sel]
            s2 = _suffix(wt[:, None, None] * xs[:, :, None] * xs[:, None, sel])[start] / s0[:, None, None]
            cross = (s2 - mean[:, :, None] * ms[:, None, :]).sum(axis=0).T  # (s, p)
            info_ss = cross[:, sel]
        else:
            cross = np.zeros((0, x.shape[1]))
            info_ss = np.zeros((0, 0))
    if cross.shape[0]:
        try:
            solved = np.linalg.solve(info_ss + 1e-9 * np.eye(info_ss.shape[0]), cross)
        except np.linalg.LinAlgError:
            return None
        variance = info_jj - (cross * solved).sum(axis=0)
    else:
        variance = info_jj
    z = np.where(variance > 1e-12, u / np.sqrt(np.maximum(variance, 1e-12)), 0.0)
    z[sel] = 0.0
    return np.asarray(z, dtype=float)
