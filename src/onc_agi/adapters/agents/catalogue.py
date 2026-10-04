"""Literature-grounded methods catalogue: the competent-analyst references beside the baseline ladder.

Every method follows the shared baseline contract of :mod:`.baselines`: post-outcome columns
are dropped, columns are median-imputed and standardised, an explicit rule says when the list
is empty, every hyperparameter is a module constant fixed before any evaluation data was seen,
and randomness is seeded per world with :func:`stable_seed`. Survival worlds
(``AnalysisInput.time`` present) are analysed with Cox models wherever the method has a Cox
form; the exceptions say so.

* :func:`adjusted_analysis` - clinical- and cohort-adjusted score tests with pairwise
  interaction screening.
* :func:`boruta` - the all-relevant random-forest selector, settings fixed by a recorded pilot.
* :func:`stability_pfer` - stability selection with the Meinshausen-Buhlmann error bound enforced.
* :func:`penalised_cox` - L1-penalised Cox regression (L1 logistic on binary worlds), penalty
  by cross-validation with the one-standard-error rule.
* :func:`invariant_prediction` - invariant causal prediction across environments.

The two-phase acquisition policy, which needs actions rather than an analysis function, is
:class:`onc_agi.adapters.agents.TwoPhaseAgent`; it shares :func:`adjusted_main_z`.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier

from onc_agi.adapters.agents.baselines import FDR, bh_select, prepared, usable
from onc_agi.services.kit import AnalysisInput
from onc_agi.services.score_test import SEPARATION_RIDGE, cox_fit, cox_terms, logistic_fit
from onc_agi.services.scoring import stable_seed

Array = NDArray[np.float64]


# --------------------------------------------------------------------------- shared design


@dataclass(frozen=True)
class Design:
    """Standardised baseline candidates plus the adjustment covariates.

    ``x`` holds every non-constant baseline feature (``ids``); ``clinical`` indexes the
    clinical indicators among them; ``strata`` holds centred dummy columns of the recruitment
    stratum (none when the revealed rows come from one stratum).
    """

    ids: list[str]
    x: Array
    clinical: list[int]
    strata: Array


def design(data: AnalysisInput) -> Design:
    ids, x = prepared(data)
    keep = [j for j in range(x.shape[1]) if x[:, j].std() > 0] if x.size else []
    kind = {f.feature_id: f.data_type for f in data.card.features}
    ids = [ids[j] for j in keep]
    x = x[:, keep] if x.size else x
    levels = sorted(set(data.stratum))
    dummies = [np.array([s == level for s in data.stratum], dtype=float) for level in levels[1:]]
    strata = np.column_stack([(d - d.mean()) / d.std() for d in dummies if d.std() > 0]) if dummies else None
    return Design(
        ids=ids,
        x=x,
        clinical=[j for j, f in enumerate(ids) if kind[f] == "clinical"],
        strata=strata if strata is not None else np.empty((len(data.y), 0)),
    )


def _two_sided(z: Array) -> Array:
    return np.asarray(2 * stats.norm.sf(np.abs(z)), dtype=float)


def _standardise(column: Array) -> Array | None:
    sd = column.std()
    return None if sd == 0 else (column - column.mean()) / sd


# --------------------------------------------------------------------------- score tests


def _ridge_cox_fit(x: Array, time: Array, event: NDArray[np.bool_], ridge: float) -> Array | None:
    """Cox coefficients under a Gaussian prior ``ridge * |beta|^2 / 2`` (Newton with step halving)."""
    beta = np.zeros(x.shape[1])
    loglik, score, info = cox_terms(x, beta, time, event)
    current = loglik
    for _ in range(50):
        try:
            step = np.linalg.solve(info + ridge * np.eye(len(beta)), score - ridge * beta)
        except np.linalg.LinAlgError:
            return None
        scale = 1.0
        while scale >= 1e-4:
            candidate = beta + scale * step
            value, cand_score, cand_info = cox_terms(x, candidate, time, event)
            value -= 0.5 * ridge * float(candidate @ candidate)
            if np.isfinite(value) and value >= current - 1e-12:
                break
            scale /= 2
        else:
            return None
        beta, previous, current, score, info = candidate, current, value, cand_score, cand_info
        if abs(current - previous) < 1e-10:
            break
    return beta if np.isfinite(beta).all() else None


def catalogue_score_z(
    x: Array, selected: Sequence[int], y: NDArray[np.int64], time: Array | None
) -> Array | None:
    """Score z of adding each column of standardised ``x`` to a model holding the ``selected`` columns.

    The catalogue's own form of :func:`onc_agi.services.score_test.candidate_score_z`,
    which it leaves unchanged. Two changes make it safe to use as a reference analyst:

    * a nuisance model that has no finite fit (separation, or a monotone likelihood in Cox) is
      refitted under the weakly informative prior :data:`SEPARATION_RIDGE` (SD 2.5 per
      standardised coefficient), as :func:`logistic_score_z` does, instead of failing;
    * on survival worlds the efficient Cox score is studentised by the robust (Lin & Wei 1989)
      variance, the sum of squared efficient score residuals, instead of the model-based
      information. The model-based variance is anti-conservative for rare or heavy-tailed
      columns (one carrier among hundreds of patients), where a single event dominates the score.

    Logistic tests keep the model-based variance. Selected columns get z = 0. ``None`` when even
    the prior-regularised nuisance fit fails.

    References: Rao (1948) Proc Camb Philos Soc 44:50; Lin & Wei (1989) J Am Stat Assoc 84:1074;
    Gelman, Jakulin, Pittau & Su (2008) Ann Appl Stat 2:1360 (weakly informative priors).
    """
    sel = list(selected)
    if time is None:
        yf = y.astype(float)
        nuisance = np.column_stack([np.ones(len(yf)), x[:, sel]])
        if sel:
            prob = logistic_fit(nuisance, yf)
            if prob is None:
                prob = logistic_fit(nuisance, yf, ridge=SEPARATION_RIDGE)
        else:
            prob = np.full(len(yf), yf.mean())
        if prob is None:
            return None
        w = prob * (1 - prob)
        u = x.T @ (yf - prob)
        cross = nuisance.T @ (x * w[:, None])
        info_ss = nuisance.T @ (nuisance * w[:, None])
        try:
            solved = np.linalg.solve(info_ss + 1e-9 * np.eye(info_ss.shape[0]), cross)
        except np.linalg.LinAlgError:
            return None
        variance = (w[:, None] * x**2).sum(axis=0) - (cross * solved).sum(axis=0)
    else:
        event = y.astype(bool)
        beta = np.zeros(len(sel))
        if sel:
            fitted = cox_fit(x[:, sel], time, event)
            if fitted is None:
                fitted = _ridge_cox_fit(x[:, sel], time, event, SEPARATION_RIDGE)
            if fitted is None:
                return None
            beta = fitted
        order = np.argsort(time, kind="stable")
        t, xs = time[order], x[order]
        ev = np.flatnonzero(event[order])
        start = np.searchsorted(t, t[ev], side="left")
        eta = xs[:, sel] @ beta if sel else np.zeros(len(xs))
        wt = np.exp(eta - (eta.max() if eta.size else 0.0))
        s0 = np.cumsum(wt[::-1])[::-1][start]
        mean = np.cumsum((wt[:, None] * xs)[::-1], axis=0)[::-1][start] / s0[:, None]  # (events, p)
        u = (xs[ev] - mean).sum(axis=0)
        # per-patient score residuals: own event term minus the risk-set compensator (Lin & Wei 1989)
        inverse = np.zeros(len(t))
        np.add.at(inverse, start, 1.0 / s0)
        weighted_mean = np.zeros_like(xs)
        np.add.at(weighted_mean, start, mean / s0[:, None])
        residual = -wt[:, None] * (xs * np.cumsum(inverse)[:, None] - np.cumsum(weighted_mean, axis=0))
        residual[ev] += xs[ev] - mean
        if sel:
            ms = mean[:, sel]
            s2 = np.cumsum((wt[:, None, None] * xs[:, sel, None] * xs[:, None, :])[::-1], axis=0)[::-1][start]
            cross = (s2 / s0[:, None, None] - ms[:, :, None] * mean[:, None, :]).sum(axis=0)  # (s, p)
            try:
                projection = np.linalg.solve(cross[:, sel] + 1e-9 * np.eye(len(sel)), cross)
            except np.linalg.LinAlgError:
                return None
            u = u - u[sel] @ projection
            residual = residual - residual[:, sel] @ projection
        variance = (residual**2).sum(axis=0)
    z = np.where(variance > 1e-12, u / np.sqrt(np.maximum(variance, 1e-12)), 0.0)
    z[sel] = 0.0
    return np.asarray(z, dtype=float) if np.isfinite(z).all() else None


# --------------------------------------------------------------------------- adjusted analysis

INTERACTION_TOP = 10  # features whose pairwise products are screened (45 pairs)


def adjusted_main_z(data: AnalysisInput, d: Design) -> Array | None:
    """Score z of every candidate given the clinical indicators and the recruitment strata.

    A molecular (non-clinical) feature is tested against a model holding every clinical
    indicator and the stratum dummies; a clinical indicator against the other indicators and
    the dummies, so the indicators are mutually adjusted. Logistic or Cox score tests
    (:func:`candidate_score_z`). ``None`` when an adjustment model cannot be fitted.
    """
    clinical = d.x[:, d.clinical]
    covariates = np.column_stack([d.strata, clinical])
    s = d.strata.shape[1]
    z = np.zeros(len(d.ids))
    molecular = [j for j in range(len(d.ids)) if j not in set(d.clinical)]
    if molecular:
        zm = catalogue_score_z(
            np.column_stack([covariates, d.x[:, molecular]]), range(covariates.shape[1]), data.y, data.time
        )
        if zm is None:
            return None
        z[molecular] = zm[covariates.shape[1] :]
    for i, j in enumerate(d.clinical):
        others = [k for k in range(covariates.shape[1]) if k != s + i]
        zc = catalogue_score_z(covariates, others, data.y, data.time)
        if zc is None:
            return None
        z[j] = zc[s + i]
    return z


def _interaction_z(data: AnalysisInput, d: Design, a: int, b: int) -> float | None:
    """Score z of the product ``x_a * x_b`` given both main effects, the indicators and the strata."""
    product = _standardise(d.x[:, a] * d.x[:, b])
    if product is None:
        return None
    nuisance = [d.strata, d.x[:, sorted(set(d.clinical) | {a, b})]]
    base = np.column_stack(nuisance)
    z = catalogue_score_z(np.column_stack([base, product]), range(base.shape[1]), data.y, data.time)
    return None if z is None else float(z[-1])


def adjusted_analysis(data: AnalysisInput) -> list[str]:
    """Covariate- and cohort-adjusted analysis with pairwise interaction screening.

    The competent-analyst reference: each candidate is tested conditionally on the clinical
    indicators (one of which is the cohort indicator) and on the recruitment stratum, by a Rao
    score test from the logistic model (Cox model on survival worlds, with the robust score
    variance; :func:`catalogue_score_z`). Interactions
    are screened hierarchically: products are formed among the ``INTERACTION_TOP`` = 10
    candidates with the strongest adjusted main effects (Kooperberg & LeBlanc 2008), and each
    is tested given both main effects and the covariates. One Benjamini-Hochberg procedure at
    5% runs over the main effects and the 45 interaction tests together, so a single error
    budget covers the whole analysis: under the global null the chance of any claim is at most
    5%. The list is the selected main effects by p-value, then the members of selected
    interactions not already listed. Empty when nothing is selected, or when even the
    regularised adjustment model cannot be fitted.

    References: Rao (1948) score test; Benjamini & Hochberg (1995) J R Stat Soc B 57:289;
    Cox (1972) J R Stat Soc B 34:187; Lin & Wei (1989) J Am Stat Assoc 84:1074; Kooperberg &
    LeBlanc (2008) Genet Epidemiol 32:255.
    """
    d = design(data)
    if not usable(data, d.x):
        return []
    z = adjusted_main_z(data, d)
    if z is None:
        return []
    p = _two_sided(z)
    top = [int(j) for j in np.argsort(p, kind="stable")[:INTERACTION_TOP]]
    pairs = list(itertools.combinations(sorted(top), 2))
    pair_p = np.ones(len(pairs))
    for i, (a, b) in enumerate(pairs):
        zi = _interaction_z(data, d, a, b)
        if zi is not None:
            pair_p[i] = float(_two_sided(np.array([zi]))[0])
    chosen = bh_select(np.concatenate([p, pair_p]), FDR)  # one budget: main effects and interactions
    ranking = [d.ids[j] for j in chosen if j < len(p)]
    for k in sorted((j - len(p) for j in chosen if j >= len(p)), key=lambda k: pair_p[k]):
        a, b = sorted(pairs[k], key=lambda j: p[j])
        ranking += [d.ids[j] for j in (a, b) if d.ids[j] not in ranking]
    return ranking


# --------------------------------------------------------------------------- Boruta

BorutaImportance = Literal["impurity", "permutation"]
# Frozen by a recorded pilot before any evaluation data: the declared default was kept.
BORUTA_IMPORTANCE: BorutaImportance = "permutation"
BORUTA_ALPHA = 0.01
BORUTA_TREES = 200
BORUTA_MAX_RUNS = 100
BORUTA_HOLDOUT = 0.368  # permutation importance is measured on rows the run's forest never saw


def _permutation_importance(
    forest: RandomForestClassifier, x: Array, y: NDArray[np.int64], rng: np.random.Generator
) -> Array:
    """Increase in held-out Brier score when each column is permuted once (Breiman 2001)."""
    base = float(np.mean((forest.predict_proba(x)[:, 1] - y) ** 2))
    n, k = x.shape
    out = np.empty(k)
    batch = max(1, 20_000 // max(n, 1))
    for start in range(0, k, batch):
        cols = range(start, min(k, start + batch))
        stacked = np.tile(x, (len(cols), 1))
        for i, c in enumerate(cols):
            stacked[i * n : (i + 1) * n, c] = rng.permutation(x[:, c])
        prob = forest.predict_proba(stacked)[:, 1].reshape(len(cols), n)
        out[start : start + len(cols)] = np.mean((prob - y) ** 2, axis=1) - base
    return out


def boruta(
    data: AnalysisInput,
    *,
    importance: BorutaImportance = BORUTA_IMPORTANCE,
    alpha: float = BORUTA_ALPHA,
    trees: int = BORUTA_TREES,
    max_runs: int = BORUTA_MAX_RUNS,
) -> list[str]:
    """Boruta all-relevant selection with a random forest (Kursa & Rudnicki 2010).

    Each run adds a permuted shadow copy of every feature not yet rejected, fits a forest
    (``max_features="sqrt"``) and records a hit for every feature whose importance beats the
    largest shadow importance. After each run a binomial test per undecided feature confirms
    it when the upper-tail p is below ``alpha / p`` and rejects it when the lower-tail p is,
    as the authors' R package does: the level is per tail and per look, with no correction
    for repeating the test after each of up to 100 runs, so the realised error exceeds the
    nominal level (about 8% of pure-null worlds gave a claim at ``alpha = 0.01`` in review).
    Tentative features left after ``max_runs`` runs are not listed. Confirmed features are
    ranked by median importance. Importance is either forest impurity decrease or, avoiding
    its bias towards continuous and many-valued features (Strobl et al. 2007), held-out
    permutation importance: each run splits the rows into a 63.2% part, drawn without
    replacement, on which the forest is grown (its trees bootstrap within that part, as a
    standard random forest does), and the remaining 36.8%, on which each column is permuted
    once. Survival worlds are analysed through the event indicator, which discards follow-up
    time. Empty when nothing is confirmed.

    References: Kursa & Rudnicki (2010) J Stat Softw 36(11); Breiman (2001) Mach Learn 45:5;
    Strobl, Boulesteix, Zeileis & Hothorn (2007) BMC Bioinformatics 8:25.
    """
    d = design(data)
    if not usable(data, d.x):
        return []
    x, y = d.x, data.y
    rng = np.random.default_rng(stable_seed("boruta", data.card.world_id))
    p = x.shape[1]
    hits = np.zeros(p)
    status = np.zeros(p, dtype=int)  # 0 tentative, 1 confirmed, -1 rejected
    history: list[Array] = []
    runs = 0
    while runs < max_runs and (status == 0).any():
        live = np.flatnonzero(status >= 0)
        shadow = np.column_stack([rng.permutation(x[:, j]) for j in live])
        both = np.column_stack([x[:, live], shadow])
        forest = RandomForestClassifier(
            n_estimators=trees, max_features="sqrt", random_state=int(rng.integers(2**31)), n_jobs=1
        )
        if importance == "impurity":
            forest.fit(both, y)
            imp = forest.feature_importances_
        else:
            order = rng.permutation(len(y))
            cut = round(len(y) * (1 - BORUTA_HOLDOUT))
            train, test = order[:cut], order[cut:]
            if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
                break
            forest.fit(both[train], y[train])
            imp = _permutation_importance(forest, both[test], y[test], rng)
        real, shad = imp[: len(live)], imp[len(live) :]
        full = np.full(p, np.nan)
        full[live] = real
        history.append(full)
        hits[live] += real > shad.max()
        runs += 1
        undecided = np.flatnonzero(status == 0)
        level = alpha / p
        status[undecided[stats.binom.sf(hits[undecided] - 1, runs, 0.5) < level]] = 1
        status[undecided[stats.binom.cdf(hits[undecided], runs, 0.5) < level]] = -1
    if not history:
        return []
    median = np.nanmedian(np.array(history), axis=0)
    return [d.ids[int(j)] for j in np.argsort(-median, kind="stable") if status[j] == 1]


# --------------------------------------------------------------------------- L1 paths (logistic, Cox)

PATH_LENGTH = 30
PATH_RATIO = 0.01  # smallest penalty as a share of the largest
SOLVER_TOL = 1e-4  # largest coefficient change (standardised scale) at convergence
SOLVER_MAX_ITER = 1000


@dataclass(frozen=True)
class _Problem:
    """A smooth loss (mean negative log-likelihood) with an optional unpenalised intercept."""

    loss: Callable[[Array], tuple[float, Array]]
    intercept: bool
    start: Array
    scale: float  # observations (logistic) or events (Cox): -scale * loss is the log-likelihood


def _problem(x: Array, y: NDArray[np.int64], time: Array | None) -> _Problem:
    """Logistic loss (intercept first) or the Breslow Cox partial likelihood per event."""
    if time is None:
        yf = y.astype(float)
        n = len(yf)

        def logistic(beta: Array) -> tuple[float, Array]:
            eta = beta[0] + x @ beta[1:]
            r = expit(eta) - yf
            value = float(np.mean(np.logaddexp(0.0, eta) - yf * eta))
            return value, np.concatenate([[r.mean()], x.T @ r / n])

        mean = float(np.clip(yf.mean(), 1e-6, 1 - 1e-6))
        start = np.zeros(x.shape[1] + 1)
        start[0] = np.log(mean / (1 - mean))
        return _Problem(logistic, True, start, float(n))
    order = np.argsort(time, kind="stable")
    t, xs = time[order], x[order]
    ev = np.flatnonzero(y.astype(bool)[order])
    start_at = np.searchsorted(t, t[ev], side="left")  # each event's risk set: sorted rows from here on
    events = max(len(ev), 1)
    event_sum = xs[ev].sum(axis=0)

    def cox(beta: Array) -> tuple[float, Array]:
        eta = xs @ beta
        top = float(eta.max()) if eta.size else 0.0
        w = np.exp(eta - top)
        s0 = np.cumsum(w[::-1])[::-1][start_at]
        value = -float(np.sum(eta[ev] - top - np.log(s0))) / events
        # sum over events of the risk-set mean of x, as one product: row j is weighted by w_j times
        # the sum of 1/s0 over the events whose risk set contains it
        share = np.zeros(len(t))
        np.add.at(share, start_at, 1.0 / s0)
        return value, -(event_sum - xs.T @ (w * np.cumsum(share))) / events

    return _Problem(cox, False, np.zeros(x.shape[1]), float(events))


def _fista(problem: _Problem, beta: Array, lam: float, lipschitz: float) -> tuple[Array, float]:
    """Accelerated proximal gradient for ``loss + lam * |beta|_1`` (Beck & Teboulle 2009).

    Backtracking on the Lipschitz constant, and momentum restarted whenever it points uphill
    (O'Donoghue & Candes 2015).
    """
    pen = slice(1, None) if problem.intercept else slice(None)
    previous, y, t = beta.copy(), beta.copy(), 1.0
    fy, gy = problem.loss(y)
    for _ in range(SOLVER_MAX_ITER):
        while True:
            z = y - gy / lipschitz
            z[pen] = np.sign(z[pen]) * np.maximum(np.abs(z[pen]) - lam / lipschitz, 0.0)
            fz, _ = problem.loss(z)
            step = z - y
            if fz <= fy + gy @ step + 0.5 * lipschitz * step @ step + 1e-12 or lipschitz > 1e12:
                break
            lipschitz *= 2.0
        moved = float(np.max(np.abs(z - previous))) if z.size else 0.0
        if (y - z) @ (z - previous) > 0:
            t = 1.0  # restart: the momentum step went uphill
        t_next = (1 + np.sqrt(1 + 4 * t * t)) / 2
        y = z + ((t - 1) / t_next) * (z - previous)
        previous, t = z, t_next
        if moved < SOLVER_TOL:
            break
        fy, gy = problem.loss(y)
    return previous, lipschitz


def _lambda_max(problem: _Problem) -> float:
    _, grad = problem.loss(problem.start)
    penalised = grad[1:] if problem.intercept else grad
    return float(np.max(np.abs(penalised))) if penalised.size else 0.0


def _l1_path(problem: _Problem, lambdas: Array, *, stop_above: int | None = None) -> list[Array]:
    """Warm-started solutions along decreasing ``lambdas`` (stops once more than ``stop_above`` are active)."""
    beta, lipschitz, out = problem.start.copy(), 1.0, []
    offset = 1 if problem.intercept else 0
    for lam in lambdas:
        beta, lipschitz = _fista(problem, beta, float(lam), lipschitz)
        out.append(beta.copy())
        if stop_above is not None and np.count_nonzero(beta[offset:]) > stop_above:
            break
    return out


def _grid(problem: _Problem) -> Array:
    top = _lambda_max(problem)
    return np.asarray(top * np.geomspace(1.0, PATH_RATIO, PATH_LENGTH), dtype=float)


def entry_order(x: Array, y: NDArray[np.int64], time: Array | None, limit: int) -> list[int]:
    """The first ``limit`` columns to enter the L1 path (ties by coefficient size)."""
    problem = _problem(x, y, time)
    offset = 1 if problem.intercept else 0
    order: list[int] = []
    for beta in _l1_path(problem, _grid(problem), stop_above=limit):
        coef = np.abs(beta[offset:])
        order += [int(j) for j in np.argsort(-coef, kind="stable") if coef[j] > 0 and j not in order]
        if len(order) >= limit:
            break
    return order[:limit]


# --------------------------------------------------------------------------- penalised Cox

CV_FOLDS = 5


def _folds(y: NDArray[np.int64], k: int, seed: int) -> NDArray[np.int64]:
    """Fold labels stratified by outcome (event) class."""
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y), dtype=np.int64)
    for cls in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == cls))
        fold[idx] = np.arange(len(idx)) % k
    return fold


def _log_lik(x: Array, y: NDArray[np.int64], time: Array | None, beta: Array) -> float:
    problem = _problem(x, y, time)
    return -problem.scale * problem.loss(beta)[0]


def penalised_cox(data: AnalysisInput) -> list[str]:
    """L1-penalised Cox regression on survival worlds (L1 logistic on binary worlds).

    The lasso path (30 penalties down to 1% of the largest, Breslow ties, accelerated proximal
    gradient) is fitted on all revealed rows; the penalty is the largest whose 5-fold
    cross-validated error is within one standard error of the minimum (the one-standard-error
    rule), where the Cox error is the cross-validated partial likelihood of Verweij & van
    Houwelingen (1993), ``l(beta_-k) - l_-k(beta_-k)`` per event in fold ``k``, and the
    logistic error is the held-out deviance. The non-zero coefficients at that penalty are
    listed, largest first; empty when none is.

    References: Tibshirani (1997) Stat Med 16:385; Verweij & van Houwelingen (1993) Stat Med
    12:2305; Simon, Friedman, Hastie & Tibshirani (2011) J Stat Softw 39(5); Hastie, Tibshirani &
    Friedman (2009) The Elements of Statistical Learning, section 7.10 (one-standard-error rule).
    """
    d = design(data)
    if not usable(data, d.x):
        return []
    x, y, time = d.x, data.y, data.time
    problem = _problem(x, y, time)
    lambdas = _grid(problem)
    if lambdas[0] <= 0:
        return []
    path = _l1_path(problem, lambdas)
    fold = _folds(y, CV_FOLDS, stable_seed("penalised-cv", data.card.world_id))
    errors = np.full((CV_FOLDS, len(lambdas)), np.nan)
    for k in range(CV_FOLDS):
        train, test = fold != k, fold == k
        t_train = None if time is None else time[train]
        if len(np.unique(y[train])) < 2 or not y[test].any():
            continue
        for i, beta in enumerate(_l1_path(_problem(x[train], y[train], t_train), lambdas)):
            if time is None:
                errors[k, i] = _problem(x[test], y[test], None).loss(beta)[0]
            else:
                cvpl = _log_lik(x, y, time, beta) - _log_lik(x[train], y[train], t_train, beta)
                errors[k, i] = -cvpl / float(y[test].sum())
    valid = ~np.isnan(errors).any(axis=1)
    if valid.sum() < 2:
        return []
    mean = errors[valid].mean(axis=0)
    se = errors[valid].std(axis=0, ddof=1) / np.sqrt(valid.sum())
    best = int(np.argmin(mean))
    chosen = int(np.flatnonzero(mean <= mean[best] + se[best])[0])
    coef = np.abs(path[chosen][1:] if problem.intercept else path[chosen])
    return [d.ids[int(j)] for j in np.argsort(-coef, kind="stable") if coef[j] > 1e-8]


# --------------------------------------------------------------------------- stability selection

STABILITY_PAIRS = 50  # complementary pairs of half-samples (100 subsamples)
STABILITY_THRESHOLD = 0.75  # selection-frequency threshold pi_thr
STABILITY_PFER = 1.0  # bound on the expected number of falsely selected features


def stability_q(p: int, *, pfer: float = STABILITY_PFER, threshold: float = STABILITY_THRESHOLD) -> int:
    """Largest per-subsample selection size ``q`` with ``q^2 / ((2 pi_thr - 1) p) <= PFER``."""
    return int(np.floor(np.sqrt(pfer * (2 * threshold - 1) * p) + 1e-12))


def pfer_bound(q: int, p: int, threshold: float = STABILITY_THRESHOLD) -> float:
    """Meinshausen-Buhlmann (2010, Theorem 1) bound on E[V] for ``q`` selections out of ``p``."""
    return q * q / ((2 * threshold - 1) * p)


def stability_pfer(data: AnalysisInput) -> list[str]:
    """Stability selection with the per-family error rate controlled at E[V] <= 1.

    On each of 100 half-samples, drawn as 50 complementary pairs, the selection is
    the first ``q`` features to enter the L1 path (logistic, or Cox on survival worlds). With
    threshold ``pi_thr = 0.75`` the Meinshausen-Buhlmann bound
    ``E[V] <= q^2 / ((2 pi_thr - 1) p)`` is held at 1 by choosing ``q = floor(sqrt(0.5 p))``
    and asserted before selection. Frequencies count every half-sample separately, so the bound
    applied is Meinshausen and Buhlmann's for subsamples of size n/2; the complementary-pair
    sampling (Shah & Samworth 2013) only reduces Monte Carlo variance and its sharper bounds
    are not used. Features selected in at least 75% of half-samples are listed by frequency.
    Empty when none is, or when ``q`` would be zero.

    References: Meinshausen & Buhlmann (2010) J R Stat Soc B 72:417; Shah & Samworth (2013)
    J R Stat Soc B 75:55; Tibshirani (1996) J R Stat Soc B 58:267.
    """
    d = design(data)
    if not usable(data, d.x):
        return []
    x, y, time = d.x, data.y, data.time
    p = x.shape[1]
    q = stability_q(p)
    if q < 1:
        return []
    if pfer_bound(q, p) > STABILITY_PFER + 1e-12:
        raise AssertionError("stability selection would exceed its error bound")
    rng = np.random.default_rng(stable_seed("stability-pfer", data.card.world_id))
    n = len(y)
    counts = np.zeros(p)
    draws = 0
    for _ in range(STABILITY_PAIRS):
        order = rng.permutation(n)
        for half in (order[: n // 2], order[n // 2 : 2 * (n // 2)]):
            draws += 1
            if len(np.unique(y[half])) < 2:
                continue
            chosen = entry_order(x[half], y[half], None if time is None else time[half], q)
            counts[chosen] += 1
    freq = counts / draws
    return [d.ids[int(j)] for j in np.argsort(-freq, kind="stable") if freq[j] >= STABILITY_THRESHOLD]


# --------------------------------------------------------------------------- invariant causal prediction

ICP_ALPHA = 0.05  # overall level, Bonferroni-split over the environment variables
ICP_PRESELECT = 8  # candidates entering the subset search, by L1-path entry order
ICP_MIN_ROWS = 20  # an environment needs this many rows ...
ICP_MIN_CLASS = 3  # ... and this many of each outcome class (events and non-events)


def _environment_variables(data: AnalysisInput, d: Design) -> list[tuple[int | None, NDArray[np.int64]]]:
    """Candidate environment partitions: the recruitment strata, then each binary clinical indicator.

    Returns ``(column, labels)`` pairs; ``column`` is the candidate index used as the
    environment (``None`` for the strata), which is then excluded from the predictors. A
    partition is kept only when every environment has ``ICP_MIN_ROWS`` rows and
    ``ICP_MIN_CLASS`` of each outcome class.
    """
    out: list[tuple[int | None, NDArray[np.int64]]] = []
    candidates: list[tuple[int | None, Sequence[object]]] = [(None, data.stratum)]
    candidates += [(j, d.x[:, j].round(9).tolist()) for j in d.clinical]
    for column, values in candidates:
        levels = sorted(set(values), key=str)
        if len(levels) < 2:
            continue
        labels = np.array([levels.index(v) for v in values], dtype=np.int64)
        ok = all(
            (labels == e).sum() >= ICP_MIN_ROWS
            and min(int(data.y[labels == e].sum()), int((1 - data.y[labels == e]).sum())) >= ICP_MIN_CLASS
            for e in range(len(levels))
        )
        if ok:
            out.append((column, labels))
    return out


def _fit_log_lik(design_x: Array, y: NDArray[np.int64], time: Array | None) -> float | None:
    """Maximised log-likelihood of a logistic (with intercept) or Cox model, ``None`` if the fit fails."""
    if time is None:
        exog = np.column_stack([np.ones(len(y)), design_x])
        prob = logistic_fit(exog, y.astype(float))
        if prob is None:
            return None
        prob = np.clip(prob, 1e-12, 1 - 1e-12)
        return float(np.sum(y * np.log(prob) + (1 - y) * np.log(1 - prob)))
    event = y.astype(bool)
    if design_x.shape[1] == 0:
        return cox_terms(np.zeros((len(y), 1)), np.zeros(1), time, event)[0]
    beta = cox_fit(design_x, time, event)
    return None if beta is None else cox_terms(design_x, beta, time, event)[0]


def _invariance_p(
    x: Array, subset: tuple[int, ...], env: NDArray[np.int64], data: AnalysisInput, null: float | None
) -> float:
    """Likelihood-ratio p that ``Y | X_S`` is the same in every environment.

    The alternative adds environment dummies and their interactions with ``X_S``. A failed
    fit accepts the set (p = 1), the conservative direction for an intersection of accepted sets.
    """
    if null is None:
        return 1.0
    dummies = np.column_stack([(env == e).astype(float) for e in range(1, int(env.max()) + 1)])
    xs = x[:, list(subset)]
    parts = [xs, dummies] + [dummies * xs[:, [j]] for j in range(xs.shape[1])]
    alternative = _fit_log_lik(np.column_stack(parts), data.y, data.time)
    if alternative is None:
        return 1.0
    df = dummies.shape[1] * (xs.shape[1] + 1)
    return float(stats.chi2.sf(max(0.0, 2 * (alternative - null)), df))


def invariant_prediction(data: AnalysisInput) -> list[str]:
    """Invariant causal prediction across environments (Peters, Buhlmann & Meinshausen 2016).

    For a set ``S`` the null hypothesis is that the conditional model of the outcome given
    ``X_S`` (logistic, or Cox on survival worlds) is the same in every environment; it is
    tested by a likelihood-ratio test against environment-specific intercepts and slopes.
    Every subset of the ``ICP_PRESELECT`` = 8 candidates first entering the pooled L1 path is
    tested, and the estimate is the intersection of the accepted sets (empty when none is
    accepted: the model is rejected). The per-feature p-value is the largest p of any set
    without the feature.

    Environments are the cohort-level partitions published with the patients: the
    recruitment strata when there are several, and each binary clinical indicator (the cohort
    indicator is one of them, and fake names do not say which). Each partition is analysed
    separately with its indicator excluded from the predictors, at level 0.05 divided by the
    number of partitions (Bonferroni), and the selections are pooled, ranked by adjusted
    p-value. The list is empty when no partition has two environments with at least 20 rows
    and 3 of each outcome class (one environment cannot reveal invariance), or when no
    feature is selected.

    Limits. ICP's coverage guarantee (the estimate lies within the causal parents with
    probability at least 1 - alpha) holds only if the environments are interventions that do
    not act on the outcome directly, and only if the parent set is inside the preselected pool;
    the data-driven L1 preselection voids the guarantee whenever it drops a parent. Strata and
    clinical indicators are not designed interventions: in these worlds an indicator often
    affects the outcome directly, every set is then rejected, and ICP abstains (on about 95-97%
    of signal worlds in development checks).

    References: Peters, Buhlmann & Meinshausen (2016) J R Stat Soc B 78:947; Heinze-Deml,
    Peters & Meinshausen (2018) J Causal Inference 6(2) (nonlinear and non-Gaussian tests).
    """
    d = design(data)
    if not usable(data, d.x):
        return []
    environments = _environment_variables(data, d)
    if not environments:
        return []
    level = ICP_ALPHA / len(environments)
    ranked = entry_order(d.x, data.y, data.time, ICP_PRESELECT + 1)
    best: dict[int, float] = {}
    null_cache: dict[tuple[int, ...], float | None] = {}
    for column, env in environments:
        pool = [j for j in ranked if j != column][:ICP_PRESELECT]
        p_by_set: dict[tuple[int, ...], float] = {}
        for size in range(len(pool) + 1):
            for subset in itertools.combinations(pool, size):
                if subset not in null_cache:
                    null_cache[subset] = _fit_log_lik(d.x[:, list(subset)], data.y, data.time)
                p_by_set[subset] = _invariance_p(d.x, subset, env, data, null_cache[subset])
        accepted = [set(s) for s, pv in p_by_set.items() if pv > level]
        if not accepted:
            continue
        for j in set.intersection(*accepted):
            pj = max(pv for s, pv in p_by_set.items() if j not in s)
            best[j] = min(best.get(j, 1.0), pj * len(environments))
    return [d.ids[j] for j in sorted(best, key=lambda j: (best[j], j))]
