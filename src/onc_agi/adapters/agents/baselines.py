"""Standard static baselines as analysis functions.

Every honest baseline ignores ``post_outcome`` features (standard practice) and
returns an ordered list with an explicit empty-list rule, so it can abstain.
Tuning uses only the data revealed in the episode (never eval or private truth).

Survival worlds (``AnalysisInput.time`` present) are analysed with Cox models where the
method has a Cox form: :func:`univariate_bh` uses the Cox score (log-rank-type) test and
:func:`forward_score` selects with Cox score tests. The penalised and tree baselines analyse
the event indicator as a binary outcome there, which is valid but discards follow-up time.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from onc_agi.core.schema import Timing
from onc_agi.services.kit import AnalysisInput
from onc_agi.services.score_test import candidate_score_z
from onc_agi.services.scoring import stable_seed

log = logging.getLogger(__name__)

FDR = 0.05


def prepared(
    data: AnalysisInput, *, include_post_outcome: bool = False
) -> tuple[list[str], NDArray[np.float64]]:
    """Feature ids and standardised, median-imputed columns (post-outcome dropped by default)."""
    timing = {f.feature_id: f.timing for f in data.card.features}
    keep = [j for j, f in enumerate(data.feature_ids) if include_post_outcome or timing[f] is Timing.BASELINE]
    x = data.x[:, keep]
    if x.size:
        x = np.where(np.isnan(x), np.nanmedian(x, axis=0), x)
        x = np.nan_to_num(x)
        sd = x.std(axis=0)
        x = (x - x.mean(axis=0)) / np.where(sd == 0, 1.0, sd)
    return [data.feature_ids[j] for j in keep], x


def _usable(data: AnalysisInput, x: NDArray[np.float64]) -> bool:
    return bool(
        x.shape[0] >= 10 and x.shape[1] > 0 and len(np.unique(data.y)) == 2 and min(np.bincount(data.y)) >= 3
    )


def _cox_score_p(
    x: NDArray[np.float64], time: NDArray[np.float64], y: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Two-sided p of each column's marginal Cox score (log-rank-type) statistic, Breslow ties."""
    z = candidate_score_z(x, [], y, time)
    if z is None:
        return np.ones(x.shape[1])
    return np.asarray(2 * stats.norm.sf(np.abs(z)), dtype=float)


def univariate_bh(data: AnalysisInput) -> list[str]:
    """Wilcoxon rank-sum per feature (Cox score test on survival worlds), Benjamini-Hochberg at 5%.

    Empty when nothing passes.
    """
    ids, x = prepared(data)
    if not _usable(data, x):
        return []
    y = data.y.astype(bool)
    if data.time is not None:
        p = _cox_score_p(x, data.time, data.y)
    else:
        p = np.array([stats.mannwhitneyu(x[y, j], x[~y, j]).pvalue for j in range(x.shape[1])])
    order = np.argsort(p)
    m = len(p)
    passed = p[order] <= FDR * np.arange(1, m + 1) / m
    k = int(np.max(np.flatnonzero(passed)) + 1) if passed.any() else 0
    return [ids[j] for j in order[:k]]


FORWARD_ALPHA = 0.05
FORWARD_MAX = 10


def forward_score(
    data: AnalysisInput, *, alpha: float = FORWARD_ALPHA, max_size: int = FORWARD_MAX
) -> list[str]:
    """Forward stepwise selection by score tests, Bonferroni entry at ``alpha / p``.

    Each step adds the feature whose score test against the current model (logistic with an
    intercept, or Cox on survival worlds) is largest, while its two-sided p clears the
    Bonferroni level; selection order is the ranking. Empty when nothing enters.
    """
    ids, x = prepared(data)
    if not _usable(data, x):
        return []
    level = float(stats.norm.isf(alpha / (2 * x.shape[1])))
    chosen: list[int] = []
    while len(chosen) < min(max_size, x.shape[1]):
        z = candidate_score_z(x, chosen, data.y, data.time)
        if z is None:
            break
        best = int(np.argmax(np.abs(z)))
        if abs(float(z[best])) <= level:
            break
        chosen.append(best)
    return [ids[j] for j in chosen]


def _penalised(data: AnalysisInput, l1_ratio: float) -> list[str]:
    ids, x = prepared(data)
    if not _usable(data, x):
        return []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = LogisticRegressionCV(
            Cs=10,
            cv=5,
            solver="saga",
            l1_ratios=[l1_ratio],
            scoring="neg_log_loss",
            max_iter=2000,
            random_state=stable_seed("cv", data.card.world_id),
            use_legacy_attributes=False,
        ).fit(x, data.y)
    coef = np.abs(model.coef_[0])
    order = [int(j) for j in np.argsort(-coef) if coef[j] > 1e-8]
    return [ids[j] for j in order]


def lasso(data: AnalysisInput) -> list[str]:
    """Cross-validated L1 logistic regression; the non-zero coefficients, largest first."""
    return _penalised(data, 1.0)


def elastic_net(data: AnalysisInput) -> list[str]:
    return _penalised(data, 0.5)


def stability_selection(data: AnalysisInput, *, draws: int = 50, cutoff: float = 0.6) -> list[str]:
    """Half-sample L1 selection frequency (Meinshausen-Buhlmann); features selected in >= 60%."""
    ids, x = prepared(data)
    if not _usable(data, x):
        return []
    rng = np.random.default_rng(stable_seed("stability", data.card.world_id))
    counts = np.zeros(x.shape[1])
    n = x.shape[0]
    for _ in range(draws):
        idx = rng.choice(n, size=n // 2, replace=False)
        if len(np.unique(data.y[idx])) < 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            fit = LogisticRegression(l1_ratio=1.0, C=0.05, solver="liblinear").fit(x[idx], data.y[idx])
        counts += np.abs(fit.coef_[0]) > 1e-8
    freq = counts / draws
    order = [int(j) for j in np.argsort(-freq) if freq[j] >= cutoff]
    return [ids[j] for j in order]


def random_forest(data: AnalysisInput) -> list[str]:
    """Impurity importance above the largest importance obtained with a permuted outcome."""
    ids, x = prepared(data)
    if not _usable(data, x):
        return []
    seed = stable_seed("rf", data.card.world_id)
    forest = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1).fit(x, data.y)
    rng = np.random.default_rng(seed)
    null = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1).fit(
        x, rng.permutation(data.y)
    )
    cut = float(null.feature_importances_.max())
    imp = forest.feature_importances_
    return [ids[int(j)] for j in np.argsort(-imp) if imp[j] > cut]


def knockoffs(data: AnalysisInput, *, fdr: float = 0.1) -> list[str]:
    """Model-X Gaussian knockoffs with lasso statistics (knockpy), knockoff (offset 0) threshold.

    The knockoff+ threshold (offset 1) cannot select fewer than about 1/q features,
    so with the sparse truths of most worlds it never selects anything; the offset-0
    threshold controls a modified FDR and can make small selections.
    """
    ids, x = prepared(data)
    if not _usable(data, x) or x.shape[1] < 2:
        return []
    from knockpy import knockoff_stats  # optional dependency
    from knockpy.knockoff_filter import KnockoffFilter

    np.random.seed(stable_seed("knockoffs", data.card.world_id) % (2**32))
    kfilter = KnockoffFilter(ksampler="gaussian", fstat="lasso")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kfilter.forward(X=x, y=data.y.astype(float), fdr=fdr)
    stat = np.asarray(kfilter.W, dtype=float)
    threshold = float(knockoff_stats.data_dependent_threshhold(W=stat, fdr=fdr, offset=0))
    order = [int(j) for j in np.argsort(-stat) if stat[j] >= threshold and np.isfinite(threshold)]
    return [ids[j] for j in order]
