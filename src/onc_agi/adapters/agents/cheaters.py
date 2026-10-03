"""Cheater and shortcut agents.

Each encodes a strategy that should *not* be rewarded. The release gates require
every one of them to score at the floor (or chance) on every released tier.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from onc_agi.adapters.agents.baselines import prepared, univariate_bh
from onc_agi.core.ports import WorldStore
from onc_agi.core.schema import GroupLabel, Tier, Timing, WorldCard
from onc_agi.services.kit import AnalysisInput
from onc_agi.services.scoring import stable_seed

FAMOUS = (
    "TP53",
    "ERBB2",
    "ESR1",
    "PGR",
    "BRCA1",
    "BRCA2",
    "MKI67",
    "GATA3",
    "FOXA1",
    "MYC",
    "PIK3CA",
    "CCND1",
)


def giant_list(data: AnalysisInput) -> list[str]:
    """List everything, post-outcome measurements included."""
    return list(data.card.feature_ids())


def always_empty(data: AnalysisInput) -> list[str]:
    return []


def random_abstain(data: AnalysisInput) -> list[str]:
    """Abstain on a coin flip, otherwise list a random ordering (no restraint skill)."""
    rng = np.random.default_rng(stable_seed("random-abstain", data.card.world_id))
    if rng.random() < 0.5:
        return []
    ids, _ = prepared(data)
    return [ids[int(j)] for j in rng.permutation(len(ids))]


def random_ranking(data: AnalysisInput) -> list[str]:
    rng = np.random.default_rng(stable_seed("random", data.card.world_id))
    ids, _ = prepared(data)
    return [ids[int(j)] for j in rng.permutation(len(ids))]


def leak_exploiter(data: AnalysisInput) -> list[str]:
    """Post-outcome features first (they predict best), then univariate hits."""
    post = [f.feature_id for f in data.card.features if f.timing is Timing.POST_OUTCOME]
    return post + [f for f in univariate_bh(data) if f not in post]


def auc_maximiser(data: AnalysisInput, top: int = 20) -> list[str]:
    """Top features by univariate AUC, post-outcome included: what a prediction leaderboard rewards."""
    ids, x = prepared(data, include_post_outcome=True)
    if x.shape[0] < 10 or len(np.unique(data.y)) < 2:
        return []
    auc = np.array([abs(roc_auc_score(data.y, x[:, j]) - 0.5) for j in range(x.shape[1])])
    return [ids[int(j)] for j in np.argsort(-auc)[:top]]


def famous_genes(data: AnalysisInput) -> list[str]:
    """Literature recall: famous cancer genes if their names are visible, else a guess."""
    present = [g for g in FAMOUS if g in data.card.feature_ids()]
    if present:
        return present
    return random_ranking(data)[:5]


def metadata_only(data: AnalysisInput, top: int = 10) -> list[str]:
    """Rank by price and data type only (never reads values or outcomes)."""
    feats = sorted(data.card.features, key=lambda f: (-f.assay_price, f.data_type, f.feature_id))
    return [f.feature_id for f in feats if f.timing is Timing.BASELINE][:top]


def _abs_corr(x: NDArray[np.float64]) -> NDArray[np.float64]:
    if x.shape[0] < 3:
        return np.zeros((x.shape[1], x.shape[1]))
    c = np.abs(np.nan_to_num(np.corrcoef(x, rowvar=False)))
    np.fill_diagonal(c, 0.0)
    return c


def variance_ranker(data: AnalysisInput, top: int = 10) -> list[str]:
    """Outcome-blind: highest raw variance first."""
    timing = {f.feature_id: f.timing for f in data.card.features}
    keep = [j for j, f in enumerate(data.feature_ids) if timing[f] is Timing.BASELINE]
    var = np.nanvar(data.x[:, keep], axis=0) if data.x.shape[0] else np.zeros(len(keep))
    return [data.feature_ids[keep[int(j)]] for j in np.argsort(-var)[:top]]


def hub_ranker(data: AnalysisInput, top: int = 10) -> list[str]:
    """Outcome-blind: most correlated-with-others first (hubs of the correlation graph)."""
    ids, x = prepared(data)
    degree = (_abs_corr(x) > 0.5).sum(axis=0)
    return [ids[int(j)] for j in np.argsort(-degree, kind="stable")[:top]]


def cluster_size_ranker(data: AnalysisInput, top: int = 10) -> list[str]:
    """Outcome-blind: features with a near-duplicate partner first (stand-in placement artefact)."""
    ids, x = prepared(data)
    best = _abs_corr(x).max(axis=0) if x.shape[1] else np.zeros(0)
    return [ids[int(j)] for j in np.argsort(-best, kind="stable")[:top]]


def synthetic_ranker(data: AnalysisInput, top: int = 10) -> list[str]:
    """Outcome-blind: derived (synthetic-looking) baseline features first."""
    derived = [
        f.feature_id for f in data.card.features if f.data_type == "derived" and f.timing is Timing.BASELINE
    ]
    return derived[:top]


# ---------------------------------------------------------------------------- learned outcome-blind ranker

_TYPES = ("expression", "copy_number", "protein", "clinical", "mutation", "lab", "derived")


def descriptors(card: WorldCard, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Outcome-free per-feature descriptors."""
    filled = np.where(np.isnan(x), np.nanmedian(x, axis=0), x) if x.size else x
    corr = _abs_corr(filled) if filled.size else np.zeros((x.shape[1], x.shape[1]))
    rows = []
    for j, meta in enumerate(card.features):
        col = filled[:, j] if filled.size else np.zeros(1)
        onehot = [1.0 if meta.data_type == t else 0.0 for t in _TYPES]
        rows.append(
            [
                float(np.log1p(np.var(col))),
                float(np.mean(col)),
                float(corr[j].max()) if corr.size else 0.0,
                float((corr[j] > 0.5).sum()) if corr.size else 0.0,
                float(meta.timing is Timing.POST_OUTCOME),
                float(np.mean(np.isnan(x[:, j]))) if x.size else 0.0,
                *onehot,
            ]
        )
    return np.asarray(rows, dtype=float)


@dataclass
class LearnedRanker:
    """Trained on public-train answer keys to predict truth from descriptors alone."""

    model: LogisticRegression | None = None
    top: int = 10

    def fit(self, store: WorldStore, world_ids: Sequence[str]) -> LearnedRanker:
        feats: list[NDArray[np.float64]] = []
        labels: list[int] = []
        for wid in world_ids:
            world = store.world(wid)
            if world.card.tier is not Tier.PUBLIC_TRAIN:
                # nothing learns from eval or private worlds
                raise ValueError(
                    f"{wid} is a {world.card.tier.value} world; training reads public-train only"
                )
            key = store.answer_key(wid)
            truth = {
                f
                for g in key.groups
                if g.label is GroupLabel.RECOVERABLE
                for p in g.parts
                for f in p.equivalence_set
            }
            feats.append(descriptors(world.card, world.x))
            labels.extend(1 if f in truth else 0 for f in world.card.feature_ids())
        y = np.asarray(labels)
        if len(np.unique(y)) < 2:
            raise ValueError("training worlds need both true and non-true features")
        self.model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(np.vstack(feats), y)
        return self

    def __call__(self, data: AnalysisInput) -> list[str]:
        if self.model is None:
            raise RuntimeError("LearnedRanker must be fitted on public-train worlds first")
        full = np.full((data.x.shape[0], len(data.card.features)), np.nan)
        index = {f: j for j, f in enumerate(data.card.feature_ids())}
        for j, f in enumerate(data.feature_ids):
            full[:, index[f]] = data.x[:, j]
        prob = self.model.predict_proba(descriptors(data.card, full))[:, 1]
        return [data.card.features[int(j)].feature_id for j in np.argsort(-prob)[: self.top]]
