"""In-memory world data shared by the engine, scorer and builder.

A :class:`WorldData` holds the revealable pool exactly as an agent may see it
(already passed through the private observation layer) plus the pre-shuffled
recruitment queues that make sequential mode deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from onc_agi.core.schema import WorldCard


@dataclass(frozen=True)
class WorldData:
    card: WorldCard
    patient_ids: tuple[str, ...]
    x: NDArray[np.float64]  # (n_pool, n_features), NaN = missing; columns follow card.features
    y: NDArray[np.int64]  # (n_pool,) binary outcome; survival worlds: the event indicator
    stratum: tuple[str, ...]
    queues: dict[str, tuple[int, ...]]  # stratum -> row indices in fixed recruitment order
    time: NDArray[np.float64] | None = None  # (n_pool,) survival follow-up in days, else None

    def __post_init__(self) -> None:
        if self.x.ndim != 2:
            raise ValueError("feature matrix must have two dimensions")
        n, p = self.x.shape
        if p != len(self.card.features):
            raise ValueError("feature matrix width differs from the world card")
        if (
            n != self.card.n_pool
            or len(self.patient_ids) != n
            or self.y.shape != (n,)
            or len(self.stratum) != n
        ):
            raise ValueError("world rows are misaligned with the world card")
        if not all(isinstance(v, str) and v for v in self.patient_ids) or len(set(self.patient_ids)) != n:
            raise ValueError("patient identifiers must be nonempty and unique")
        if not np.isin(self.y, (0, 1)).all():
            raise ValueError("outcome values must be binary")
        if not np.issubdtype(self.x.dtype, np.number) or np.iscomplexobj(self.x):
            raise ValueError("feature matrix must contain real numeric measurements")
        if np.isinf(self.x).any():
            raise ValueError("feature matrix cannot contain infinite measurements")
        if len(set(self.card.strata)) != len(self.card.strata) or not all(self.card.strata):
            raise ValueError("card strata must be nonempty and unique")
        if (self.time is not None) != (self.card.outcome_type == "survival") or (
            self.time is not None and self.time.shape != (n,)
        ):
            raise ValueError("survival worlds (and only they) carry one follow-up time per row")
        if set(self.queues) != set(self.card.strata):
            raise ValueError("recruitment queues must cover exactly the card strata")
        if any(
            not isinstance(i, (int, np.integer)) or isinstance(i, bool)
            for q in self.queues.values()
            for i in q
        ):
            raise ValueError("recruitment queues must contain integer row indices")
        seen = sorted(i for q in self.queues.values() for i in q)
        if seen != list(range(n)):
            raise ValueError("recruitment queues must partition the pool")
        if any(self.stratum[i] != s for s, q in self.queues.items() for i in q):
            raise ValueError("recruitment queues disagree with each row's stratum")
        if self.card.stratum_sizes and self.card.stratum_sizes != {s: len(q) for s, q in self.queues.items()}:
            raise ValueError("published stratum sizes disagree with the pool")

    def column(self, feature_id: str) -> int:
        return self.card.feature_ids().index(feature_id)
