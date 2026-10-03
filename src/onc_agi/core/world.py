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
    y: NDArray[np.int64]  # (n_pool,) binary outcome
    stratum: tuple[str, ...]
    queues: dict[str, tuple[int, ...]]  # stratum -> row indices in fixed recruitment order

    def __post_init__(self) -> None:
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
        if set(self.queues) != set(self.card.strata):
            raise ValueError("recruitment queues must cover exactly the card strata")
        seen = sorted(i for q in self.queues.values() for i in q)
        if seen != list(range(n)):
            raise ValueError("recruitment queues must partition the pool")

    def column(self, feature_id: str) -> int:
        return self.card.feature_ids().index(feature_id)
