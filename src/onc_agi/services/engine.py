"""Episode engine: the single owner of what an agent sees and what it has spent.

Both modes run through :class:`Episode`:

* **full access** - ``reset`` reveals the whole revealable pool with every feature
  measured; only ``submit`` remains.
* **sequential** - ``reset`` reveals nothing; ``recruit`` pops patients from a
  fixed, pre-shuffled queue per stratum (revealing their outcome); ``assay``
  measures features on every recruited patient not yet measured; ``submit`` ends
  the episode. Prices are published and truth-blind; spend is checked against the
  truth-blind budget before any state changes.

Requests carry identifiers: replaying an identical request returns the original
response without charging again; reusing an identifier for a different payload
is a conflict.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from onc_agi.core.errors import ArenaError
from onc_agi.core.schema import (
    Action,
    Assay,
    EpisodeStatus,
    ErrorCode,
    Mode,
    Observation,
    Recruit,
    Reset,
    RevealedData,
    Submit,
)
from onc_agi.core.world import WorldData


@dataclass(frozen=True)
class EpisodeView:
    """Numeric view of revealed data for in-process agents (no per-cell validation)."""

    world_id: str
    mode: Mode
    step: int
    status: EpisodeStatus
    budget: float
    spent: float
    available: tuple[str, ...]
    rows: tuple[int, ...]
    outcome: NDArray[np.int64]
    stratum: tuple[str, ...]
    feature_ids: tuple[str, ...]
    x: NDArray[np.float64]  # (len(rows), len(feature_ids)); NaN where not measured
    measured: tuple[bool, ...]  # per feature: measured on at least one recruited row
    time: NDArray[np.float64] | None = None  # survival worlds: follow-up of revealed rows

    def to_observation(self, patient_ids: tuple[str, ...]) -> Observation:
        """Validated wire form (measured columns only)."""
        columns = {
            fid: tuple(None if np.isnan(v) else float(v) for v in self.x[:, j])
            for j, fid in enumerate(self.feature_ids)
            if self.measured[j]
        }
        return Observation(
            world_id=self.world_id,
            mode=self.mode,
            step=self.step,
            status=self.status,
            budget=self.budget,
            spent=self.spent,
            available_actions=self.available,
            revealed=RevealedData(
                patient_ids=tuple(patient_ids[i] for i in self.rows),
                outcome=tuple(int(v) for v in self.outcome),
                stratum=self.stratum,
                columns=columns,
                time=None if self.time is None else tuple(float(v) for v in self.time),
            ),
        )


@dataclass
class Episode:
    world: WorldData
    step: int = 0
    status: EpisodeStatus = EpisodeStatus.ACTIVE
    spent: float = 0.0
    submission: tuple[str, ...] | None = None
    _recruited: list[int] = field(default_factory=list)
    _cursor: dict[str, int] = field(default_factory=dict)
    _measured: NDArray[np.bool_] | None = None
    _requests: dict[str, tuple[str, EpisodeView]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n, p = self.world.x.shape
        self._measured = np.zeros((n, p), dtype=bool)
        self._cursor = {s: 0 for s in self.world.card.strata}
        if self.world.card.mode is Mode.FULL_ACCESS:
            self._recruited = list(range(n))
            self._measured[:, :] = True

    # ------------------------------------------------------------------ public API

    @property
    def mode(self) -> Mode:
        return self.world.card.mode

    def view(self) -> EpisodeView:
        assert self._measured is not None
        rows = np.array(self._recruited, dtype=np.int64)
        fids = self.world.card.feature_ids()
        if rows.size:
            x = np.where(self._measured[rows], self.world.x[rows], np.nan)
            measured = tuple(bool(v) for v in self._measured[rows].any(axis=0))
            y = self.world.y[rows]
        else:
            x = np.empty((0, len(fids)))
            measured = tuple(False for _ in fids)
            y = np.empty(0, dtype=np.int64)
        return EpisodeView(
            world_id=self.world.card.world_id,
            mode=self.mode,
            step=self.step,
            status=self.status,
            budget=self.world.card.budget,
            spent=self.spent,
            available=self._available(),
            rows=tuple(int(r) for r in rows),
            outcome=y,
            stratum=tuple(self.world.stratum[r] for r in self._recruited),
            feature_ids=fids,
            x=x,
            measured=measured,
            time=None if self.world.time is None else self.world.time[rows],
        )

    def apply(self, action: Action) -> EpisodeView:
        """Apply one action and return the resulting view (idempotent per request id)."""
        fingerprint = action.model_dump_json()
        previous = self._requests.get(action.request_id)
        if previous is not None:
            if previous[0] != fingerprint:
                raise ArenaError(
                    ErrorCode.REQUEST_CONFLICT, f"request {action.request_id} reused for a different action"
                )
            return previous[1]
        if isinstance(action, Reset):
            if action.world_id != self.world.card.world_id:
                raise ArenaError(ErrorCode.UNKNOWN_WORLD, f"episode is for {self.world.card.world_id}")
            if self.step > 0:
                # Reset is idempotent: it never rewinds or advances a started episode.
                result = self.view()
                self._requests[action.request_id] = (fingerprint, result)
                return result
        elif self.status is EpisodeStatus.SUBMITTED:
            raise ArenaError(ErrorCode.EPISODE_CLOSED, "the episode has already been submitted")
        elif action.kind not in self._available():
            raise ArenaError(
                ErrorCode.ACTION_NOT_AVAILABLE, f"{action.kind} is not available in {self.mode.value} mode"
            )
        if isinstance(action, Recruit):
            self._recruit(action)
        elif isinstance(action, Assay):
            self._assay(action)
        elif isinstance(action, Submit):
            self._submit(action)
        self.step += 1
        result = self.view()
        self._requests[action.request_id] = (fingerprint, result)
        return result

    # ------------------------------------------------------------------ actions

    def _available(self) -> tuple[str, ...]:
        if self.status is EpisodeStatus.SUBMITTED:
            return ()
        if self.mode is Mode.FULL_ACCESS:
            return ("submit",)
        return ("recruit", "assay", "submit")

    def _charge(self, cost: float) -> None:
        if self.spent + cost > self.world.card.budget + 1e-9:
            raise ArenaError(
                ErrorCode.OVER_BUDGET,
                f"cost {cost:.2f} exceeds remaining budget {self.world.card.budget - self.spent:.2f}",
            )
        self.spent += cost

    def _recruit(self, action: Recruit) -> None:
        if action.stratum not in self.world.queues:
            raise ArenaError(ErrorCode.UNKNOWN_STRATUM, f"unknown stratum {action.stratum!r}")
        queue = self.world.queues[action.stratum]
        start = self._cursor[action.stratum]
        rows = queue[start : start + action.count]
        if not rows:
            raise ArenaError(ErrorCode.ACTION_NOT_AVAILABLE, f"stratum {action.stratum!r} is exhausted")
        self._charge(len(rows) * self.world.card.prices.recruit_per_patient)
        self._cursor[action.stratum] = start + len(rows)
        self._recruited.extend(rows)

    def _assay(self, action: Assay) -> None:
        assert self._measured is not None
        fids = self.world.card.feature_ids()
        unknown = [f for f in action.feature_ids if f not in fids]
        if unknown:
            raise ArenaError(ErrorCode.UNKNOWN_FEATURE, f"unknown features: {unknown[:5]}")
        rows = np.array(self._recruited, dtype=np.int64)
        cols = [fids.index(f) for f in dict.fromkeys(action.feature_ids)]
        cost = 0.0
        for j in cols:
            unmeasured = int((~self._measured[rows, j]).sum()) if rows.size else 0
            cost += unmeasured * self.world.card.features[j].assay_price
        self._charge(cost)
        if rows.size:
            for j in cols:
                self._measured[rows, j] = True

    def _submit(self, action: Submit) -> None:
        fids = set(self.world.card.feature_ids())
        unknown = [f for f in action.ranking if f not in fids]
        if unknown:
            raise ArenaError(ErrorCode.UNKNOWN_FEATURE, f"unknown features: {unknown[:5]}")
        self.submission = action.ranking
        self.status = EpisodeStatus.SUBMITTED
