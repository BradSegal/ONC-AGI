"""Ports implemented by infrastructure adapters."""

from __future__ import annotations

from typing import Protocol

from onc_agi.core.schema import AnswerKey, Tier, TraceEvent, WorldCard
from onc_agi.core.world import WorldData


class WorldStore(Protocol):
    """Read access to built worlds. Answer keys are only read by the scorer."""

    def world_ids(self, tier: Tier) -> tuple[str, ...]: ...

    def card(self, world_id: str) -> WorldCard: ...

    def world(self, world_id: str) -> WorldData: ...

    def answer_key(self, world_id: str) -> AnswerKey: ...


class TraceSink(Protocol):
    """Receives one trace event per applied action."""

    def record(self, event: TraceEvent) -> None: ...
