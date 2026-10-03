"""Ports implemented by infrastructure adapters, and the records they carry."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Protocol

from pydantic import Field

from onc_agi.core.schema import AnswerKey, Frozen, Scorecard, Tier, TraceEvent, WorldCard
from onc_agi.core.world import WorldData

Sha256 = str  # lowercase hex digest


class WorldStore(Protocol):
    """Read access to built worlds. Answer keys are only read by the scorer."""

    def world_ids(self, tier: Tier) -> tuple[str, ...]: ...

    def card(self, world_id: str) -> WorldCard: ...

    def world(self, world_id: str) -> WorldData: ...

    def answer_key(self, world_id: str) -> AnswerKey: ...


class TraceSink(Protocol):
    """Receives one trace event per applied action."""

    def record(self, event: TraceEvent) -> None: ...


RecordingKind = Literal["assistant", "reasoning", "tool_call", "tool_result", "action", "note"]


class RecordingEvent(Frozen):
    """One line of an agent recording: what the agent said, ran, saw or did on one world.

    The arena traces actions server-side but never an agent's reasoning; a recording is the
    client's own account, so ``explain`` can say where a world was won or lost.
    """

    scorecard_id: str | None = None
    world_id: str
    turn: int = Field(ge=0, description="Per-world event counter, in emission order.")
    kind: RecordingKind
    content: str
    tool: str | None = None
    error: bool = False
    ts: datetime


class RunRecord(Frozen):
    """How one world of a recorded scorecard ended (one per world, played or not)."""

    kind: Literal["run"] = "run"
    scorecard_id: str | None = None
    world_id: str
    ranking: tuple[str, ...] = ()
    submitted: bool = False
    spent: float = Field(default=0.0, ge=0.0)
    tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0.0)
    model: str | None = None
    error: str | None = Field(default=None, description="Why the world ended without a normal finish.")


class RecordingHeader(Frozen):
    """The scorecard a recording belongs to (first line of the file)."""

    kind: Literal["scorecard"] = "scorecard"
    scorecard_id: str
    agent: str
    tier: Tier
    track: str
    world_ids: tuple[str, ...]
    tags: tuple[str, ...] = ()
    model: str | None = None
    harness: str | None = None
    opened_at: datetime


class RecordingSink(Protocol):
    """Receives an agent's recording events (thread-safe: worlds are played concurrently)."""

    def event(
        self,
        world_id: str,
        kind: RecordingKind,
        content: str,
        *,
        tool: str | None = None,
        error: bool = False,
    ) -> None: ...


class OpenRecord(Frozen):
    """Server-side record of an open scorecard. Never holds a raw API key."""

    scorecard_id: str
    agent: str
    track: str
    tier: Tier
    world_ids: tuple[str, ...]
    tags: tuple[str, ...] = ()
    owner_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    opened_at: datetime


class ClosedRecord(Frozen):
    """Server-side record of a closed scorecard: exactly what ``close`` returned."""

    scorecard: Scorecard
    owner_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    closed_at: datetime
    expired: bool = Field(default=False, description="Closed by the server's TTL sweep, not the client.")


class ScorecardArchive(Protocol):
    """Durable scorecard state: open records, per-scorecard traces and closed scorecards.

    An archive belongs to one server process (the ledger may be shared; the archive not).
    """

    def acquire(self) -> None:
        """Claim exclusive service ownership, failing immediately if already owned."""
        ...

    def release(self) -> None:
        """Release ownership after all service work has stopped."""
        ...

    def save_open(self, record: OpenRecord) -> None: ...

    def open_records(self) -> tuple[OpenRecord, ...]: ...

    def trace(self, scorecard_id: str) -> TraceSink: ...

    def events(self, scorecard_id: str) -> tuple[TraceEvent, ...]: ...

    def save_closed(self, record: ClosedRecord) -> None:
        """Persist the closed scorecard, then retire its open record."""
        ...

    def closed_ids(self) -> tuple[str, ...]: ...

    def load_closed(self, scorecard_id: str) -> ClosedRecord | None: ...
