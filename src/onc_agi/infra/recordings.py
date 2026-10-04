"""Run records on disk: the agent's JSONL recording, the server trace and the closed scorecard.

Both tracks, the open track (``onc-agi play``) and the standard track (the Inspect task), write the
same three files into a run directory::

    recording.jsonl   the agent's own account: one scorecard header, its events, one run per world
    trace.jsonl       the server trace: one TraceEvent per applied action, with request and
                      response digests (``GET /v1/scorecards/{sid}/trace``)
    scorecard.json    the closed scorecard

The recording (the ARC-AGI-3 ``Recorder`` counterpart)::

    {"kind": "scorecard", "scorecard_id": "sc-...", "agent": ..., "world_ids": [...], ...}
    {"scorecard_id": "sc-...", "world_id": "w", "turn": 0, "kind": "assistant", "content": ..., "ts": ...}
    {"kind": "run", "world_id": "w", "ranking": [...], "submitted": true, "spent": 0.0, ...}

Event kinds are ``assistant``, ``reasoning``, ``tool_call``, ``tool_result``, ``action`` (the
exact action JSON sent to the arena; ``error`` when refused) and ``note``. One file holds one
scorecard. Worlds are played on several threads, so every write happens under one lock and is
flushed at once: a crash loses at most the line in flight. Replaying the trace against the world
bundles verifies the run (:func:`onc_agi.services.replay.verify_run`).
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Annotated, TextIO

from pydantic import Field, TypeAdapter

from onc_agi.core.ports import RecordingEvent, RecordingHeader, RecordingKind, RunRecord
from onc_agi.core.schema import Scorecard, TraceEvent

RECORDING_FILE = "recording.jsonl"
TRACE_FILE = "trace.jsonl"
SCORECARD_FILE = "scorecard.json"

_Line = Annotated[RecordingEvent | RunRecord | RecordingHeader, Field(discriminator="kind")]
_LINE: TypeAdapter[RecordingEvent | RunRecord | RecordingHeader] = TypeAdapter(_Line)


class RecordingWriter:
    """Thread-safe writer (a :class:`~onc_agi.core.ports.RecordingSink`).

    Refuses an existing file: a recording is evidence of one run and is never appended to.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.scorecard_id: str | None = None
        self._turns: dict[str, int] = {}
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._handle: TextIO | None = path.open("x", encoding="utf-8")
        except FileExistsError:
            raise FileExistsError(
                f"recording {path} already exists; record each run to a new place"
            ) from None

    def __enter__(self) -> RecordingWriter:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None

    def _write(self, line: str) -> None:
        if self._handle is None:
            raise RuntimeError(f"recording {self.path} is closed")
        self._handle.write(line + "\n")
        self._handle.flush()

    def header(self, record: RecordingHeader) -> None:
        """Write the scorecard header; later events carry its scorecard id."""
        with self._lock:
            if self.scorecard_id is not None:
                raise RuntimeError("a recording holds exactly one scorecard")
            self.scorecard_id = record.scorecard_id
            self._write(record.model_dump_json())

    def event(
        self,
        world_id: str,
        kind: RecordingKind,
        content: str,
        *,
        tool: str | None = None,
        error: bool = False,
    ) -> None:
        with self._lock:
            turn = self._turns.get(world_id, 0)
            self._turns[world_id] = turn + 1
            event = RecordingEvent(
                scorecard_id=self.scorecard_id,
                world_id=world_id,
                turn=turn,
                kind=kind,
                content=content,
                tool=tool,
                error=error,
                ts=datetime.now(UTC),
            )
            self._write(event.model_dump_json())

    def run(self, record: RunRecord) -> None:
        with self._lock:
            if record.scorecard_id is None and self.scorecard_id is not None:
                record = record.model_copy(update={"scorecard_id": self.scorecard_id})
            self._write(record.model_dump_json())

    def append(self, event: RecordingEvent) -> None:
        """Write an event made elsewhere (a converted transcript) exactly as given."""
        with self._lock:
            self._turns[event.world_id] = max(self._turns.get(event.world_id, 0), event.turn + 1)
            self._write(event.model_dump_json())


@dataclass(frozen=True)
class Recording:
    header: RecordingHeader | None
    events: tuple[RecordingEvent, ...]
    runs: tuple[RunRecord, ...]


def recording_path(path: Path) -> Path:
    """A recording file, or the ``recording.jsonl`` inside a record directory."""
    return path / RECORDING_FILE if path.is_dir() else path


def read_recording(path: Path) -> Recording:
    """Read and validate a recording (fails fast on a malformed line or a second scorecard)."""
    file = recording_path(path)
    header: RecordingHeader | None = None
    events: list[RecordingEvent] = []
    runs: list[RunRecord] = []
    for number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = _LINE.validate_json(line)
        except ValueError as exc:
            raise ValueError(f"{file}:{number}: not a recording line ({exc})") from exc
        if isinstance(item, RecordingHeader):
            if header is not None:
                raise ValueError(f"{file}:{number}: a recording holds exactly one scorecard")
            header = item
        elif isinstance(item, RunRecord):
            runs.append(item)
        else:
            events.append(item)
    if header is not None:  # runs are written as worlds end; report them in scorecard order
        order = {w: i for i, w in enumerate(header.world_ids)}
        runs.sort(key=lambda r: order.get(r.world_id, len(order)))
    return Recording(header, tuple(events), tuple(runs))


def write_trace(path: Path, events: Sequence[TraceEvent]) -> None:
    """Write a server trace (refuses an existing file, like a recording)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.writelines(event.model_dump_json() + "\n" for event in events)


def read_trace(path: Path) -> tuple[TraceEvent, ...]:
    file = path / TRACE_FILE if path.is_dir() else path
    return tuple(
        TraceEvent.model_validate_json(line) for line in file.read_text().splitlines() if line.strip()
    )


@dataclass(frozen=True)
class RunFiles:
    """The three files of a run directory (the trace and scorecard when present)."""

    recording: Recording
    trace: tuple[TraceEvent, ...] | None
    scorecard: Scorecard | None


def save_run(directory: Path, scorecard: Scorecard, trace: Sequence[TraceEvent] | None) -> None:
    """Complete a run directory beside its recording: the closed scorecard and the server trace
    (``None`` when the arena keeps none, so the run cannot be verified by replay)."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / SCORECARD_FILE).write_text(scorecard.model_dump_json(indent=1))
    if trace is not None:
        write_trace(directory / TRACE_FILE, trace)


def read_run(directory: Path) -> RunFiles:
    trace, card = directory / TRACE_FILE, directory / SCORECARD_FILE
    return RunFiles(
        read_recording(directory),
        read_trace(trace) if trace.exists() else None,
        Scorecard.model_validate_json(card.read_text()) if card.exists() else None,
    )
