"""JSONL agent recordings: one scorecard header, the agents' events, one run record per world.

The ARC-AGI-3 ``Recorder`` counterpart, generalising the live workspace's transcript sidecar::

    {"kind": "scorecard", "scorecard_id": "sc-...", "agent": ..., "world_ids": [...], ...}
    {"scorecard_id": "sc-...", "world_id": "w", "turn": 0, "kind": "assistant", "content": ..., "ts": ...}
    {"kind": "run", "world_id": "w", "ranking": [...], "submitted": true, "spent": 0.0, ...}

One file holds one scorecard. Worlds are played on several threads, so every write
happens under one lock and is flushed at once: a crash loses at most the line in flight.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Annotated, TextIO

from pydantic import Field, TypeAdapter

from onc_agi.core.ports import RecordingEvent, RecordingHeader, RecordingKind, RunRecord

RECORDING_FILE = "recording.jsonl"

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
