"""JSONL trace recorder shared by every harness (one line per applied action)."""

from __future__ import annotations

from pathlib import Path

from onc_agi.core.schema import TraceEvent


class TraceRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: TraceEvent) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json() + "\n")

    def events(self) -> list[TraceEvent]:
        if not self.path.exists():
            return []
        return [TraceEvent.model_validate_json(line) for line in self.path.read_text().splitlines() if line]
