"""File :class:`~onc_agi.core.ports.ScorecardArchive` (B3, B4).

Layout under ``root``::

    open/<sid>.json      OpenRecord, written at open, removed once the scorecard is closed
    traces/<sid>.jsonl   one TraceEvent per applied action (replayed to restore an episode)
    closed/<sid>.json    ClosedRecord: the exposure-filtered scorecard close returned

Records are written atomically (temp file + ``os.replace``). The closed record is written
before the open record is removed, so a crash in between leaves both and the closed one
wins. Scorecard ids reach this module from URLs; ids outside a conservative alphabet are
never turned into paths.
"""

from __future__ import annotations

import fcntl
import logging
import os
import re
from pathlib import Path

from onc_agi.core.ports import ClosedRecord, OpenRecord
from onc_agi.core.schema import TraceEvent
from onc_agi.infra.ledger import atomic_write_text
from onc_agi.infra.recorder import TraceRecorder

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]{0,95}")

log = logging.getLogger(__name__)


def _safe(scorecard_id: str) -> bool:
    return _SAFE_ID.fullmatch(scorecard_id) is not None


class FileScorecardArchive:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._owner_fd: int | None = None
        for sub in ("open", "traces", "closed"):
            (root / sub).mkdir(parents=True, exist_ok=True)

    def acquire(self) -> None:
        """One active service owns an archive, including two services in the same process."""
        if self._owner_fd is not None:
            raise RuntimeError("scorecard archive already belongs to an active service")
        fd = os.open(self.root / ".owner.lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise RuntimeError("scorecard archive already belongs to an active service") from exc
        except BaseException:
            os.close(fd)
            raise
        self._owner_fd = fd

    def release(self) -> None:
        if self._owner_fd is not None:
            fd, self._owner_fd = self._owner_fd, None
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def _path(self, kind: str, scorecard_id: str, suffix: str) -> Path:
        if not _safe(scorecard_id):
            raise ValueError(f"unsafe scorecard id {scorecard_id!r}")
        return self.root / kind / f"{scorecard_id}{suffix}"

    def save_open(self, record: OpenRecord) -> None:
        atomic_write_text(self._path("open", record.scorecard_id, ".json"), record.model_dump_json(indent=1))

    def open_records(self) -> tuple[OpenRecord, ...]:
        return tuple(
            OpenRecord.model_validate_json(p.read_text()) for p in sorted((self.root / "open").glob("*.json"))
        )

    def trace(self, scorecard_id: str) -> TraceRecorder:
        return TraceRecorder(self._path("traces", scorecard_id, ".jsonl"))

    def events(self, scorecard_id: str) -> tuple[TraceEvent, ...]:
        """Recorded actions in order. A crash mid-append can leave a torn final line with no
        newline; it is dropped, which is consistent: the client never saw that response and
        retries the same request id. A malformed line anywhere else is corruption and raises."""
        path = self._path("traces", scorecard_id, ".jsonl")
        if not path.exists():
            return ()
        text = path.read_text()
        lines = text.splitlines()
        if lines and not text.endswith("\n"):
            try:
                TraceEvent.model_validate_json(lines[-1])
            except ValueError:
                if self._owner_fd is None:
                    raise RuntimeError("repairing a torn trace requires archive ownership") from None
                log.warning("dropping torn final trace line of %s", scorecard_id)
                lines = lines[:-1]
                atomic_write_text(path, "".join(line + "\n" for line in lines))
            else:
                # A complete event can survive a crash before its newline is written.
                # Separate it from the next append without dropping the recorded action.
                if self._owner_fd is not None:
                    atomic_write_text(path, text + "\n")
        return tuple(TraceEvent.model_validate_json(line) for line in lines if line)

    def save_closed(self, record: ClosedRecord) -> None:
        sid = record.scorecard.scorecard_id
        atomic_write_text(self._path("closed", sid, ".json"), record.model_dump_json(indent=1))
        self._path("open", sid, ".json").unlink(missing_ok=True)

    def closed_ids(self) -> tuple[str, ...]:
        """Every closed scorecard on record (operator-side ingestion)."""
        return tuple(p.stem for p in sorted((self.root / "closed").glob("*.json")))

    def load_closed(self, scorecard_id: str) -> ClosedRecord | None:
        if not _safe(scorecard_id):
            return None
        path = self._path("closed", scorecard_id, ".json")
        return ClosedRecord.model_validate_json(path.read_text()) if path.exists() else None
