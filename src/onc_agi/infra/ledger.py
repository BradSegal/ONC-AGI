"""JSON-file :class:`~onc_agi.services.scorecards.Ledger` (server-side state).

Concurrency: every public method runs under :meth:`JsonLedger.lock`, a reentrant
lock made of a :class:`threading.RLock` (threads of one server) and, at depth 0, an
exclusive ``fcntl.flock`` on the sidecar ``<ledger>.lock`` file (several server
processes, or an Inspect run, sharing one ledger). Writes go to a unique temp file in
the ledger's directory, are fsynced and then atomically replace the ledger, so a reader
never sees a partial file and concurrent writers never share a temp path.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path

from onc_agi.core.schema import Tier


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically: unique temp file, flush+fsync, ``os.replace``."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


class JsonLedger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock_path = path.with_name(path.name + ".lock")
        self._mutex = threading.RLock()
        self._depth = 0
        self._fd: int | None = None
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock():
            if not path.exists():
                self._write({"used": {}, "openings": {}})

    @contextlib.contextmanager
    def lock(self) -> Iterator[None]:
        """Hold the ledger exclusively (reentrant; cross-thread and cross-process)."""
        with self._mutex:
            if self._depth == 0:
                fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                except BaseException:
                    os.close(fd)
                    raise
                self._fd = fd
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
                if self._depth == 0 and self._fd is not None:
                    fd, self._fd = self._fd, None
                    try:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                    finally:
                        os.close(fd)

    def _read(self) -> dict[str, dict[str, list[str]]]:
        data: dict[str, dict[str, list[str]]] = json.loads(self.path.read_text())
        return data

    def _write(self, data: dict[str, dict[str, list[str]]]) -> None:
        atomic_write_text(self.path, json.dumps(data, indent=1, sort_keys=True))

    def used(self, tier: Tier) -> set[str]:
        with self.lock():
            return set(self._read()["used"].get(tier.value, []))

    def mark_used(self, tier: Tier, world_ids: list[str]) -> None:
        with self.lock():
            data = self._read()
            data["used"].setdefault(tier.value, []).extend(world_ids)
            self._write(data)

    def openings(self, api_key: str, tier: Tier) -> list[str]:
        with self.lock():
            return list(self._read()["openings"].get(f"{api_key}|{tier.value}", []))

    def record_opening(self, api_key: str, tier: Tier, when: str) -> None:
        with self.lock():
            data = self._read()
            data["openings"].setdefault(f"{api_key}|{tier.value}", []).append(when)
            self._write(data)
