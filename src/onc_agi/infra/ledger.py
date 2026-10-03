"""JSON-file :class:`~onc_agi.services.scorecards.Ledger` (server-side state).

Every read-modify-write holds a process-local lock and an exclusive ``flock`` on a sidecar
lock file, and replaces the ledger atomically from a unique temporary file, so concurrent
openings (threads or server processes) can neither lose updates nor corrupt the JSON.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar

from onc_agi.core.schema import Tier

Data = dict[str, dict[str, list[str]]]
T = TypeVar("T")


class JsonLedger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked():
            if not path.exists():
                self._write({"used": {}, "openings": {}})

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self._lock, open(self.path.with_suffix(".lock"), "a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _read(self) -> Data:
        data: Data = json.loads(self.path.read_text())
        return data

    def _write(self, data: Data) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name, suffix=".tmp")
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(data, indent=1, sort_keys=True))
        os.replace(tmp, self.path)

    def transaction(self, update: Callable[[Data], T]) -> T:
        """Apply ``update`` to the ledger under the lock and persist it (atomic read-modify-write)."""
        with self._locked():
            data = self._read()
            result = update(data)
            self._write(data)
            return result

    def used(self, tier: Tier) -> set[str]:
        with self._locked():
            return set(self._read()["used"].get(tier.value, []))

    def mark_used(self, tier: Tier, world_ids: list[str]) -> None:
        self.transaction(lambda data: data["used"].setdefault(tier.value, []).extend(world_ids))

    def openings(self, api_key: str, tier: Tier) -> list[str]:
        with self._locked():
            return list(self._read()["openings"].get(f"{api_key}|{tier.value}", []))

    def record_opening(self, api_key: str, tier: Tier, when: str) -> None:
        self.transaction(lambda data: data["openings"].setdefault(f"{api_key}|{tier.value}", []).append(when))
