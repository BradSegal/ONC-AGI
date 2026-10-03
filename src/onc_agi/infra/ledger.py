"""JSON-file :class:`~onc_agi.services.scorecards.Ledger` (server-side state)."""

from __future__ import annotations

import json
from pathlib import Path

from onc_agi.core.schema import Tier


class JsonLedger:
    def __init__(self, path: Path) -> None:
        self.path = path
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            self._write({"used": {}, "openings": {}})

    def _read(self) -> dict[str, dict[str, list[str]]]:
        data: dict[str, dict[str, list[str]]] = json.loads(self.path.read_text())
        return data

    def _write(self, data: dict[str, dict[str, list[str]]]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
        tmp.replace(self.path)

    def used(self, tier: Tier) -> set[str]:
        return set(self._read()["used"].get(tier.value, []))

    def mark_used(self, tier: Tier, world_ids: list[str]) -> None:
        data = self._read()
        data["used"].setdefault(tier.value, []).extend(world_ids)
        self._write(data)

    def openings(self, api_key: str, tier: Tier) -> list[str]:
        return list(self._read()["openings"].get(f"{api_key}|{tier.value}", []))

    def record_opening(self, api_key: str, tier: Tier, when: str) -> None:
        data = self._read()
        data["openings"].setdefault(f"{api_key}|{tier.value}", []).append(when)
        self._write(data)
