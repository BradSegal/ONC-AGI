"""World bundles on disk and the filesystem :class:`WorldStore`.

Layout (one directory per world under ``<root>/<tier>/<world_id>/``)::

    card.json         public WorldCard
    pool.parquet      patient_id, stratum, outcome, then one column per feature (card order)
    queues.json       recruitment order per stratum
    answer_key.json   present only for public-train worlds

Eval and private answer keys live in a separate ``keys`` directory that is never
shipped with bundles.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from onc_agi.core.errors import ArenaError
from onc_agi.core.schema import AnswerKey, ErrorCode, Tier, WorldCard
from onc_agi.core.world import WorldData

_RESERVED = ("patient_id", "stratum", "outcome")


def write_world(root: Path, world: WorldData, key: AnswerKey | None, *, keys_dir: Path | None = None) -> Path:
    """Write ``world`` atomically enough for our use: into a temp dir, then rename."""
    target = root / world.card.tier.value / world.card.world_id
    if target.exists():
        raise FileExistsError(f"world {world.card.world_id} already exists; bundles are immutable")
    tmp = target.with_name(target.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)  # debris from an interrupted earlier write
    tmp.mkdir(parents=True)
    try:
        _write_into(tmp, world, key, keys_dir)
        tmp.rename(target)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return target


def _write_into(tmp: Path, world: WorldData, key: AnswerKey | None, keys_dir: Path | None) -> None:
    frame = pd.DataFrame(world.x, columns=list(world.card.feature_ids()))
    frame.insert(0, "outcome", world.y)
    frame.insert(0, "stratum", list(world.stratum))
    frame.insert(0, "patient_id", list(world.patient_ids))
    frame.to_parquet(tmp / "pool.parquet", index=False)
    (tmp / "card.json").write_text(world.card.model_dump_json(indent=1))
    (tmp / "queues.json").write_text(json.dumps({k: list(v) for k, v in world.queues.items()}))
    if key is not None:
        if world.card.tier is Tier.PUBLIC_TRAIN:
            (tmp / "answer_key.json").write_text(key.model_dump_json(indent=1))
        else:
            if keys_dir is None:
                raise ValueError("eval and private answer keys need a separate keys_dir")
            keys_dir.mkdir(parents=True, exist_ok=True)
            (keys_dir / f"{world.card.world_id}.json").write_text(key.model_dump_json(indent=1))


def read_world(directory: Path) -> WorldData:
    card = WorldCard.model_validate_json((directory / "card.json").read_text())
    frame = pd.read_parquet(directory / "pool.parquet")
    if tuple(frame.columns[:3]) != _RESERVED or tuple(frame.columns[3:]) != card.feature_ids():
        raise ValueError(f"pool columns in {directory} do not match the world card")
    queues = {
        k: tuple(int(i) for i in v) for k, v in json.loads((directory / "queues.json").read_text()).items()
    }
    return WorldData(
        card=card,
        patient_ids=tuple(str(v) for v in frame["patient_id"]),
        x=frame.iloc[:, 3:].to_numpy(dtype=np.float64),
        y=frame["outcome"].to_numpy(dtype=np.int64),
        stratum=tuple(str(v) for v in frame["stratum"]),
        queues=queues,
    )


class FileWorldStore:
    """Filesystem implementation of :class:`onc_agi.core.ports.WorldStore`."""

    def __init__(self, root: Path, keys_dir: Path | None = None) -> None:
        self.root = root
        self.keys_dir = keys_dir
        self._cache: dict[str, WorldData] = {}

    def _dir(self, world_id: str) -> Path:
        for tier in Tier:
            candidate = self.root / tier.value / world_id
            if (candidate / "card.json").exists():
                return candidate
        raise ArenaError(ErrorCode.UNKNOWN_WORLD, f"no world {world_id!r}")

    def world_ids(self, tier: Tier) -> tuple[str, ...]:
        base = self.root / tier.value
        if not base.exists():
            return ()
        return tuple(sorted(p.name for p in base.iterdir() if (p / "card.json").exists()))

    def card(self, world_id: str) -> WorldCard:
        return WorldCard.model_validate_json((self._dir(world_id) / "card.json").read_text())

    def world(self, world_id: str) -> WorldData:
        if world_id not in self._cache:
            self._cache[world_id] = read_world(self._dir(world_id))
        return self._cache[world_id]

    def answer_key(self, world_id: str) -> AnswerKey:
        local = self._dir(world_id) / "answer_key.json"
        if local.exists():
            return AnswerKey.model_validate_json(local.read_text())
        if self.keys_dir is not None:
            private = self.keys_dir / f"{world_id}.json"
            if private.exists():
                return AnswerKey.model_validate_json(private.read_text())
        raise ArenaError(ErrorCode.UNKNOWN_WORLD, f"no answer key available for {world_id!r}")
