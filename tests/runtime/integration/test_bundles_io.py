"""World bundles on disk and the filesystem store."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from arena_factories import planted_world
from onc_agi.core.errors import ArenaError
from onc_agi.core.schema import ErrorCode, Mode, Tier, TraceEvent
from onc_agi.infra.bundles import FileWorldStore, read_world, write_world
from onc_agi.infra.ledger import JsonLedger
from onc_agi.infra.recorder import TraceRecorder


def test_a_world_round_trips_exactly_including_missing_cells(tmp_path: Path) -> None:
    world, key = planted_world("w-00", signal=True, seed=1, mode=Mode.SEQUENTIAL)
    world.x[3, 2] = np.nan
    path = write_world(tmp_path, world, key)
    back = read_world(path)
    assert back.card == world.card
    assert (
        back.patient_ids == world.patient_ids
        and back.stratum == world.stratum
        and back.queues == world.queues
    )
    np.testing.assert_array_equal(back.x, world.x)
    np.testing.assert_array_equal(back.y, world.y)
    assert back.x.dtype == np.float64 and back.y.dtype == np.int64


def test_public_train_bundles_carry_their_answer_key(tmp_path: Path) -> None:
    world, key = planted_world("w-00", signal=True, seed=1)
    path = write_world(tmp_path, world, key)
    assert sorted(p.name for p in path.iterdir()) == [
        "answer_key.json",
        "card.json",
        "pool.parquet",
        "queues.json",
    ]
    assert FileWorldStore(tmp_path).answer_key("w-00") == key


@pytest.mark.parametrize("tier", [Tier.PUBLIC_EVAL, Tier.PRIVATE])
def test_hidden_tier_keys_never_enter_the_bundle(tmp_path: Path, tier: Tier) -> None:
    world, key = planted_world("w-00", signal=True, seed=1, tier=tier)
    keys = tmp_path / "keys"
    path = write_world(tmp_path / "bundles", world, key, keys_dir=keys)
    assert not (path / "answer_key.json").exists()
    assert not any("answer_key" in p.name for p in (tmp_path / "bundles").rglob("*"))
    assert "equivalence_set" not in (path / "card.json").read_text()
    with pytest.raises(ArenaError) as err:
        FileWorldStore(tmp_path / "bundles").answer_key("w-00")
    assert err.value.code is ErrorCode.UNKNOWN_WORLD
    assert FileWorldStore(tmp_path / "bundles", keys).answer_key("w-00") == key


def test_hidden_tier_keys_require_a_separate_keys_directory(tmp_path: Path) -> None:
    world, key = planted_world("w-00", signal=True, seed=1, tier=Tier.PRIVATE)
    with pytest.raises(ValueError, match="keys_dir"):
        write_world(tmp_path, world, key)


def test_bundles_are_never_overwritten_and_a_refused_write_leaves_no_debris(tmp_path: Path) -> None:
    world, key = planted_world("w-00", signal=True, seed=1)
    write_world(tmp_path, world, key)
    for _ in range(2):
        with pytest.raises(FileExistsError):
            write_world(tmp_path, world, key)
    assert sorted(p.name for p in (tmp_path / "public_train").iterdir()) == ["w-00"]


def test_a_pool_whose_columns_disagree_with_the_card_is_rejected(tmp_path: Path) -> None:
    world, key = planted_world("w-00", signal=True, seed=1)
    path = write_world(tmp_path, world, key)
    frame = pd.read_parquet(path / "pool.parquet")
    frame.rename(columns={"f03": "zz"}).to_parquet(path / "pool.parquet", index=False)
    with pytest.raises(ValueError, match="do not match"):
        read_world(path)


def test_fractional_outcomes_are_rejected_before_integer_conversion(tmp_path: Path) -> None:
    world, key = planted_world("w-fractional", signal=True, seed=1)
    path = write_world(tmp_path, world, key)
    frame = pd.read_parquet(path / "pool.parquet")
    frame["outcome"] = frame["outcome"].astype(float)
    frame.loc[0, "outcome"] = 0.5
    frame.to_parquet(path / "pool.parquet", index=False)
    with pytest.raises(ValueError, match="binary"):
        read_world(path)


def test_missing_patient_identifiers_are_rejected_before_string_conversion(tmp_path: Path) -> None:
    world, key = planted_world("w-missing-id", signal=True, seed=1)
    path = write_world(tmp_path, world, key)
    frame = pd.read_parquet(path / "pool.parquet")
    frame.loc[0, "patient_id"] = None
    frame.to_parquet(path / "pool.parquet", index=False)
    with pytest.raises(ValueError, match="patient identifiers"):
        read_world(path)


def test_store_lists_worlds_per_tier_and_refuses_unknown_worlds(tmp_path: Path) -> None:
    for i, tier in enumerate([Tier.PUBLIC_TRAIN, Tier.PUBLIC_TRAIN, Tier.PUBLIC_EVAL]):
        world, key = planted_world(f"w-{i:02d}", signal=True, seed=i, tier=tier)
        write_world(tmp_path, world, key, keys_dir=tmp_path / "keys")
    store = FileWorldStore(tmp_path)
    assert store.world_ids(Tier.PUBLIC_TRAIN) == ("w-00", "w-01")
    assert store.world_ids(Tier.PUBLIC_EVAL) == ("w-02",)
    assert store.world_ids(Tier.PRIVATE) == ()
    assert store.card("w-02").tier is Tier.PUBLIC_EVAL
    assert store.world("w-01") is store.world("w-01")
    with pytest.raises(ArenaError) as err:
        store.card("w-99")
    assert err.value.code is ErrorCode.UNKNOWN_WORLD


def test_the_json_ledger_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "state" / "ledger.json"
    ledger = JsonLedger(path)
    ledger.mark_used(Tier.PRIVATE, ["a", "b"])
    ledger.record_opening("key-1", Tier.PRIVATE, "2026-10-03")
    again = JsonLedger(path)
    assert again.used(Tier.PRIVATE) == {"a", "b"} and again.used(Tier.PUBLIC_EVAL) == set()
    assert again.openings("key-1", Tier.PRIVATE) == ["2026-10-03"]
    assert again.openings("key-2", Tier.PRIVATE) == []


def test_the_trace_recorder_appends_and_reads_back_events(tmp_path: Path) -> None:
    recorder = TraceRecorder(tmp_path / "t" / "trace.jsonl")
    assert recorder.events() == []
    events = [
        TraceEvent(
            world_id="w-00",
            step=i,
            action_kind="reset",
            request_id=f"r{i}",
            request_sha256="0" * 64,
            response_sha256="1" * 64,
            spent=0.0,
        )
        for i in range(3)
    ]
    for event in events:
        recorder.record(event)
    assert TraceRecorder(recorder.path).events() == events
