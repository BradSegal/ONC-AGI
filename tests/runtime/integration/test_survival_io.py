"""Survival worlds and missing cells across boundaries: bundles, HTTP and the Inspect harness."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from arena_factories import InMemoryStore, planted_world, survival_world
from fastapi.testclient import TestClient
from onc_agi.adapters.client import ArenaClient, view_from_observation
from onc_agi.core.schema import Mode, Reset, Submit, Tier
from onc_agi.infra.bundles import read_world, write_world
from onc_agi.infra.ledger import JsonLedger
from onc_agi.services.engine import Episode
from onc_agi.services.scorecards import ScorecardService

KEY = "test-key-0001"


def test_survival_bundles_round_trip_with_a_time_column(tmp_path: Path) -> None:
    world, key = survival_world("w-surv", signal=True, seed=1, missing=0.1)
    path = write_world(tmp_path, world, key)
    import pandas as pd

    columns = list(pd.read_parquet(path / "pool.parquet").columns)
    assert columns[:4] == ["patient_id", "stratum", "outcome", "time"]
    again = read_world(path)
    assert again.time is not None and world.time is not None
    np.testing.assert_array_equal(again.time, world.time)
    np.testing.assert_array_equal(np.isnan(again.x), np.isnan(world.x))
    assert again.card == world.card


def test_binary_bundles_keep_the_v1_column_layout(tmp_path: Path) -> None:
    world, key = planted_world("w-bin", signal=True, seed=2)
    import pandas as pd

    path = write_world(tmp_path, world, key)
    assert list(pd.read_parquet(path / "pool.parquet").columns)[:4] == [
        "patient_id",
        "stratum",
        "outcome",
        "f00",
    ]
    assert read_world(path).time is None


@pytest.fixture
def served(tmp_path: Path) -> tuple[TestClient, InMemoryStore]:
    from onc_agi.adapters.http import create_app

    store = InMemoryStore.of(
        survival_world("s-00", signal=True, seed=3, missing=0.1),
        planted_world("b-00", signal=True, seed=4),
    )
    service = ScorecardService(store, JsonLedger(tmp_path / "ledger.json"))
    return TestClient(create_app(service)), store


def test_http_observations_carry_time_only_for_survival_worlds(served) -> None:  # type: ignore[no-untyped-def]
    http, store = served
    client = ArenaClient("http://testserver", KEY, client=http)
    sid, cards = client.open("surv", Tier.PUBLIC_TRAIN, 2)
    by_type = {c.outcome_type: c for c in cards}
    assert set(by_type) == {"survival", "binary"}
    for outcome_type, card in by_type.items():
        raw = http.post(
            f"/v1/scorecards/{sid}/worlds/{card.world_id}/actions",
            json={"action": {"kind": "reset", "request_id": "r", "world_id": card.world_id}},
            headers={"X-Arena-Key": KEY},
        ).json()
        assert ("time" in raw["revealed"]) is (outcome_type == "survival")
        card_json = http.get(
            f"/v1/scorecards/{sid}/worlds/{card.world_id}", headers={"X-Arena-Key": KEY}
        ).json()
        assert card_json == raw
    surv = by_type["survival"]
    obs = client.state(sid, surv.world_id)
    view = view_from_observation(surv, obs)
    world = store.world(surv.world_id)
    assert view.time is not None and world.time is not None
    np.testing.assert_allclose(view.time, world.time)
    assert any(v is None for column in obs.revealed.columns.values() for v in column)
    client.act(sid, surv.world_id, Submit(request_id="s", ranking=("f00",)))


def test_the_inspect_harness_explains_and_writes_follow_up_time() -> None:
    inspect_task = pytest.importorskip("onc_agi.adapters.inspect_task")
    world, _ = survival_world("s-ins", signal=True, seed=5)
    text = inspect_task.card_text(world.card)
    assert "survival outcome" in text and "time (days; outcome is the event indicator)" in text
    assert "censored" in text
    view = Episode(world).apply(Reset(request_id="r", world_id="s-ins"))
    header = inspect_task._revealed_csv(view).splitlines()[0].split(",")
    assert header[:4] == ["patient_id", "stratum", "outcome", "time"]
    binary, _ = planted_world("b-ins", signal=True, seed=5)
    assert "binary outcome" in inspect_task.card_text(binary.card)
    assert "time" not in inspect_task.card_text(binary.card).split("Features")[0]


def test_sequential_survival_worlds_with_missing_cells_play_over_http(tmp_path: Path) -> None:
    """Over HTTP a missing cell and an unassayed cell are both null; the staged agent still finishes."""
    from onc_agi.adapters.agents import make_agent
    from onc_agi.adapters.http import create_app

    store = InMemoryStore.of(survival_world("s-seq", signal=True, seed=6, mode=Mode.SEQUENTIAL, missing=0.1))
    service = ScorecardService(store, JsonLedger(tmp_path / "ledger.json"))
    client = ArenaClient("http://testserver", KEY, client=TestClient(create_app(service)))
    sid, (card,) = client.open("seq", Tier.PUBLIC_TRAIN, 1)
    final = client.play(make_agent("seq_univariate_bh"), sid, card, max_steps=40)
    assert final.status.value == "submitted"
