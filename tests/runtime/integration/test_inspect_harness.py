"""Inspect standard harness: what every model sees, may do and is scored on.

Model calls and Docker are not exercised here; tools run against fake Inspect
context objects so their host-side behaviour can be checked directly.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("inspect_ai")

import yaml
from arena_factories import LEAK, fid, planted_world
from onc_agi.adapters import inspect_task
from onc_agi.core.schema import Mode, Recruit, Tier
from onc_agi.infra.bundles import write_world
from onc_agi.services.engine import Episode
from onc_agi.services.scoring import score_world


@pytest.fixture
def store_root(tmp_path: Path) -> Path:
    root = tmp_path / "store"
    for i in range(4):
        mode = Mode.SEQUENTIAL if i % 2 else Mode.FULL_ACCESS
        write_world(root, *planted_world(f"w-{i:02d}", signal=i != 2, seed=i, n_pool=60, mode=mode))
    return root


def test_the_task_card_states_the_premise_every_feature_and_the_leak_rule() -> None:
    world, _ = planted_world("w-00", signal=True, seed=1)
    text = inspect_task.card_text(world.card)
    assert world.card.premise in text
    assert all(f"- {f.feature_id}," in text for f in world.card.features)
    assert f"- {LEAK}, expression, post_outcome" in text
    assert "zeroes this world" in text and "/data/revealed.csv" in text


def test_the_sequential_task_card_states_budget_and_recruit_price() -> None:
    world, _ = planted_world("w-00", signal=True, seed=1, mode=Mode.SEQUENTIAL)
    text = inspect_task.card_text(world.card)
    assert f"Budget {world.card.budget:.0f} USD" in text and "per patient" in text


def test_the_revealed_csv_carries_only_measured_columns_and_no_patient_identity() -> None:
    world, _ = planted_world("w-00", signal=True, seed=1, mode=Mode.SEQUENTIAL)
    ep = Episode(world)
    ep.apply(Recruit(request_id="a", count=5))
    from onc_agi.core.schema import Assay

    view = ep.apply(Assay(request_id="b", feature_ids=(fid(2),)))
    header, *rows = inspect_task._revealed_csv(view).strip().splitlines()
    assert header.split(",") == ["patient_id", "stratum", "outcome", fid(2)]
    assert len(rows) == 5
    assert not any(pid in "".join(rows) for pid in world.patient_ids)


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("f01, f02", ("f01", "f02")), ("f01\nf02,f01", ("f01", "f02")), ("", ()), (" , ,", ())],
)
def test_submitted_answers_parse_into_an_ordered_unique_ranking(
    answer: str, expected: tuple[str, ...]
) -> None:
    assert inspect_task._ranking(answer) == expected


def test_the_sandbox_has_no_network_and_bounded_resources() -> None:
    compose = yaml.safe_load(inspect_task.SANDBOX_COMPOSE.read_text())
    service = compose["services"]["default"]
    assert service["network_mode"] == "none"
    assert "mem_limit" in service and "cpus" in service


@pytest.mark.parametrize(("mode", "extra_tools"), [(Mode.FULL_ACCESS, 0), (Mode.SEQUENTIAL, 2)])
def test_tasks_select_worlds_by_mode_with_a_safety_timeout(
    store_root: Path, mode: Mode, extra_tools: int
) -> None:
    make = inspect_task.arena_full_access if mode is Mode.FULL_ACCESS else inspect_task.arena_sequential
    task = make(str(store_root))
    ids = [s.metadata["world_id"] for s in task.dataset]
    assert ids == ["w-00", "w-02"] if mode is Mode.FULL_ACCESS else ids == ["w-01", "w-03"]
    assert task.time_limit == inspect_task.SAFETY_TIMEOUT_SECONDS
    assert task.metadata["harness"] == inspect_task.HARNESS


class FakeStore:
    def __init__(self, key: str) -> None:
        self.key = key

    def get(self, name: str) -> str:
        return self.key


class FakeSandbox:
    def __init__(self) -> None:
        self.files: dict[str, str] = {}

    async def write_file(self, path: str, content: str) -> None:
        self.files[path] = content


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch) -> tuple[Episode, FakeSandbox]:
    world, _ = planted_world("w-01", signal=True, seed=1, mode=Mode.SEQUENTIAL, n_pool=60)
    episode = Episode(world)
    box = FakeSandbox()
    inspect_task._EPISODES["sample-1"] = episode
    monkeypatch.setattr(inspect_task, "store", lambda: FakeStore("sample-1"))
    monkeypatch.setattr(inspect_task, "sandbox", lambda: box)
    yield episode, box
    inspect_task._EPISODES.pop("sample-1", None)


def test_host_tools_act_on_the_episode_and_refresh_the_sandbox_copy(
    host: tuple[Episode, FakeSandbox],
) -> None:
    episode, box = host
    message = asyncio.run(inspect_task.recruit()(count=10))
    assert "10 patients revealed" in message and len(episode.view().rows) == 10
    asyncio.run(inspect_task.assay()(feature_ids=f"{fid(0)}, {fid(1)}"))
    assert box.files["/data/revealed.csv"].splitlines()[0].endswith(f"{fid(0)},{fid(1)}")


def test_refused_tool_calls_come_back_to_the_model_as_tool_errors(host: tuple[Episode, FakeSandbox]) -> None:
    from inspect_ai.tool import ToolError

    with pytest.raises(ToolError):
        asyncio.run(inspect_task.assay()(feature_ids="ghost"))


def test_the_scorer_scores_the_final_answer_with_the_arena_scorer(store_root: Path) -> None:
    score_fn = inspect_task.arena_scorer(str(store_root))
    state = SimpleNamespace(
        metadata={"world_id": "w-00"},
        uuid="no-episode",
        output=SimpleNamespace(completion=f"{fid(0)}, ghost"),
    )
    score = asyncio.run(score_fn(state, None))
    assert score.value == 1.0 and score.answer == fid(0)
    from onc_agi.infra.bundles import FileWorldStore

    expected = score_world((fid(0),), FileWorldStore(store_root).answer_key("w-00"))
    assert score.metadata["world_score"]["find"] == expected.find


def test_inspect_logs_convert_to_a_standard_track_scorecard(store_root: Path) -> None:
    from onc_agi.infra.bundles import FileWorldStore

    worlds = FileWorldStore(store_root)
    samples = []
    for wid, ranking in [("w-00", (fid(0),)), ("w-02", ())]:
        ws = score_world(ranking, worlds.answer_key(wid))
        samples.append(
            SimpleNamespace(
                scores={"arena_scorer": SimpleNamespace(metadata={"world_score": ws.model_dump(mode="json")})}
            )
        )
    log = SimpleNamespace(
        samples=samples,
        stats=SimpleNamespace(model_usage={"m": SimpleNamespace(total_tokens=1234)}),
        eval=SimpleNamespace(run_id="run-1", model="provider/model-x"),
    )
    card = inspect_task.log_to_scorecard(log, agent="model-x", tier=Tier.PUBLIC_TRAIN)  # type: ignore[arg-type]
    assert card.track == "standard" and card.harness == inspect_task.HARNESS
    assert card.model == "provider/model-x" and card.tokens == 1234
    assert card.discovery_score == pytest.approx(1.0)
