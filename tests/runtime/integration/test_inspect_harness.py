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
from onc_agi.core.errors import ArenaError
from onc_agi.core.schema import Mode, Recruit, Tier
from onc_agi.infra.archive import FileScorecardArchive
from onc_agi.infra.bundles import write_world
from onc_agi.services.engine import Episode
from onc_agi.services.scoring import score_world


@pytest.fixture(autouse=True)
def release_task_services():  # type: ignore[no-untyped-def]
    yield
    for run in tuple(inspect_task._RUNS.values()):
        run.service.shutdown()
    inspect_task._RUNS.clear()


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
    sizes = world.card.stratum_sizes or {
        s: world.card.n_pool // len(world.card.strata) for s in world.card.strata
    }
    assert all(
        f"{s}: {n}" in text for s, n in sizes.items()
    ), "the model must know which strata it can recruit"
    assert "efficiency" in text and "refreshed after every recruit or assay" in text


@pytest.mark.parametrize("make", ["recruit", "assay"])
def test_host_tool_schemas_require_every_parameter_for_strict_providers(make: str) -> None:
    """OpenAI-compatible strict function calling rejects a schema whose properties are not all required."""
    from inspect_ai.tool import ToolDef

    params = ToolDef(getattr(inspect_task, make)()).parameters
    assert set(params.required) == set(params.properties)


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
    def __init__(self, values: dict[str, str]) -> None:
        self.values = dict(values)

    def get(self, name: str) -> str:
        return self.values[name]

    def set(self, name: str, value: str) -> None:
        self.values[name] = value


class FakeSandbox:
    def __init__(self) -> None:
        self.files: dict[str, str] = {}

    async def write_file(self, path: str, content: str) -> None:
        self.files[path] = content


def _sample(task: object, world_id: str) -> SimpleNamespace:
    sid = task.metadata["scorecard_id"]  # type: ignore[attr-defined]
    return SimpleNamespace(metadata={"world_id": world_id, "scorecard_id": sid}, uuid=world_id)


@pytest.fixture
def host(store_root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[object, FakeSandbox]:
    task = inspect_task.arena_sequential(str(store_root), ledger=str(store_root.parent / "ledger.json"))
    sid = task.metadata["scorecard_id"]
    box = FakeSandbox()
    monkeypatch.setattr(inspect_task, "store", lambda: FakeStore({"scorecard_id": sid, "world_id": "w-01"}))
    monkeypatch.setattr(inspect_task, "sandbox", lambda: box)
    yield task, box
    inspect_task._RUNS.pop(sid, None)


def _episode(task: object, world_id: str) -> Episode:
    run = inspect_task._RUNS[task.metadata["scorecard_id"]]  # type: ignore[attr-defined]
    return run.service.episode(run.scorecard_id, world_id)


def test_each_task_run_is_one_scorecard_opened_through_the_service(store_root: Path) -> None:
    task = inspect_task.arena_full_access(str(store_root), ledger=str(store_root.parent / "ledger.json"))
    sid = task.metadata["scorecard_id"]
    assert {s.metadata["scorecard_id"] for s in task.dataset} == {sid}
    assert inspect_task._RUNS[sid].service.ledger is not None and task.epochs == 1
    inspect_task._RUNS.pop(sid)


def test_eval_tiers_refuse_to_run_without_the_servers_ledger(store_root: Path) -> None:
    with pytest.raises(ValueError, match="fresh worlds"):
        inspect_task.arena_full_access(str(store_root), tier="public_eval", n_worlds=1)


def test_failed_task_open_releases_the_archive(store_root: Path) -> None:
    root = store_root.parent / "archive"
    with pytest.raises(ArenaError):
        inspect_task.arena_full_access(str(store_root), n_worlds=999, archive=str(root))
    archive = FileScorecardArchive(root)
    archive.acquire()
    archive.release()


def test_host_tools_act_on_the_episode_and_refresh_the_sandbox_copy(host: tuple[object, FakeSandbox]) -> None:
    task, box = host
    asyncio.run(inspect_task.setup_world()(SimpleNamespace(metadata=_sample(task, "w-01").metadata), None))
    message = asyncio.run(inspect_task.recruit()(count=10, stratum="all"))
    assert "10 patients revealed" in message and len(_episode(task, "w-01").view().rows) == 10
    asyncio.run(inspect_task.assay()(feature_ids=f"{fid(0)}, {fid(1)}"))
    assert box.files["/data/revealed.csv"].splitlines()[0].endswith(f"{fid(0)},{fid(1)}")


def test_refused_tool_calls_come_back_to_the_model_as_tool_errors(host: tuple[object, FakeSandbox]) -> None:
    from inspect_ai.tool import ToolError

    task, _ = host
    asyncio.run(inspect_task.setup_world()(SimpleNamespace(metadata=_sample(task, "w-01").metadata), None))
    with pytest.raises(ToolError):
        asyncio.run(inspect_task.assay()(feature_ids="ghost"))


def test_the_scorer_scores_the_final_answer_with_the_arena_scorer(store_root: Path) -> None:
    task = inspect_task.arena_full_access(str(store_root), ledger=str(store_root.parent / "ledger.json"))
    state = SimpleNamespace(
        **_sample(task, "w-00").__dict__, output=SimpleNamespace(completion=f"{fid(0)}, ghost")
    )
    score = asyncio.run(inspect_task.arena_scorer()(state, None))
    assert score.value == 1.0 and score.answer == fid(0)
    from onc_agi.infra.bundles import FileWorldStore

    expected = score_world((fid(0),), FileWorldStore(store_root).answer_key("w-00"))
    assert score.metadata["world_score"]["find"] == expected.find
    assert _episode(task, "w-00").submission == (fid(0),)
    inspect_task._RUNS.pop(task.metadata["scorecard_id"])


def test_eval_tier_logs_carry_no_per_world_results(tmp_path: Path) -> None:
    root = tmp_path / "store"
    for i in range(6):
        world, key = planted_world(f"e-{i:02d}", signal=i % 3 != 0, seed=i, n_pool=60, tier=Tier.PUBLIC_EVAL)
        write_world(root, world, key, keys_dir=tmp_path / "keys")
    task = inspect_task.arena_full_access(
        str(root),
        tier="public_eval",
        n_worlds=5,
        keys_dir=str(tmp_path / "keys"),
        ledger=str(tmp_path / "ledger.json"),
        api_key="operator-1",
        min_eval_worlds=1,
    )
    sample = task.dataset[0]
    state = SimpleNamespace(
        metadata=sample.metadata, uuid=sample.id, output=SimpleNamespace(completion=fid(0))
    )
    score = asyncio.run(inspect_task.arena_scorer()(state, None))
    assert "world_score" not in (score.metadata or {})
    inspect_task._RUNS.pop(task.metadata["scorecard_id"])


def _log(task: object, samples: list[SimpleNamespace], usage: dict[str, SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(
        samples=samples,
        stats=SimpleNamespace(model_usage=usage),
        eval=SimpleNamespace(run_id="run-1", model="provider/model-x", metadata=task.metadata),  # type: ignore[attr-defined]
    )


def _usage(**kw: int | float | None) -> SimpleNamespace:
    base = dict(
        input_tokens=100,
        output_tokens=10,
        total_tokens=1234,
        input_tokens_cache_read=1000,
        input_tokens_cache_write=None,
        total_cost=None,
    )
    return SimpleNamespace(**(base | kw))


def test_close_scorecard_records_model_harness_tokens_and_cache_aware_cost(store_root: Path) -> None:
    task = inspect_task.arena_full_access(str(store_root), ledger=str(store_root.parent / "ledger.json"))
    price = inspect_task.Price(input=1e-6, output=2e-6, cache_read=1e-7)
    card = inspect_task.close_scorecard(
        _log(task, [], {"provider/model-x": _usage()}), agent="x", prices={"provider/model-x": price}  # type: ignore[arg-type]
    )
    assert card.track == "standard" and card.harness == inspect_task.HARNESS and card.agent == "x"
    assert card.model == "provider/model-x" and card.tokens == 1234
    assert card.cost_usd == pytest.approx(100 * 1e-6 + 10 * 2e-6 + 1000 * 1e-7)
    assert "oracle" in card.versions and task.metadata["scorecard_id"] not in inspect_task._RUNS


def test_failed_close_releases_owned_archive_and_run_state(
    store_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = store_root.parent / "archive"
    task = inspect_task.arena_full_access(str(store_root), archive=str(root))
    sid = task.metadata["scorecard_id"]

    def failed(*args: object, **kwargs: object) -> None:
        raise OSError("injected close write failure")

    monkeypatch.setattr(inspect_task._RUNS[sid].service, "close", failed)
    with pytest.raises(OSError, match="close write"):
        inspect_task.close_scorecard(_log(task, [], {}))  # type: ignore[arg-type]
    assert sid not in inspect_task._RUNS
    archive = FileScorecardArchive(root)
    archive.acquire()
    archive.release()


def test_provider_reported_cost_wins_and_unknown_cost_stays_none() -> None:
    log = SimpleNamespace(stats=SimpleNamespace(model_usage={"m": _usage(total_cost=0.5)}))
    assert inspect_task.usage(log) == (1234, 0.5)  # type: ignore[arg-type]
    log = SimpleNamespace(stats=SimpleNamespace(model_usage={"m": _usage()}))
    assert inspect_task.usage(log) == (1234, None)  # type: ignore[arg-type]


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
        stats=SimpleNamespace(model_usage={"m": _usage()}),
        eval=SimpleNamespace(run_id="run-1", model="provider/model-x"),
    )
    card = inspect_task.log_to_scorecard(log, agent="model-x", tier=Tier.PUBLIC_TRAIN)  # type: ignore[arg-type]
    assert card.track == "standard" and card.harness == inspect_task.HARNESS
    assert card.model == "provider/model-x" and card.tokens == 1234
    assert card.discovery_score == pytest.approx(1.0)
