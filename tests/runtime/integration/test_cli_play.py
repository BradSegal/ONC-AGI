"""``onc-agi play``, ``onc-agi worlds`` and ``onc-agi explain`` in-process on the bundled fixture worlds."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from onc_agi.adapters import cli
from onc_agi.core.schema import Tier
from onc_agi.infra.bundles import FileWorldStore
from onc_agi.infra.recordings import read_recording

FIXTURES = FileWorldStore(cli.fixture_store())
IDS = FIXTURES.world_ids(Tier.PUBLIC_TRAIN)


@pytest.fixture(autouse=True)
def _no_remote(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep a developer's ARENA_URL/ARENA_KEY or .env from sending these runs to a server."""
    monkeypatch.delenv("ARENA_URL", raising=False)
    monkeypatch.delenv("ARENA_KEY", raising=False)
    monkeypatch.chdir(tmp_path)


def test_play_a_builtin_agent_on_three_fixture_worlds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "card.json"
    assert cli.main(["play", "--agent", "random", "--n", "3", "--workers", "2", "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("random ") and "3 worlds, 3 submitted, 0 errors, 0 unplayed" in printed
    card = json.loads(out.read_text())
    assert card["agent"] == "random" and card["n_worlds"] == 3 and card["track"] == "open"


def test_play_named_worlds_with_a_recording_then_explain_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    chosen = [IDS[0], IDS[-1]]
    record = tmp_path / "run"
    args = ["play", "--agent", "univariate_bh", "--worlds", ",".join(chosen), "--record", str(record)]
    assert cli.main([*args, "--tags", "kit,test"]) == 0
    assert f"recording in {record}" in capsys.readouterr().out
    assert {p.name for p in record.iterdir()} == {
        "recording.jsonl",
        "scorecard.json",
        "explanations.md",
        "explanations.json",
    }
    recording = read_recording(record)
    assert recording.header is not None and list(recording.header.world_ids) == chosen
    assert recording.header.tags == ("kit", "test")
    assert [r.world_id for r in recording.runs] == chosen and all(r.submitted for r in recording.runs)
    assert {e.kind for e in recording.events} == {"action"}
    scorecard = json.loads((record / "scorecard.json").read_text())
    explained = json.loads((record / "explanations.json").read_text())
    by_world = {w["world_id"]: w for w in scorecard["worlds"]}
    for x in explained:
        if not x["is_null"]:
            assert x["find"] == pytest.approx(by_world[x["world_id"]]["find"])
    assert cli.main(["explain", "--record", str(record), "--json", str(tmp_path / "x.json")]) == 0
    report = capsys.readouterr().out
    assert report.startswith(f"# univariate_bh on {scorecard['scorecard_id']}") and chosen[0] in report
    assert json.loads((tmp_path / "x.json").read_text()) == explained
    assert cli.main(args) == 2  # a recording is never overwritten
    assert "already exists" in capsys.readouterr().err


def test_play_a_class_spec_in_sequential_mode(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "mine.py"
    source.write_text(
        "from onc_agi.adapters.agents import make_agent\n"
        "from onc_agi.services.kit import PipelineAgent\n"
        "class Mine(PipelineAgent):\n"
        "    def __init__(self):\n"
        "        base = make_agent('univariate_bh')\n"
        "        super().__init__('mine', base.analyst)\n"
    )
    assert cli.main(["play", "--agent", f"{source}:Mine", "--n", "2", "--mode", "sequential"]) == 0
    assert capsys.readouterr().out.startswith("mine ")


def test_worlds_lists_public_train_ids_modes_and_sizes(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["worlds"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == f"{len(IDS)} public-train worlds"
    first = lines[0].split()
    card = FIXTURES.card(IDS[0])
    assert first[:3] == [IDS[0], card.mode.value, str(card.n_pool)]


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["play", "--agent", "nobody", "--n", "1"], "unknown agent 'nobody'"),
        (["play", "--agent", "oracle", "--url", "http://arena.test", "--key", "k" * 8], "only in-process"),
        (["play", "--agent", "random", "--url", "http://arena.test"], "needs a key"),
        (["play", "--agent", "random", "--url", "http://a.test", "--store", "s"], "not both"),
        (["play", "--agent", "llm"], "--agent llm needs --profile"),
        (["play", "--agent", "random", "--tier", "public_eval"], "give --n"),
    ],
)
def test_play_refuses_bad_requests_before_opening_a_scorecard(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(argv) == 2
    assert message in capsys.readouterr().err


def test_play_llm_fails_fast_without_its_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert cli.main(["play", "--agent", "llm", "--profile", "openrouter-luna", "--n", "1"]) == 2
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err


def test_play_reports_world_errors_with_a_nonzero_exit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "bad.py"
    source.write_text(
        "from onc_agi.services.kit import Agent\n"
        "class Bad(Agent):\n"
        "    name = 'bad'\n"
        "    def choose_action(self, card, view):\n"
        "        raise RuntimeError('no idea')\n"
    )
    assert cli.main(["play", "--agent", f"{source}:Bad", "--n", "2"]) == 1
    captured = capsys.readouterr()
    assert "2 errors" in captured.out and "RuntimeError: no idea" in captured.err


def test_explain_refuses_non_public_worlds_without_the_operator_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from arena_factories import planted_world
    from onc_agi.core.ports import RunRecord
    from onc_agi.infra.bundles import write_world
    from onc_agi.infra.recordings import RecordingWriter

    world, key = planted_world("e-00", signal=True, seed=1, tier=Tier.PUBLIC_EVAL)
    write_world(tmp_path / "store", world, key, keys_dir=tmp_path / "keys")
    with RecordingWriter(tmp_path / "rec.jsonl") as rec:
        rec.run(RunRecord(world_id="e-00", ranking=("f00",), submitted=True))
    base = ["explain", "--record", str(tmp_path / "rec.jsonl"), "--store", str(tmp_path / "store")]
    assert cli.main([*base, "--keys", str(tmp_path / "keys")]) == 2
    assert "public-train only" in capsys.readouterr().err
    assert cli.main([*base, "--keys", str(tmp_path / "keys"), "--operator"]) == 0
    assert "OPERATOR ONLY" in capsys.readouterr().out
