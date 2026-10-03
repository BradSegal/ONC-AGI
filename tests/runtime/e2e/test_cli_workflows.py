"""Consumer workflows through the ``arena`` command: smoke, evaluate+replay, conformance."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from onc_agi.adapters import cli

ARENA = [sys.executable, "-m", "onc_agi.adapters.cli"]


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([*ARENA, *args], capture_output=True, text=True, timeout=600, check=False)


@pytest.fixture(scope="module")
def fixtures() -> Path:
    root = cli.fixture_store()
    if not (root / "public_train").exists():
        pytest.skip("packaged fixture worlds are not built")
    return root


def test_the_smoke_command_passes_and_prints_every_scorecard(fixtures: Path) -> None:
    result = run("smoke")
    assert result.returncode == 0, result.stderr
    assert "smoke passed" in result.stdout
    for name in cli.SMOKE_AGENTS:
        assert any(line.startswith(name) for line in result.stdout.splitlines()), name


def test_an_offline_evaluation_trace_replays_to_the_same_score(fixtures: Path, tmp_path: Path) -> None:
    trace, card_json = tmp_path / "trace.jsonl", tmp_path / "card.json"
    evaluated = run(
        "evaluate",
        "--agent",
        "univariate_bh",
        "--store",
        str(fixtures),
        "--trace",
        str(trace),
        "--json",
        str(card_json),
    )
    assert evaluated.returncode == 0, evaluated.stderr
    card = json.loads(card_json.read_text())
    assert card["agent"] == "univariate_bh" and card["n_worlds"] == len(card["worlds"]) > 0
    replayed = run("replay", "--trace", str(trace), "--store", str(fixtures))
    assert replayed.returncode == 0, replayed.stdout + replayed.stderr
    assert "MISMATCH" not in replayed.stdout
    finds = {line.split(":")[0]: line for line in replayed.stdout.splitlines()}
    assert set(finds) == {w["world_id"] for w in card["worlds"]}


def test_a_tampered_trace_fails_replay(fixtures: Path, tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    assert (
        run(
            "evaluate", "--agent", "random", "--store", str(fixtures), "--trace", str(trace), "--n", "2"
        ).returncode
        == 0
    )
    lines = trace.read_text().splitlines()
    event = json.loads(lines[-1])
    event["response_sha256"] = "0" * 64
    trace.write_text("\n".join([*lines[:-1], json.dumps(event)]) + "\n")
    replayed = run("replay", "--trace", str(trace), "--store", str(fixtures))
    assert replayed.returncode == 1 and "MISMATCH" in replayed.stdout


def test_the_conformance_suite_passes_in_process(fixtures: Path, tmp_path: Path) -> None:
    result = run("conformance", "--ledger", str(tmp_path / "ledger.json"))
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout[result.stdout.index("{") :])
    assert report["ok"] is True and all(report["checks"].values())
