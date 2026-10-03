"""The ``onc-agi`` command in-process: exit codes and outputs that harness authors script against."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from arena_factories import planted_world
from onc_agi.adapters import cli
from onc_agi.core.schema import Mode, Tier
from onc_agi.infra.bundles import write_world


@pytest.fixture
def small_store(tmp_path: Path) -> Path:
    root = tmp_path / "store"
    for i in range(6):
        mode = Mode.SEQUENTIAL if i % 2 else Mode.FULL_ACCESS
        write_world(root, *planted_world(f"w-{i:02d}", signal=i % 3 != 2, seed=i, n_pool=80, mode=mode))
    return root


def test_smoke_passes_on_a_store_with_known_answers(
    small_store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["smoke", "--store", str(small_store)]) == 0
    out = capsys.readouterr().out
    assert "6 public-train worlds" in out and "smoke passed" in out


def test_smoke_refuses_an_empty_store(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["smoke", "--store", str(tmp_path)]) == 2
    assert "no public-train" in capsys.readouterr().err


def test_smoke_fails_when_the_fixtures_cannot_define_a_score(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "store"
    for i in range(3):
        write_world(root, *planted_world(f"w-{i:02d}", signal=True, seed=i, n_pool=80))
    assert cli.main(["smoke", "--store", str(root)]) == 1
    assert "smoke FAILED" in capsys.readouterr().err


def test_evaluate_trace_and_replay_round_trip(
    small_store: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trace, out = tmp_path / "t.jsonl", tmp_path / "card.json"
    args = [
        "evaluate",
        "--agent",
        "univariate_bh",
        "--store",
        str(small_store),
        "--trace",
        str(trace),
        "--json",
        str(out),
    ]
    assert cli.main(args) == 0
    card = json.loads(out.read_text())
    assert card["agent"] == "univariate_bh" and card["n_worlds"] == 6
    assert cli.main(["replay", "--trace", str(trace), "--store", str(small_store)]) == 0
    assert "MISMATCH" not in capsys.readouterr().out


def test_evaluate_limits_the_number_of_worlds(small_store: Path, tmp_path: Path) -> None:
    out = tmp_path / "card.json"
    assert (
        cli.main(
            ["evaluate", "--agent", "random", "--store", str(small_store), "--n", "3", "--json", str(out)]
        )
        == 0
    )
    assert json.loads(out.read_text())["n_worlds"] == 3


def test_conformance_reports_every_check(
    small_store: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["conformance", "--store", str(small_store), "--ledger", str(tmp_path / "ledger.json")])
    report = json.loads(capsys.readouterr().out)
    assert code == 0 and report["ok"] is True
    assert "resume_returns_same_state" in report["checks"]
    assert {"restart_restores_open_scorecard", "foreign_key_refused_like_unknown_id[close]"} <= set(
        report["checks"]
    )
    assert (tmp_path / "ledger.archive" / "closed").is_dir()  # default archive sits beside the ledger


def test_serve_accepts_archive_and_ttl_flags(
    small_store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import uvicorn

    served: dict[str, object] = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: served.update(app=app, port=port))
    base = ["serve", "--store", str(small_store), "--ledger", str(tmp_path / "l.json"), "--port", "9"]
    assert cli.main([*base, "--archive", str(tmp_path / "arch"), "--ttl-hours", "0.5"]) == 0
    assert served["port"] == 9 and (tmp_path / "arch" / "open").is_dir()
    assert cli.main([*base, "--ttl-hours", "0"]) == 0 and (tmp_path / "l.archive").is_dir()
    assert cli.main([*base, "--ttl-hours", "-1"]) == 2


def test_serve_refuses_eval_worlds_without_issued_keys(
    small_store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: None)
    world, key = planted_world("e-00", signal=True, seed=9, n_pool=80, tier=Tier.PUBLIC_EVAL)
    write_world(small_store, world, key, keys_dir=tmp_path / "answer-keys")
    base = ["serve", "--store", str(small_store), "--ledger", str(tmp_path / "l.json")]
    assert cli.main(base) == 2
    keys = tmp_path / "keys.txt"
    keys.write_text("issued-key\n")
    assert cli.main([*base, "--api-keys", str(keys)]) == 0
    assert cli.main([*base, "--allow-unissued-keys"]) == 0


def test_unknown_agents_are_rejected_by_the_parser(small_store: Path) -> None:
    with pytest.raises(SystemExit):
        cli.main(["evaluate", "--agent", "nobody", "--store", str(small_store)])
