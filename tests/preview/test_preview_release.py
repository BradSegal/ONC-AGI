"""Export hygiene: the manifest matches the exported bytes and the release scan catches planted leaks."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import scan_release

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((ROOT / "EXPORT.json").read_text())


def test_every_exported_file_matches_its_recorded_hash() -> None:
    for entry in MANIFEST["files"]:
        assert hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest() == entry["sha256"], entry[
            "path"
        ]


def test_runtime_contains_only_exported_files_and_the_fixture_store() -> None:
    exported = {entry["path"] for entry in MANIFEST["files"]}
    for path in (ROOT / "src" / "onc_agi").rglob("*"):
        rel = str(path.relative_to(ROOT))
        if path.is_file() and "__pycache__" not in rel and "/fixtures/" not in rel:
            assert rel in exported, f"{rel} is not an exported file"


def test_manifest_records_no_absolute_paths() -> None:
    text = json.dumps(MANIFEST)
    assert "/home/" not in text and "/Users/" not in text


@pytest.fixture
def scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "tools", repo / "tools")
    (repo / "README.md").write_text("clean\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    monkeypatch.setattr(scan_release, "ROOT", repo)
    return repo


def commit(repo: Path) -> None:
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.org", "commit", "-qm", "x"],
        cwd=repo,
        check=True,
    )


def test_scan_passes_a_clean_tree(scratch: Path) -> None:
    commit(scratch)
    assert scan_release.scan() == []


@pytest.mark.parametrize(
    ("rel", "content", "reason"),
    [
        ("src/x.py", "from private_generator.services import generator\n", "private"),
        ("src/y.py", "PATH = '/" + "home/someone/secret'\n", "absolute home path"),
        ("keys/w.json", "{}", "forbidden path"),
        ("examples/k.json", json.dumps({"groups": [], "reject_set": []}), "answer-key"),
        ("src/z.py", "TOKEN = 'ghp_" + "a" * 30 + "'\n", "GitHub token"),
    ],
)
def test_scan_catches_planted_leaks(scratch: Path, rel: str, content: str, reason: str) -> None:
    path = scratch / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    commit(scratch)
    assert any(reason in problem for problem in scan_release.scan())


def test_scan_remembers_forbidden_paths_deleted_from_history(scratch: Path) -> None:
    (scratch / "data").mkdir()
    (scratch / "data" / "cohort.csv").write_text("x\n")
    commit(scratch)
    shutil.rmtree(scratch / "data")
    commit(scratch)
    assert any("data/cohort.csv" in problem for problem in scan_release.scan())
