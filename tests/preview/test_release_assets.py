"""Release assets: world sets carry the data notice and match their pins; the gate needs green CI."""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path
from typing import Any

import pytest
import release_gate
import world_sets

ROOT = Path(__file__).resolve().parents[2]
NOTICE = (ROOT / "DATA-NOTICE.md").read_bytes()


def _archive(path: Path, members: dict[str, bytes], symlink: str | None = None) -> str:
    with tarfile.open(path, "w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        if symlink:
            link = tarfile.TarInfo(symlink)
            link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
            archive.addfile(link)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _names(path: Path) -> list[str]:
    with tarfile.open(path, "r:gz") as archive:
        return archive.getnames()


def test_declared_world_sets_are_pinned_and_the_notice_names_every_source() -> None:
    sets = world_sets.load_spec(world_sets.SPEC)
    assert sets and all(len(item["sha256"]) == 64 for item in sets)
    assert all(len(c["sha256"]) == 64 for item in sets for c in item["companions"])
    text = NOTICE.decode()
    for source in ("tcga_brca", "tcga_pancan", "metabric", "msk_impact", "scanb", "nhanes", "ODbL"):
        assert source in text


def test_package_adds_the_notice_first_and_keeps_every_member(tmp_path: Path) -> None:
    sources, out = tmp_path / "sources", tmp_path / "out"
    sources.mkdir()
    members = {"manifests/m.json": b"{}", "public_train/w1/card.json": b"{}"}
    pin = _archive(sources / "set.tar.gz", members)
    world_sets.package(
        [{"asset": "set.tar.gz", "sha256": pin, "companions": []}], NOTICE, sources, out, mtime=0
    )
    assert _names(out / "set.tar.gz") == ["DATA-NOTICE.md", *members]
    assert world_sets.archive_notice(out / "set.tar.gz") == NOTICE
    first = (out / "set.tar.gz").read_bytes()
    world_sets.package(
        [{"asset": "set.tar.gz", "sha256": pin, "companions": []}], NOTICE, sources, out, mtime=0
    )
    assert (out / "set.tar.gz").read_bytes() == first, "packaging is deterministic"


def test_package_refuses_a_missing_or_altered_companion(tmp_path: Path) -> None:
    pin = _archive(tmp_path / "set.tar.gz", {"a.json": b"{}"})
    companion = {"asset": "set-excluded.txt", "sha256": "1" * 64}
    sets = [{"asset": "set.tar.gz", "sha256": pin, "companions": [companion]}]
    with pytest.raises(world_sets.WorldSetError, match=r"missing source asset set-excluded\.txt"):
        world_sets.package(sets, NOTICE, tmp_path, tmp_path / "o", 0)
    (tmp_path / "set-excluded.txt").write_text("changed")
    with pytest.raises(world_sets.WorldSetError, match="does not match the pinned"):
        world_sets.package(sets, NOTICE, tmp_path, tmp_path / "o", 0)


def test_package_refuses_a_source_that_differs_from_its_pin(tmp_path: Path) -> None:
    _archive(tmp_path / "set.tar.gz", {"a.json": b"{}"})
    with pytest.raises(world_sets.WorldSetError, match="does not match the pinned"):
        world_sets.package(
            [{"asset": "set.tar.gz", "sha256": "0" * 64, "companions": []}],
            NOTICE,
            tmp_path,
            tmp_path / "o",
            0,
        )


@pytest.mark.parametrize(
    ("members", "symlink", "message"),
    [
        ({"../escape.json": b"{}"}, None, "unsafe member path"),
        ({"a.json": b"{}"}, "link", "not a regular file"),
        ({"DATA-NOTICE.md": b"another notice"}, None, "different DATA-NOTICE.md"),
    ],
)
def test_package_refuses_unsafe_or_conflicting_members(
    tmp_path: Path, members: dict[str, bytes], symlink: str | None, message: str
) -> None:
    pin = _archive(tmp_path / "set.tar.gz", members, symlink)
    with pytest.raises(world_sets.WorldSetError, match=message):
        world_sets.package(
            [{"asset": "set.tar.gz", "sha256": pin, "companions": []}], NOTICE, tmp_path, tmp_path / "o", 0
        )


def test_verify_requires_complete_checksums_and_the_notice(tmp_path: Path) -> None:
    sources, out = tmp_path / "sources", tmp_path / "out"
    sources.mkdir()
    (sources / "set-certified.txt").write_text("w1\n")
    companion = {
        "asset": "set-certified.txt",
        "sha256": world_sets.sha256_file(sources / "set-certified.txt"),
    }
    pin = _archive(sources / "set.tar.gz", {"a.json": b"{}"})
    sets = [{"asset": "set.tar.gz", "sha256": pin, "companions": [companion]}]
    world_sets.package(sets, NOTICE, sources, out, mtime=0)
    assert (out / "set-certified.txt").read_text() == "w1\n"
    assert (out / "DATA-NOTICE.md").read_bytes() == NOTICE
    (out / "onc_agi-1.0-py3-none-any.whl").write_bytes(b"wheel")
    world_sets.write_checksums(out)
    world_sets.verify(out, sets, NOTICE)

    (out / "stray.txt").write_text("unlisted")
    with pytest.raises(world_sets.WorldSetError, match="missing from SHA256SUMS"):
        world_sets.verify(out, sets, NOTICE)
    (out / "stray.txt").unlink()
    (out / "set-certified.txt").write_text("w1\nw2\n")  # a companion that no longer matches its pin
    world_sets.write_checksums(out)
    with pytest.raises(world_sets.WorldSetError, match="differs from its pin"):
        world_sets.verify(out, sets, NOTICE)
    (out / "set-certified.txt").write_text("w1\n")
    world_sets.write_checksums(out)
    (out / "onc_agi-1.0-py3-none-any.whl").write_bytes(b"tampered")
    with pytest.raises(world_sets.WorldSetError, match="does not match SHA256SUMS"):
        world_sets.verify(out, sets, NOTICE)
    _archive(out / "set.tar.gz", {"a.json": b"{}"})  # an archive without the notice
    world_sets.write_checksums(out)
    with pytest.raises(world_sets.WorldSetError, match="first member is not"):
        world_sets.verify(out, sets, NOTICE)


def _api(runs: list[dict[str, Any]], jobs: dict[str, str]) -> Any:
    def api(path: str) -> dict[str, Any]:
        if "/jobs" in path:
            return {"jobs": [{"name": n, "conclusion": c} for n, c in jobs.items()]}
        return {"workflow_runs": runs}

    return api


SHA = "a" * 40
GREEN = {"id": 2, "head_sha": SHA, "status": "completed", "conclusion": "success", "created_at": "2"}


@pytest.mark.parametrize(
    ("runs", "jobs", "refusal"),
    [
        ([], {}, "no ci.yml run"),
        ([{**GREEN, "head_sha": "b" * 40}], {}, "no ci.yml run"),
        ([{**GREEN, "status": "in_progress", "conclusion": None}], {}, "not completed"),
        ([{**GREEN, "conclusion": "failure"}], {}, "concluded failure"),
        (
            [GREEN, {**GREEN, "id": 3, "created_at": "3", "conclusion": "cancelled"}],
            {},
            "concluded cancelled",
        ),
        ([GREEN], {"checks": "success"}, "job site is absent"),
        ([GREEN], {"checks": "success", "site": "skipped"}, "job site is skipped"),
        ([GREEN], {"checks": "success", "site": None}, "job site is None"),
    ],
)
def test_gate_refuses_anything_but_a_green_run_with_every_job(
    runs: list[dict[str, Any]], jobs: dict[str, str], refusal: str
) -> None:
    reasons = release_gate.ci_refusals(_api(runs, jobs), "o/r", SHA, "ci.yml", ["checks", "site"], False)
    assert any(refusal in reason for reason in reasons), reasons


def test_gate_passes_a_green_run_and_refuses_a_tag_that_is_not_the_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    api = _api([GREEN], {"checks": "success", "site": "success"})
    assert release_gate.ci_refusals(api, "o/r", SHA, "ci.yml", ["checks", "site"], True) == []
    argv = ["--repo", "o/r", "--sha", SHA, "--job", "checks", "--tag", "v0.0.0-not-the-version"]
    assert release_gate.main(argv, api=api) == 1
    assert "does not match pyproject version" in capsys.readouterr().err
