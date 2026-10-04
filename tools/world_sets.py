"""Prepare the world-set release assets: pinned sources in, notice-bearing archives and SHA256SUMS out.

World sets live outside git. ``release/world-sets.json`` names the release they are fetched from
and pins the SHA-256 of each archive and of its companion files (the certified and excluded world-id
lists). The release workflow downloads them and then runs:

    python3 tools/world_sets.py package --sources <downloaded> --out <dir>
    python3 tools/world_sets.py checksums <dir with every release asset>
    python3 tools/world_sets.py verify <same dir>

``package`` refuses any file whose hash differs from its pin, and an archive that holds anything
but regular files and directories under relative paths. It rewrites each archive with
``DATA-NOTICE.md`` as its first member (an existing notice must match this repository's exactly),
copies the companions unchanged and adds ``DATA-NOTICE.md`` itself. ``checksums`` writes
``SHA256SUMS`` for every file in the directory. ``verify`` re-checks every hash, refuses unlisted
files, and confirms each archive carries the notice and each companion still matches its pin. Every command
exits non-zero on the first problem, so a release step that uses it cannot pass by omission.
Standard library only, so it runs on a bare runner.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "release" / "world-sets.json"
NOTICE = ROOT / "DATA-NOTICE.md"
NOTICE_NAME = "DATA-NOTICE.md"
SUMS = "SHA256SUMS"


class WorldSetError(Exception):
    """A world-set asset failed a release check."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _pin(path: Path, item: Any) -> dict[str, str]:
    asset = item.get("asset", "") if isinstance(item, dict) else ""
    pin = item.get("sha256", "") if isinstance(item, dict) else ""
    if not asset or "/" in asset or asset in (NOTICE_NAME, SUMS) or len(pin) != 64:
        raise WorldSetError(f"{path}: each file needs a plain asset name and a 64-hex sha256: {item}")
    return {"asset": asset, "sha256": pin}


def load_spec(path: Path) -> list[dict[str, Any]]:
    """The declared sets, each ``{"asset", "sha256", "companions": [{"asset", "sha256"}]}``."""
    spec = json.loads(path.read_text(encoding="utf-8"))
    sets = spec.get("sets")
    if not spec.get("source_release") or not isinstance(sets, list) or not sets:
        raise WorldSetError(f"{path}: needs a source_release and at least one set")
    return [
        {**_pin(path, item), "companions": [_pin(path, c) for c in item.get("companions", [])]}
        for item in sets
    ]


def _verified_source(sources: Path, item: dict[str, str]) -> Path:
    source = sources / item["asset"]
    if not source.is_file():
        raise WorldSetError(f"missing source asset {item['asset']}")
    actual = sha256_file(source)
    if actual != item["sha256"]:
        raise WorldSetError(f"{item['asset']}: sha256 {actual} does not match the pinned {item['sha256']}")
    return source


def _check_member(archive: str, member: tarfile.TarInfo) -> None:
    parts = PurePosixPath(member.name).parts
    if member.name.startswith("/") or ".." in parts or not parts:
        raise WorldSetError(f"{archive}: unsafe member path {member.name!r}")
    if not (member.isfile() or member.isdir()):
        raise WorldSetError(f"{archive}: member {member.name!r} is not a regular file or directory")


def package_archive(source: Path, target: Path, notice: bytes, mtime: int) -> int:
    """Stream ``source`` into ``target`` with the notice first; return the member count copied."""
    copied = 0
    with (
        source.open("rb") as raw_in,
        tarfile.open(fileobj=raw_in, mode="r|gz") as src,
        target.open("wb") as raw_out,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw_out, compresslevel=6, mtime=0) as gz,
        tarfile.open(fileobj=gz, mode="w|", format=tarfile.PAX_FORMAT) as out,
    ):
        info = tarfile.TarInfo(NOTICE_NAME)
        info.size, info.mode, info.mtime = len(notice), 0o644, mtime
        out.addfile(info, io.BytesIO(notice))
        for member in src:
            _check_member(source.name, member)
            if member.isfile():
                handle = src.extractfile(member)
                assert handle is not None  # isfile() guarantees a payload
                if PurePosixPath(member.name) == PurePosixPath(NOTICE_NAME):
                    if handle.read() != notice:
                        raise WorldSetError(f"{source.name}: carries a different {NOTICE_NAME}")
                    continue
                out.addfile(member, handle)
            else:
                out.addfile(member)
            copied += 1
    if copied == 0:
        raise WorldSetError(f"{source.name}: archive is empty")
    return copied


def archive_notice(path: Path) -> bytes | None:
    """The bytes of the archive's first member if it is the data notice, else None."""
    with tarfile.open(path, mode="r|gz") as archive:
        first = next(iter(archive), None)
        if first is None or first.name != NOTICE_NAME or not first.isfile():
            return None
        handle = archive.extractfile(first)
        return handle.read() if handle is not None else None


def package(sets: list[dict[str, Any]], notice: bytes, sources: Path, out: Path, mtime: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for item in sets:
        source = _verified_source(sources, item)
        target = out / item["asset"]
        count = package_archive(source, target, notice, mtime)
        if archive_notice(target) != notice:
            raise WorldSetError(f"{target.name}: notice missing after packaging")
        print(f"packaged {item['asset']}: source sha256 verified, {count} members, notice added")
        for companion in item["companions"]:
            shutil.copyfile(_verified_source(sources, companion), out / companion["asset"])
            print(f"copied {companion['asset']}: sha256 verified")
    (out / NOTICE_NAME).write_bytes(notice)


def write_checksums(directory: Path) -> Path:
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.name != SUMS)
    if not files:
        raise WorldSetError(f"{directory}: no assets to checksum")
    sums = directory / SUMS
    sums.write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in files), encoding="utf-8")
    print(f"wrote {SUMS} for {len(files)} assets")
    return sums


def verify(directory: Path, sets: list[dict[str, Any]], notice: bytes) -> None:
    sums = directory / SUMS
    if not sums.is_file():
        raise WorldSetError(f"{directory}: no {SUMS}")
    listed: dict[str, str] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, sep, name = line.partition("  ")
        if not sep or len(digest) != 64 or not name or name in listed:
            raise WorldSetError(f"{SUMS}: malformed or duplicate line {line!r}")
        listed[name] = digest
    present = {p.name for p in directory.iterdir() if p.is_file() and p.name != SUMS}
    if unlisted := sorted(present - set(listed)):
        raise WorldSetError(f"assets missing from {SUMS}: {unlisted}")
    for name, digest in sorted(listed.items()):
        path = directory / name
        if not path.is_file():
            raise WorldSetError(f"{SUMS} lists {name}, which is absent")
        if sha256_file(path) != digest:
            raise WorldSetError(f"{name}: sha256 does not match {SUMS}")
    for item in sets:
        if item["asset"] not in listed:
            raise WorldSetError(f"world set {item['asset']} is not among the release assets")
        if archive_notice(directory / item["asset"]) != notice:
            raise WorldSetError(f"{item['asset']}: first member is not this repository's {NOTICE_NAME}")
        for companion in item["companions"]:
            if listed.get(companion["asset"]) != companion["sha256"]:
                raise WorldSetError(f"{companion['asset']} is absent or differs from its pin")
    if NOTICE_NAME not in listed or (directory / NOTICE_NAME).read_bytes() != notice:
        raise WorldSetError(f"{NOTICE_NAME} is absent or differs from this repository's")
    print(f"verified {len(listed)} assets against {SUMS}; {len(sets)} world sets carry {NOTICE_NAME}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--spec", type=Path, default=SPEC, help="world-set declaration")
    parser.add_argument("--notice", type=Path, default=NOTICE, help="data notice to place in each archive")
    sub = parser.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("package", help="verify pinned sources and add the data notice")
    pack.add_argument("--sources", type=Path, required=True)
    pack.add_argument("--out", type=Path, required=True)
    sub.add_parser("checksums", help="write SHA256SUMS for a release directory").add_argument(
        "dir", type=Path
    )
    sub.add_parser("verify", help="check SHA256SUMS and the notice in each world set").add_argument(
        "dir", type=Path
    )
    args = parser.parse_args(argv)
    try:
        sets = load_spec(args.spec)
        notice = args.notice.read_bytes()
        if args.command == "package":
            package(sets, notice, args.sources, args.out, int(os.environ.get("SOURCE_DATE_EPOCH", "0")))
        elif args.command == "checksums":
            write_checksums(args.dir)
        else:
            verify(args.dir, sets, notice)
    except (WorldSetError, OSError, tarfile.TarError, json.JSONDecodeError) as exc:
        print(f"world sets: REFUSED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
