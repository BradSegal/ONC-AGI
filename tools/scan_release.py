"""Release scan: nothing private, secret or machine-specific is in the tree or its history.

Checks every file tracked by git, or every file in the working tree when there is
no commit yet, and every path that ever existed in the history:

* no import of the private generator package or of PyPlasmode;
* no absolute home paths, private keys or credential-shaped tokens;
* answer-key content (groups plus a reject set) only in public-train fixtures and
  the public-train example payload;
* Parquet only inside the packaged fixture store, and no file over 5 MB;
* no ``data/``, ``tickets/``, ``design/`` or ``keys/`` directories, and no ``.env``.

It then runs ``gitleaks`` over the full history and the current working tree. A missing scanner fails closed. Exit code 0
means every check passed.

    uv run python tools/scan_release.py
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORE = "src/onc_agi/fixtures/store/"
KEY_EXAMPLE = "examples/payloads/answer_key_public_train.json"
FORBIDDEN_DIRS = ("data/", "tickets/", "design/", "keys/", "research/")
MAX_BYTES = 5 * 1024 * 1024
TEXT_RULES = (
    (
        re.compile(r"^\s*(from|import)\s+(private_generator|pyplasmode)\b", re.M),
        "imports a private or excluded package",
    ),
    (re.compile(r"/home/[A-Za-z0-9_.-]+/|/Users/[A-Za-z0-9_.-]+/|[A-Z]:\\\\Users\\\\"), "absolute home path"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
    (re.compile(r"\b(ghp|gho|ghs|github_pat)_[A-Za-z0-9_]{20,}"), "GitHub token"),
    (re.compile(r"\bsk-(ant-)?[A-Za-z0-9_-]{20,}"), "API key"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AWS access key"),
)
SELF = "tools/scan_release.py"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def current_files() -> list[str]:
    """Tracked files plus untracked files that are not ignored (what the next commit could add)."""
    tracked = git("ls-files", "-z").split("\0")
    untracked = git("ls-files", "--others", "--exclude-standard", "-z").split("\0")
    return sorted((set(tracked) | set(untracked)) - {""})


def history_paths() -> set[str]:
    return {line for line in git("log", "--all", "--name-only", "--pretty=format:").splitlines() if line}


def is_answer_key(path: Path) -> bool:
    try:
        data = json.loads(path.read_text())
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(data, dict) and "groups" in data and "reject_set" in data


def scan() -> list[str]:
    problems: list[str] = []
    files = current_files()
    for rel in sorted(set(files) | history_paths()):
        if rel.startswith(FORBIDDEN_DIRS) or Path(rel).name == ".env":
            problems.append(f"{rel}: forbidden path (present now or in history)")
    for rel in files:
        path = ROOT / rel
        if not path.is_file():
            continue
        if path.stat().st_size > MAX_BYTES:
            problems.append(f"{rel}: larger than 5 MB")
        if rel.endswith(".parquet") and not rel.startswith(STORE + "public_train/"):
            problems.append(f"{rel}: data file outside the public-train fixture store")
        public_key = rel.startswith(STORE + "public_train/") or rel == KEY_EXAMPLE
        if rel.endswith(".json") and not public_key and is_answer_key(path):
            problems.append(f"{rel}: answer-key content outside public train")
        if rel == SELF or rel.endswith(".parquet"):
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        for pattern, reason in TEXT_RULES:
            if pattern.search(text):
                problems.append(f"{rel}: {reason}")
    return problems


def gitleaks() -> list[str]:
    if shutil.which("gitleaks") is None:
        return ["gitleaks is required: install it before running the release scan"]
    git("rev-parse", "--verify", "HEAD")
    problems = []
    with tempfile.TemporaryDirectory(prefix="onc-secret-scan-") as directory:
        candidate = Path(directory)
        for rel in current_files():
            source = ROOT / rel
            if source.is_symlink():
                problems.append(f"{rel}: symlinks are not permitted in the release candidate")
            elif source.is_file():
                target = candidate / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        for mode, root, extra in (("git", ROOT, ["--log-opts=--all"]), ("dir", candidate, [])):
            args = ["gitleaks", mode, str(root), "--no-banner", "--redact", *extra]
            result = subprocess.run(args, capture_output=True, text=True, check=False)
            if result.returncode:
                problems.append(f"gitleaks {mode}: {result.stdout.strip() or result.stderr.strip()}")
    return problems


def main() -> int:
    problems = scan() + gitleaks()
    for problem in problems:
        print(problem)
    print(f"release scan: {'FAILED' if problems else 'passed'} ({len(current_files())} files)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
