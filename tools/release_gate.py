"""Release gate: build release assets only from a commit whose CI and site build are green.

    python3 tools/release_gate.py --repo OWNER/REPO --sha SHA --workflow ci.yml --job checks --job site \
        [--tag v1.0.0] [--main-ref origin/main]

Reads the commit's runs of the CI workflow through the GitHub API (``gh api``, authenticated by
``GH_TOKEN``) and passes only when the most recent run for exactly that commit has completed
with success and every named job in it succeeded. With ``--tag`` (a real release) the run must
come from a push to ``main``, the commit must be reachable from ``--main-ref``, and the tag must
equal ``v`` plus the version in ``pyproject.toml``. On a pass it writes ``version`` and
``prerelease`` to ``$GITHUB_OUTPUT`` when that is set. A missing, queued, running, failed,
cancelled or skipped run or job refuses the release; nothing is inferred from absence.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
Api = Callable[[str], Any]


def gh_api(path: str) -> Any:
    result = subprocess.run(["gh", "api", path], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"gh api {path} failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def ci_refusals(api: Api, repo: str, sha: str, workflow: str, jobs: list[str], main_only: bool) -> list[str]:
    """Reasons the commit's CI does not authorise a release; empty when it does."""
    query = f"repos/{repo}/actions/workflows/{workflow}/runs?head_sha={sha}&per_page=100"
    if main_only:
        query += "&event=push&branch=main"
    runs = [run for run in api(query).get("workflow_runs", []) if run.get("head_sha") == sha]
    if not runs:
        scope = "a push to main" if main_only else "any event"
        return [f"no {workflow} run for {sha} from {scope}"]
    run = max(runs, key=lambda item: (item.get("created_at") or "", item.get("id") or 0))
    where = f"{workflow} run {run.get('id')} (attempt {run.get('run_attempt')})"
    if run.get("status") != "completed":
        return [f"{where} is {run.get('status')}, not completed"]
    if run.get("conclusion") != "success":
        return [f"{where} concluded {run.get('conclusion')}"]
    listed = api(f"repos/{repo}/actions/runs/{run['id']}/jobs?filter=latest&per_page=100").get("jobs", [])
    outcome = {job.get("name"): job.get("conclusion") for job in listed}
    return [
        f"{where}: job {name} is {outcome.get(name, 'absent')}"
        for name in jobs
        if outcome.get(name) != "success"
    ]


def package_version() -> str:
    return str(tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"])


def tag_refusals(tag: str, sha: str, main_ref: str) -> list[str]:
    reasons: list[str] = []
    version = package_version()
    if tag != f"v{version}":
        reasons.append(f"tag {tag} does not match pyproject version {version} (expected v{version})")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", sha, main_ref], cwd=ROOT, check=False)
    if ancestor.returncode != 0:
        reasons.append(f"{sha} is not reachable from {main_ref}")
    return reasons


def main(argv: list[str] | None = None, api: Api = gh_api) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--workflow", default="ci.yml")
    parser.add_argument("--job", action="append", required=True, help="a CI job that must succeed")
    parser.add_argument("--tag", help="release tag; requires CI from a push to main")
    parser.add_argument("--main-ref", default="origin/main")
    args = parser.parse_args(argv)
    try:
        reasons = ci_refusals(api, args.repo, args.sha, args.workflow, args.job, main_only=bool(args.tag))
        if args.tag:
            reasons += tag_refusals(args.tag, args.sha, args.main_ref)
    except (RuntimeError, KeyError, ValueError, OSError) as exc:
        reasons = [f"gate could not be evaluated: {exc}"]
    for reason in reasons:
        print(f"release gate: REFUSED: {reason}", file=sys.stderr)
    if reasons:
        return 1
    version = package_version()
    print(
        f"release gate: PASSED: {args.workflow} green for {args.sha} (jobs: {', '.join(args.job)}); version {version}"
    )
    if output := os.environ.get("GITHUB_OUTPUT"):
        prerelease = re.search(r"(a|b|rc|\.dev)\d", version) is not None  # PEP 440 pre-release
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"version={version}\nprerelease={str(prerelease).lower()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
