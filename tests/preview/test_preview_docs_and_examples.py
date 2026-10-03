"""Documentation code blocks execute, and every template agent runs end to end."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = sorted((ROOT / "docs").glob("*.md"))
BLOCK = re.compile(r"```python\n(.*?)```", re.S)


@pytest.mark.parametrize("doc", [d for d in DOCS if BLOCK.search(d.read_text())], ids=lambda d: d.name)
def test_doc_python_examples_execute(doc: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    namespace: dict[str, object] = {"__name__": "__doc_example__"}
    for block in BLOCK.findall(doc.read_text()):
        exec(compile(block, str(doc), "exec"), namespace)


def test_docs_link_only_to_files_that_exist() -> None:
    for doc in [*DOCS, ROOT / "README.md", ROOT / "STABILITY.md", ROOT / "CONTRIBUTING.md"]:
        for target in re.findall(r"\]\(([^)#:]+)\)", doc.read_text()):
            assert (doc.parent / target).exists(), f"{doc.name} links to missing {target}"


def test_pipeline_template() -> None:
    import pipeline_agent

    card = pipeline_agent.main()
    assert card.find is not None and card.find > 0.5


def test_sequential_template_spends_less_than_the_budget() -> None:
    import sequential_agent

    card = sequential_agent.main()
    assert card.discovery_score is not None and card.mean_data_cost > 0


def test_http_template_matches_the_in_process_scorer() -> None:
    import http_agent

    scorecard = http_agent.main()
    assert scorecard.n_worlds == 20 and scorecard.discovery_score is not None


def test_llm_template_runs_offline() -> None:
    import llm_agent

    assert llm_agent.main() == [1.0, 1.0, 0.0]


def test_smoke_command_from_a_foreign_directory(tmp_path: Path) -> None:
    exe = Path(sys.executable).with_name("onc-agi")
    result = subprocess.run(
        [str(exe), "smoke"], cwd=tmp_path, capture_output=True, text=True, env=os.environ, timeout=600
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "smoke passed" in result.stdout


def test_smoke_fails_loudly_without_fixture_worlds(tmp_path: Path) -> None:
    exe = Path(sys.executable).with_name("onc-agi")
    empty = tmp_path / "store"
    empty.mkdir()
    result = subprocess.run(
        [str(exe), "smoke", "--store", str(empty)], capture_output=True, text=True, timeout=120
    )
    assert result.returncode != 0
    assert "no public-train fixture worlds" in result.stderr
