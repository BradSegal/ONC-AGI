"""Version labels follow behaviour: comment, docstring and formatting edits keep them; code edits move them."""

from __future__ import annotations

import ast
import shutil
from pathlib import Path

import onc_agi
import pytest
from onc_agi.core.digest import behaviour_label
from onc_agi.services import scoring

PACKAGE = Path(onc_agi.__file__).parent
LABELLED = ("services/scoring.py", "services/engine.py", "adapters/inspect_task.py", "adapters/agents/llm.py")


def _replace_spans(text: str, spans: list[tuple[int, int, int, int, str]]) -> str:
    """Replace (line, col, end_line, end_col) source spans, last first so earlier offsets hold."""
    lines = text.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line.encode()))
    raw = text.encode()
    for line, col, end_line, end_col, new in sorted(spans, reverse=True):
        raw = raw[: starts[line - 1] + col] + new.encode() + raw[starts[end_line - 1] + end_col :]
    return raw.decode()


def recommented(text: str) -> str:
    return (
        "# a new leading comment\n"
        + text.replace("\n\n\n", "\n\n\n# a new comment between blocks\n", 1)
        + "# end\n"
    )


def redocumented(text: str) -> str:
    spans = [
        (doc.lineno, doc.col_offset, doc.end_lineno or doc.lineno, doc.end_col_offset or 0, '"""Reworded."""')
        for node in ast.walk(ast.parse(text))
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        and ast.get_docstring(node) is not None
        for doc in node.body[:1]
    ]
    assert spans, "the module documents something"
    return _replace_spans(text, spans)


def reformatted(text: str) -> str:
    return ast.unparse(ast.parse(text))  # new quotes, wrapping and spacing; comments gone


def mutated(text: str) -> str:
    """Increment the first integer literal the code uses: a behavioural edit."""
    first = next(
        n
        for n in ast.walk(ast.parse(text))
        if isinstance(n, ast.Constant) and type(n.value) is int and n.end_lineno == n.lineno
    )
    return _replace_spans(
        text,
        [(first.lineno, first.col_offset, first.lineno, first.end_col_offset or 0, str(first.value + 1))],
    )


@pytest.mark.unit
@pytest.mark.parametrize("module", LABELLED)
def test_comment_docstring_and_format_edits_keep_the_label_and_code_edits_change_it(
    module: str, tmp_path: Path
) -> None:
    source = PACKAGE / module
    copy = tmp_path / source.name
    shutil.copy(source, copy)
    original = behaviour_label("probe-1.0", copy)
    assert original == behaviour_label("probe-1.0", source)
    for edit in (recommented, redocumented, reformatted):
        copy.write_text(edit(source.read_text()))
        assert copy.read_bytes() != source.read_bytes()
        assert behaviour_label("probe-1.0", copy) == original, edit.__name__
    copy.write_text(mutated(source.read_text()))
    assert behaviour_label("probe-1.0", copy) != original


@pytest.mark.unit
def test_the_sandbox_image_label_ignores_comments_and_spacing_but_not_instructions(tmp_path: Path) -> None:
    dockerfile = PACKAGE / "adapters" / "sandbox" / "Dockerfile"
    copy = tmp_path / "Dockerfile"
    text = dockerfile.read_text()
    copy.write_text("# reworded header\n\n" + text.replace("WORKDIR /data", "WORKDIR    /data  "))
    assert behaviour_label("probe-1.0", copy) == behaviour_label("probe-1.0", dockerfile)
    copy.write_text(text.replace("WORKDIR /data", "WORKDIR /tmp"))
    assert behaviour_label("probe-1.0", copy) != behaviour_label("probe-1.0", dockerfile)


@pytest.mark.unit
def test_shipped_labels_are_behaviour_labels_of_their_modules() -> None:
    core = PACKAGE / "core"
    scorer = behaviour_label("scorer-1.0", PACKAGE / "services" / "scoring.py", core / "schema.py")
    assert scorer == scoring.SCORER_VERSION
    engine_sources = (core / "world.py", core / "schema.py", core / "errors.py")
    assert (
        behaviour_label("engine-1.0", PACKAGE / "services" / "engine.py", *engine_sources)
        == scoring.ENGINE_VERSION
    )
    # the scorer's label moves when the answer-key contract it reads changes
    assert behaviour_label("scorer-1.0", PACKAGE / "services" / "scoring.py") != scoring.SCORER_VERSION
    inspect_task = pytest.importorskip("onc_agi.adapters.inspect_task")
    sandbox = PACKAGE / "adapters" / "sandbox" / "Dockerfile"
    assert (
        behaviour_label("inspect-standard-1.1", PACKAGE / "adapters" / "inspect_task.py", sandbox)
        == inspect_task.HARNESS
    )
