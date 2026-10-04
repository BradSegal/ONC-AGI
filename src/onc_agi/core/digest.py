"""Canonical JSON digests used by traces and replay, and behaviour labels for versioned code."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# The package's own top-level name. The public export renames it (module and distribution spelling),
# so it is normalised out of labels: identical code in either copy carries the same label.
_PACKAGE = __name__.split(".")[0]


def canonical_sha256(payload: Any) -> str:
    """SHA-256 of canonical JSON, so replays can verify request and response bytes."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()


class _Behaviour(ast.NodeTransformer):
    """Drops statements that do nothing: bare strings (docstrings) and ``pass``."""

    def visit_Expr(self, node: ast.Expr) -> ast.Expr | None:
        return None if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str) else node

    def visit_Pass(self, node: ast.Pass) -> None:
        return None


def _behaviour(path: Path) -> str:
    """What a source file does, without its comments, docstrings or layout.

    Python is reduced to its AST dump without bare string statements (docstrings) or ``pass``;
    any other text file (the sandbox Dockerfile) to its non-comment lines, whitespace collapsed.
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix != ".py":
        lines = (" ".join(line.split()) for line in text.splitlines())
        return "\n".join(line for line in lines if line and not line.startswith("#"))
    tree = _Behaviour().visit(ast.parse(text))
    # Python 3.13 omits empty fields from dumps by default; keep them so labels match across versions.
    dump = ast.dump(tree, show_empty=True) if sys.version_info >= (3, 13) else ast.dump(tree)  # type: ignore[call-arg]
    return dump.replace(_PACKAGE, "<package>").replace(_PACKAGE.replace("_", "-"), "<package>")


def behaviour_label(name: str, *sources: Path, extra: Iterable[str] = ()) -> str:
    """``<name>+<8 hex>`` over what ``sources`` do (and any ``extra`` text, such as a config).

    Edits to comments, docstrings or formatting leave the label unchanged; any change to the
    code, a string the code uses, or ``extra`` changes it. Rows are comparable only within one label.
    """
    h = hashlib.sha256()
    for part in (*(_behaviour(p) for p in sources), *extra):
        h.update(part.encode())
        h.update(b"\x1f")
    return f"{name}+{h.hexdigest()[:8]}"
