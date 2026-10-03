"""Hexagonal layer rules for the public package.

* core imports only the standard library, pydantic, numpy, scipy and core itself;
* services never import infra or adapters;
* infra never imports services or adapters;
* nothing public imports the private ``private_generator`` package, and neither do these tests.

The checker parses source with :mod:`ast`; nothing is imported, so a forbidden
import cannot hide behind a failing import.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterator
from pathlib import Path

import onc_agi
import pytest

PACKAGE_ROOT = Path(onc_agi.__file__).parent
TESTS_ROOT = Path(__file__).resolve().parents[1]
CORE_THIRD_PARTY = {"pydantic", "numpy", "scipy"}


def imported_modules(path: Path) -> Iterator[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.lineno, node.module


def layer_of(path: Path, root: Path) -> str | None:
    parts = path.relative_to(root).parts
    return parts[0] if len(parts) > 1 and parts[0] in {"core", "services", "infra", "adapters"} else None


def violations(root: Path, package: str = "onc_agi") -> list[str]:
    """Every layer-rule violation under ``root``, as ``path:line imports module (rule)``."""
    found: list[str] = []
    for path in sorted(root.rglob("*.py")):
        layer = layer_of(path, root)
        for line, module in imported_modules(path):
            top = module.split(".")[0]
            rule = None
            if top == "private_generator":
                rule = "public package imports the private generator"
            elif layer == "core":
                internal_ok = module == f"{package}.core" or module.startswith(f"{package}.core.")
                if top == package and not internal_ok:
                    rule = "core imports an outer layer"
                elif top != package and top not in sys.stdlib_module_names and top not in CORE_THIRD_PARTY:
                    rule = "core imports a non-whitelisted dependency"
            elif layer == "services" and module.startswith((f"{package}.infra", f"{package}.adapters")):
                rule = "services import infra or adapters"
            elif layer == "infra" and module.startswith((f"{package}.services", f"{package}.adapters")):
                rule = "infra imports services or adapters"
            if rule:
                found.append(f"{path.relative_to(root)}:{line} imports {module} ({rule})")
    return found


def test_public_package_respects_layer_rules() -> None:
    assert violations(PACKAGE_ROOT) == []


def test_the_public_runtime_does_not_depend_on_pyplasmode() -> None:
    """Evaluator bridges live in the private package; the public runtime stands alone."""
    offenders = [
        f"{p.relative_to(PACKAGE_ROOT)}:{line}"
        for p in sorted(PACKAGE_ROOT.rglob("*.py"))
        for line, module in imported_modules(p)
        if module.split(".")[0] == "pyplasmode"
    ]
    assert offenders == []
    manifest = (PACKAGE_ROOT.parents[1] / "pyproject.toml").read_text()
    assert "pyplasmode" not in manifest


def test_public_tests_never_import_the_private_generator() -> None:
    offenders = [
        f"{p.relative_to(TESTS_ROOT)}:{line}"
        for p in sorted(TESTS_ROOT.rglob("*.py"))
        for line, module in imported_modules(p)
        if module.split(".")[0] == "private_generator"
    ]
    assert offenders == []


@pytest.mark.parametrize(
    ("relative", "source", "rule"),
    [
        ("core/bad.py", "from onc_agi.services import scoring\n", "core imports an outer layer"),
        ("core/bad.py", "import pandas\n", "core imports a non-whitelisted dependency"),
        ("services/bad.py", "from onc_agi.infra.bundles import read_world\n", "services import infra"),
        ("services/bad.py", "import onc_agi.adapters.cli\n", "services import infra or adapters"),
        ("infra/bad.py", "from onc_agi.services.engine import Episode\n", "infra imports services"),
        (
            "adapters/bad.py",
            "from private_generator.services import generator\n",  # scan: allow generator_import
            "imports the private generator",
        ),
    ],
)
def test_checker_rejects_a_deliberately_forbidden_import(
    tmp_path: Path, relative: str, source: str, rule: str
) -> None:
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_text(source)
    found = violations(tmp_path)
    assert len(found) == 1 and rule in found[0]


@pytest.mark.parametrize(
    ("relative", "source"),
    [
        (
            "core/ok.py",
            "import json\nimport numpy as np\nfrom pydantic import BaseModel\nfrom onc_agi.core.schema import Tier\n",
        ),
        (
            "services/ok.py",
            "from onc_agi.core.ports import WorldStore\nimport statsmodels.api as sm\n",
        ),
        (
            "adapters/ok.py",
            "from onc_agi.infra.bundles import read_world\nfrom onc_agi.services import kit\n",
        ),
        ("core/rel.py", "from .schema import Tier\n"),
    ],
)
def test_checker_accepts_permitted_imports(tmp_path: Path, relative: str, source: str) -> None:
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_text(source)
    assert violations(tmp_path) == []
