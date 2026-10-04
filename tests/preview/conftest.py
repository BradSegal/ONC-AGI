"""ONC-AGI release checks: fixtures, published contract artefacts, docs, examples and export hygiene."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / "tools", ROOT / "examples" / "agents", ROOT / "tests" / "preview" / "support"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        if "preview" in Path(str(item.fspath)).parts:
            item.add_marker(pytest.mark.preview)
