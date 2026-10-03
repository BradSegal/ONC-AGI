"""Shared test configuration for the public ``onc_agi`` package.

Every test is marked by the pyramid layer of the directory it lives in
(``unit``, ``integration``, ``statistical``, ``adversarial``, ``e2e``), so a
layer can be selected with ``-m``. These tests never import the private
``private_generator`` package: the public package must be testable on its own.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from arena_factories import InMemoryStore, standard_world

_LAYERS = ("unit", "integration", "statistical", "adversarial", "e2e")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        parts = Path(str(item.fspath)).parts
        for layer in _LAYERS:
            if layer in parts:
                item.add_marker(getattr(pytest.mark, layer))
                break


@pytest.fixture
def store() -> InMemoryStore:
    """Six small public-train worlds (four with signal, two null) with known answers."""
    return InMemoryStore.of(*(standard_world(i) for i in range(6)))
