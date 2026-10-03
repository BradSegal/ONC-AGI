"""Agent specs for ``onc-agi play --agent``: a registry name, ``module:Class`` or ``path.py:Class``.

A class spec names an :class:`Agent` subclass with a no-argument constructor; the
factory builds a fresh instance per world. Every spec is resolved, and one instance
built and closed, before any scorecard is opened, so a bad spec fails fast.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path

from onc_agi.adapters.agents import AGENT_NAMES, BASELINES, AgentFactory, make_agent
from onc_agi.core.ports import WorldStore
from onc_agi.services.kit import Agent


class AgentSpecError(ValueError):
    """The spec does not name a playable agent."""


def builtin_names() -> tuple[str, ...]:
    return (*AGENT_NAMES, *(f"seq_{b}" for b in BASELINES))


def _load_class(spec: str) -> type[Agent]:
    target, _, attr = spec.rpartition(":")
    if not target or not attr:
        raise AgentSpecError(f"{spec!r}: expected module:Class or path.py:Class")
    if target.endswith(".py"):
        path = Path(target)
        if not path.is_file():
            raise AgentSpecError(f"{spec!r}: no file {path}")
        name = f"arena_agent_{path.stem}"
        module_spec = importlib.util.spec_from_file_location(name, path)
        if module_spec is None or module_spec.loader is None:
            raise AgentSpecError(f"{spec!r}: cannot load {path}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[name] = module  # dataclasses and pickling look modules up by name
        module_spec.loader.exec_module(module)
    else:
        try:
            module = importlib.import_module(target)
        except ImportError as exc:
            raise AgentSpecError(f"{spec!r}: cannot import {target} ({exc})") from exc
    cls = getattr(module, attr, None)
    if not (inspect.isclass(cls) and issubclass(cls, Agent)):
        raise AgentSpecError(f"{spec!r}: {attr} is not an Agent subclass in {target}")
    if inspect.isabstract(cls):
        raise AgentSpecError(f"{spec!r}: {attr} is abstract (implement choose_action)")
    return cls


def resolve_agent(spec: str, store: WorldStore | None = None) -> tuple[AgentFactory, str]:
    """A per-world agent factory and the agent's name for the scorecard.

    ``store`` is the local world store; ``None`` means play over HTTP, where the oracle
    (which reads answer keys) is refused.
    """
    if spec == "oracle" and store is None:
        raise AgentSpecError(
            "the oracle reads answer keys, so it plays only in-process (--store), not over HTTP"
        )
    if spec in builtin_names():
        factory: AgentFactory = lambda: make_agent(spec, store)  # noqa: E731
    elif ":" in spec:
        cls = _load_class(spec)
        factory = cls
    else:
        raise AgentSpecError(
            f"unknown agent {spec!r}: use a built-in name ({', '.join(builtin_names())}), llm,"
            " module:Class or path.py:Class"
        )
    try:
        probe = factory()
    except TypeError as exc:
        raise AgentSpecError(f"{spec!r}: the agent needs a no-argument constructor ({exc})") from exc
    try:
        name = probe.name if probe.name != Agent.name else type(probe).__name__
    finally:
        probe.close()
    return factory, name
