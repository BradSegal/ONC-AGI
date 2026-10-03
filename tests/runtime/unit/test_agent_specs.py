"""``onc-agi play --agent`` specs: built-in names, module:Class and path.py:Class, failing fast."""

from __future__ import annotations

from pathlib import Path

import pytest
from arena_factories import InMemoryStore
from onc_agi.adapters.agents.specs import AgentSpecError, resolve_agent
from onc_agi.services.kit import Agent, PipelineAgent

AGENT_SOURCE = """
from onc_agi.core.schema import Submit
from onc_agi.services.kit import Agent


class Abstainer(Agent):
    name = "abstainer"

    def choose_action(self, card, view):
        return Submit(request_id=self.request_id(), ranking=())


class Nameless(Abstainer):
    name = "agent"


class NeedsArgs(Abstainer):
    def __init__(self, threshold):
        super().__init__()


class NotAnAgent:
    pass
"""


@pytest.fixture
def agent_file(tmp_path: Path) -> Path:
    path = tmp_path / "my_agents.py"
    path.write_text(AGENT_SOURCE)
    return path


def test_builtin_names_resolve_to_fresh_instances(store: InMemoryStore) -> None:
    factory, name = resolve_agent("univariate_bh", store)
    assert name == "univariate_bh" and isinstance(factory(), PipelineAgent)
    assert factory() is not factory()
    factory, name = resolve_agent("seq_lasso", None)
    assert name == "seq_lasso"
    _, name = resolve_agent("oracle", store)
    assert name == "oracle"


def test_the_oracle_is_refused_over_http() -> None:
    with pytest.raises(AgentSpecError, match="only in-process"):
        resolve_agent("oracle", None)


def test_a_path_spec_loads_an_agent_subclass(agent_file: Path) -> None:
    factory, name = resolve_agent(f"{agent_file}:Abstainer")
    assert name == "abstainer" and isinstance(factory(), Agent)
    _, name = resolve_agent(f"{agent_file}:Nameless")
    assert name == "Nameless"  # the default agent name falls back to the class name


def test_a_module_spec_loads_an_agent_subclass(agent_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(agent_file.parent))
    factory, name = resolve_agent("my_agents:Abstainer")
    assert name == "abstainer" and type(factory()).__name__ == "Abstainer"


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("nobody", "unknown agent 'nobody'"),
        ("no_such_module_xyz:Agent", "cannot import no_such_module_xyz"),
        ("{file}:Missing", "Missing is not an Agent subclass"),
        ("{file}:NotAnAgent", "NotAnAgent is not an Agent subclass"),
        ("{file}:NeedsArgs", "no-argument constructor"),
        ("{dir}/absent.py:Abstainer", "no file"),
        ("onc_agi.services.kit:Agent", "abstract"),
        (":Abstainer", "expected module:Class"),
    ],
)
def test_bad_specs_fail_fast_with_a_clear_message(agent_file: Path, spec: str, message: str) -> None:
    with pytest.raises(AgentSpecError, match=message):
        resolve_agent(spec.format(file=agent_file, dir=agent_file.parent))
