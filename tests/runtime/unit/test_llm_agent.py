"""The LLM agent template against a fake OpenAI-compatible server and a fake sandbox (no network, no Docker)."""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from arena_factories import InMemoryStore, fid, planted_world
from onc_agi.adapters.agents.llm import (
    TOOL_SPECS,
    ChatClient,
    HarnessConfig,
    LLMToolAgent,
    harness_label,
)
from onc_agi.adapters.profiles import Profile
from onc_agi.adapters.swarm import LocalArena, SwarmResult, run_swarm
from onc_agi.core.schema import Mode, Tier
from onc_agi.infra.recordings import Recording, RecordingWriter, read_recording

pytest.importorskip("inspect_ai")  # the agent shows the standard harness's card ([inspect] extra)

_ids = itertools.count()
Reply = dict[str, Any] | httpx.Response


def call(name: str, **arguments: Any) -> dict[str, Any]:
    return {
        "id": f"call-{next(_ids)}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def reply(
    *calls: dict[str, Any],
    content: str | None = None,
    reasoning: str | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content, "refusal": None}
    if calls:
        message["tool_calls"] = list(calls)
    if reasoning is not None:
        message["reasoning"] = reasoning
    return {
        "choices": [{"message": message, "finish_reason": "tool_calls"}],
        "usage": usage or {"prompt_tokens": 100, "completion_tokens": 20},
    }


class FakeModel:
    """Scripted chat-completions endpoint; keeps every request body."""

    def __init__(self, *replies: Reply) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        self.headers: list[httpx.Headers] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        self.requests.append(json.loads(request.content))
        self.headers.append(request.headers)
        if not self.replies:
            raise AssertionError("the agent asked the model more often than scripted")
        nxt = self.replies.pop(0)
        return nxt if isinstance(nxt, httpx.Response) else httpx.Response(200, json=nxt)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


class FakeSandbox:
    def __init__(self, output: str = "top: f00 z=6.1, f03 z=1.2") -> None:
        self.output = output
        self.csv: list[str] = []
        self.runs: list[tuple[list[str], str | None]] = []
        self.closed = False

    def write_data(self, csv: str) -> None:
        self.csv.append(csv)

    def run(self, argv: list[str], stdin: str | None = None) -> tuple[str, bool]:
        self.runs.append((argv, stdin))
        return self.output, False

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def profile(monkeypatch: pytest.MonkeyPatch) -> Profile:
    monkeypatch.setenv("FAKE_LLM_KEY", "sk-test")
    return Profile(
        name="fake",
        base_url="http://llm.test/v1",
        model="org/model-1",
        api_key_env="FAKE_LLM_KEY",
        generate={"max_tokens": 500, "temperature": 0.2},
        extra_body={"usage": {"include": True}},
        prices_per_mtok={"input": 1.0, "output": 2.0, "cache_read": 0.5},
    )


def _play(
    profile: Profile,
    model: FakeModel,
    tmp_path: Path,
    *,
    mode: Mode = Mode.FULL_ACCESS,
    config: HarnessConfig | None = None,
    signal: bool = True,
) -> tuple[SwarmResult, list[FakeSandbox], Recording]:
    store = InMemoryStore.of(planted_world("w-00", signal=signal, seed=3, n_pool=40, mode=mode))
    boxes: list[FakeSandbox] = []

    def sandbox(_: HarnessConfig) -> FakeSandbox:
        boxes.append(FakeSandbox())
        return boxes[-1]

    def factory() -> LLMToolAgent:
        return LLMToolAgent(
            profile, config, http=model.client(), sandbox_factory=sandbox, sleep=lambda _: None
        )

    with RecordingWriter(tmp_path / "rec.jsonl") as recorder:
        result = run_swarm(
            LocalArena.over_store(store),
            factory,
            agent_name="llm-fake",
            tier=Tier.PUBLIC_TRAIN,
            n_worlds=1,
            recorder=recorder,
            model=profile.model,
            harness="h",
        )
    return result, boxes, read_recording(tmp_path / "rec.jsonl")


def _tool_messages(request: dict[str, Any]) -> list[dict[str, Any]]:
    return [m for m in request["messages"] if m["role"] == "tool"]


def test_a_python_result_reaches_the_model_and_submit_keeps_only_valid_ids(
    profile: Profile, tmp_path: Path
) -> None:
    py = call("python", code="import pandas as pd; print(pd.read_csv('/data/revealed.csv').shape)")
    model = FakeModel(
        reply(py, content="Let me look.", reasoning="Screen marginals first."),
        reply(call("submit", answer=f"{fid(0)}, NOT_A_FEATURE,\n{fid(0)}, {fid(3)}")),
    )
    result, (box,), recording = _play(profile, model, tmp_path)
    (run,) = result.runs
    assert run.submitted and run.ranking == (fid(0), fid(3)) and run.error is None
    first, second = model.requests
    assert [m["role"] for m in first["messages"]] == ["system", "user"]
    assert "World w-00" in first["messages"][1]["content"]  # the standard harness's task card
    assert [t["function"]["name"] for t in first["tools"]] == ["python", "bash", "submit"]
    (tool,) = _tool_messages(second)
    assert tool["tool_call_id"] == py["id"] and tool["content"] == box.output
    assert second["messages"][2]["tool_calls"][0]["id"] == py["id"]  # the reply is kept verbatim...
    assert "refusal" not in second["messages"][2]  # ...without null fields
    assert box.runs == [(["python", "-"], json.loads(py["function"]["arguments"])["code"])]
    assert box.csv and box.csv[0].startswith("patient_id,stratum,outcome")
    assert box.closed  # close() released the sandbox
    kinds = [e.kind for e in recording.events]
    assert kinds.count("reasoning") == 1 and kinds.count("assistant") == 1
    assert [e.tool for e in recording.events if e.kind == "tool_call"] == ["python", "submit"]
    (reasoning,) = [e for e in recording.events if e.kind == "reasoning"]
    assert reasoning.content == "Screen marginals first."


def test_the_request_carries_the_profile_and_strict_tool_schemas(profile: Profile, tmp_path: Path) -> None:
    model = FakeModel(reply(call("submit", answer="")))
    _play(profile, model, tmp_path)
    (body,) = model.requests
    assert body["model"] == "org/model-1" and body["max_tokens"] == 500 and body["temperature"] == 0.2
    assert body["usage"] == {"include": True}  # extra_body is merged into the body
    assert model.headers[0]["authorization"] == "Bearer sk-test"
    for spec in TOOL_SPECS.values():
        params = spec["function"]["parameters"]
        assert sorted(params["required"]) == sorted(params["properties"])
        assert params["additionalProperties"] is False
    assert "stratum" in TOOL_SPECS["recruit"]["function"]["parameters"]["required"]


def test_a_refused_assay_goes_back_to_the_model_as_a_tool_error(profile: Profile, tmp_path: Path) -> None:
    recruit = call("recruit", count=10, stratum="all")
    bad = call("assay", feature_ids="NOPE_1, NOPE_2")
    good = call("assay", feature_ids=f"{fid(0)},{fid(1)}")
    model = FakeModel(
        reply(recruit),
        reply(bad),
        reply(good, call("python", code="print(1)")),
        reply(call("submit", answer=fid(0))),
    )
    result, (box,), recording = _play(profile, model, tmp_path, mode=Mode.SEQUENTIAL)
    (run,) = result.runs
    assert run.submitted and run.error is None and run.ranking == (fid(0),)
    assert run.spent == pytest.approx(10 * 1.0 + 10 * 0.5 * 2)
    names = [t["function"]["name"] for t in model.requests[0]["tools"]]
    assert names == ["python", "bash", "recruit", "assay", "submit"]
    recruited = _tool_messages(model.requests[1])[-1]
    assert recruited["content"].startswith("Recruited; 10 patients revealed")
    refused = _tool_messages(model.requests[2])[-1]
    assert refused["tool_call_id"] == bad["id"] and refused["content"].startswith("unknown_feature:")
    measured, ran = _tool_messages(model.requests[3])[-2:]
    assert measured["content"].startswith("Measured 2 features") and ran["content"] == box.output
    assert box.csv[-1].splitlines()[0] == f"patient_id,stratum,outcome,{fid(0)},{fid(1)}"  # refreshed
    errors = [e for e in recording.events if e.kind == "tool_result" and e.error]
    assert len(errors) == 1 and errors[0].tool == "assay"
    assert sum(1 for e in recording.events if e.kind == "action" and e.error) == 1


def test_bad_arguments_and_unknown_tools_come_back_as_errors(profile: Profile, tmp_path: Path) -> None:
    broken = {"id": "call-x", "type": "function", "function": {"name": "python", "arguments": "{not json"}}
    model = FakeModel(
        reply(broken, call("recruit", count=5, stratum="all")),  # recruit is not offered in full access
        reply(call("submit", answer="")),
    )
    result, _, _ = _play(profile, model, tmp_path)
    errors = [m["content"] for m in _tool_messages(model.requests[1])]
    assert errors[0].startswith("Arguments were not a valid JSON object")
    assert errors[1].startswith("Unknown tool 'recruit'")
    assert result.runs[0].submitted and result.runs[0].ranking == ()


def test_exhausting_max_turns_ends_the_world_unsubmitted(profile: Profile, tmp_path: Path) -> None:
    model = FakeModel(reply(content="Thinking..."), reply(content="Still thinking..."))
    result, (box,), recording = _play(profile, model, tmp_path, config=HarnessConfig(max_turns=2))
    (run,) = result.runs
    assert not run.submitted and run.error is None and run.ranking == ()
    assert result.scorecard.worlds[0].listed == 0  # scored as an empty submission
    assert model.requests[1]["messages"][-1] == {"role": "user", "content": HarnessConfig().continue_prompt}
    assert box.closed
    (note,) = [e for e in recording.events if e.kind == "note"]
    assert "max_turns=2" in note.content


def test_usage_and_provider_cost_are_captured(profile: Profile, tmp_path: Path) -> None:
    model = FakeModel(
        reply(
            call("python", code="1"), usage={"prompt_tokens": 1000, "completion_tokens": 50, "cost": 0.004}
        ),
        reply(
            call("submit", answer=""), usage={"prompt_tokens": 1200, "completion_tokens": 30, "cost": 0.001}
        ),
    )
    result, _, _ = _play(profile, model, tmp_path)
    (run,) = result.runs
    assert run.tokens == 2280 and run.cost_usd == pytest.approx(0.005) and run.model == "org/model-1"
    card = result.scorecard
    assert card.tokens == 2280 and card.cost_usd == pytest.approx(0.005) and card.model == "org/model-1"


def test_without_a_reported_cost_the_profile_prices_apply(profile: Profile, tmp_path: Path) -> None:
    usage = {"prompt_tokens": 1000, "completion_tokens": 100, "prompt_tokens_details": {"cached_tokens": 400}}
    model = FakeModel(reply(call("submit", answer=""), usage=usage))
    result, _, _ = _play(profile, model, tmp_path)
    expected = (600 * 1.0 + 400 * 0.5 + 100 * 2.0) / 1e6
    assert result.runs[0].cost_usd == pytest.approx(expected)


def test_an_unpriced_reply_makes_the_cost_unknown(profile: Profile, tmp_path: Path) -> None:
    unpriced = Profile(name="free", base_url="http://llm.test/v1", model="m")  # no key, no prices
    model = FakeModel(reply(call("submit", answer="")))
    result, _, _ = _play(unpriced, model, tmp_path)
    assert result.runs[0].cost_usd is None and result.runs[0].tokens == 120
    assert "authorization" not in model.headers[0]


def _client(model: FakeModel, profile: Profile, slept: list[float]) -> ChatClient:
    return ChatClient(profile, http=model.client(), retries=3, backoff=1.0, sleep=slept.append)


def test_transient_failures_are_retried_with_backoff(profile: Profile) -> None:
    ok = reply(content="done")
    model = FakeModel(httpx.Response(429), httpx.Response(503, text="busy"), {"error": {"code": 502}}, ok)
    slept: list[float] = []
    completion = _client(model, profile, slept).complete([{"role": "user", "content": "hi"}], [])
    assert completion.message["content"] == "done" and slept == [1.0, 2.0, 4.0]


def test_connection_errors_are_retried_then_reported(profile: Profile) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    slept: list[float] = []
    client = ChatClient(
        profile, http=httpx.Client(transport=httpx.MockTransport(refuse)), retries=2, sleep=slept.append
    )
    with pytest.raises(RuntimeError, match="gave up after 3 attempts"):
        client.complete([], [])
    assert len(slept) == 2


def test_client_errors_are_not_retried(profile: Profile) -> None:
    model = FakeModel(httpx.Response(400, text="bad tools"))
    slept: list[float] = []
    with pytest.raises(RuntimeError, match="HTTP 400: bad tools"):
        _client(model, profile, slept).complete([], [])
    assert slept == []


def test_a_model_error_ends_only_its_world(profile: Profile, tmp_path: Path) -> None:
    model = FakeModel(httpx.Response(401, text="no key"))
    result, (box,), _ = _play(profile, model, tmp_path)
    assert "HTTP 401" in (result.runs[0].error or "") and box.closed


def test_the_harness_config_loads_validates_and_labels(tmp_path: Path) -> None:
    path = tmp_path / "h.toml"
    path.write_text('[harness]\nmax_turns = 7\ntools = ["python"]\nsystem_prompt = "Be brief."\n')
    config = HarnessConfig.load(path)
    assert config.max_turns == 7 and config.tools == ("python",) and config.system_prompt == "Be brief."
    assert HarnessConfig.load(None) == HarnessConfig()
    label = harness_label(config)
    assert re.fullmatch(r"openai-tools-1\.0\+[0-9a-f]{8}", label)
    assert label != harness_label(HarnessConfig()) and harness_label(HarnessConfig()) == harness_label(
        HarnessConfig()
    )
    path.write_text("[harness]\nworkers = 4\n")
    with pytest.raises(ValueError, match="unknown \\[harness\\] keys \\['workers'\\]"):
        HarnessConfig.load(path)
    with pytest.raises(ValueError, match="tools may only name"):
        HarnessConfig(tools=("python", "submit"))
    with pytest.raises(ValueError, match="positive"):
        HarnessConfig(max_turns=0)


def test_one_agent_plays_one_world(profile: Profile) -> None:
    from onc_agi.services.engine import Episode

    model = FakeModel(reply(call("python", code="1")), reply(call("python", code="2")))
    agent = LLMToolAgent(profile, http=model.client(), sandbox_factory=lambda _: FakeSandbox())
    a, _ = planted_world("w-00", signal=True, seed=1)
    b, _ = planted_world("w-01", signal=True, seed=2)
    agent_choose: Callable[..., Any] = agent.choose_action
    with pytest.raises(AssertionError):  # the scripted model runs out: no arena action yet
        agent_choose(a.card, Episode(a).view())
    with pytest.raises(RuntimeError, match="one world"):
        agent_choose(b.card, Episode(b).view())
    agent.close()


def test_a_missing_key_fails_when_the_agent_is_built(monkeypatch: pytest.MonkeyPatch) -> None:
    from onc_agi.adapters.profiles import ProfileError

    monkeypatch.delenv("ABSENT_LLM_KEY", raising=False)
    with pytest.raises(ProfileError, match="ABSENT_LLM_KEY"):
        LLMToolAgent(Profile(name="p", base_url="http://x", model="m", api_key_env="ABSENT_LLM_KEY"))
