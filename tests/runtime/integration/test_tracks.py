"""Both tracks, both modes, in-process and over HTTP: one run format, verified by server replay.

The open track is the kit's LLM agent (``onc-agi play --agent llm``) through the swarm runner; the
standard track is the Inspect task (``onc-agi standard``). Each run writes the same run directory
(recording, server trace, scorecard), and :func:`verify_run` replays the server trace against the
world bundles and checks the scorecard and the recording against it.

No paid model and no Docker: one scripted policy plays both tracks, behind a fake
OpenAI-compatible endpoint (open track) and Inspect's ``mockllm`` (standard track), and the
analysis sandbox is an in-memory stand-in.
"""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

pytest.importorskip("inspect_ai")

from inspect_ai.model import ChatMessage, GenerateConfig, ModelOutput, get_model
from inspect_ai.tool import ToolChoice, ToolInfo
from inspect_ai.util import ExecResult, SandboxEnvironment, sandboxenv
from onc_agi.adapters import cli, inspect_task
from onc_agi.adapters.agents.llm import HarnessConfig, LLMToolAgent, harness_label
from onc_agi.adapters.client import ArenaClient
from onc_agi.adapters.http import serve_in_thread
from onc_agi.adapters.profiles import Profile
from onc_agi.adapters.swarm import LocalArena, run_swarm, server_copies, server_trace
from onc_agi.core.schema import GroupLabel, Mode, Scorecard, Tier, TraceEvent
from onc_agi.infra.archive import FileScorecardArchive
from onc_agi.infra.bundles import FileWorldStore
from onc_agi.infra.ledger import JsonLedger
from onc_agi.infra.recordings import RECORDING_FILE, RecordingWriter, read_run, save_run, write_trace
from onc_agi.services.explain import explain_run
from onc_agi.services.replay import RunVerification, verify_run
from onc_agi.services.scorecards import ScorecardService

STORE = FileWorldStore(cli.fixture_store())
KEY = "track-key-0001"
N_WORLDS = 2

# ------------------------------------------------------------------------------------- the policy


def _answer(world_id: str) -> str:
    key = STORE.answer_key(world_id)
    return ",".join(
        dict.fromkeys(
            p.true_feature for g in key.groups if g.label is GroupLabel.RECOVERABLE for p in g.parts
        )
    )


def policy(world_id: str, step: int) -> tuple[str, dict[str, Any]]:
    """The scripted model: what it calls at its ``step``-th turn on a world (both tracks)."""
    card = STORE.card(world_id)
    if card.mode is Mode.FULL_ACCESS:
        script = [("python", {"code": "print('screen')"}), ("submit", {"answer": _answer(world_id)})]
    else:
        stratum = card.strata[0]
        size = (card.stratum_sizes or {}).get(stratum, card.n_pool // len(card.strata))
        script = [
            ("recruit", {"count": min(12, size), "stratum": stratum}),
            ("assay", {"feature_ids": "ghost-feature"}),  # refused: comes back as a tool error
            ("assay", {"feature_ids": ",".join(card.feature_ids()[:3])}),
            ("python", {"code": "print('screen')"}),
            ("submit", {"answer": _answer(world_id)}),
        ]
    return script[min(step, len(script) - 1)]


def _world(text: str) -> str:
    match = re.search(r"World (\S+): ", text)
    assert match, "the task card names its world"
    return match.group(1)


_calls = itertools.count()


class OpenAIEndpoint:
    """A fake OpenAI-compatible endpoint playing :func:`policy` (open track)."""

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        messages = body["messages"]
        world_id = _world(next(m["content"] for m in messages if m["role"] == "user"))
        step = sum(m["role"] == "assistant" for m in messages)
        name, args = policy(world_id, step)
        call = {
            "id": f"call-{next(_calls)}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }
        message = {
            "role": "assistant",
            "content": f"step {step}",
            "reasoning": f"plan for {world_id}",
            "tool_calls": [call],
        }
        return httpx.Response(
            200,
            json={
                "choices": [{"message": message}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 5, "cost": 0.001},
            },
        )


def inspect_policy(
    messages: list[ChatMessage], tools: list[ToolInfo], tool_choice: ToolChoice, config: GenerateConfig
) -> ModelOutput:
    """The same policy as Inspect ``mockllm`` outputs (standard track)."""
    world_id = _world(next(m.text for m in messages if m.role == "user"))
    step = sum(m.role == "assistant" for m in messages)
    name, args = policy(world_id, step)
    return ModelOutput.for_tool_call("mockllm/model", name, args, content=f"step {step}")


@sandboxenv(name="arena-test")
class MemorySandbox(SandboxEnvironment):
    """In-memory stand-in for the analysis sandbox: files are kept, commands print a screen."""

    def __init__(self) -> None:
        super().__init__()
        self.files: dict[str, str | bytes] = {}

    async def exec(
        self, cmd: list[str], input: str | bytes | None = None, *args: Any, **kwargs: Any
    ) -> ExecResult[str]:
        return ExecResult(success=True, returncode=0, stdout="top: f00 z=6.1", stderr="")

    async def write_file(self, file: str, contents: str | bytes) -> None:
        self.files[file] = contents

    async def read_file(self, file: str, text: bool = True) -> Any:
        return self.files[file]

    @classmethod
    async def sample_init(
        cls, task_name: str, config: Any, metadata: dict[str, str]
    ) -> dict[str, SandboxEnvironment]:
        return {"default": cls()}

    @classmethod
    async def sample_cleanup(
        cls, task_name: str, config: Any, environments: dict[str, SandboxEnvironment], interrupted: bool
    ) -> None:
        return None


class FakeAnalysisBox:
    def write_data(self, csv: str) -> None:
        self.csv = csv

    def run(self, argv: list[str], stdin: str | None = None) -> tuple[str, bool]:
        return "top: f00 z=6.1", False

    def close(self) -> None:
        pass


# ------------------------------------------------------------------------------------- the arenas


def _service(root: Path) -> ScorecardService:
    return ScorecardService(
        STORE, JsonLedger(root / "ledger.json"), archive=FileScorecardArchive(root / "archive")
    )


@pytest.fixture
def server(tmp_path: Path) -> Iterator[str]:
    """An arena server over real HTTP (with an archive, so it keeps traces)."""
    with serve_in_thread(_service(tmp_path / "server")) as url:
        yield url


@pytest.fixture(autouse=True)
def memory_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inspect_task, "SANDBOX", "arena-test")


@pytest.fixture
def profile(monkeypatch: pytest.MonkeyPatch) -> Profile:
    monkeypatch.setenv("TRACK_LLM_KEY", "sk-test")
    return Profile(
        name="script", base_url="http://llm.test/v1", model="script/policy", api_key_env="TRACK_LLM_KEY"
    )


def _worlds(mode: Mode) -> tuple[str, ...]:
    return tuple(w for w in STORE.world_ids(Tier.PUBLIC_TRAIN) if STORE.card(w).mode is mode)[:N_WORLDS]


Copies = tuple[list[TraceEvent], Scorecard]


def play_open(record: Path, mode: Mode, profile: Profile, url: str | None, tmp_path: Path) -> Copies:
    """The open track: the kit's LLM agent through the swarm runner, recording a run directory;
    returns the server's own trace and scorecard, fetched again after the run."""
    arena: LocalArena | ArenaClient = (
        ArenaClient(url, KEY) if url else LocalArena(_service(tmp_path / "local"))
    )
    endpoint = OpenAIEndpoint()
    config = HarnessConfig(max_turns=8)

    def factory() -> LLMToolAgent:
        return LLMToolAgent(
            profile,
            config,
            http=httpx.Client(transport=httpx.MockTransport(endpoint)),
            sandbox_factory=lambda _: FakeAnalysisBox(),
        )

    with RecordingWriter(record / RECORDING_FILE) as writer:
        result = run_swarm(
            arena,
            factory,
            agent_name="llm-script",
            tier=Tier.PUBLIC_TRAIN,
            world_ids=_worlds(mode),
            workers=2,
            recorder=writer,
            model=profile.model,
            harness=harness_label(config),
        )
    assert not result.errors
    save_run(record, result.scorecard, server_trace(arena, result.scorecard.scorecard_id))
    copies = server_copies(arena, result.scorecard.scorecard_id)
    assert copies is not None
    return copies


def play_standard(
    record: Path, mode: Mode, url: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Copies | None:
    """The standard track: the Inspect task, closed and recorded by :func:`run_standard`; over
    HTTP, returns the server's own copies (an in-process arena ends with the run)."""
    where: dict[str, Any] = {"url": url} if url else {"store_root": str(cli.fixture_store())}
    if url:
        monkeypatch.setenv("ARENA_KEY", KEY)
    inspect_task.run_standard(
        mode=mode,
        model=get_model("mockllm/model", custom_outputs=inspect_policy),
        record=record,
        log_dir=tmp_path / "logs",
        world_ids=",".join(_worlds(mode)),
        message_limit=14,
        **where,
    )
    if url is None:
        return None
    sid = read_run(record).recording.header.scorecard_id  # type: ignore[union-attr]
    return server_copies(ArenaClient(url, KEY), sid)


# ------------------------------------------------------------------------------------- AC-01


@pytest.mark.parametrize("transport", ["local", "http"])
@pytest.mark.parametrize("mode", [Mode.FULL_ACCESS, Mode.SEQUENTIAL], ids=lambda m: m.value)
@pytest.mark.parametrize("track", ["open", "standard"])
def test_every_track_plays_every_mode_locally_and_over_http_and_its_trace_verifies_by_replay(
    track: str,
    mode: Mode,
    transport: str,
    request: pytest.FixtureRequest,
    profile: Profile,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = request.getfixturevalue("server") if transport == "http" else None
    record = tmp_path / "run"
    copies = (
        play_open(record, mode, profile, url, tmp_path)
        if track == "open"
        else play_standard(record, mode, url, tmp_path, monkeypatch)
    )
    assert (copies is None) == (track == "standard" and transport == "local")

    run = read_run(record)
    assert run.scorecard is not None and run.trace and run.recording.header is not None
    header, scorecard = run.recording.header, run.scorecard
    assert header.track == ("open" if track == "open" else "standard") == scorecard.track
    assert list(header.world_ids) == [w.world_id for w in scorecard.worlds] == list(_worlds(mode))
    verification = verify_run(
        run.trace,
        STORE,
        scorecard=scorecard,
        header=header,
        runs=run.recording.runs,
        events=run.recording.events,
        server_trace=copies[0] if copies else None,
        server_scorecard=copies[1] if copies else None,
    )
    assert verification.ok, verification.mismatches
    assert verification.authority == ("server" if copies else "local")
    assert ("verified against the server" if copies else "consistent") in verification.summary()
    assert len(verification.reports) == N_WORLDS and all(
        r.submission is not None for r in verification.reports
    )
    # one format: the same event kinds and an action trail that is the server trace's own
    kinds = {e.kind for e in run.recording.events}
    assert {"assistant", "tool_call", "tool_result", "action"} <= kinds
    if track == "open":
        assert "reasoning" in kinds  # captured from the provider's reasoning field
    if mode is Mode.SEQUENTIAL:  # the refused assay is recorded as an error, and absent from the trace
        assert any(
            e.kind == "action" and e.error and "ghost-feature" in e.content for e in run.recording.events
        )
        assert not any("ghost-feature" in (t.action_json or "") for t in run.trace)
    # the run explains itself, and its explanations agree with the scorer
    explanations = explain_run(run.recording.events, run.recording.runs, STORE)
    for x, scored in zip(explanations, scorecard.worlds, strict=True):
        assert x.find == (None if scored.is_null else pytest.approx(scored.find))
        assert x.outcome in ("found", "partial", "correct_abstain")


# ------------------------------------------------------------------------------------- tampering


@pytest.fixture
def recorded(profile: Profile, tmp_path: Path) -> tuple[Path, Copies]:
    record = tmp_path / "run"
    return record, play_open(record, Mode.SEQUENTIAL, profile, None, tmp_path)


def _verify(record: Path, copies: Copies | None = None, **change: Any) -> RunVerification:
    run = read_run(record)
    assert run.trace is not None
    files: dict[str, Any] = {
        "scorecard": run.scorecard,
        "header": run.recording.header,
        "runs": run.recording.runs,
        "events": run.recording.events,
    }
    files |= change
    trace = files.pop("trace", run.trace)
    server = {"server_trace": copies[0], "server_scorecard": copies[1]} if copies else {}
    return verify_run(trace, STORE, **files, **server)


def _rewrite_trace(record: Path, change: Any) -> None:
    trace = read_run(record).trace
    assert trace is not None
    (record / "trace.jsonl").unlink()
    write_trace(record / "trace.jsonl", change(list(trace)))


def _joined(result: RunVerification) -> str:
    assert not result.ok
    return " | ".join(result.mismatches)


def test_an_untouched_run_verifies_against_the_server_and_is_otherwise_only_consistent(
    recorded: tuple[Path, Copies],
) -> None:
    record, copies = recorded
    assert _verify(record, copies).authority == "server" and _verify(record, copies).ok
    local = _verify(record)
    assert local.ok and local.authority == "local" and "not checked against the server" in local.summary()


def test_a_tampered_response_digest_fails_replay(recorded: tuple[Path, Copies]) -> None:
    record, _ = recorded

    def change(trace: list[TraceEvent]) -> list[TraceEvent]:
        trace[1] = trace[1].model_copy(update={"response_sha256": "0" * 64})
        return trace

    _rewrite_trace(record, change)
    assert "response digest differs" in _joined(_verify(record))


def test_a_dropped_action_disagrees_with_the_recording_and_the_scorecard(
    recorded: tuple[Path, Copies],
) -> None:
    record, _ = recorded
    _rewrite_trace(record, lambda trace: [e for e in trace if e.action_kind != "submit"])
    joined = _joined(_verify(record))
    assert "recorded submitted=True, the trace disagrees" in joined
    assert "the recorded actions are not the traced ones" in joined
    assert "the scorecard is not the replayed one" in joined


@pytest.mark.parametrize(
    "field",
    ["discovery_score", "find", "restraint", "n_worlds", "per_tier", "leak_rate", "interval", "versions"],
)
def test_any_doctored_scorecard_aggregate_fails_even_without_per_world_scores(
    recorded: tuple[Path, Copies], field: str
) -> None:
    """The whole scorecard is recomputed from the replay, so aggregates are checked on every tier."""
    record, _ = recorded
    card = read_run(record).scorecard
    assert card is not None
    doctored = {
        "discovery_score": (card.discovery_score or 0.0) + 0.1,
        "find": (card.find or 0.0) + 0.1,
        "restraint": (card.restraint or 0.0) - 0.1,
        "n_worlds": card.n_worlds,  # unchanged count, but the per-world list removed below
        "per_tier": card.per_tier[:-1],
        "leak_rate": 1.0,
        "interval": card.interval.model_copy(update={"high": 0.99}),
        "versions": {**card.versions, "scorer": "scorer-0.9+00000000"},
    }[field]
    update: dict[str, Any] = {field: doctored} | ({"worlds": ()} if field == "n_worlds" else {})
    if field == "n_worlds":
        update["discovery_score"] = 0.99  # as an eval-tier scorecard: aggregates only
    joined = _joined(_verify(record, scorecard=card.model_copy(update=update)))
    assert "the scorecard is not the replayed one" in joined


def test_a_world_dropped_from_the_recording_and_trace_fails(recorded: tuple[Path, Copies]) -> None:
    record, _ = recorded
    run = read_run(record)
    assert run.recording.header is not None and run.trace is not None
    gone = run.recording.header.world_ids[0]
    runs = [r for r in run.recording.runs if r.world_id != gone]
    trace = [e for e in run.trace if e.world_id != gone]
    assert "missing ['" + gone in _joined(_verify(record, runs=runs, trace=trace))
    header = run.recording.header.model_copy(update={"world_ids": run.recording.header.world_ids[1:]})
    assert "the scorecard counts 2 worlds, the recording 1" in _joined(
        _verify(record, header=header, runs=runs, trace=trace)
    )


def test_missing_parts_fail(recorded: tuple[Path, Copies]) -> None:
    record, _ = recorded
    assert "the recording lists no runs" in _joined(_verify(record, runs=()))
    assert "no scorecard header" in _joined(_verify(record, header=None))
    assert "the run has no scorecard" in _joined(_verify(record, scorecard=None))


def test_a_fabricated_trace_fails_against_the_server_and_is_only_consistent_without_it(
    recorded: tuple[Path, Copies],
) -> None:
    """A trace and scorecard fabricated together agree with each other, but not with the server."""
    record, copies = recorded
    run = read_run(record)
    assert run.trace is not None and run.scorecard is not None and run.recording.header is not None
    wid = run.recording.header.world_ids[0]
    fake = [e for e in run.trace if not (e.world_id == wid and e.action_kind == "submit")]
    local = _verify(record, trace=fake)
    assert not local.ok  # the honest scorecard no longer matches either
    joined = _joined(_verify(record, copies, trace=fake))
    assert "the local trace is not the server's" in joined


def test_malformed_recorded_and_traced_actions_are_mismatches_not_crashes(
    recorded: tuple[Path, Copies],
) -> None:
    record, _ = recorded
    run = read_run(record)
    assert run.trace is not None
    events = [
        e.model_copy(update={"content": "{not json"}) if e.kind == "action" and i == 0 else e
        for i, e in enumerate(run.recording.events)
    ]
    assert "is not an action" in _joined(_verify(record, events=events))
    broken = [
        e.model_copy(update={"action_json": '{"kind": "recruit"}'}) if i == 1 else e
        for i, e in enumerate(run.trace)
    ]
    assert "unreadable action" in _joined(_verify(record, trace=broken))
    illegal = [
        (
            e.model_copy(update={"action_json": e.action_json.replace('"count":12', '"count":99999')})
            if e.action_kind == "recruit" and e.action_json
            else e
        )
        for e in run.trace
    ]
    joined = _joined(_verify(record, trace=illegal))
    assert "refuses this action" in joined or "request digest differs" in joined


def test_arena_replay_fails_without_a_trace_or_runs_and_checks_against_a_server(
    recorded: tuple[Path, Copies], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record, _ = recorded
    store = str(cli.fixture_store())
    assert cli.main(["replay", "--record", str(record), "--store", store]) == 0
    assert "consistent" in capsys.readouterr().out
    no_runs = tmp_path / "no-runs"
    no_runs.mkdir()
    lines = (record / "recording.jsonl").read_text().splitlines()
    (no_runs / "recording.jsonl").write_text(
        "\n".join(line for line in lines if '"kind":"run"' not in line) + "\n"
    )
    for name in ("trace.jsonl", "scorecard.json"):
        (no_runs / name).write_text((record / name).read_text())
    assert cli.main(["replay", "--record", str(no_runs), "--store", store]) == 1
    assert "the recording lists no runs" in capsys.readouterr().out
    (record / "trace.jsonl").unlink()
    assert cli.main(["replay", "--record", str(record), "--store", store]) == 1
    assert "NOT VERIFIED: the run has no server trace" in capsys.readouterr().out


def test_arena_replay_with_a_url_checks_the_servers_copies(
    server: str, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ARENA_URL", raising=False)
    record = tmp_path / "played"
    base = ["play", "--agent", "univariate_bh", "--url", server, "--key", KEY, "--n", "2"]
    assert cli.main([*base, "--record", str(record)]) == 0
    assert "verified against the server" in capsys.readouterr().out
    check = [
        "replay",
        "--record",
        str(record),
        "--store",
        str(cli.fixture_store()),
        "--url",
        server,
        "--key",
        KEY,
    ]
    assert cli.main(check) == 0 and "verified against the server" in capsys.readouterr().out
    card = json.loads((record / "scorecard.json").read_text())
    card["discovery_score"] = 0.99
    (record / "scorecard.json").write_text(json.dumps(card))
    assert cli.main(check) == 1
    assert "the local scorecard is not the server's" in capsys.readouterr().out


# ------------------------------------------------------------------------------------- the trace surface


def test_the_trace_is_served_to_its_owner_once_the_scorecard_is_closed(server: str) -> None:
    from onc_agi.core.errors import ArenaError
    from onc_agi.core.schema import ErrorCode, Reset

    owner = ArenaClient(server, KEY)
    sid, cards = owner.open("trace-check", Tier.PUBLIC_TRAIN, 1)
    owner.act(sid, cards[0].world_id, Reset(request_id="r", world_id=cards[0].world_id))
    with pytest.raises(ArenaError) as err:
        owner.trace(sid)
    assert err.value.code is ErrorCode.ACTION_NOT_AVAILABLE  # still open
    owner.close(sid)
    (event,) = owner.trace(sid)
    assert event.scorecard_id == sid and event.action_kind == "reset"
    with pytest.raises(ArenaError) as foreign:
        ArenaClient(server, KEY + "-other").trace(sid)
    with pytest.raises(ArenaError) as unknown:
        owner.trace("sc-" + "0" * 16)
    assert foreign.value.code is unknown.value.code  # a foreign key learns nothing


def test_a_service_without_an_archive_says_it_keeps_no_trace(tmp_path: Path) -> None:
    from onc_agi.core.errors import ArenaError

    service = ScorecardService(STORE, JsonLedger(tmp_path / "ledger.json"))
    sid, _ = service.open(KEY, agent="a", track="open", tier=Tier.PUBLIC_TRAIN, n_worlds=1)
    service.close(sid)
    with pytest.raises(ArenaError, match="keeps no traces"):
        service.trace(sid)


def test_arena_standard_runs_closes_records_and_verifies_from_the_command_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ARENA_URL", raising=False)
    monkeypatch.chdir(tmp_path)
    record = tmp_path / "std"
    argv = ["standard", "--model", "mockllm/model", "--mode", "full_access", "--n", "2", "--seed", "3"]
    assert (
        cli.main([*argv, "--message-limit", "3", "--record", str(record), "--json", str(tmp_path / "c.json")])
        == 0
    )
    out = capsys.readouterr().out
    assert out.startswith("inspect-standard ") and f"{record}: consistent: 2 worlds" in out
    assert {p.name for p in record.iterdir()} >= {
        "recording.jsonl",
        "trace.jsonl",
        "scorecard.json",
        "explanations.md",
    }
    assert cli.main(["replay", "--record", str(record), "--store", str(cli.fixture_store())]) == 0
    assert cli.main([*argv, "--record", str(record)]) == 2  # a run directory is never overwritten
    assert "already exists" in capsys.readouterr().err


def test_a_non_arena_error_body_is_a_typed_error_not_a_crash() -> None:
    from onc_agi.core.errors import ArenaError
    from onc_agi.core.schema import ErrorCode

    def older_server(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Not Found"})

    http = httpx.Client(base_url="http://arena.test", transport=httpx.MockTransport(older_server))
    client = ArenaClient("http://arena.test", KEY, client=http)
    with pytest.raises(ArenaError) as err:
        client.trace("sc-0000")
    assert err.value.code is ErrorCode.ACTION_NOT_AVAILABLE and "not an arena error" in err.value.message
    assert server_trace(client, "sc-0000") is None and server_copies(client, "sc-0000") is None


def test_play_against_a_server_without_the_trace_route_degrades_to_unverified(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An older server (such as the hosted arena before this route) still plays, records and closes."""
    from onc_agi.adapters import http

    original = http.create_app

    def without_trace(service: ScorecardService) -> Any:
        app = original(service)
        app.router.routes = [r for r in app.router.routes if not getattr(r, "path", "").endswith("/trace")]
        return app

    monkeypatch.setattr(http, "create_app", without_trace)
    monkeypatch.delenv("ARENA_URL", raising=False)
    record = tmp_path / "old"
    with serve_in_thread(_service(tmp_path / "older")) as url:
        argv = ["play", "--agent", "univariate_bh", "--url", url, "--key", KEY, "--n", "2"]
        assert cli.main([*argv, "--record", str(record)]) == 0
    captured = capsys.readouterr()
    assert "not an arena error" in captured.err
    assert "NOT VERIFIED: the run has no server trace" in captured.out
    assert not (record / "trace.jsonl").exists() and (record / "scorecard.json").exists()
    assert cli.main(["replay", "--record", str(record), "--store", str(cli.fixture_store())]) == 1


def test_an_eval_tier_run_is_verified_from_its_aggregates_alone(tmp_path: Path) -> None:
    """Eval scorecards list no worlds: the recomputed aggregates and the world count carry the check."""
    from arena_factories import planted_world
    from onc_agi.adapters.agents import make_agent
    from onc_agi.infra.bundles import write_world

    root, keys = tmp_path / "eval-store", tmp_path / "keys"
    for i in range(6):
        world, key = planted_world(f"e-{i:02d}", signal=i % 3 != 0, seed=i, n_pool=60, tier=Tier.PUBLIC_EVAL)
        write_world(root, world, key, keys_dir=keys)
    store = FileWorldStore(root, keys)
    service = ScorecardService(
        store,
        JsonLedger(tmp_path / "l.json"),
        archive=FileScorecardArchive(tmp_path / "a"),
        min_eval_worlds=1,
    )
    arena = LocalArena(service, KEY)
    record = tmp_path / "eval-run"
    with RecordingWriter(record / RECORDING_FILE) as writer:
        result = run_swarm(
            arena,
            lambda: make_agent("univariate_bh"),
            agent_name="bh",
            tier=Tier.PUBLIC_EVAL,
            n_worlds=4,
            recorder=writer,
        )
    card, sid = result.scorecard, result.scorecard.scorecard_id
    assert card.worlds == () and card.n_worlds == 4
    save_run(record, card, server_trace(arena, sid))
    copies = server_copies(arena, sid)
    run = read_run(record)
    assert run.trace is not None and run.recording.header is not None

    def check(**change: Any) -> RunVerification:
        files: dict[str, Any] = {
            "scorecard": card,
            "header": run.recording.header,
            "runs": run.recording.runs,
            "events": run.recording.events,
        } | change
        trace = files.pop("trace", run.trace)
        return verify_run(trace, store, **files)

    assert check().ok
    assert (
        copies is not None
        and verify_run(
            run.trace,
            store,
            scorecard=card,
            header=run.recording.header,
            runs=run.recording.runs,
            server_trace=copies[0],
            server_scorecard=copies[1],
        ).authority
        == "server"
    )
    doctored = card.model_copy(update={"discovery_score": 0.99, "find": 0.99})
    assert "discovery_score, find" in _joined(check(scorecard=doctored))
    gone = run.recording.header.world_ids[0]
    shrunk = run.recording.header.model_copy(update={"world_ids": run.recording.header.world_ids[1:]})
    joined = _joined(
        check(
            header=shrunk,
            runs=[r for r in run.recording.runs if r.world_id != gone],
            trace=[e for e in run.trace if e.world_id != gone],
        )
    )
    assert "the scorecard counts 4 worlds, the recording 3" in joined
    service.shutdown()
