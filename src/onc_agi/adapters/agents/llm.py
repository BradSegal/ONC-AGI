"""LLM agent template: an OpenAI-compatible tool-calling model as an :class:`Agent` (open track).

The ARC "LLM template" counterpart. By default it shows the model exactly what the
Inspect standard harness shows - the same task card (``inspect_task.card_text``), the
same ``/data/revealed.csv`` (``inspect_task._revealed_csv``), the same tools and the same
no-network analysis image - so the open and standard tracks differ only where a
``[harness]`` table says they should::

    [harness]
    max_turns = 40            # model calls per world; exhausting them ends the world unsubmitted
    tool_timeout = 180        # seconds per python/bash call
    output_limit = 12000      # characters of tool output returned to the model
    tools = ["python", "bash"]  # sandbox tools; recruit/assay join in sequential worlds, submit always
    system_prompt = "..."     # replaces the default system prompt
    continue_prompt = "..."   # sent when the model replies without calling a tool

How a world runs: :meth:`LLMToolAgent.choose_action` refreshes the sandbox CSV from the
current view, then runs the chat loop - python and bash execute in the world's sandbox -
until the model calls an arena tool (recruit, assay or submit), whose action it returns.
The runner applies it; the next call reports the result to the model. A refused action
comes back through :meth:`LLMToolAgent.on_refused` and reaches the model as a tool error,
as in the standard harness. The model is reached over plain HTTP (``httpx``):
``POST {base_url}/chat/completions`` with the profile's parameters, retried with
backoff on 429, 5xx and connection errors.

The standard track for model comparison remains the Inspect task (fixed prompt and tools);
this template is the open track's starting point.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import tomllib
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from importlib import resources
from pathlib import Path
from typing import Any, Protocol

import httpx
from pydantic import ValidationError

from onc_agi.adapters.profiles import Profile
from onc_agi.core.errors import ArenaError
from onc_agi.core.schema import Action, Assay, Mode, Recruit, Submit, WorldCard
from onc_agi.services.engine import EpisodeView
from onc_agi.services.kit import Agent, EndEpisode, Usage

HARNESS_NAME = "openai-tools-1.0"

SYSTEM_PROMPT = """You are a careful biomedical data scientist working in an analysis sandbox.
You have tools to run Python or bash on the data, and a submit tool to give your final answer.
Each message may call one or more tools. Inspect the data before concluding, prefer simple robust
statistics, and keep tool output short (print summaries, not whole tables). When you have an answer,
call submit with the ordered, comma-separated feature ids (most likely driver first), or an empty
string if nothing can be found. Only the submit call counts as your answer."""

CONTINUE_PROMPT = "Please continue with the tools, and finish by calling submit."

SANDBOX_TOOLS = ("python", "bash")
ARENA_TOOLS = ("recruit", "assay", "submit")
RETRY_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class HarnessConfig:
    max_turns: int = 40
    tool_timeout: int = 180
    output_limit: int = 12_000
    tools: tuple[str, ...] = SANDBOX_TOOLS
    system_prompt: str = SYSTEM_PROMPT
    continue_prompt: str = CONTINUE_PROMPT

    def __post_init__(self) -> None:
        if min(self.max_turns, self.tool_timeout, self.output_limit) < 1:
            raise ValueError("max_turns, tool_timeout and output_limit must be positive")
        if not set(self.tools) <= set(SANDBOX_TOOLS):
            raise ValueError(
                f"[harness] tools may only name {list(SANDBOX_TOOLS)}; arena tools are added per world"
            )

    @classmethod
    def from_table(cls, table: dict[str, Any]) -> HarnessConfig:
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(table) - known)
        if unknown:
            raise ValueError(f"unknown [harness] keys {unknown}; known: {sorted(known)}")
        if "tools" in table:
            table = {**table, "tools": tuple(table["tools"])}
        return cls(**table)

    @classmethod
    def load(cls, path: Path | None) -> HarnessConfig:
        if path is None:
            return cls()
        return cls.from_table(dict(tomllib.loads(path.read_text(encoding="utf-8")).get("harness", {})))

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def harness_label(config: HarnessConfig) -> str:
    """``openai-tools-1.0+<8 hex>`` over the config, this module and the standard harness label,
    so a prompt, tool, card or image change changes the label."""
    h = hashlib.sha256(config.digest().encode())
    h.update(Path(__file__).read_bytes())
    h.update(_standard_harness().HARNESS.encode())
    return f"{HARNESS_NAME}+{h.hexdigest()[:8]}"


def _standard_harness() -> Any:
    """The Inspect harness module: its card and CSV are what this agent shows (needs the extra)."""
    try:
        from onc_agi.adapters import inspect_task
    except ImportError as exc:  # inspect_ai is the optional [inspect] extra
        raise ImportError(
            "the LLM agent shows the standard harness's task card: pip install 'onc-agi[inspect]'"
        ) from exc
    return inspect_task


def _function(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    # strict-compatible: every property required, nothing else allowed
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


TOOL_SPECS: dict[str, dict[str, Any]] = {
    "python": _function(
        "python",
        "Run Python 3 code in the sandbox (numpy, pandas, scipy, scikit-learn, statsmodels; no network)."
        " The revealed data are in /data/revealed.csv. Each call is a fresh process: print what you need.",
        {"code": {"type": "string", "description": "Python code to run."}},
    ),
    "bash": _function(
        "bash",
        "Run a bash command in the sandbox (no network).",
        {"cmd": {"type": "string", "description": "The command."}},
    ),
    "recruit": _function(
        "recruit",
        "Recruit patients (their outcome and stratum are revealed).",
        {
            "count": {
                "type": "integer",
                "description": "Number of patients to recruit from the stratum's queue.",
            },
            "stratum": {
                "type": "string",
                "description": 'Stratum to recruit from, as named on the task card ("all" when there is one).',
            },
        },
    ),
    "assay": _function(
        "assay",
        "Measure features on every recruited patient not yet measured.",
        {"feature_ids": {"type": "string", "description": "Comma-separated feature ids to measure."}},
    ),
    "submit": _function(
        "submit",
        "Submit the ordered, comma-separated feature ids (or an empty string to abstain).",
        {"answer": {"type": "string", "description": "Ordered, comma-separated feature ids, or empty."}},
    ),
}


# ---------------------------------------------------------------------------------------- sandbox


class Sandbox(Protocol):
    """Where python and bash run for one world."""

    def write_data(self, csv: str) -> None: ...

    def run(self, argv: list[str], stdin: str | None = None) -> tuple[str, bool]:
        """Run a command; return (combined output, failed)."""
        ...

    def close(self) -> None: ...


SandboxFactory = Callable[[HarnessConfig], Sandbox]
_IMAGE_LOCK = threading.Lock()
_IMAGES: dict[str, str] = {}


def sandbox_image() -> str:
    """Build (once) the standard harness's analysis image and return its tag."""
    context = Path(str(resources.files("onc_agi") / "adapters" / "sandbox"))
    tag = "onc-agi-sandbox:" + hashlib.sha256((context / "Dockerfile").read_bytes()).hexdigest()[:12]
    with _IMAGE_LOCK:  # concurrent worlds share one build
        if tag not in _IMAGES:
            if subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode != 0:
                subprocess.run(
                    ["docker", "build", "-q", "-t", tag, str(context)], check=True, capture_output=True
                )
            _IMAGES[tag] = tag
    return tag


class DockerSandbox:
    """One no-network container per world, with the standard harness compose file's limits.

    It runs as the host user, so ``/data`` (a bind-mounted temp dir) stays removable; :meth:`close`
    removes the container and the directory.
    """

    def __init__(self, image: str, timeout: int) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="arena-world-"))
        self.timeout = timeout
        try:
            self.container = subprocess.run(
                ["docker", "run", "-d", "--rm", "--init", "--network", "none", "--cpus", "2", "--memory", "4g",
                 "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
                 "-v", f"{self.dir}:/data", "-w", "/data", image, "tail", "-f", "/dev/null"],
                check=True, capture_output=True, text=True,
            ).stdout.strip()  # fmt: skip
        except BaseException:
            shutil.rmtree(self.dir, ignore_errors=True)
            raise

    @classmethod
    def start(cls, config: HarnessConfig) -> DockerSandbox:
        return cls(sandbox_image(), config.tool_timeout)

    def write_data(self, csv: str) -> None:
        (self.dir / "revealed.csv").write_text(csv)

    def run(self, argv: list[str], stdin: str | None = None) -> tuple[str, bool]:
        try:
            done = subprocess.run(
                ["docker", "exec", "-i", self.container, *argv],
                input=stdin, capture_output=True, text=True, timeout=self.timeout,
            )  # fmt: skip
        except subprocess.TimeoutExpired:
            return f"Timed out after {self.timeout} s.", True
        out = (done.stdout + (("\n" + done.stderr) if done.stderr else "")).strip() or "(no output)"
        return out, done.returncode != 0

    def close(self) -> None:
        subprocess.run(["docker", "rm", "-f", self.container], capture_output=True)
        shutil.rmtree(self.dir, ignore_errors=True)


# ---------------------------------------------------------------------------------------- model


@dataclass(frozen=True)
class Completion:
    message: dict[str, Any]
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None


class ChatClient:
    """Chat completions over ``httpx`` for one profile, with retries and cost accounting."""

    def __init__(
        self,
        profile: Profile,
        *,
        http: httpx.Client | None = None,
        retries: int = 5,
        backoff: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.profile = profile
        self.http = http or httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0))
        self.retries, self.backoff, self.sleep = retries, backoff, sleep
        self.headers = profile.headers()

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Completion:
        body = {**self.profile.request_body(), "messages": messages, "tools": tools}
        url = f"{self.profile.base_url}/chat/completions"
        last = "no attempt made"
        for attempt in range(self.retries + 1):
            if attempt:
                self.sleep(self.backoff * 2 ** (attempt - 1))
            try:
                response = self.http.post(url, json=body, headers=self.headers)
            except httpx.TransportError as exc:
                last = f"{type(exc).__name__}: {exc}"
                continue
            if response.status_code in RETRY_STATUS or response.status_code >= 500:
                last = f"HTTP {response.status_code}: {response.text[:500]}"
                continue
            if response.status_code >= 400:
                raise RuntimeError(
                    f"{url} refused the request: HTTP {response.status_code}: {response.text[:1000]}"
                )
            data = response.json()
            if not data.get("choices"):  # some routers return 200 with an upstream error body
                last = f"no choices in response: {json.dumps(data)[:500]}"
                continue
            return self._completion(data)
        raise RuntimeError(f"{url}: gave up after {self.retries + 1} attempts ({last})")

    def _completion(self, data: dict[str, Any]) -> Completion:
        message = {k: v for k, v in data["choices"][0]["message"].items() if v is not None}
        usage = data.get("usage") or {}
        prompt, completion = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
        return Completion(message, prompt, completion, self._cost(usage, prompt, completion))

    def _cost(self, usage: dict[str, Any], prompt: int, completion: int) -> float | None:
        """The provider's reported cost (OpenRouter: ``usage.cost``), else the profile's prices."""
        if usage.get("cost") is not None:
            return float(usage["cost"])
        price = self.profile.price()
        if price is None or not usage:
            return None
        details = usage.get("prompt_tokens_details") or {}
        cached = int(details.get("cached_tokens") or 0)
        written = int(details.get("cache_write_tokens") or 0)
        read = price.input if price.cache_read is None else price.cache_read
        write = price.input if price.cache_write is None else price.cache_write
        uncached = max(0, prompt - cached - written)
        return uncached * price.input + cached * read + written * write + completion * price.output

    def close(self) -> None:
        self.http.close()


# ---------------------------------------------------------------------------------------- agent


@dataclass(frozen=True)
class _Pending:
    """An arena tool call whose action the runner is applying."""

    call_id: str
    tool: str
    action: Action
    n_features: int = 0


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n... [truncated {len(text) - limit} characters]"


def _ranking(answer: str, valid: set[str]) -> tuple[str, ...]:
    listed = dict.fromkeys(f.strip() for f in answer.replace("\n", ",").split(",") if f.strip())
    return tuple(f for f in listed if f in valid)


class LLMToolAgent(Agent):
    """One world, one model conversation, one sandbox. Build a fresh instance per world."""

    def __init__(
        self,
        profile: Profile,
        config: HarnessConfig | None = None,
        *,
        http: httpx.Client | None = None,
        sandbox_factory: SandboxFactory | None = None,
        name: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__()
        self.profile = profile
        self.config = config or HarnessConfig()
        self.name = name or f"llm-{profile.name}"
        self._chat = ChatClient(profile, http=http, sleep=sleep)  # resolves the key: fails fast
        self._sandbox_factory = sandbox_factory or DockerSandbox.start
        self._sandbox: Sandbox | None = None
        self._card: WorldCard | None = None
        self._tools: tuple[str, ...] = ()
        self._messages: list[dict[str, Any]] = []
        self._queue: list[dict[str, Any]] = []  # tool calls of the last reply not yet answered
        self._pending: _Pending | None = None
        self.turns = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd: float | None = 0.0  # None once any reply is unpriced

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self._messages

    def usage(self) -> Usage:
        return Usage(self.prompt_tokens + self.completion_tokens, self.cost_usd, self.profile.model)

    # ------------------------------------------------------------------ the Agent interface

    def choose_action(self, card: WorldCard, view: EpisodeView) -> Action:
        harness = _standard_harness()
        if self._card is None:
            self._start(card, harness.card_text(card))
        elif card.world_id != self._card.world_id:
            raise RuntimeError("an LLMToolAgent plays one world; build a fresh one per world")
        assert self._sandbox is not None
        self._sandbox.write_data(harness._revealed_csv(view))
        if self._pending is not None:  # the runner applied the arena action we returned
            self._reply(
                self._pending.call_id, self._pending.tool, self._accepted(self._pending, view), error=False
            )
            self._pending = None
        while True:
            while self._queue:
                action = self._handle(self._queue.pop(0), card)
                if action is not None:
                    return action
            if self.turns >= self.config.max_turns:
                raise EndEpisode(f"max_turns={self.config.max_turns} reached without a submission")
            self._ask(card.world_id)

    def on_refused(self, action: Action, error: ArenaError) -> None:
        """Hand the refusal to the model as the tool's error, as the standard harness does."""
        if self._pending is None or self._pending.action != action:
            raise error
        self._reply(
            self._pending.call_id, self._pending.tool, f"{error.code.value}: {error.message}", error=True
        )
        self._pending = None

    def close(self) -> None:
        if self._sandbox is not None:
            self._sandbox.close()
            self._sandbox = None
        self._chat.close()

    # ------------------------------------------------------------------ the conversation

    def _start(self, card: WorldCard, card_text: str) -> None:
        self._card = card
        tools = [t for t in SANDBOX_TOOLS if t in self.config.tools]
        if card.mode is Mode.SEQUENTIAL:
            tools += ["recruit", "assay"]
        self._tools = (*tools, "submit")
        self._messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": card_text},
        ]
        self._sandbox = self._sandbox_factory(self.config)

    def _ask(self, wid: str) -> None:
        self.turns += 1
        reply = self._chat.complete(self._messages, [TOOL_SPECS[t] for t in self._tools])
        self.prompt_tokens += reply.prompt_tokens
        self.completion_tokens += reply.completion_tokens
        if self.cost_usd is not None:
            self.cost_usd = None if reply.cost_usd is None else self.cost_usd + reply.cost_usd
        message = reply.message
        self._messages.append(message)
        reasoning = message.get("reasoning") or message.get("reasoning_content")
        if reasoning:
            self.record(wid, "reasoning", str(reasoning))
        if message.get("content"):
            self.record(wid, "assistant", str(message["content"]))
        self._queue = list(message.get("tool_calls") or [])
        if not self._queue:
            self._messages.append({"role": "user", "content": self.config.continue_prompt})

    def _reply(self, call_id: str, tool: str, text: str, *, error: bool) -> None:
        assert self._card is not None
        self.record(self._card.world_id, "tool_result", text, tool=tool, error=error)
        self._messages.append({"role": "tool", "tool_call_id": call_id, "content": text})

    def _handle(self, call: dict[str, Any], card: WorldCard) -> Action | None:
        """Run a sandbox tool and answer it, or return the arena action a tool call asks for."""
        call_id = str(call.get("id", ""))
        function = call.get("function") or {}
        name = str(function.get("name", ""))
        try:
            args = json.loads(function.get("arguments") or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except ValueError as exc:
            self._reply(call_id, name, f"Arguments were not a valid JSON object: {exc}", error=True)
            return None
        shown = str(args.get("code") or args.get("cmd")) if name in SANDBOX_TOOLS else json.dumps(args)
        self.record(card.world_id, "tool_call", shown, tool=name)
        if name not in self._tools:
            self._reply(
                call_id, name, f"Unknown tool {name!r}; available: {', '.join(self._tools)}.", error=True
            )
            return None
        if name in SANDBOX_TOOLS:
            assert self._sandbox is not None
            argv = ["python", "-"] if name == "python" else ["bash", "-lc", str(args.get("cmd", ""))]
            stdin = str(args.get("code", "")) if name == "python" else None
            out, failed = self._sandbox.run(argv, stdin=stdin)
            self._reply(call_id, name, _truncate(out, self.config.output_limit), error=failed)
            return None
        try:
            action, n = self._arena_action(name, args, card)
        except (ValidationError, ValueError, TypeError, KeyError) as exc:
            self._reply(call_id, name, f"invalid_payload: {exc}", error=True)
            return None
        self._pending = _Pending(call_id, name, action, n)
        return action

    def _arena_action(self, name: str, args: dict[str, Any], card: WorldCard) -> tuple[Action, int]:
        if name == "recruit":
            stratum = str(args.get("stratum") or "all")
            return Recruit(request_id=self.request_id(), count=int(args["count"]), stratum=stratum), 0
        if name == "assay":
            ids = tuple(
                dict.fromkeys(f.strip() for f in str(args.get("feature_ids", "")).split(",") if f.strip())
            )
            return Assay(request_id=self.request_id(), feature_ids=ids), len(ids)
        ranking = _ranking(str(args.get("answer", "")), set(card.feature_ids()))
        return Submit(request_id=self.request_id(), ranking=ranking), 0

    @staticmethod
    def _accepted(pending: _Pending, view: EpisodeView) -> str:
        spent = f"spent {view.spent:.0f} of {view.budget:.0f} USD."
        if pending.tool == "recruit":
            return f"Recruited; {len(view.rows)} patients revealed; {spent}"
        if pending.tool == "assay":
            return f"Measured {pending.n_features} features; {spent}"
        return "Submitted."
