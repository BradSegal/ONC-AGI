"""Inspect AI standard harness.

One task per mode. Every model sees the same premise and task card, the same
tools and the same limits:

* ``python`` / ``bash`` run inside a Docker sandbox with **no network**; the
  revealed data are written there as ``/data/revealed.csv``;
* ``recruit`` / ``assay`` (sequential mode) and ``submit`` run on the **host**,
  where the episode and answer keys live, so nothing scorer-side ever enters the
  sandbox;
* a safety timeout per world; tokens and cost are reported, not capped.

Each task run is one arena scorecard, played through the same arena surface as the open
track (``onc-agi play``): in-process over a :class:`ScorecardService` on ``store_root``, or over
HTTP against an arena server (``url=``, with the key in the environment variable ``key_env``,
default ``ARENA_KEY``, so it never enters the log). Either way the draw, the caps on eval tiers
, the server trace and the close are the server's own. In-process eval tiers need the
ledger the server uses (``ledger=``), so a world drawn here is never drawn again there. On public
train, ``n_worlds=`` plays a stratified sample seeded by ``seed=`` (default 0) and ``world_ids=``
plays named worlds: comma-separated ids, or a published id-list file.

Per-world results are attached to the log only for in-process public-train runs; otherwise
they exist only in the closed scorecard. :func:`close_scorecard` closes the run's scorecard
after the eval and, with ``record=``, writes the run directory every track writes: the
recording (the Inspect transcript plus the actions the arena applied), the server trace and
the scorecard. :func:`run_standard` does all of it in one call (``onc-agi standard``).
:func:`log_to_scorecard` remains for logs made without a scorecard.
Any OpenAI-compatible endpoint works through Inspect's ``openai-api/<name>/<model>``
provider (``<NAME>_BASE_URL`` and ``<NAME>_API_KEY`` in the environment).
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.log import EvalLog
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import Generate, Solver, TaskState, basic_agent, solver
from inspect_ai.tool import Tool, ToolError, bash, python, tool
from inspect_ai.util import sandbox, store

from onc_agi.adapters.client import ArenaClient, view_from_observation
from onc_agi.adapters.inspect_log import transcript
from onc_agi.adapters.profiles import Price
from onc_agi.adapters.swarm import LocalArena, server_trace
from onc_agi.core.digest import behaviour_label
from onc_agi.core.errors import ArenaError
from onc_agi.core.ports import RecordingEvent, RecordingHeader
from onc_agi.core.schema import (
    Action,
    Assay,
    EpisodeStatus,
    Mode,
    Observation,
    Recruit,
    Reset,
    Scorecard,
    Submit,
    Tier,
    WorldCard,
    WorldScore,
)
from onc_agi.infra.archive import FileScorecardArchive
from onc_agi.infra.bundles import FileWorldStore, world_id_list
from onc_agi.infra.ledger import JsonLedger
from onc_agi.infra.recordings import RECORDING_FILE, RecordingWriter, save_run
from onc_agi.services import scoring
from onc_agi.services.engine import EpisodeView
from onc_agi.services.scorecards import MIN_EVAL_WORLDS, ScorecardService

SANDBOX_COMPOSE = Path(__file__).parent / "sandbox" / "compose.yaml"
SANDBOX: str | tuple[str, str] = ("docker", str(SANDBOX_COMPOSE))  # tests substitute an in-memory one


# What every model sees and can do: a card, prompt, tool or image change changes the label.
HARNESS = behaviour_label("inspect-standard-1.1", Path(__file__), SANDBOX_COMPOSE.with_name("Dockerfile"))
SAFETY_TIMEOUT_SECONDS = 1800
OPERATOR_KEY = "inspect-standard"  # caller identity for caps when the harness runs in-process


class _Run:
    """One task run's scorecard: the arena it plays (in-process or over HTTP) and what it applied.

    A plain class, not a dataclass: ``inspect eval FILE@task`` loads this file outside
    ``sys.modules``, where dataclasses cannot resolve postponed annotations.
    """

    def __init__(
        self,
        arena: LocalArena | ArenaClient,
        scorecard_id: str,
        tier: Tier,
        agent: str,
        cards: dict[str, WorldCard],
        store: FileWorldStore | None,  # in-process only: per-world public-train results in the log
    ) -> None:
        self.arena, self.scorecard_id, self.tier, self.agent = arena, scorecard_id, tier, agent
        self.cards, self.store = cards, store
        self.opened_at = datetime.now(UTC)
        self.actions: dict[str, list[tuple[str, str | None]]] = {}  # world -> (action JSON, refusal)

    @property
    def service(self) -> ScorecardService | None:
        return self.arena.service if isinstance(self.arena, LocalArena) else None

    def act(self, world_id: str, action: Action) -> Observation:
        """Apply an action through the arena, keeping it (and any refusal) for the recording."""
        try:
            obs = self.arena.act(self.scorecard_id, world_id, action)
        except ArenaError as exc:
            refusal = f"{exc.code.value}: {exc.message}"
            self.actions.setdefault(world_id, []).append((action.model_dump_json(), refusal))
            raise
        self.actions.setdefault(world_id, []).append((action.model_dump_json(), None))
        return obs

    def shutdown(self) -> None:
        if isinstance(self.arena, LocalArena):
            self.arena.shutdown()
        else:
            self.arena.http.close()


_RUNS: dict[str, _Run] = {}  # host-side state, keyed by scorecard id


_SURVIVAL_COLUMN = {"survival": "time (days; outcome is the event indicator), "}


def _survival_lines(card: WorldCard) -> list[str]:
    """How to read a survival outcome (nothing for binary cards, whose text is unchanged)."""
    if card.outcome_type != "survival" or card.horizon_days is None:
        return []
    return [
        "The outcome is time to event: outcome = 1 if the event was observed and 0 if censored; time is "
        f"the follow-up in days (to the event or to censoring, at most {card.horizon_days:.0f})."
    ]


def card_text(card: WorldCard) -> str:
    lines = [
        card.premise,
        "",
        f"World {card.world_id}: {card.n_pool} patients, {card.outcome_type} outcome, mode {card.mode.value}.",
        *_survival_lines(card),
    ]
    if card.mode is Mode.SEQUENTIAL:
        sizes = card.stratum_sizes or {s: card.n_pool // len(card.strata) for s in card.strata}
        lines += [
            f"Budget {card.budget:.0f} USD. Nothing is revealed yet: recruit patients (outcome and stratum"
            f" revealed, {card.prices.recruit_per_patient:.0f} USD per patient), then assay features on them"
            " (the feature's price per recruited patient not yet measured).",
            "Strata (name: patients available): " + ", ".join(f"{s}: {n}" for s, n in sizes.items()) + ".",
            "/data/revealed.csv holds the recruited patients and assayed features (columns: patient_id, stratum,"
            f" outcome, {_SURVIVAL_COLUMN.get(card.outcome_type, '')}then features) and is refreshed after every"
            " recruit or assay.",
            "Spending counts: Find is scaled by your efficiency against a reference cost, so acquire what the"
            " question needs rather than everything.",
        ]
    else:
        lines.append(
            "All data are already in /data/revealed.csv (columns: patient_id, stratum, outcome, "
            f"{_SURVIVAL_COLUMN.get(card.outcome_type, '')}then features)."
        )
    lines.append("Features (id, data type, timing, assay price per patient):")
    lines += [
        f"- {f.feature_id}, {f.data_type}, {f.timing.value}, {f.assay_price:.0f}" for f in card.features
    ]
    lines += [
        "",
        "post_outcome features are measured after the outcome; listing one as a driver zeroes this world.",
        "Analyse with the python or bash tools (no network). Finish by calling submit with a comma-separated,"
        " ordered list of feature ids (most likely driver first), or an empty string if nothing can be found.",
    ]
    return "\n".join(lines)


def _revealed_csv(view: EpisodeView) -> str:
    frame = pd.DataFrame(view.x, columns=list(view.feature_ids))
    frame = frame.loc[:, [f for f, m in zip(view.feature_ids, view.measured, strict=True) if m]]
    if view.time is not None:
        frame.insert(0, "time", view.time)
    frame.insert(0, "outcome", view.outcome)
    frame.insert(0, "stratum", list(view.stratum))
    frame.insert(0, "patient_id", [f"r{i}" for i in range(len(view.rows))])
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue()


async def _sync(view: EpisodeView) -> None:
    await sandbox().write_file("/data/revealed.csv", _revealed_csv(view))


def _here() -> tuple[_Run, str]:
    """The scorecard and world of the running sample."""
    return _RUNS[str(store().get("scorecard_id"))], str(store().get("world_id"))


def _act(action: Action) -> EpisodeView:
    """Apply a tool action; a refused action returns to the model as a tool error."""
    run, world_id = _here()
    try:
        obs = run.act(world_id, action)
    except ArenaError as exc:
        raise ToolError(f"{exc.code.value}: {exc.message}") from exc
    return view_from_observation(run.cards[world_id], obs)


def _step() -> int:
    run, world_id = _here()
    return run.arena.state(run.scorecard_id, world_id).step


@solver
def setup_world() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        store().set("scorecard_id", str(state.metadata["scorecard_id"]))
        store().set("world_id", str(state.metadata["world_id"]))
        await _sync(_act(Reset(request_id="harness-reset", world_id=str(state.metadata["world_id"]))))
        return state

    return solve


@tool
def recruit() -> Tool:
    async def execute(count: int, stratum: str) -> str:
        """Recruit patients (their outcome and stratum are revealed).

        Args:
            count: Number of patients to recruit from the stratum's queue.
            stratum: Stratum to recruit from, as named on the task card ("all" when there is one).
        """
        view = _act(Recruit(request_id=f"recruit-{_step()}", count=count, stratum=stratum))
        await _sync(view)
        return (
            f"Recruited; {len(view.rows)} patients revealed; spent {view.spent:.0f} of {view.budget:.0f} USD."
        )

    return execute


@tool
def assay() -> Tool:
    async def execute(feature_ids: str) -> str:
        """Measure features on every recruited patient not yet measured.

        Args:
            feature_ids: Comma-separated feature ids to measure.
        """
        ids = tuple(f.strip() for f in feature_ids.split(",") if f.strip())
        view = _act(Assay(request_id=f"assay-{_step()}", feature_ids=ids))
        await _sync(view)
        return f"Measured {len(ids)} features; spent {view.spent:.0f} of {view.budget:.0f} USD."

    return execute


def _ranking(answer: str, valid: set[str] | None = None) -> tuple[str, ...]:
    """A submitted answer as an ordered ranking: comma- or line-separated ids, first mention kept,
    and (given ``valid``) only ids the world has. Both tracks parse answers here."""
    listed = dict.fromkeys(f.strip() for f in answer.replace("\n", ",").split(",") if f.strip())
    return tuple(f for f in listed if valid is None or f in valid)


@scorer(metrics=[mean()])
def arena_scorer() -> Any:
    async def score(state: TaskState, target: Target) -> Score:
        run = _RUNS[str(state.metadata["scorecard_id"])]
        world_id = str(state.metadata["world_id"])
        card = run.cards[world_id]
        answer = state.output.completion if state.output else ""
        valid = set(card.feature_ids())
        ranking = _ranking(answer, valid)
        obs = run.arena.state(run.scorecard_id, world_id)
        if obs.status is EpisodeStatus.ACTIVE:
            obs = run.act(world_id, Submit(request_id="harness-submit", ranking=ranking))
        metadata: dict[str, Any] = {"spent": obs.spent}
        if run.tier is not Tier.PUBLIC_TRAIN or run.store is None:
            # eval results exist only as the closed scorecard; over HTTP the server scores
            return Score(value=0.0, answer=",".join(ranking), metadata=metadata | {"exposure": "scorecard"})
        world_score = scoring.score_world(
            ranking,
            run.store.answer_key(world_id),
            spent=obs.spent,
            sequential=card.mode is Mode.SEQUENTIAL,
        )
        value = float(world_score.restrained) if world_score.is_null else world_score.find
        return Score(
            value=value,
            answer=",".join(ranking),
            metadata=metadata | {"world_score": world_score.model_dump(mode="json")},
        )

    return score


def _arena(
    store_root: str | None,
    url: str | None,
    key_env: str,
    keys_dir: str | None,
    ledger: str | None,
    archive: str | None,
    api_key: str,
    min_eval_worlds: int,
) -> tuple[LocalArena | ArenaClient, FileWorldStore | None]:
    """The arena a task run plays: a server over HTTP, or an in-process service on a store."""
    if (store_root is None) == (url is None):
        raise ValueError("give store_root= (play in-process) or url= (play a server), not both")
    if url is not None:
        key = os.environ.get(key_env, "")
        if not key:
            raise ValueError(f"playing a server needs its key in the environment variable {key_env}")
        return ArenaClient(url, key), None
    assert store_root is not None
    worlds = FileWorldStore(Path(store_root), Path(keys_dir) if keys_dir else None)
    scratch = tempfile.TemporaryDirectory(prefix="arena-inspect-")  # the ledger and archive, unless given
    service = ScorecardService(
        worlds,
        JsonLedger(Path(ledger) if ledger else Path(scratch.name) / "ledger.json"),
        archive=FileScorecardArchive(Path(archive) if archive else Path(scratch.name) / "archive"),
        min_eval_worlds=min_eval_worlds,
    )
    local = LocalArena(service, api_key)
    local._tmp = scratch
    return local, worlds


def _task(
    store_root: str | None,
    tier: str,
    mode: Mode,
    n_worlds: int | None,
    keys_dir: str | None,
    message_limit: int,
    ledger: str | None,
    archive: str | None,
    agent: str,
    api_key: str,
    min_eval_worlds: int,
    world_ids: str | None = None,
    seed: int | None = None,
    url: str | None = None,
    key_env: str = "ARENA_KEY",
) -> Task:
    tier_ = Tier(tier)
    if url is None and tier_ is not Tier.PUBLIC_TRAIN and (ledger is None or n_worlds is None):
        raise ValueError(
            f"{tier} runs draw fresh worlds: pass the server's ledger= (and archive=) and n_worlds="
        )
    if world_ids is not None and n_worlds is not None:
        raise ValueError("give n_worlds= (a seeded stratified sample) or world_ids=, not both")
    named = world_id_list(world_ids) if world_ids is not None else None
    arena, worlds = _arena(store_root, url, key_env, keys_dir, ledger, archive, api_key, min_eval_worlds)
    sid: str | None = None
    try:
        if worlds is not None and not worlds.world_ids(tier_):
            # `inspect eval` runs a task from its file's directory, so a relative path finds nothing
            raise ValueError(f"no {tier} worlds under store_root={store_root!r}; pass an absolute path")
        if n_worlds is None and named is None:
            n_worlds = sum(1 for c in arena.worlds(tier_) if c.mode is mode)
        sid, cards = arena.open(
            agent, tier_, n_worlds, track="standard", mode=mode, world_ids=named, seed=seed
        )
        _RUNS[sid] = _Run(arena, sid, tier_, agent, {c.world_id: c for c in cards}, worlds)
        samples = [
            Sample(
                input=card_text(card),
                metadata={"world_id": card.world_id, "scorecard_id": sid},
                id=card.world_id,
            )
            for card in cards
        ]
        tools: list[Tool] = [python(timeout=180), bash(timeout=180)]
        if mode is Mode.SEQUENTIAL:
            tools += [recruit(), assay()]
        return Task(
            dataset=samples,
            solver=basic_agent(
                init=setup_world(),
                tools=tools,
                message_limit=message_limit,
                submit_description="Submit the ordered, comma-separated feature ids (or an empty string to abstain).",
            ),
            scorer=arena_scorer(),
            sandbox=SANDBOX,
            time_limit=SAFETY_TIMEOUT_SECONDS,
            epochs=1,  # one episode per world per scorecard; repeat by opening another scorecard
            metadata={"harness": HARNESS, "tier": tier, "mode": mode.value, "scorecard_id": sid},
        )
    except BaseException:
        run = _RUNS.pop(sid, None) if sid is not None else None
        (run or _Run(arena, "", tier_, agent, {}, None)).shutdown()
        raise


@task
def arena_full_access(
    store_root: str | None = None,
    tier: str = "public_train",
    n_worlds: int | None = None,
    keys_dir: str | None = None,
    message_limit: int = 60,
    ledger: str | None = None,
    archive: str | None = None,
    agent: str = "inspect-standard",
    api_key: str = OPERATOR_KEY,
    min_eval_worlds: int = MIN_EVAL_WORLDS,
    world_ids: str | None = None,
    seed: int | None = None,
    url: str | None = None,
    key_env: str = "ARENA_KEY",
) -> Task:
    return _task(
        store_root,
        tier,
        Mode.FULL_ACCESS,
        n_worlds,
        keys_dir,
        message_limit,
        ledger,
        archive,
        agent,
        api_key,
        min_eval_worlds,
        world_ids,
        seed,
        url,
        key_env,
    )


@task
def arena_sequential(
    store_root: str | None = None,
    tier: str = "public_train",
    n_worlds: int | None = None,
    keys_dir: str | None = None,
    message_limit: int = 80,
    ledger: str | None = None,
    archive: str | None = None,
    agent: str = "inspect-standard",
    api_key: str = OPERATOR_KEY,
    min_eval_worlds: int = MIN_EVAL_WORLDS,
    world_ids: str | None = None,
    seed: int | None = None,
    url: str | None = None,
    key_env: str = "ARENA_KEY",
) -> Task:
    return _task(
        store_root,
        tier,
        Mode.SEQUENTIAL,
        n_worlds,
        keys_dir,
        message_limit,
        ledger,
        archive,
        agent,
        api_key,
        min_eval_worlds,
        world_ids,
        seed,
        url,
        key_env,
    )


Prices = Mapping[str, Price]  # model name as Inspect reports it -> price


def usage(log: EvalLog, prices: Prices | None = None) -> tuple[int | None, float | None]:
    """Total tokens and cost of a run. Cost comes from the provider's own accounting when every
    model reports it, otherwise from ``prices`` (all input tokens: uncached, cache writes and cache
    reads, each at its own price); ``None`` when neither is available."""
    per_model = dict(log.stats.model_usage or {}) if log.stats else {}
    if not per_model:
        return None, None
    tokens = sum(u.total_tokens for u in per_model.values())
    if all(getattr(u, "total_cost", None) is not None for u in per_model.values()):
        return tokens, float(sum(u.total_cost or 0.0 for u in per_model.values()))
    if prices is None or not all(m in prices for m in per_model):
        return tokens, None
    cost = 0.0
    for name, u in per_model.items():
        p = prices[name]
        cost += u.input_tokens * p.input + u.output_tokens * p.output
        cost += (u.input_tokens_cache_read or 0) * (p.input if p.cache_read is None else p.cache_read)
        cost += (u.input_tokens_cache_write or 0) * (p.input if p.cache_write is None else p.cache_write)
    return tokens, cost


def close_scorecard(
    log: EvalLog,
    *,
    agent: str | None = None,
    prices: Prices | None = None,
    service: ScorecardService | None = None,
    record: Path | None = None,
) -> Scorecard:
    """Close the scorecard a task run opened, recording model, harness, tokens and cost.

    In the process that ran the eval the scorecard is still held here; elsewhere pass a
    ``service`` rebuilt from the same store, ledger and archive (it restores open scorecards).
    With ``record`` (in the process that ran the eval), write the run directory: the recording,
    the server trace and the scorecard, as ``onc-agi play --record`` writes them.
    """
    sid = str((log.eval.metadata or {})["scorecard_id"])
    tokens, cost = usage(log, prices)
    if service is not None:
        if record is not None:
            raise ValueError("a run is recorded by the process that ran it")
        card = service.close(sid, model=log.eval.model, harness=HARNESS, tokens=tokens, cost_usd=cost)
        return card if agent is None else card.model_copy(update={"agent": agent})
    run = _RUNS[sid]
    try:
        if isinstance(run.arena, LocalArena):
            card = run.arena.close(sid, model=log.eval.model, harness=HARNESS, tokens=tokens, cost_usd=cost)
        else:  # a server's close takes no client claims; they annotate the returned copy
            labels = {"model": log.eval.model, "harness": HARNESS, "tokens": tokens, "cost_usd": cost}
            card = run.arena.close(sid).model_copy(update={k: v for k, v in labels.items() if v is not None})
        if agent is not None:
            card = card.model_copy(update={"agent": agent})
        if record is not None:
            _record(record, run, log, card)
    finally:
        _RUNS.pop(sid, None)
        run.shutdown()
    return card


def _with_actions(
    events: Sequence[RecordingEvent], actions: Sequence[tuple[str, str | None]]
) -> list[RecordingEvent]:
    """One world's Inspect transcript with the actions the arena applied placed where they happened:
    the reset first, each recruit or assay after its tool call, the harness's submit last."""
    if not events and not actions:
        return []
    wid = events[0].world_id if events else ""
    base = events[0] if events else None
    queue = list(actions)
    out: list[RecordingEvent] = []

    def kind(item: tuple[str, str | None]) -> str:
        return str(json.loads(item[0]).get("kind"))

    def emit() -> None:
        action, refusal = queue.pop(0)
        content = action if refusal is None else f"{action}\nrefused: {refusal}"
        out.append(_event(base, wid, "action", content, error=refusal is not None))

    if queue and kind(queue[0]) == "reset":
        emit()
    for event in events:
        out.append(event)
        if (
            event.kind == "tool_call"
            and event.tool in ("recruit", "assay")
            and queue
            and kind(queue[0]) == event.tool
        ):
            emit()
    while queue:
        emit()
    return [e.model_copy(update={"turn": i}) for i, e in enumerate(out)]


def _event(
    base: RecordingEvent | None, world_id: str, kind: Any, content: str, *, error: bool
) -> RecordingEvent:
    return RecordingEvent(
        scorecard_id=base.scorecard_id if base else None,
        world_id=world_id,
        turn=0,
        kind=kind,
        content=content,
        error=error,
        ts=base.ts if base else datetime.now(UTC),
    )


def _record(directory: Path, run: _Run, log: EvalLog, card: Scorecard) -> None:
    """Write the run directory: recording (transcript and applied actions), server trace, scorecard."""
    events, runs = transcript(log)
    header = RecordingHeader(
        scorecard_id=run.scorecard_id,
        agent=card.agent,
        tier=run.tier,
        track="standard",
        world_ids=tuple(run.cards),
        model=log.eval.model,
        harness=HARNESS,
        opened_at=run.opened_at,
    )
    by_world: dict[str, list[RecordingEvent]] = {}
    for event in events:
        by_world.setdefault(event.world_id, []).append(event)
    with RecordingWriter(directory / RECORDING_FILE) as writer:
        writer.header(header)
        for wid in run.cards:
            for event in _with_actions(by_world.get(wid, []), run.actions.get(wid, [])):
                writer.append(event.model_copy(update={"scorecard_id": run.scorecard_id, "world_id": wid}))
        for record in sorted(runs, key=lambda r: list(run.cards).index(r.world_id)):
            writer.run(record)
    save_run(directory, card, server_trace(run.arena, run.scorecard_id))


class StandardRun(NamedTuple):
    scorecard: Scorecard
    log_path: Path
    record: Path | None


def run_standard(
    *,
    mode: Mode,
    model: Any,
    record: Path | None = None,
    log_dir: Path = Path("logs"),
    model_config: Mapping[str, Any] | None = None,
    prices: Prices | None = None,
    agent: str | None = None,
    max_samples: int = 4,
    **task_args: Any,
) -> StandardRun:
    """One standard-track scorecard end to end: run the Inspect task for ``mode`` with ``model``
    (an Inspect model or model string), close the scorecard and, with ``record``, write the run
    directory. ``task_args`` are the task's parameters (``store_root`` or ``url``, ``tier``,
    ``n_worlds`` or ``world_ids``, ``seed``, ``message_limit``, ...); ``model_config`` holds
    Inspect ``GenerateConfig`` fields (:meth:`Profile.inspect_config`)."""
    from inspect_ai import eval as inspect_eval

    make = arena_full_access if mode is Mode.FULL_ACCESS else arena_sequential
    if agent is not None:
        task_args.setdefault("agent", agent)
    logs = inspect_eval(
        make(**task_args),
        model=model,
        log_dir=str(log_dir),
        max_samples=max_samples,
        display="none",
        **dict(model_config or {}),
    )
    log = logs[0]
    if log.status != "success":
        sid = (log.eval.metadata or {}).get("scorecard_id")
        run = _RUNS.pop(str(sid), None)
        if run is not None:
            run.shutdown()
        raise RuntimeError(f"the Inspect run ended with status {log.status}: {log.error}")
    card = close_scorecard(log, agent=agent, prices=prices, record=record)
    return StandardRun(card, Path(log.location), record)


def log_to_scorecard(log: EvalLog, *, agent: str, tier: Tier, prices: Prices | None = None) -> Scorecard:
    """Standard-track scorecard from an Inspect log alone (no scorecard service), with the model,
    tokens and cost recorded. Public train only: eval logs carry no per-world results."""
    scores: list[WorldScore] = []
    for sample in log.samples or []:
        for s in (sample.scores or {}).values():
            if s.metadata and "world_score" in s.metadata:
                scores.append(WorldScore.model_validate(s.metadata["world_score"]))
    tokens, cost = usage(log, prices)
    return scoring.aggregate(
        scores,
        scorecard_id=f"inspect-{log.eval.run_id}",
        agent=agent,
        tier=tier,
        track="standard",
        model=log.eval.model,
        harness=HARNESS,
        tokens=tokens,
        cost_usd=cost,
    )
