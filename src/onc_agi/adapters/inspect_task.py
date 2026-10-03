"""Inspect AI standard harness.

One task per mode. Every model sees the same premise and task card, the same
tools and the same limits:

* ``python`` / ``bash`` run inside a Docker sandbox with **no network**; the
  revealed data are written there as ``/data/revealed.csv``;
* ``recruit`` / ``assay`` (sequential mode) and ``submit`` run on the **host**,
  where the episode and answer keys live, so nothing scorer-side ever enters the
  sandbox;
* a safety timeout per world; tokens and cost are reported, not capped.

Each task run is one arena scorecard, opened through :class:`ScorecardService`
exactly as the HTTP interface opens one: the same fresh never-reused draws and
caps on eval tiers, the same server-side traces, and the same close.
Eval tiers therefore need the ledger the server uses (``ledger=``), so a world
drawn here is never drawn again there. Per-world results are attached to the
log only on public train. :func:`close_scorecard` closes the run's scorecard
after the eval; :func:`log_to_scorecard` remains for logs made without one.
Any OpenAI-compatible endpoint works through Inspect's ``openai-api/<name>/<model>``
provider (``<NAME>_BASE_URL`` and ``<NAME>_API_KEY`` in the environment).
"""

from __future__ import annotations

import hashlib
import io
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
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

from onc_agi.core.errors import ArenaError
from onc_agi.core.schema import (
    Action,
    Assay,
    Mode,
    Recruit,
    Reset,
    Scorecard,
    Submit,
    Tier,
    WorldCard,
    WorldScore,
)
from onc_agi.infra.archive import FileScorecardArchive
from onc_agi.infra.bundles import FileWorldStore
from onc_agi.infra.ledger import JsonLedger
from onc_agi.services import scoring
from onc_agi.services.engine import EpisodeView
from onc_agi.services.scorecards import MIN_EVAL_WORLDS, ScorecardService

SANDBOX_COMPOSE = Path(__file__).parent / "sandbox" / "compose.yaml"


def _harness_label() -> str:
    """``inspect-standard-1.1+<8 hex>`` over this module and the sandbox image: what every model sees
    and can do. A card, prompt, tool or image change changes the label, as the scorer's label does.
    """
    h = hashlib.sha256(Path(__file__).read_bytes())
    h.update((Path(__file__).parent / "sandbox" / "Dockerfile").read_bytes())
    return f"inspect-standard-1.1+{h.hexdigest()[:8]}"


HARNESS = _harness_label()
SAFETY_TIMEOUT_SECONDS = 1800
OPERATOR_KEY = "inspect-standard"  # caller identity for caps; the harness runs operator-side


@dataclass(frozen=True)
class _Run:
    service: ScorecardService
    scorecard_id: str
    tier: Tier


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
        return run.service.act(run.scorecard_id, world_id, action)
    except ArenaError as exc:
        raise ToolError(f"{exc.code.value}: {exc.message}") from exc


def _step() -> int:
    run, world_id = _here()
    return run.service.episode(run.scorecard_id, world_id).step


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


def _ranking(answer: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(f.strip() for f in answer.replace("\n", ",").split(",") if f.strip()))


@scorer(metrics=[mean()])
def arena_scorer() -> Any:
    async def score(state: TaskState, target: Target) -> Score:
        run = _RUNS[str(state.metadata["scorecard_id"])]
        world_id = str(state.metadata["world_id"])
        episode = run.service.episode(run.scorecard_id, world_id)
        answer = state.output.completion if state.output else ""
        valid = set(episode.world.card.feature_ids())
        ranking = tuple(f for f in _ranking(answer) if f in valid)
        if episode.submission is None:
            run.service.act(run.scorecard_id, world_id, Submit(request_id="harness-submit", ranking=ranking))
        if run.tier is not Tier.PUBLIC_TRAIN:  # eval results exist only as the closed scorecard
            return Score(value=0.0, answer=",".join(ranking), metadata={"exposure": "scorecard only"})
        world_score = scoring.score_world(
            episode.submission or (),
            run.service.store.answer_key(world_id),
            spent=episode.spent,
            sequential=episode.mode is Mode.SEQUENTIAL,
        )
        value = float(world_score.restrained) if world_score.is_null else world_score.find
        return Score(
            value=value,
            answer=",".join(ranking),
            metadata={"world_score": world_score.model_dump(mode="json")},
        )

    return score


def _task(
    store_root: str,
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
) -> Task:
    worlds = FileWorldStore(Path(store_root), Path(keys_dir) if keys_dir else None)
    tier_ = Tier(tier)
    if tier_ is not Tier.PUBLIC_TRAIN and (ledger is None or n_worlds is None):
        raise ValueError(
            f"{tier} runs draw fresh worlds: pass the server's ledger= (and archive=) and n_worlds="
        )
    if n_worlds is None:
        n_worlds = sum(1 for w in worlds.world_ids(tier_) if worlds.card(w).mode is mode)
    service = ScorecardService(
        worlds,
        JsonLedger(
            Path(ledger) if ledger else Path(tempfile.mkdtemp(prefix="arena-inspect-")) / "ledger.json"
        ),
        archive=FileScorecardArchive(Path(archive)) if archive else None,
        min_eval_worlds=min_eval_worlds,
    )
    sid: str | None = None
    try:
        sid, cards = service.open(
            api_key, agent=agent, track="standard", tier=tier_, n_worlds=n_worlds, mode=mode
        )
        _RUNS[sid] = _Run(service, sid, tier_)
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
            sandbox=("docker", str(SANDBOX_COMPOSE)),
            time_limit=SAFETY_TIMEOUT_SECONDS,
            epochs=1,  # one episode per world per scorecard; repeat by opening another scorecard
            metadata={"harness": HARNESS, "tier": tier, "mode": mode.value, "scorecard_id": sid},
        )
    except BaseException:
        if sid is not None:
            _RUNS.pop(sid, None)
        service.shutdown()
        raise


@task
def arena_full_access(
    store_root: str,
    tier: str = "public_train",
    n_worlds: int | None = None,
    keys_dir: str | None = None,
    message_limit: int = 60,
    ledger: str | None = None,
    archive: str | None = None,
    agent: str = "inspect-standard",
    api_key: str = OPERATOR_KEY,
    min_eval_worlds: int = MIN_EVAL_WORLDS,
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
    )


@task
def arena_sequential(
    store_root: str,
    tier: str = "public_train",
    n_worlds: int | None = None,
    keys_dir: str | None = None,
    message_limit: int = 80,
    ledger: str | None = None,
    archive: str | None = None,
    agent: str = "inspect-standard",
    api_key: str = OPERATOR_KEY,
    min_eval_worlds: int = MIN_EVAL_WORLDS,
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
    )


class Price(NamedTuple):
    """USD per token. Cached input is often billed differently; unset cache prices fall back to ``input``."""

    input: float
    output: float
    cache_read: float | None = None
    cache_write: float | None = None


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
) -> Scorecard:
    """Close the scorecard a task run opened, recording model, harness, tokens and cost.

    In the process that ran the eval the scorecard is still held here; elsewhere pass a
    ``service`` rebuilt from the same store, ledger and archive (it restores open scorecards).
    """
    sid = str((log.eval.metadata or {})["scorecard_id"])
    svc = service or _RUNS[sid].service
    try:
        tokens, cost = usage(log, prices)
        card = svc.close(sid, model=log.eval.model, harness=HARNESS, tokens=tokens, cost_usd=cost)
    finally:
        _RUNS.pop(sid, None)
        if service is None:
            svc.shutdown()
    return card if agent is None else card.model_copy(update={"agent": agent})


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
