"""Inspect AI standard harness.

One task per mode. Every model sees the same premise and task card, the same
tools and the same limits:

* ``python`` / ``bash`` run inside a Docker sandbox with **no network**; the
  revealed data are written there as ``/data/revealed.csv``;
* ``recruit`` / ``assay`` (sequential mode) and ``submit`` run on the **host**,
  where the episode and answer keys live, so nothing scorer-side ever enters the
  sandbox;
* a safety timeout per world; tokens and cost are reported, not capped.

The scorer reuses the arena scorer, and :func:`log_to_scorecard` converts an
Inspect log into a standard-track :class:`Scorecard` plus shared-schema traces.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

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
from onc_agi.infra.bundles import FileWorldStore
from onc_agi.services import scoring
from onc_agi.services.engine import Episode, EpisodeView

SANDBOX_COMPOSE = Path(__file__).parent / "sandbox" / "compose.yaml"
HARNESS = "inspect-standard-1.0"
SAFETY_TIMEOUT_SECONDS = 1800

_EPISODES: dict[str, Episode] = {}  # host-side state, keyed by Inspect sample uuid


def card_text(card: WorldCard) -> str:
    lines = [
        card.premise,
        "",
        f"World {card.world_id}: {card.n_pool} patients, binary outcome, mode {card.mode.value}.",
        (
            f"Budget {card.budget:.0f} USD; recruiting costs {card.prices.recruit_per_patient:.0f} USD per patient."
            if card.mode is Mode.SEQUENTIAL
            else "All data are already in /data/revealed.csv (columns: patient_id, stratum, outcome, then features)."
        ),
        "Features (id, data type, timing, assay price per patient):",
    ]
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
    frame.insert(0, "outcome", view.outcome)
    frame.insert(0, "stratum", list(view.stratum))
    frame.insert(0, "patient_id", [f"r{i}" for i in range(len(view.rows))])
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue()


async def _sync(view: EpisodeView) -> None:
    await sandbox().write_file("/data/revealed.csv", _revealed_csv(view))


def _episode() -> Episode:
    return _EPISODES[str(store().get("episode_key"))]


def _apply(episode: Episode, action: Action) -> EpisodeView:
    """Apply a tool action; a refused action returns to the model as a tool error."""
    try:
        return episode.apply(action)
    except ArenaError as exc:
        raise ToolError(f"{exc.code.value}: {exc.message}") from exc


@solver
def setup_world(store_root: str) -> Solver:
    worlds = FileWorldStore(Path(store_root))

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        world_id = str(state.metadata["world_id"])
        episode = Episode(worlds.world(world_id))
        key = f"{state.uuid}"
        _EPISODES[key] = episode
        store().set("episode_key", key)
        view = episode.apply(Reset(request_id="harness-reset", world_id=world_id))
        await _sync(view)
        return state

    return solve


@tool
def recruit() -> Tool:
    async def execute(count: int, stratum: str = "all") -> str:
        """Recruit patients (their outcome and stratum are revealed).

        Args:
            count: Number of patients to recruit from the stratum's queue.
            stratum: Stratum to recruit from.
        """
        episode = _episode()
        view = _apply(episode, Recruit(request_id=f"recruit-{episode.step}", count=count, stratum=stratum))
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
        episode = _episode()
        ids = tuple(f.strip() for f in feature_ids.split(",") if f.strip())
        view = _apply(episode, Assay(request_id=f"assay-{episode.step}", feature_ids=ids))
        await _sync(view)
        return f"Measured {len(ids)} features; spent {view.spent:.0f} of {view.budget:.0f} USD."

    return execute


def _ranking(answer: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(f.strip() for f in answer.replace("\n", ",").split(",") if f.strip()))


@scorer(metrics=[mean()])
def arena_scorer(store_root: str, keys_dir: str | None = None) -> Any:
    worlds = FileWorldStore(Path(store_root), Path(keys_dir) if keys_dir else None)

    async def score(state: TaskState, target: Target) -> Score:
        world_id = str(state.metadata["world_id"])
        # the episode is finished once scored; drop it so host state does not grow with the eval
        episode = _EPISODES.pop(state.uuid, None) or Episode(worlds.world(world_id))
        answer = state.output.completion if state.output else ""
        valid = set(worlds.card(world_id).feature_ids())
        ranking = tuple(f for f in _ranking(answer) if f in valid)
        if episode.submission is None:
            episode.apply(Submit(request_id="harness-submit", ranking=ranking))
        world_score = scoring.score_world(
            episode.submission or (),
            worlds.answer_key(world_id),
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
    store_root: str, tier: str, mode: Mode, n_worlds: int | None, keys_dir: str | None, message_limit: int
) -> Task:
    worlds = FileWorldStore(Path(store_root), Path(keys_dir) if keys_dir else None)
    ids = [w for w in worlds.world_ids(Tier(tier)) if worlds.card(w).mode is mode][: n_worlds or None]
    samples = [Sample(input=card_text(worlds.card(w)), metadata={"world_id": w}, id=w) for w in ids]
    tools: list[Tool] = [python(timeout=180), bash(timeout=180)]
    if mode is Mode.SEQUENTIAL:
        tools += [recruit(), assay()]
    return Task(
        dataset=samples,
        solver=basic_agent(
            init=setup_world(store_root),
            tools=tools,
            message_limit=message_limit,
            submit_description="Submit the ordered, comma-separated feature ids (or an empty string to abstain).",
        ),
        scorer=arena_scorer(store_root, keys_dir),
        sandbox=("docker", str(SANDBOX_COMPOSE)),
        time_limit=SAFETY_TIMEOUT_SECONDS,
        metadata={"harness": HARNESS, "tier": tier, "mode": mode.value},
    )


@task
def arena_full_access(
    store_root: str,
    tier: str = "public_train",
    n_worlds: int | None = None,
    keys_dir: str | None = None,
    message_limit: int = 60,
) -> Task:
    return _task(store_root, tier, Mode.FULL_ACCESS, n_worlds, keys_dir, message_limit)


@task
def arena_sequential(
    store_root: str,
    tier: str = "public_train",
    n_worlds: int | None = None,
    keys_dir: str | None = None,
    message_limit: int = 80,
) -> Task:
    return _task(store_root, tier, Mode.SEQUENTIAL, n_worlds, keys_dir, message_limit)


def log_to_scorecard(log: EvalLog, *, agent: str, tier: Tier) -> Scorecard:
    """Standard-track scorecard from an Inspect log, with token usage and the model recorded."""
    scores: list[WorldScore] = []
    for sample in log.samples or []:
        for s in (sample.scores or {}).values():
            if s.metadata and "world_score" in s.metadata:
                scores.append(WorldScore.model_validate(s.metadata["world_score"]))
    tokens = sum(u.total_tokens for u in (log.stats.model_usage or {}).values()) if log.stats else None
    return scoring.aggregate(
        scores,
        scorecard_id=f"inspect-{log.eval.run_id}",
        agent=agent,
        tier=tier,
        track="standard",
        model=log.eval.model,
        harness=HARNESS,
        tokens=tokens,
    )
