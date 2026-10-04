"""Read an Inspect log of the standard harness as recording events and run records.

The standard track's transcript in the shape of every recording (``infra/recordings.py``), so
one explanation path and one run-directory format serve both tracks. Needs the optional
``inspect`` extra; imported only when an Inspect log is read.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from onc_agi.core.ports import RecordingEvent, RecordingKind, RunRecord

if TYPE_CHECKING:
    from inspect_ai.log import EvalLog


def from_inspect_log(path: Path) -> tuple[list[RecordingEvent], list[RunRecord]]:
    try:
        from inspect_ai.log import read_eval_log
    except ImportError as exc:
        raise ImportError("explaining an Inspect log needs: pip install 'onc-agi[inspect]'") from exc
    return transcript(read_eval_log(str(path)))


def transcript(log: EvalLog) -> tuple[list[RecordingEvent], list[RunRecord]]:
    """Each sample's messages as recording events (reasoning, text, tool calls and results, in order)
    and one run record per world (the scored answer, spend, tokens and cost)."""
    sid = (log.eval.metadata or {}).get("scorecard_id")
    events: list[RecordingEvent] = []
    runs: list[RunRecord] = []
    for sample in log.samples or []:
        wid = str(sample.metadata["world_id"])
        said: list[tuple[RecordingKind, str, str | None, bool]] = []
        for message in sample.messages:
            if message.role == "assistant":
                for part in message.content if isinstance(message.content, list) else []:
                    if part.type == "reasoning" and not part.redacted and (part.summary or part.reasoning):
                        said.append(("reasoning", str(part.summary or part.reasoning), None, False))
                if message.text:
                    said.append(("assistant", message.text, None, False))
                for call in message.tool_calls or []:
                    args: dict[str, Any] = dict(call.arguments or {})
                    body = str(args.get("code") or args.get("cmd") or json.dumps(args))
                    said.append(("tool_call", body, call.function, False))
            elif message.role == "tool":
                error = getattr(message, "error", None)
                text = message.text + (f"\n{error.message}" if error else "")
                said.append(("tool_result", text, getattr(message, "function", None), bool(error)))
        stamp = datetime.now(UTC)  # Inspect messages carry no timestamps of their own
        events += [
            RecordingEvent(
                scorecard_id=sid, world_id=wid, turn=i, kind=k, content=c, tool=t, error=e, ts=stamp
            )
            for i, (k, c, t, e) in enumerate(said)
        ]
        score = next(iter((sample.scores or {}).values()), None)
        answer = str(score.answer) if score and score.answer else ""
        metadata = (score.metadata or {}) if score else {}
        world_score = metadata.get("world_score")
        spent = metadata.get("spent", (world_score or {}).get("spent", 0.0))
        usages = list((sample.model_usage or {}).values())
        tokens = sum(u.total_tokens for u in usages) or None
        costs = [getattr(u, "total_cost", None) for u in usages]
        runs.append(
            RunRecord(
                scorecard_id=sid,
                world_id=wid,
                ranking=tuple(f for f in answer.split(",") if f),
                submitted=score is not None,  # the harness's scorer submits the final answer
                spent=float(spent),
                tokens=tokens,
                cost_usd=float(sum(c or 0.0 for c in costs)) if usages and None not in costs else None,
                model=log.eval.model,
                error=(
                    None
                    if score is not None
                    else f"no score: {sample.error.message if sample.error else 'the sample ended unscored'}"
                ),
            )
        )
    return events, runs
