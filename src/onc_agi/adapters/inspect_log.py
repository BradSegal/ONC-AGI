"""Read an Inspect log of the standard harness as recording events and run records (for ``explain``).

Needs the optional ``inspect`` extra; imported only when an Inspect log is explained.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from onc_agi.core.ports import RecordingEvent, RecordingKind, RunRecord


def from_inspect_log(path: Path) -> tuple[list[RecordingEvent], list[RunRecord]]:
    try:
        from inspect_ai.log import read_eval_log
    except ImportError as exc:
        raise ImportError("explaining an Inspect log needs: pip install 'onc-agi[inspect]'") from exc

    log = read_eval_log(str(path))
    sid = (log.eval.metadata or {}).get("scorecard_id")
    events: list[RecordingEvent] = []
    runs: list[RunRecord] = []
    for sample in log.samples or []:
        wid = str(sample.metadata["world_id"])
        said: list[tuple[RecordingKind, str, str | None, bool]] = []
        called_submit = False
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
                    called_submit |= call.function == "submit"
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
        world_score = (score.metadata or {}).get("world_score") if score else None
        tokens = sum(u.total_tokens for u in (sample.model_usage or {}).values()) or None
        runs.append(
            RunRecord(
                scorecard_id=sid,
                world_id=wid,
                ranking=tuple(f for f in answer.split(",") if f),
                # the harness's scorer submits the final answer even without a submit call
                submitted=called_submit or bool(answer),
                spent=float((world_score or {}).get("spent", 0.0)),
                tokens=tokens,
                model=log.eval.model,
            )
        )
    return events, runs
