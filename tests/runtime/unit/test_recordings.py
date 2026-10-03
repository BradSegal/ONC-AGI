"""Agent recordings: JSONL round trip, per-world turns, thread safety and fail-fast reading."""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest
from onc_agi.core.ports import RecordingHeader, RunRecord
from onc_agi.core.schema import Tier
from onc_agi.infra.recordings import RECORDING_FILE, RecordingWriter, read_recording

HEADER = RecordingHeader(
    scorecard_id="sc-1",
    agent="a",
    tier=Tier.PUBLIC_TRAIN,
    track="open",
    world_ids=("w-0", "w-1"),
    model="m",
    opened_at=datetime(2026, 10, 3, tzinfo=UTC),
)


def test_a_recording_round_trips(tmp_path: Path) -> None:
    with RecordingWriter(tmp_path / RECORDING_FILE) as rec:
        rec.header(HEADER)
        rec.event("w-0", "assistant", "Screen first.")
        rec.event("w-0", "tool_call", "print(1)", tool="python")
        rec.event("w-1", "reasoning", "Null?")
        rec.event("w-0", "tool_result", "boom", tool="python", error=True)
        rec.run(RunRecord(world_id="w-0", ranking=("f1",), submitted=True, spent=3.0, tokens=9, cost_usd=0.1))
    data = read_recording(tmp_path)  # a record directory resolves to its recording.jsonl
    assert data.header == HEADER
    assert [(e.world_id, e.turn, e.kind) for e in data.events] == [
        ("w-0", 0, "assistant"),
        ("w-0", 1, "tool_call"),
        ("w-1", 0, "reasoning"),
        ("w-0", 2, "tool_result"),
    ]
    assert all(e.scorecard_id == "sc-1" for e in data.events) and data.events[3].error
    (run,) = data.runs
    assert run.scorecard_id == "sc-1" and run.ranking == ("f1",) and run.cost_usd == 0.1


def test_concurrent_writers_keep_every_line_whole_and_turns_dense(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    with RecordingWriter(path) as rec:
        rec.header(HEADER)

        def write(w: str) -> None:
            for i in range(200):
                rec.event(w, "note", f"{w}-{i}" * 20)

        threads = [threading.Thread(target=write, args=(f"w-{k}",)) for k in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    data = read_recording(path)
    assert len(data.events) == 1200
    for k in range(6):
        turns = [e.turn for e in data.events if e.world_id == f"w-{k}"]
        assert turns == list(range(200))


def test_a_recording_is_never_appended_to(tmp_path: Path) -> None:
    RecordingWriter(tmp_path / "r.jsonl").close()
    with pytest.raises(FileExistsError, match="already exists"):
        RecordingWriter(tmp_path / "r.jsonl")


def test_one_recording_holds_one_scorecard(tmp_path: Path) -> None:
    with RecordingWriter(tmp_path / "r.jsonl") as rec:
        rec.header(HEADER)
        with pytest.raises(RuntimeError, match="exactly one"):
            rec.header(HEADER)
    path = tmp_path / "two.jsonl"
    line = HEADER.model_dump_json()
    path.write_text(line + "\n" + line + "\n")
    with pytest.raises(ValueError, match="exactly one scorecard"):
        read_recording(path)


def test_a_malformed_line_fails_with_its_location(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(HEADER.model_dump_json() + '\n{"kind": "telepathy"}\n')
    with pytest.raises(ValueError, match=r"bad.jsonl:2"):
        read_recording(path)


def test_writing_after_close_fails(tmp_path: Path) -> None:
    rec = RecordingWriter(tmp_path / "r.jsonl")
    rec.close()
    with pytest.raises(RuntimeError, match="closed"):
        rec.event("w", "note", "late")
