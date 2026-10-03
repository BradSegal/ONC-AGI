"""``arena`` command line: smoke, evaluate, serve, replay and conformance."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from importlib import resources
from pathlib import Path

from onc_agi.adapters.agents import AGENT_NAMES, CHEATERS, make_agent
from onc_agi.core.schema import Scorecard, Tier, TraceEvent
from onc_agi.infra.bundles import FileWorldStore
from onc_agi.infra.recorder import TraceRecorder
from onc_agi.services.kit import evaluate
from onc_agi.services.replay import replay

SMOKE_AGENTS = (
    "oracle",
    "univariate_bh",
    "seq_univariate_bh",
    "stability",
    "random",
    "giant_list",
    "always_empty",
    "leak_exploiter",
    "random_abstain",
)
FLOOR_MARGIN = 0.02


def fixture_store() -> Path:
    return Path(str(resources.files("onc_agi") / "fixtures" / "store"))


def _fmt(value: float | None, spec: str) -> str:
    return "  n/a " if value is None else format(value, spec)


def _row(card: Scorecard) -> str:
    return (
        f"{card.agent:18s} DS {_fmt(card.discovery_score, '6.3f')}  "
        f"[{_fmt(card.interval.low, '+.2f')},{_fmt(card.interval.high, '+.2f')}]  "
        f"Find {_fmt(card.find, '5.2f')}  Restraint {_fmt(card.restraint, '+5.2f')}  "
        f"Strict {_fmt(card.strict_discovery_score, '5.2f')}  Leak {card.leak_rate:4.2f}"
    )


def cmd_smoke(args: argparse.Namespace) -> int:
    store = FileWorldStore(Path(args.store) if args.store else fixture_store())
    if not store.world_ids(Tier.PUBLIC_TRAIN):
        print("smoke: no public-train fixture worlds found", file=sys.stderr)
        return 2
    failures: list[str] = []
    print(f"ONC-AGI smoke - {len(store.world_ids(Tier.PUBLIC_TRAIN))} public-train worlds")
    for name in SMOKE_AGENTS:
        card, _ = evaluate(make_agent(name, store), store, Tier.PUBLIC_TRAIN, bootstrap_draws=200)
        print(_row(card))
        if card.discovery_score is None or card.discovery_score_unfloored is None:
            failures.append(f"{name}: score undefined (fixtures need signal and null worlds)")
        elif name == "oracle" and card.discovery_score < 0.99:
            failures.append("oracle below 0.99")
        elif (name in CHEATERS or name == "random") and card.discovery_score_unfloored > FLOOR_MARGIN:
            failures.append(f"{name} above the floor")
    if failures:
        print("smoke FAILED: " + "; ".join(failures), file=sys.stderr)
        return 1
    print("smoke passed")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    store = FileWorldStore(Path(args.store), Path(args.keys) if args.keys else None)
    tier = Tier(args.tier)
    ids = store.world_ids(tier)[: args.n] if args.n else None
    recorder = TraceRecorder(Path(args.trace)) if args.trace else None
    card, _ = evaluate(make_agent(args.agent, store), store, tier, world_ids=ids, recorder=recorder)
    print(_row(card))
    if args.json:
        Path(args.json).write_text(card.model_dump_json(indent=1))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from onc_agi.adapters.http import create_app
    from onc_agi.infra.ledger import JsonLedger
    from onc_agi.services.scorecards import ScorecardService

    store = FileWorldStore(Path(args.store), Path(args.keys) if args.keys else None)
    allowed = (
        frozenset(line.strip() for line in Path(args.api_keys).read_text().splitlines() if line.strip())
        if args.api_keys
        else None
    )
    trace_dir = Path(args.ledger).with_suffix(".traces")
    recorders: dict[str, TraceRecorder] = {}

    def sink(sid: str) -> TraceRecorder:
        return recorders.setdefault(sid, TraceRecorder(trace_dir / f"{sid}.jsonl"))

    service = ScorecardService(store, JsonLedger(Path(args.ledger)), traces=sink, allowed_keys=allowed)
    uvicorn.run(create_app(service), host=args.host, port=args.port)
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    store = FileWorldStore(Path(args.store), Path(args.keys) if args.keys else None)
    events = TraceRecorder(Path(args.trace)).events()
    by_world: dict[str, list[TraceEvent]] = {}
    for event in events:
        by_world.setdefault(event.world_id, []).append(event)
    bad = 0
    for world_id, evs in by_world.items():
        report = replay(evs, store.world(world_id), store.answer_key(world_id))
        status = "ok" if report.ok else "MISMATCH " + "; ".join(report.mismatches)
        find = f"find={report.score.find:.3f}" if report.score else ""
        print(f"{world_id}: {report.steps} steps {find} {status}")
        bad += not report.ok
    return 1 if bad else 0


def cmd_conformance(args: argparse.Namespace) -> int:
    from fastapi.testclient import TestClient

    from onc_agi.adapters.client import ArenaClient
    from onc_agi.adapters.conformance import run_conformance
    from onc_agi.adapters.http import create_app
    from onc_agi.infra.ledger import JsonLedger
    from onc_agi.services.scorecards import ScorecardService

    store = FileWorldStore(Path(args.store) if args.store else fixture_store())
    if args.url:
        client = ArenaClient(args.url, args.key)
    else:
        trace_dir = Path(args.ledger).with_suffix(".traces")
        recorders: dict[str, TraceRecorder] = {}

        def sink(sid: str) -> TraceRecorder:
            return recorders.setdefault(sid, TraceRecorder(trace_dir / f"{sid}.jsonl"))

        app = create_app(ScorecardService(store, JsonLedger(Path(args.ledger)), traces=sink))
        client = ArenaClient("http://testserver", args.key, client=TestClient(app))
        report = run_conformance(client, store, server_traces=lambda sid: sink(sid).events())
        print(json.dumps({"ok": report.ok, "checks": report.checks, "details": report.details}, indent=1))
        return 0 if report.ok else 1
    report = run_conformance(client, store)
    print(json.dumps({"ok": report.ok, "checks": report.checks, "details": report.details}, indent=1))
    return 0 if report.ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="onc-agi", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("smoke", help="run reference, baseline and cheater agents on fixture worlds")
    p.add_argument("--store")
    p.set_defaults(func=cmd_smoke)
    p = sub.add_parser("evaluate", help="evaluate one agent on a tier")
    p.add_argument(
        "--agent", required=True, choices=sorted(set(AGENT_NAMES) | {f"seq_{b}" for b in AGENT_NAMES})
    )
    p.add_argument("--store", required=True)
    p.add_argument("--keys")
    p.add_argument("--tier", default="public_train", choices=[t.value for t in Tier])
    p.add_argument("--n", type=int)
    p.add_argument("--trace")
    p.add_argument("--json")
    p.set_defaults(func=cmd_evaluate)
    p = sub.add_parser("serve", help="serve the HTTP interface")
    p.add_argument("--store", required=True)
    p.add_argument("--keys")
    p.add_argument("--ledger", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument(
        "--api-keys", help="file of issued API keys, one per line (required for eval/private serving)"
    )
    p.set_defaults(func=cmd_serve)
    p = sub.add_parser("replay", help="verify a trace against its world bundles")
    p.add_argument("--trace", required=True)
    p.add_argument("--store", required=True)
    p.add_argument("--keys")
    p.set_defaults(func=cmd_replay)
    p = sub.add_parser("conformance", help="run the custom-harness conformance suite")
    p.add_argument("--store")
    p.add_argument("--url")
    p.add_argument("--key", default="conformance-key")
    p.add_argument("--ledger", default=".arena-conformance-ledger.json")
    p.set_defaults(func=cmd_conformance)
    args = parser.parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
