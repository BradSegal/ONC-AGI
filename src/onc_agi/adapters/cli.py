"""``onc-agi`` command line: smoke, evaluate, serve, replay, conformance, and the agents kit (play, worlds, explain)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from datetime import timedelta
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import httpx

from onc_agi.adapters.agents import AGENT_NAMES, BASELINES, CHEATERS, AgentFactory, make_agent
from onc_agi.core.errors import ArenaError
from onc_agi.core.ports import RecordingEvent, RunRecord
from onc_agi.core.schema import Mode, Scorecard, Tier, TraceEvent, WorldCard
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


def _archive_dir(args: argparse.Namespace) -> Path:
    return Path(args.archive) if args.archive else Path(args.ledger).with_suffix(".archive")


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from onc_agi.adapters.http import create_app
    from onc_agi.infra.archive import FileScorecardArchive
    from onc_agi.infra.ledger import JsonLedger
    from onc_agi.services.scorecards import ScorecardService

    if args.ttl_hours < 0:
        print("serve: --ttl-hours must be >= 0 (0 disables expiry)", file=sys.stderr)
        return 2
    store = FileWorldStore(Path(args.store), Path(args.keys) if args.keys else None)
    eval_tiers = [t.value for t in (Tier.PUBLIC_EVAL, Tier.PRIVATE) if store.world_ids(t)]
    if eval_tiers and not args.api_keys and not args.allow_unissued_keys:
        # per-key caps bound nothing when any invented key mints fresh caps
        print(
            f"serve: the store holds {', '.join(eval_tiers)} worlds; pass --api-keys "
            "(or --allow-unissued-keys for development)",
            file=sys.stderr,
        )
        return 2
    allowed = (
        frozenset(line.strip() for line in Path(args.api_keys).read_text().splitlines() if line.strip())
        if args.api_keys
        else None
    )
    service = ScorecardService(
        store,
        JsonLedger(Path(args.ledger)),
        allowed_keys=allowed,
        archive=FileScorecardArchive(_archive_dir(args)),  # open records, traces, closed scorecards
        ttl=timedelta(hours=args.ttl_hours) if args.ttl_hours > 0 else None,
        min_eval_worlds=args.min_eval_worlds,
    )
    try:
        uvicorn.run(create_app(service), host=args.host, port=args.port)
    finally:
        service.shutdown()
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
    from onc_agi.infra.archive import FileScorecardArchive
    from onc_agi.infra.ledger import JsonLedger
    from onc_agi.services.scorecards import ScorecardService

    store = FileWorldStore(Path(args.store) if args.store else fixture_store())
    if args.url:
        report = run_conformance(ArenaClient(args.url, args.key), store)
    else:
        active_service: ScorecardService | None = None
        active_client: ArenaClient | None = None

        def boot() -> tuple[ArenaClient, FileScorecardArchive]:
            nonlocal active_service, active_client
            if active_service is not None:
                active_service.shutdown()
            if active_client is not None:
                active_client.http.close()
            archive = FileScorecardArchive(_archive_dir(args))
            active_service = ScorecardService(store, JsonLedger(Path(args.ledger)), archive=archive)
            app = create_app(active_service)
            transport = cast("httpx.Client", TestClient(app))  # an httpx.Client subclass
            active_client = ArenaClient("http://testserver", args.key, client=transport)
            return active_client, archive

        client, archive = boot()
        try:
            report = run_conformance(
                client, store, server_traces=lambda sid: archive.events(sid), restart=lambda: boot()[0]
            )
        finally:
            if active_service is not None:
                active_service.shutdown()
            if active_client is not None:
                active_client.http.close()
    print(json.dumps({"ok": report.ok, "checks": report.checks, "details": report.details}, indent=1))
    return 0 if report.ok else 1


# ---------------------------------------------------------------------------------------- agents kit


def _local_store(args: argparse.Namespace) -> FileWorldStore:
    return FileWorldStore(
        Path(args.store) if args.store else fixture_store(), Path(args.keys) if args.keys else None
    )


def _remote(args: argparse.Namespace) -> tuple[str | None, str | None]:
    """``--url``/``--key``, defaulting to ``ARENA_URL``/``ARENA_KEY`` (``.env`` is read first)."""
    from onc_agi.adapters.profiles import load_dotenv

    load_dotenv()
    return args.url or os.environ.get("ARENA_URL") or None, args.key or os.environ.get("ARENA_KEY") or None


def _llm_agent(args: argparse.Namespace) -> tuple[AgentFactory, str, str, str]:
    """Factory, scorecard agent name, model and harness label of ``--agent llm``."""
    from onc_agi.adapters.agents.llm import HarnessConfig, LLMToolAgent, harness_label
    from onc_agi.adapters.profiles import load_profile, profiles_path

    profiles = Path(args.profiles) if args.profiles else None
    if not args.profile:
        raise ValueError(f"--agent llm needs --profile NAME (profiles file: {profiles_path(profiles)})")
    profile = load_profile(args.profile, args.set, profiles)
    profile.api_key()  # a missing key fails here, before a scorecard is opened
    config = HarnessConfig.load(Path(args.harness_config) if args.harness_config else None)
    return (
        (lambda: LLMToolAgent(profile, config)),
        f"llm-{profile.name}",
        profile.model,
        harness_label(config),
    )


def _write_explanations(
    directory: Path,
    events: Sequence[RecordingEvent],
    runs: Sequence[RunRecord],
    store: FileWorldStore,
    title: str,
) -> bool:
    from onc_agi.services.explain import explain_run, report_json, report_markdown

    known = set(store.world_ids(Tier.PUBLIC_TRAIN))
    if not runs or any(r.world_id not in known for r in runs):
        return False  # played remotely on worlds this store does not hold
    explanations = explain_run(events, runs, store)
    (directory / "explanations.md").write_text(report_markdown(explanations, title=title))
    (directory / "explanations.json").write_text(report_json(explanations))
    return True


def cmd_play(args: argparse.Namespace) -> int:
    from onc_agi.adapters.agents.specs import AgentSpecError, resolve_agent
    from onc_agi.adapters.client import ArenaClient
    from onc_agi.adapters.profiles import ProfileError
    from onc_agi.adapters.swarm import Arena, LocalArena, run_swarm
    from onc_agi.infra.recordings import RECORDING_FILE, RecordingWriter, read_recording

    url, key = _remote(args)
    if url and args.store:
        print("play: give --url (remote server) or --store (in-process), not both", file=sys.stderr)
        return 2
    tier, mode = Tier(args.tier), Mode(args.mode) if args.mode else None
    world_ids = tuple(w.strip() for w in args.worlds.split(",") if w.strip()) if args.worlds else None
    tags = tuple(t.strip() for t in args.tags.split(",") if t.strip()) if args.tags else ()
    store = None if url else _local_store(args)
    model = harness = None
    try:
        if args.agent == "llm":
            factory, name, model, harness = _llm_agent(args)
        else:
            factory, name = resolve_agent(args.agent, store)
    except (AgentSpecError, ProfileError, ValueError, ImportError) as exc:
        print(f"play: {exc}", file=sys.stderr)
        return 2
    arena: Arena
    if url:
        if not key:
            print("play: playing a server needs a key (--key or ARENA_KEY)", file=sys.stderr)
            return 2
        arena = ArenaClient(url, key)
    else:
        assert store is not None
        arena = LocalArena.over_store(store)
    n = args.n
    if n is None and world_ids is None:
        if tier is not Tier.PUBLIC_TRAIN:
            print(f"play: {tier.value} worlds are drawn fresh; give --n", file=sys.stderr)
            return 2
        n = sum(1 for c in arena.worlds(tier) if mode is None or c.mode is mode)
    record = Path(args.record) if args.record else None
    try:
        recorder = RecordingWriter(record / RECORDING_FILE) if record else None
    except FileExistsError as exc:
        print(f"play: {exc}", file=sys.stderr)
        return 2
    try:
        result = run_swarm(
            arena,
            factory,
            agent_name=name,
            tier=tier,
            n_worlds=n if world_ids is None else None,
            world_ids=world_ids,
            mode=mode,
            tags=tags,
            workers=args.workers,
            recorder=recorder,
            budget_usd=args.budget_usd,
            model=model,
            harness=harness,
        )
    except ArenaError as exc:
        print(f"play: the arena refused the scorecard: {exc}", file=sys.stderr)
        return 2
    finally:
        if recorder is not None:
            recorder.close()
    card = result.scorecard
    print(_row(card))
    submitted = sum(r.submitted for r in result.runs)
    cost = "" if card.cost_usd is None else f", cost {card.cost_usd:.4f} USD"
    tokens = "" if card.tokens is None else f", {card.tokens} tokens"
    print(
        f"{card.scorecard_id}: {len(result.runs)} worlds, {submitted} submitted, "
        f"{len(result.errors)} errors, {len(result.unplayed)} unplayed{tokens}{cost}"
    )
    if result.budget_reached:
        print(
            f"budget {args.budget_usd} USD reached: {len(result.unplayed)} worlds never started"
            " (they score as empty submissions)"
        )
    for wid, error in result.errors.items():
        print(f"  {wid}: {error}", file=sys.stderr)
    if record is not None:
        (record / "scorecard.json").write_text(card.model_dump_json(indent=1))
        if tier is Tier.PUBLIC_TRAIN:
            data = read_recording(record)
            explained = _write_explanations(
                record,
                data.events,
                data.runs,
                store or FileWorldStore(fixture_store()),
                f"{name} on {card.scorecard_id}",
            )
            print(
                f"recording in {record}"
                + ("" if explained else " (explanations skipped: worlds not in the local store)")
            )
    if args.json:
        Path(args.json).write_text(card.model_dump_json(indent=1))
    return 1 if result.errors else 0


def _cards(args: argparse.Namespace, tier: Tier) -> list[WorldCard]:
    from onc_agi.adapters.client import ArenaClient

    url, key = _remote(args)
    if url:
        if not key:
            raise ValueError("listing a server's worlds needs a key (--key or ARENA_KEY)")
        return ArenaClient(url, key).worlds(tier)
    store = _local_store(args)
    return [store.card(w) for w in store.world_ids(tier)]


def cmd_worlds(args: argparse.Namespace) -> int:
    try:
        cards = _cards(args, Tier.PUBLIC_TRAIN)
    except (ValueError, ArenaError) as exc:
        print(f"worlds: {exc}", file=sys.stderr)
        return 2
    for c in cards:
        print(
            f"{c.world_id:32s} {c.mode.value:12s} {c.n_pool:6d} patients {len(c.features):6d} features"
            f"  strata {','.join(c.strata)}"
        )
    print(f"{len(cards)} public-train worlds")
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    from onc_agi.infra.recordings import read_recording
    from onc_agi.services.explain import explain_run, report_json, report_markdown

    store = _local_store(args)
    events: Sequence[RecordingEvent]
    runs: Sequence[RunRecord]
    if args.record:
        data = read_recording(Path(args.record))
        events, runs = data.events, data.runs
        title = f"{data.header.agent} on {data.header.scorecard_id}" if data.header else str(args.record)
    else:
        from onc_agi.adapters.inspect_log import from_inspect_log

        events, runs = from_inspect_log(Path(args.inspect_log))
        title = f"Inspect log {Path(args.inspect_log).name}"
    try:
        explanations = explain_run(events, runs, store, operator=args.operator)
    except (PermissionError, ArenaError) as exc:
        print(f"explain: {exc}", file=sys.stderr)
        return 2
    print(report_markdown(explanations, title=title), end="")
    if args.json:
        Path(args.json).write_text(report_json(explanations))
    return 0


def _add_play(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser(
        "play",
        help="play any agent on one scorecard, in-process or against a server (parallel worlds, recordings)",
    )
    p.add_argument(
        "--agent", required=True, help="built-in name, llm, module:Class or path.py:Class (an Agent subclass)"
    )
    p.add_argument("--profile", help="model profile for --agent llm")
    p.add_argument("--profiles", help="profiles TOML (default ./profiles.toml, else the packaged example)")
    p.add_argument("--set", action="append", default=[], metavar="K=V", help="override a profile parameter")
    p.add_argument("--harness-config", help="TOML file with a [harness] table for --agent llm")
    p.add_argument("--url", help="arena server (default $ARENA_URL); without it, play in-process")
    p.add_argument("--key", help="API key for --url (default $ARENA_KEY)")
    p.add_argument("--store", help="world store for in-process play (default: the bundled fixtures)")
    p.add_argument("--keys", help=argparse.SUPPRESS)
    p.add_argument("--tier", default="public_train", choices=[t.value for t in Tier])
    worlds = p.add_mutually_exclusive_group()
    worlds.add_argument("--n", type=int, help="number of worlds (default: every listed world)")
    worlds.add_argument("--worlds", help="comma-separated public-train world ids")
    p.add_argument("--mode", choices=[m.value for m in Mode])
    p.add_argument("--workers", type=int, default=4, help="worlds played in parallel")
    p.add_argument("--tags", help="comma-separated scorecard tags")
    p.add_argument("--record", help="directory for the recording, scorecard and explanations")
    p.add_argument("--budget-usd", type=float, help="start no new world once reported cost exceeds this")
    p.add_argument("--json", help="write the scorecard JSON here")
    p.set_defaults(func=cmd_play)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="onc-agi", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("smoke", help="run reference, baseline and cheater agents on fixture worlds")
    p.add_argument("--store")
    p.set_defaults(func=cmd_smoke)
    p = sub.add_parser("evaluate", help="evaluate one agent on a tier")
    p.add_argument(
        "--agent", required=True, choices=sorted(set(AGENT_NAMES) | {f"seq_{b}" for b in BASELINES})
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
    p.add_argument(
        "--allow-unissued-keys",
        action="store_true",
        help="serve eval/private worlds to any key (development only; caps then bound nothing)",
    )
    p.add_argument("--archive", help="scorecard archive directory (default: <ledger>.archive)")
    p.add_argument(
        "--min-eval-worlds",
        type=int,
        default=40,
        help="minimum eval/private draw size (default: 40; operator policy, not statistical assurance)",
    )
    p.add_argument(
        "--ttl-hours", type=float, default=24.0, help="auto-close open scorecards after this long (0: never)"
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
    p.add_argument("--archive", help="in-process scorecard archive (default: <ledger>.archive)")
    p.set_defaults(func=cmd_conformance)
    _add_play(sub)
    p = sub.add_parser("worlds", help="list public-train worlds (ids, modes, sizes)")
    p.add_argument("--url", help="arena server (default $ARENA_URL); without it, the local store")
    p.add_argument("--key", help="API key for --url (default $ARENA_KEY)")
    p.add_argument("--store", help="world store (default: the bundled fixtures)")
    p.add_argument("--keys", help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_worlds)
    p = sub.add_parser("explain", help="explain each world's outcome from a recording or an Inspect log")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--record", help="recording directory or recording.jsonl")
    source.add_argument("--inspect-log", help="Inspect log of the standard harness")
    p.add_argument("--store", help="world store with the answer keys (default: the bundled fixtures)")
    p.add_argument("--keys", help="answer keys of non-public worlds (operators only)")
    p.add_argument(
        "--operator", action="store_true", help="allow non-public worlds (report marked operator-only)"
    )
    p.add_argument("--json", help="write the explanations as JSON here")
    p.set_defaults(func=cmd_explain)
    args = parser.parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
