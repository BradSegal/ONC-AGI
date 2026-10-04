"""``onc-agi`` command line: smoke, evaluate, serve, replay, conformance, and the agents kit (play,
standard, worlds, subset, explain)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from datetime import timedelta
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    import httpx

from onc_agi.adapters.agents import AGENT_NAMES, BASELINES, CHEATERS, AgentFactory, make_agent
from onc_agi.core.errors import ArenaError
from onc_agi.core.ports import RecordingEvent, RunRecord
from onc_agi.core.schema import Mode, Scorecard, Tier, TraceEvent, WorldCard
from onc_agi.infra.bundles import FileWorldStore, world_id_list
from onc_agi.infra.recorder import TraceRecorder
from onc_agi.infra.recordings import read_run
from onc_agi.services import sampling
from onc_agi.services.kit import check_world_ids, evaluate
from onc_agi.services.replay import replay, verify_run
from onc_agi.services.sampling import WorldProfile

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


def _sample(profiles: Sequence[WorldProfile], n: int, seed: int, what: str) -> tuple[str, ...]:
    """A seeded stratified sample of ``n`` worlds; its mix by source, family and mode goes to stderr."""
    sample = sampling.stratified_sample(profiles, n, seed)
    print(f"{what}: {sample.summary()}", file=sys.stderr)
    return sample.world_ids


def cmd_evaluate(args: argparse.Namespace) -> int:
    store = FileWorldStore(Path(args.store), Path(args.keys) if args.keys else None)
    tier = Tier(args.tier)
    try:
        ids = check_world_ids(store, tier, world_id_list(args.worlds)) if args.worlds else None
        if args.n:
            ids = _sample(sampling.world_profiles(store, tier), args.n, args.seed, "evaluate")
    except (ValueError, ArenaError) as exc:
        print(f"evaluate: {exc}", file=sys.stderr)
        return 2
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
    if args.record:
        server = None
        url, key = args.url, args.key or os.environ.get("ARENA_KEY")  # only an explicit --url
        if url:
            from onc_agi.adapters.client import ArenaClient
            from onc_agi.adapters.swarm import server_copies

            if not key:
                print("replay: checking against a server needs its key (--key or ARENA_KEY)", file=sys.stderr)
                return 2
            run = read_run(Path(args.record))
            sid = (
                run.scorecard.scorecard_id
                if run.scorecard
                else run.recording.header.scorecard_id if run.recording.header else None
            )
            server = server_copies(ArenaClient(url, key), sid) if sid else None
            if server is None:
                print(f"{args.record}: NOT VERIFIED: the server's copies are unavailable", file=sys.stderr)
                return 1
        outcome = verify_and_explain(Path(args.record), store, title=None, explain=False, server=server)
        return 0 if outcome in ("verified", "consistent") else 1
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


RunStatus = Literal["verified", "consistent", "unverified", "failed"]


def verify_and_explain(
    record: Path,
    store: FileWorldStore,
    *,
    title: str | None,
    explain: bool = True,
    server: tuple[Sequence[TraceEvent], Scorecard] | None = None,
) -> RunStatus:
    """Verify a run directory by replay (:func:`verify_run`), then explain each world.

    ``verified``: checked against the server's own trace and scorecard (``server``); ``consistent``:
    the local files agree with each other and the world bundles, but no server copy was compared;
    ``unverified``: nothing could be checked (no server trace, or no answer keys for the worlds);
    ``failed``: a mismatch. Replay and explanations both need the worlds' answer keys in ``store``.
    """
    from onc_agi.services.explain import explain_run, report_json, report_markdown

    run = read_run(record)
    header, runs = run.recording.header, run.recording.runs
    ids = list(header.world_ids) if header is not None else [r.world_id for r in runs]
    if run.trace is None:
        print(f"{record}: NOT VERIFIED: the run has no server trace")
        return "unverified"
    try:
        held = bool(ids) and all(store.answer_key(w) is not None for w in ids)
    except ArenaError:
        held = False
    if not held:
        print(f"{record}: NOT VERIFIED: the store lacks these worlds' answer keys")
        return "unverified"
    verification = verify_run(
        run.trace,
        store,
        scorecard=run.scorecard,
        header=header,
        runs=runs,
        events=run.recording.events,
        server_trace=server[0] if server else None,
        server_scorecard=server[1] if server else None,
    )
    print(f"{record}: {verification.summary()}")
    if not verification.ok:
        return "failed"
    if explain and title is not None and all(store.card(w).tier is Tier.PUBLIC_TRAIN for w in ids):
        explanations = explain_run(run.recording.events, runs, store)
        (record / "explanations.md").write_text(report_markdown(explanations, title=title))
        (record / "explanations.json").write_text(report_json(explanations))
        print(f"explanations in {record / 'explanations.md'}")
    return "verified" if verification.authority == "server" else "consistent"


def cmd_play(args: argparse.Namespace) -> int:
    from onc_agi.adapters.agents.specs import AgentSpecError, resolve_agent
    from onc_agi.adapters.client import ArenaClient
    from onc_agi.adapters.profiles import ProfileError
    from onc_agi.adapters.swarm import LocalArena, run_swarm, server_copies, server_trace
    from onc_agi.infra.recordings import RECORDING_FILE, RecordingWriter, save_run

    url, key = _remote(args)
    if url and args.store:
        print("play: give --url (remote server) or --store (in-process), not both", file=sys.stderr)
        return 2
    tier, mode = Tier(args.tier), Mode(args.mode) if args.mode else None
    world_ids = world_id_list(args.worlds) if args.worlds else None
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
    arena: LocalArena | ArenaClient
    if url:
        if not key:
            print("play: playing a server needs a key (--key or ARENA_KEY)", file=sys.stderr)
            return 2
        arena = ArenaClient(url, key)
    else:
        assert store is not None
        arena = LocalArena.over_store(store)
    n = args.n
    if n is None and world_ids is None and tier is Tier.PUBLIC_TRAIN:
        n = sum(1 for c in arena.worlds(tier) if mode is None or c.mode is mode)
    elif n is not None and store is not None and tier is Tier.PUBLIC_TRAIN:
        # in-process: draw here so the mix can be reported; a server draws the same way itself
        profiles = [p for p in sampling.world_profiles(store, tier) if mode is None or p.mode == mode.value]
        try:
            world_ids = _sample(profiles, n, args.seed or 0, "play")
        except ValueError as exc:
            print(f"play: {exc}", file=sys.stderr)
            return 2
    # an eval tier without --n: a server with fixed sets scores the whole set; others refuse
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
            seed=args.seed if world_ids is None else None,
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
    status: RunStatus = "consistent"
    if record is not None:
        save_run(record, card, server_trace(arena, card.scorecard_id))
        print(f"recording, trace and scorecard in {record}")
        status = verify_and_explain(
            record,
            store or FileWorldStore(fixture_store()),
            title=f"{name} on {card.scorecard_id}",
            server=server_copies(arena, card.scorecard_id),
        )
    if args.json:
        Path(args.json).write_text(card.model_dump_json(indent=1))
    return 1 if result.errors or status == "failed" else 0


def cmd_standard(args: argparse.Namespace) -> int:
    """The standard track end to end: the Inspect task for one model, then close, record and verify."""
    try:
        from onc_agi.adapters.inspect_task import run_standard
    except ImportError as exc:
        print(f"standard: the Inspect harness needs the [inspect] extra ({exc})", file=sys.stderr)
        return 2
    from onc_agi.adapters.profiles import ProfileError, load_profile

    url, key = _remote(args)
    if url and args.store:
        print("standard: give --url (remote server) or --store (in-process), not both", file=sys.stderr)
        return 2
    if bool(args.profile) == bool(args.model):
        print(
            "standard: give --profile NAME (a model profile) or --model (an Inspect model)", file=sys.stderr
        )
        return 2
    task_args: dict[str, object] = {"tier": args.tier, "seed": args.seed}
    if url:
        if not key:
            print("standard: playing a server needs a key (--key or ARENA_KEY)", file=sys.stderr)
            return 2
        os.environ["ARENA_KEY"] = key  # the task reads it from the environment, never from its arguments
        task_args["url"] = url
    else:
        task_args["store_root"] = args.store or str(fixture_store())
        if args.keys:
            task_args["keys_dir"] = args.keys
    if args.n is not None:
        task_args["n_worlds"] = args.n
    if args.worlds:
        task_args["world_ids"] = args.worlds
    if args.message_limit:
        task_args["message_limit"] = args.message_limit
    try:
        if args.profile:
            profile = load_profile(args.profile, args.set, Path(args.profiles) if args.profiles else None)
            model, config, price = profile.inspect_model(), profile.inspect_config(), profile.price()
            if profile.extra_headers:
                print(
                    f"standard: headers {sorted(profile.extra_headers)} are not sent on the standard track"
                    " (Inspect would write their values into its log)",
                    file=sys.stderr,
                )
            prices = {model: price} if price else None  # Inspect reports usage under the model string
        else:
            model, config, prices = args.model, {}, None
        record = Path(args.record) if args.record else None
        if record is not None and (record / "recording.jsonl").exists():
            raise FileExistsError(
                f"recording {record / 'recording.jsonl'} already exists; record each run to a new place"
            )
        result = run_standard(
            mode=Mode(args.mode),
            model=model,
            model_config=config,
            prices=prices,
            record=record,
            log_dir=Path(args.log_dir),
            agent=args.agent,
            max_samples=args.max_samples,
            **task_args,
        )
    except (ProfileError, ValueError, FileExistsError, ArenaError, RuntimeError) as exc:
        print(f"standard: {exc}", file=sys.stderr)
        return 2
    card = result.scorecard
    print(_row(card))
    print(f"{card.scorecard_id}: {card.n_worlds} worlds; Inspect log {result.log_path}")
    status: RunStatus = "consistent"
    if record is not None:
        print(f"recording, trace and scorecard in {record}")
        server = None
        if url:  # an in-process arena ends with the run; a server keeps its own copies
            from onc_agi.adapters.client import ArenaClient
            from onc_agi.adapters.swarm import server_copies

            server = server_copies(ArenaClient(url, os.environ["ARENA_KEY"]), card.scorecard_id)
        status = verify_and_explain(
            record,
            _local_store(args) if not url else FileWorldStore(fixture_store()),
            title=f"{card.agent} on {card.scorecard_id}",
            server=server,
        )
    if args.json:
        Path(args.json).write_text(card.model_dump_json(indent=1))
    return 1 if status == "failed" else 0


def _cards(args: argparse.Namespace, tier: Tier) -> list[WorldCard]:
    from onc_agi.adapters.client import ArenaClient

    url, key = _remote(args)
    if url:
        if not key:
            raise ValueError("listing a server's worlds needs a key (--key or ARENA_KEY)")
        return ArenaClient(url, key).worlds(tier)
    store = _local_store(args)
    return [store.card(w) for w in store.world_ids(tier)]


def _profiles(args: argparse.Namespace, tier: Tier) -> list[tuple[WorldProfile, WorldCard]]:
    """Profiles and cards of a tier's worlds; a server's listing shows cards only (source and family unknown)."""
    if _remote(args)[0]:
        return [(sampling.world_profile(c), c) for c in _cards(args, tier)]
    store = _local_store(args)
    return [(p, store.card(p.world_id)) for p in sampling.world_profiles(store, tier)]


def cmd_worlds(args: argparse.Namespace) -> int:
    try:
        rows = _profiles(args, Tier.PUBLIC_TRAIN)
    except (ValueError, ArenaError) as exc:
        print(f"worlds: {exc}", file=sys.stderr)
        return 2
    print(f"{'world_id':32s} {'mode':12s} {'source':16s} {'family':18s} {'rows x features':>15s}  strata")
    for p, c in rows:
        size = f"{p.rows} x {p.features}"
        print(
            f"{p.world_id:32s} {p.mode:12s} {p.source:16s} {p.family:18s} {size:>15s}  {','.join(c.strata)}"
        )
    print(f"{len(rows)} public-train worlds")
    return 0


def cmd_subset(args: argparse.Namespace) -> int:
    """Draw a seeded stratified subset across one or more stores and write it as a world-id list."""
    tier, mode = Tier(args.tier), Mode(args.mode) if args.mode else None
    roots = [Path(s) for s in args.store] or [fixture_store()]
    profiles = [
        p
        for root in roots
        for p in sampling.world_profiles(FileWorldStore(root), tier)
        if mode is None or p.mode == mode.value
    ]
    try:
        sample = sampling.stratified_sample(profiles, args.n, args.seed)
    except ValueError as exc:
        print(f"subset: {exc} (stores: {', '.join(map(str, roots))})", file=sys.stderr)
        return 2
    print(sample.table(), file=sys.stderr)
    text = "".join(f"{w}\n" for w in sample.world_ids)
    if not args.out:
        print(text, end="")
        return 0
    out = Path(args.out)
    if out.exists() and out.read_text() != text:
        print(f"subset: {out} already lists other worlds; published id lists never change", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {len(sample.world_ids)} world ids to {out}", file=sys.stderr)
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
    worlds.add_argument(
        "--n", type=int, help="number of worlds, a stratified sample (default: every listed world)"
    )
    worlds.add_argument(
        "--worlds", help="public-train world ids: comma-separated, or an id-list file (one per line)"
    )
    p.add_argument("--seed", type=int, help="seed of the --n public-train sample (default 0)")
    p.add_argument("--mode", choices=[m.value for m in Mode])
    p.add_argument("--workers", type=int, default=4, help="worlds played in parallel")
    p.add_argument("--tags", help="comma-separated scorecard tags")
    p.add_argument("--record", help="directory for the recording, scorecard and explanations")
    p.add_argument("--budget-usd", type=float, help="start no new world once reported cost exceeds this")
    p.add_argument("--json", help="write the scorecard JSON here")
    p.set_defaults(func=cmd_play)


def _add_standard(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser(
        "standard",
        help="play the standard track (the Inspect task) for one model, in-process or against a server",
    )
    model = p.add_mutually_exclusive_group()
    model.add_argument("--profile", help="model profile (as for play --agent llm)")
    model.add_argument("--model", help="an Inspect model string instead of a profile (e.g. openai/gpt-4o)")
    p.add_argument("--profiles", help="profiles TOML (default ./profiles.toml, else the packaged example)")
    p.add_argument("--set", action="append", default=[], metavar="K=V", help="override a profile parameter")
    p.add_argument("--mode", required=True, choices=[m.value for m in Mode])
    p.add_argument("--url", help="arena server (default $ARENA_URL); without it, play in-process")
    p.add_argument("--key", help="API key for --url (default $ARENA_KEY)")
    p.add_argument("--store", help="world store for in-process play (default: the bundled fixtures)")
    p.add_argument("--keys", help=argparse.SUPPRESS)
    p.add_argument("--tier", default="public_train", choices=[t.value for t in Tier])
    worlds = p.add_mutually_exclusive_group()
    worlds.add_argument("--n", type=int, help="number of worlds, a stratified sample (default: every world)")
    worlds.add_argument("--worlds", help="world ids: comma-separated, or an id-list file (one per line)")
    p.add_argument("--seed", type=int, help="seed of the --n public-train sample (default 0)")
    p.add_argument("--agent", help="scorecard agent label (default: inspect-standard)")
    p.add_argument(
        "--message-limit", type=int, help="messages per world (task default: 60 full access, 80 sequential)"
    )
    p.add_argument("--max-samples", type=int, default=4, help="worlds played in parallel")
    p.add_argument("--log-dir", default="logs", help="Inspect log directory (default ./logs)")
    p.add_argument("--record", help="directory for the recording, trace, scorecard and explanations")
    p.add_argument("--json", help="write the scorecard JSON here")
    p.set_defaults(func=cmd_standard)


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
    worlds = p.add_mutually_exclusive_group()
    worlds.add_argument("--n", type=int, help="number of worlds, a seeded stratified sample (default: all)")
    worlds.add_argument("--worlds", help="world ids: comma-separated, or an id-list file (one per line)")
    p.add_argument("--seed", type=int, default=0, help="seed of the --n sample (default 0)")
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
    p = sub.add_parser("replay", help="verify a trace, or a recorded run, against its world bundles")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--trace", help="a trace JSONL (arena evaluate --trace)")
    source.add_argument(
        "--record", help="a run directory (arena play/standard --record): trace, scorecard, recording"
    )
    p.add_argument("--store", required=True)
    p.add_argument("--keys")
    p.add_argument("--url", help="with --record: check against this server's own trace and scorecard")
    p.add_argument("--key", help="API key for --url (default $ARENA_KEY)")
    p.set_defaults(func=cmd_replay)
    p = sub.add_parser("conformance", help="run the custom-harness conformance suite")
    p.add_argument("--store")
    p.add_argument("--url")
    p.add_argument("--key", default="conformance-key")
    p.add_argument("--ledger", default=".arena-conformance-ledger.json")
    p.add_argument("--archive", help="in-process scorecard archive (default: <ledger>.archive)")
    p.set_defaults(func=cmd_conformance)
    _add_play(sub)
    _add_standard(sub)
    p = sub.add_parser("worlds", help="list public-train worlds (ids, modes, sources, families, sizes)")
    p.add_argument("--url", help="arena server (default $ARENA_URL); without it, the local store")
    p.add_argument("--key", help="API key for --url (default $ARENA_KEY)")
    p.add_argument("--store", help="world store (default: the bundled fixtures)")
    p.add_argument("--keys", help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_worlds)
    p = sub.add_parser(
        "subset", help="draw a seeded stratified world subset (source x family x mode) and write its id list"
    )
    p.add_argument(
        "--store",
        action="append",
        default=[],
        help="world store; repeat to draw across packs (default: fixtures)",
    )
    p.add_argument("--tier", default="public_train", choices=[t.value for t in Tier])
    p.add_argument("--mode", choices=[m.value for m in Mode], help="draw from one mode only")
    p.add_argument("--n", type=int, required=True, help="number of worlds")
    p.add_argument("--seed", type=int, default=0, help="sample seed (default 0)")
    p.add_argument(
        "--out", help="write the id list here (default: stdout); an existing list is never changed"
    )
    p.set_defaults(func=cmd_subset)
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
