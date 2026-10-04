"""Build the site's data files from the real runtime.

* ``src/data/smoke.json`` - the stdout of the public ``onc-agi smoke`` command, run in
  process on the packaged fixture worlds; the page prints it verbatim as expected output.
* ``src/data/oracle.json`` - the oracle's method demonstrated on a toy world: the
  null maximum statistic over permuted outcomes, its 95th-percentile threshold,
  and detection rates of the planted cause and its stand-in as the sample grows.
* ``src/data/journey.json`` - one recorded sequential episode on a toy world, scored, and
  the same policy played on every toy world.
* ``src/data/play.json`` - the journey world as an in-browser task: the evidence a visitor
  sees after buying each stage of patients, and the answer-key facts that let the page score
  any ordered list exactly (checked here against the real scorer on every list of up to three).
* ``src/data/failures.json`` - failure cases for "what makes it hard": real toy worlds (leak, no signal) and seeded synthetic demonstrations.
* ``src/data/hero.json`` - the opening scene's two toy worlds (one planted cause, one none) in full.
* ``src/data/catalogue.json`` - one full-access toy world per mechanism: its roles from the
  answer key, full-pool evidence, what a naive marginal screen lists and scores, and a small
  standardised data matrix for a picture.

Run from the repository root::

    uv run python site/scripts/prepare_data.py
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import build_toy  # noqa: E402
from onc_agi.adapters.cli import fixture_store  # noqa: E402
from onc_agi.adapters.cli import main as cli_main  # noqa: E402
from onc_agi.core.schema import Scorecard, Tier  # noqa: E402
from onc_agi.infra.bundles import FileWorldStore  # noqa: E402
from onc_agi.services import scoring  # noqa: E402
from onc_agi.services.kit import evaluate  # noqa: E402

OUT = ROOT / "site" / "src" / "data"

MECHANIC = {
    "toy-driver": "generating",
    "toy-stand-in": "stand_in",
    "toy-wrong-type": "wrong_data_type",
    "toy-confounder": "observed_confounder",
    "toy-leak": "leak",
    "toy-interaction": "interaction",
    "toy-module": "module",
    "toy-neutral": "neutral_group",
    "toy-null-a": "no_signal",
    "toy-null-b": "no_signal",
}


def r(x: float | None, digits: int = 4) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), digits)


def summary(card: Scorecard) -> dict[str, Any]:
    return {
        "discovery_score": r(card.discovery_score),
        "unfloored": r(card.discovery_score_unfloored),
        "low": r(card.interval.low),
        "high": r(card.interval.high),
        "find": r(card.find),
        "restraint": r(card.restraint),
        "strict": r(card.strict_discovery_score),
        "leak_rate": r(card.leak_rate),
        "data_cost": r(card.mean_data_cost, 1),
        "n_worlds": card.n_worlds,
    }


def smoke() -> dict[str, Any]:
    """The public ``onc-agi smoke`` command, run in process; its stdout is kept verbatim."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = cli_main(["smoke"])
    if code != 0:
        raise SystemExit(f"onc-agi smoke exited with {code}")
    return {"command": "uv run onc-agi smoke", "output": out.getvalue()}


# --------------------------------------------------------------------------- oracle demonstration


def world_signal(spec_name: str, suffix: str = "full") -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    spec = next(s for s in build_toy.SPECS if s.world == spec_name)
    rng = np.random.default_rng(build_toy.seed_of(f"{build_toy.BUILDER_VERSION}:{spec.world}-{suffix}"))
    draft = build_toy.Draft(rng)
    build_toy.base_columns(draft)
    signal, groups = spec.plant(draft)
    names = list(draft.columns)
    x = np.column_stack([draft.columns[n] for n in names])
    truth = [p[0] for g in groups for p in g.parts]
    return signal, names, x, truth


def draw_y(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    lo, hi = -20.0, 20.0
    a = 0.0
    for _ in range(50):
        a = (lo + hi) / 2
        if np.mean(1 / (1 + np.exp(-(a + signal)))) > build_toy.PREVALENCE:
            hi = a
        else:
            lo = a
    return (rng.random(len(signal)) < 1 / (1 + np.exp(-(a + signal)))).astype(float)


def zstats(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xs = (x - x.mean(0)) / np.where(x.std(0) == 0, 1, x.std(0))
    ys = (y - y.mean()) / (y.std() or 1)
    return np.abs(xs.T @ ys) / math.sqrt(len(y))


def wilson(k: int, m: int) -> tuple[float, float]:
    if m == 0:
        return 0.0, 1.0
    p, z = k / m, 1.96
    centre = (p + z * z / (2 * m)) / (1 + z * z / m)
    half = z * math.sqrt(p * (1 - p) / m + z * z / (4 * m * m)) / (1 + z * z / m)
    return centre - half, centre + half


def public_names(world_id: str, names: list[str], x: np.ndarray) -> dict[str, str]:
    """Map the builder's internal column names to the world's public (fake) ids by exact values."""
    world = FileWorldStore(fixture_store()).world(world_id)
    fids = list(world.card.feature_ids())
    out: dict[str, str] = {}
    for k, name in enumerate(names):
        for j, fid in enumerate(fids):
            if np.allclose(world.x[:, j], x[:, k]):
                out[name] = fid
                break
    return out


def oracle() -> dict[str, Any]:
    signal, names, x, _truth = world_signal("toy-stand-in", "seq")
    ids = public_names(JOURNEY_WORLD, names, x)
    rng = np.random.default_rng(11)
    n_total = len(signal)
    null_max = []
    for _ in range(400):
        y0 = (rng.random(n_total) < build_toy.PREVALENCE).astype(float)
        null_max.append(float(zstats(x, y0).max()))
    threshold = float(np.quantile(null_max, 0.95))
    cause, partner = names.index("e01"), names.index("e02")
    grid = [30, 45, 60, 80, 100, 130, 160, 200, 240]
    m = 200
    curves: dict[str, list[dict[str, float]]] = {"cause": [], "stand_in": [], "unrelated": []}
    unrelated = names.index("e07")
    replicate_z: list[float] = []
    for n in grid:
        hits = {"cause": 0, "stand_in": 0, "unrelated": 0}
        for _ in range(m):
            rows = rng.choice(n_total, size=n, replace=False)
            y = draw_y(signal[rows], rng)
            z = zstats(x[rows], y)
            hits["cause"] += z[cause] > threshold
            hits["stand_in"] += z[partner] > threshold
            hits["unrelated"] += z[unrelated] > threshold
            if n == grid[-1]:
                replicate_z.append(round(float(z[cause]), 3))
        for key, k in hits.items():
            lo, hi = wilson(int(k), m)
            curves[key].append({"n": n, "rate": round(k / m, 4), "low": round(lo, 4), "high": round(hi, 4)})
    return {
        "world": JOURNEY_WORLD,
        "features": {"cause": ids.get("e01"), "stand_in": ids.get("e02"), "unrelated": ids.get("e07")},
        "statistic": "marginal |z| (illustration; the benchmark oracle refits the true-term model)",
        "replicates": m,
        "null_draws": len(null_max),
        "threshold": round(threshold, 4),
        "null_max": [round(v, 3) for v in sorted(null_max)],
        "replicate_z": replicate_z,
        "curves": curves,
        "bands": {"recoverable_lower": 0.8, "neutral_upper": 0.2},
    }


# --------------------------------------------------------------------------- the journey: one task, end to end

JOURNEY_WORLD = "toy-stand-in-seq"
STAGES = (30, 60, 90, 120, 160, 200, 240)


def welch_log10p(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """-log10 p of a Welch t-test per column (NaN columns get 0)."""
    from scipy import stats

    out = np.zeros(x.shape[1])
    yb = y.astype(bool)
    if yb.all() or not yb.any():
        return out
    for j in range(x.shape[1]):
        col = x[:, j]
        ok = ~np.isnan(col)
        a, b = col[ok & yb], col[ok & ~yb]
        if len(a) > 2 and len(b) > 2:
            p = stats.ttest_ind(a, b, equal_var=False).pvalue
            out[j] = -math.log10(max(float(p), 1e-300))
    return out


class JourneyAgent:
    """The journey's policy as a runtime Agent, so it can play every toy world.

    Sequential worlds: recruit in stages (30, 60, 90 ... patients, split across strata by
    their published sizes), assay everything on newcomers, test each baseline measurement
    (Welch t, Bonferroni 5%), and submit when the list repeats or the pool is exhausted.
    Full-access worlds: test once and submit.
    """

    def __new__(cls) -> Any:
        from onc_agi.core.schema import Assay, Mode, Recruit, Submit, Timing
        from onc_agi.services.kit import Agent

        class _Journey(Agent):
            name = "journey-agent"

            def __init__(self) -> None:
                super().__init__()
                self.stage = 0
                self.previous: tuple[str, ...] | None = None
                self.pending: list[Any] = []

            def _ranking(self, card: Any, view: Any) -> tuple[str, ...]:
                timing = {f.feature_id: f.timing for f in card.features}
                fids = list(view.feature_ids)
                base = [j for j, f in enumerate(fids) if timing[f] is Timing.BASELINE]
                ev = welch_log10p(view.x, view.outcome)
                cut = -math.log10(0.05 / len(base))
                return tuple(fids[j] for j in sorted(base, key=lambda j: -ev[j]) if ev[j] > cut)

            def choose_action(self, card: Any, view: Any) -> Any:
                if view.mode is Mode.FULL_ACCESS:
                    return Submit(request_id=self.request_id(), ranking=self._ranking(card, view))
                if not view.rows:
                    self.stage, self.previous, self.pending = 0, None, []
                if self.pending:
                    return self.pending.pop(0)
                if view.rows:
                    current = self._ranking(card, view)
                    done = len(view.rows) >= card.n_pool
                    if done or (current and current == self.previous):
                        return Submit(request_id=self.request_id(), ranking=current)
                    self.previous = current
                target = min(card.n_pool, STAGES[min(self.stage, len(STAGES) - 1)] * card.n_pool // 240)
                self.stage += 1
                sizes = card.stratum_sizes or {st: card.n_pool for st in card.strata}
                taken = {st: view.stratum.count(st) for st in card.strata}
                actions: list[Any] = []
                for st in card.strata:
                    want = round(target * sizes[st] / max(1, sum(sizes.values()))) - taken[st]
                    want = min(want, sizes[st] - taken[st])
                    if want > 0:
                        actions.append(Recruit(request_id=self.request_id(), count=want, stratum=st))
                if not actions:
                    return Submit(request_id=self.request_id(), ranking=self.previous or ())
                actions.append(Assay(request_id=self.request_id(), feature_ids=card.feature_ids()))
                first, *rest = actions
                self.pending = rest
                return first

        return _Journey()


def across_worlds() -> dict[str, Any]:
    """The journey policy on every toy world: one world is an example, many make a score."""
    store = FileWorldStore(fixture_store())
    ids = store.world_ids(Tier.PUBLIC_TRAIN)
    card, results_ = evaluate(JourneyAgent(), store, Tier.PUBLIC_TRAIN, world_ids=ids, bootstrap_draws=400)
    worlds = []
    for res in results_:
        sc = res.score
        assert sc is not None
        worlds.append(
            {
                "world": res.world_id,
                "mechanism": MECHANIC[res.world_id.rsplit("-", 1)[0]],
                "mode": "sequential" if res.world_id.endswith("-seq") else "full_access",
                "null": sc.is_null,
                "listed": list(res.ranking),
                "find": round(sc.find, 4),
                "restrained": sc.restrained,
                "abstained": sc.abstained,
                "leaked": sc.leaked,
                "spent": round(sc.spent, 1),
                "efficiency": round(sc.efficiency, 4),
            }
        )
    return {"summary": summary(card), "worlds": worlds}


def journey() -> dict[str, Any]:
    """A real sequential episode on one toy world, recorded action by action, then scored."""
    from onc_agi.core.schema import Assay, Recruit, Reset, Submit, Timing
    from onc_agi.services.engine import Episode

    store = FileWorldStore(fixture_store())
    world = store.world(JOURNEY_WORLD)
    key = store.answer_key(JOURNEY_WORLD)
    card = world.card
    fids = list(card.feature_ids())
    timing = {f.feature_id: f.timing for f in card.features}
    baseline = [j for j, f in enumerate(fids) if timing[f] is Timing.BASELINE]
    agent_threshold = -math.log10(0.05 / len(baseline))  # the agent's own Bonferroni line

    part = key.groups[0].parts[0]
    truth = part.true_feature
    equivalent = [f for f in part.equivalence_set if f != truth]

    ep = Episode(world)
    ep.apply(Reset(request_id="j-0", world_id=card.world_id))
    steps: list[dict[str, Any]] = []
    previous: tuple[str, ...] | None = None
    ranking: tuple[str, ...] = ()
    rid = 1
    for target in STAGES:
        count = target - len(ep.view().rows)
        view = ep.apply(Recruit(request_id=f"j-{rid}", count=count, stratum="all"))
        rid += 1
        steps.append(
            {"kind": "recruit", "count": count, "patients": len(view.rows), "spent": round(view.spent, 2)}
        )
        view = ep.apply(Assay(request_id=f"j-{rid}", feature_ids=card.feature_ids()))
        rid += 1
        steps.append(
            {
                "kind": "assay",
                "features": len(fids),
                "patients": len(view.rows),
                "spent": round(view.spent, 2),
            }
        )
        evidence = welch_log10p(view.x, view.outcome)
        order = sorted(baseline, key=lambda j: -evidence[j])
        ranking = tuple(fids[j] for j in order if evidence[j] > agent_threshold)
        steps.append(
            {
                "kind": "analyse",
                "patients": len(view.rows),
                "spent": round(view.spent, 2),
                "evidence": {fids[j]: round(float(evidence[j]), 3) for j in range(len(fids))},
                "ranking": list(ranking),
            }
        )
        if ranking and ranking == previous:
            break
        previous = ranking
    view = ep.apply(Submit(request_id=f"j-{rid}", ranking=ranking))
    steps.append(
        {
            "kind": "submit",
            "ranking": list(ranking),
            "spent": round(view.spent, 2),
            "patients": len(view.rows),
        }
    )

    def breakdown(listed: list[str]) -> dict[str, Any]:
        reps = list(scoring.representatives(listed, key))
        top = scoring.top_representatives(listed, key)
        credit = scoring._credit(top, key)
        chance_raw, _chance_exact = scoring.chance_recovery(key, listed)
        score = scoring.score_world(listed, key, spent=ep.spent, sequential=True)
        return {
            "listed": listed,
            "clusters": {f: key.clusters[f] for f in listed},
            "representatives": reps,
            "top_r": top,
            "raw": round(credit.raw, 4),
            "exact": round(credit.exact, 4),
            "chance": round(chance_raw, 4),
            "find": round(score.find, 4),
            "find_exact": round(score.find_exact, 4),
            "leaked": score.leaked,
            "abstained": score.abstained,
            "efficiency": round(score.efficiency, 4),
        }

    post = [f for f in fids if timing[f] is Timing.POST_OUTCOME]
    variants = {
        "agent": breakdown(list(ranking)),
        "substitute_only": breakdown(equivalent[:1]),
        "with_leak": breakdown([truth, post[0]]),
        "empty": breakdown([]),
        # "List everything", in an order that carries no knowledge (seeded shuffle).
        "everything": breakdown([fids[j] for j in np.random.default_rng(5).permutation(baseline)]),
    }
    # One shuffled "list everything" can get lucky; its expectation over orders is what matters.
    blind_rng = np.random.default_rng(17)
    blind = [
        scoring.score_world(
            [fids[j] for j in blind_rng.permutation(baseline)], key, spent=ep.spent, sequential=True
        )
        for _ in range(500)
    ]
    variants["everything"]["expected_find_signed"] = round(float(np.mean([b.find_signed for b in blind])), 4)
    variants["everything"]["lucky_share"] = round(float(np.mean([b.raw_recovery > 0 for b in blind])), 4)
    full_evidence = welch_log10p(world.x, world.y.astype(np.int64))
    z = (world.x - world.x.mean(0)) / np.where(world.x.std(0) == 0, 1, world.x.std(0))
    return {
        "world": card.world_id,
        "mechanism": "stand_in",
        "pool": {
            "z": [[round(float(v), 1) for v in row] for row in z],
            "outcome": [int(v) for v in world.y],
            "queue": list(world.queues["all"]),
        },
        "across": across_worlds(),
        "card": {
            "n_pool": card.n_pool,
            "budget": card.budget,
            "recruit_price": card.prices.recruit_per_patient,
            "strata": list(card.strata),
            "features": [
                {"id": f.feature_id, "type": f.data_type, "timing": f.timing.value, "price": f.assay_price}
                for f in card.features
            ],
        },
        "key": {
            "truth": truth,
            "equivalent": equivalent,
            "exact_recoverable": part.exact_recoverable,
            "reject_set": list(key.reject_set),
            "depth": key.depth,
            "reference_cost": key.reference_cost,
        },
        "agent": {
            "rule": "Welch t-test per baseline measurement, Bonferroni at 5%; stop when the list repeats"
        },
        "agent_threshold": round(agent_threshold, 3),
        "steps": steps,
        "full_pool_evidence": {fids[j]: round(float(full_evidence[j]), 3) for j in range(len(fids))},
        "scoring": variants,
    }


# --------------------------------------------------------------------------- try one: the journey world in the browser

PLAY_RULE = (
    "leaked = any listed feature is in reject; "
    "reps = the first-listed feature of each cluster, after removing neutral features; "
    "abstained = no reps; "
    "find = 0 if is_null or leaked or abstained, else find_first[reps[0]].find (depth == 1: only the "
    "first representative earns credit, and chance depends only on its stratum); "
    "efficiency = 1 if spent <= 0 else min(1, reference_cost / spent)"
)


def bonferroni_line(card: Any) -> float:
    """The journey agent's evidence line: -log10(0.05 / number of baseline measurements)."""
    from onc_agi.core.schema import Timing

    baseline = [f for f in card.features if f.timing is Timing.BASELINE]
    return -math.log10(0.05 / len(baseline))


def feature_rows(card: Any, price: bool = True) -> list[dict[str, Any]]:
    """The card's measurements in the shape of ``journey.card.features``."""
    rows = []
    for f in card.features:
        row: dict[str, Any] = {"id": f.feature_id, "type": f.data_type, "timing": f.timing.value}
        if price:
            row["price"] = f.assay_price
        rows.append(row)
    return rows


def stage_view(world: Any, target: int) -> Any:
    """A fresh real episode: reset, recruit ``target`` patients, assay every measurement on them."""
    from onc_agi.core.schema import Assay, Recruit, Reset
    from onc_agi.services.engine import Episode

    card = world.card
    ep = Episode(world)
    ep.apply(Reset(request_id="p-0", world_id=card.world_id))
    ep.apply(Recruit(request_id="p-1", count=target, stratum="all"))
    return ep.apply(Assay(request_id="p-2", feature_ids=card.feature_ids()))


def browser_score(
    listed: list[str], rules: dict[str, Any], reference_cost: float, spent: float
) -> dict[str, Any]:
    """The page's scoring rule (``PLAY_RULE``), re-implemented on the published data alone."""
    reject, neutral = set(rules["reject"]), set(rules["neutral"])
    leaked = any(f in reject for f in listed)
    reps: list[str] = []
    seen: set[int] = set()
    for f in listed:
        if f in neutral or rules["clusters"][f] in seen:
            continue
        seen.add(rules["clusters"][f])
        reps.append(f)
    abstained = not reps
    find = 0.0 if rules["is_null"] or leaked or abstained else float(rules["find_first"][reps[0]]["find"])
    efficiency = 1.0 if spent <= 0 else min(1.0, reference_cost / spent)
    return {"find": find, "leaked": leaked, "abstained": abstained, "efficiency": efficiency}


def check_browser_rule(
    fids: list[str], key: Any, rules: dict[str, Any], spends: list[float], longest: int = 3
) -> int:
    """Every ordered list of up to ``longest`` distinct features (and the empty list), at each spend:
    the browser rule must reproduce ``score_world`` exactly. Returns the number of lists checked."""
    from itertools import permutations

    lists: list[tuple[str, ...]] = [()]
    for k in range(1, longest + 1):
        lists.extend(permutations(fids, k))
    for listed in lists:
        for spent in spends:
            real = scoring.score_world(list(listed), key, spent=spent, sequential=True)
            page = browser_score(list(listed), rules, key.reference_cost, spent)
            ok = (
                abs(real.find - page["find"]) <= 1e-9
                and abs(real.efficiency - page["efficiency"]) <= 1e-9
                and real.leaked == page["leaked"]
                and real.abstained == page["abstained"]
            )
            if not ok:
                raise SystemExit(
                    f"browser rule disagrees with score_world on {listed} at spent {spent}: {page}"
                )
    return len(lists)


def play(trip: dict[str, Any]) -> dict[str, Any]:
    """The journey world as a task the visitor plays: buy a stage, read the evidence, list, be scored."""
    store = FileWorldStore(fixture_store())
    world = store.world(JOURNEY_WORLD)
    key = store.answer_key(JOURNEY_WORLD)
    card = world.card
    fids = list(card.feature_ids())
    if key.depth != 1:
        raise SystemExit(f"{JOURNEY_WORLD} has depth {key.depth}; the browser rule assumes depth 1")

    stages = []
    spent_at: dict[int, float] = {}
    for target in STAGES:
        view = stage_view(world, target)
        evidence = welch_log10p(view.x, view.outcome)
        spent_at[target] = float(view.spent)
        stages.append(
            {
                "patients": len(view.rows),
                "spent": round(view.spent, 2),
                "evidence": {fids[j]: round(float(evidence[j]), 3) for j in range(len(fids))},
            }
        )
    analysed = {s["patients"]: s for s in trip["steps"] if s["kind"] == "analyse"}
    for stage in stages:
        step = analysed.get(stage["patients"])
        if step is not None and (step["evidence"] != stage["evidence"] or step["spent"] != stage["spent"]):
            raise SystemExit(f"play stage {stage['patients']} disagrees with the journey's analyse step")
    if not {30, 60, 90} <= set(analysed):
        raise SystemExit("the journey no longer analyses 30, 60 and 90 patients; recheck play.json")

    neutral = scoring.neutral_features(key)
    find_first = {}
    for f in fids:
        single = scoring.score_world([f], key, spent=key.reference_cost, sequential=True)
        find_first[f] = {
            "find": r(single.find),
            "find_exact": r(single.find_exact),
            "chance": r(single.chance_recovery),
            "raw": r(single.raw_recovery),
        }
    rules = {
        "depth": key.depth,
        "is_null": key.is_null,
        "reject": list(key.reject_set),
        "neutral": sorted(neutral),
        "clusters": {f: key.clusters[f] for f in fids},
        "find_first": find_first,
        "rule": PLAY_RULE,
    }
    below = max(t for t in STAGES if spent_at[t] < key.reference_cost)
    above = min(t for t in STAGES if spent_at[t] > key.reference_cost)
    checked = check_browser_rule(fids, key, rules, [spent_at[below], spent_at[above]])
    print(
        f"play.json: browser rule matches score_world on {checked} lists "
        f"({checked - 1} non-empty + the empty list) at spends {spent_at[below]:g} and {spent_at[above]:g}"
    )
    return {
        "world": card.world_id,
        "n_pool": card.n_pool,
        "budget": card.budget,
        "reference_cost": key.reference_cost,
        "recruit_price": card.prices.recruit_per_patient,
        "features": feature_rows(card),
        "agent_threshold": round(bonferroni_line(card), 3),
        "stages": stages,
        "scoring": rules,
        "check": {"lists": checked, "longest": 3, "spends": [spent_at[below], spent_at[above]]},
    }


# --------------------------------------------------------------------------- what makes it hard: one world per mechanism

CATALOGUE = (
    "toy-driver",
    "toy-stand-in",
    "toy-wrong-type",
    "toy-confounder",
    "toy-leak",
    "toy-interaction",
    "toy-module",
    "toy-neutral",
    "toy-null-a",
    "toy-null-b",
)
PREVIEW_ROWS = 48

# Columns the builder plants as decoys (tools/build_toy.py). The answer key does not name them: it holds
# only what earns credit (truth, equivalents), what is ignored (neutral) and what zeroes a world (reject).
BUILDER_DECOYS: dict[str, dict[str, str]] = {
    "toy-wrong-type": {"e06": "expression target driven by the copy-number cause; a downstream correlate"},
    "toy-confounder": {
        "e08": "expression shifted by the confounding subtype; a correlate only",
        "e09": "expression shifted by the confounding subtype; a correlate only",
    },
    "toy-leak": {"q01": "the post-outcome lab column built from the outcome itself"},
}

NOTES: dict[str, str] = {
    "toy-driver": "One single-credit group; nothing else is planted.",
    "toy-stand-in": "The truth's near-duplicate is in its equivalence set, so either earns credit "
    "(exact_recoverable is false).",
    "toy-wrong-type": "The key names only the copy-number cause. Its downstream expression target is not "
    "in the key; it is identified from the builder (builder_decoys) and earns nothing.",
    "toy-confounder": "The key's truth is the observed confounder itself (a clinical subtype indicator). "
    "The expression columns it shifts are not in the key; they are identified from the builder "
    "(builder_decoys) and earn nothing.",
    "toy-leak": "Every world has three post-outcome columns in reject; the key does not say which one "
    "tracks the outcome. The builder identifies it (builder_decoys).",
    "toy-interaction": "A joint group: both parts must be listed within the top R = 2 representatives, "
    "else the group earns nothing. Both partners are named by the key.",
    "toy-module": "Weighted coverage: each listed member earns its share of the absolute weights.",
    "toy-neutral": "A second planted effect too small to recover is a neutral group: listing it neither "
    "helps nor hurts.",
    "toy-null-a": "No signal: the correct answer is an empty list (one recruitment stratum).",
    "toy-null-b": "No signal, in a world with two recruitment strata. Kept as its own entry.",
}


def builder_ids(spec_name: str, world: Any) -> dict[str, str]:
    """The builder's internal column names mapped to the full-access world's public ids, by replaying
    ``build_toy.build_one`` up to its naming step (same seed, same draws); checked against the world."""
    spec = next(s for s in build_toy.SPECS if s.world == spec_name)
    rng = np.random.default_rng(build_toy.seed_of(f"{build_toy.BUILDER_VERSION}:{spec_name}-full"))
    draft = build_toy.Draft(rng)
    build_toy.base_columns(draft)
    signal, _groups = spec.plant(draft)
    y = build_toy.outcome(signal, rng)
    build_toy.post_outcome(draft, y, leak=spec_name == "toy-leak")
    internal = list(draft.columns)
    order = [internal[i] for i in rng.permutation(len(internal))]
    public = dict(zip(order, build_toy.fake_names(len(order), rng), strict=True))
    x = np.column_stack([draft.columns[f] for f in order])
    if [public[f] for f in order] != list(world.card.feature_ids()) or not (
        np.array_equal(x, world.x) and np.array_equal(y, world.y)
    ):
        raise SystemExit(f"replaying the builder does not reproduce {spec_name}-full")
    return public


def naive_screen(world: Any, key: Any, baseline_only: bool = True) -> dict[str, Any]:
    """A marginal screen on the full pool (Welch t per measurement, Bonferroni 5%), scored full-access.

    ``baseline_only`` is the journey agent's rule; without it the screen also tests post-outcome
    measurements, the shortcut that a leak punishes.
    """
    from onc_agi.core.schema import Timing

    card = world.card
    fids = list(card.feature_ids())
    timing = {f.feature_id: f.timing for f in card.features}
    tested = [j for j, f in enumerate(fids) if timing[f] is Timing.BASELINE or not baseline_only]
    ev = welch_log10p(world.x, world.y)
    cut = bonferroni_line(card) if baseline_only else -math.log10(0.05 / len(tested))
    ranking = [fids[j] for j in sorted(tested, key=lambda j: -ev[j]) if ev[j] > cut]
    sc = scoring.score_world(ranking, key, sequential=False)
    scope = "baseline measurement" if baseline_only else "measurement, post-outcome included"
    return {
        "rule": f"Welch t-test per {scope}, on the full pool, Bonferroni at 5%",
        "threshold": round(cut, 3),
        "ranking": ranking,
        "find": r(sc.find),
        "find_exact": r(sc.find_exact),
        "leaked": sc.leaked,
        "abstained": sc.abstained,
        "restrained": sc.restrained,
    }


def catalogue(trip: dict[str, Any]) -> list[dict[str, Any]]:
    """One full-access toy world per mechanism, with its roles, evidence and the naive screen's score."""
    store = FileWorldStore(fixture_store())
    across = {w["world"]: w for w in trip["across"]["worlds"]}
    specs = {s.world: s for s in build_toy.SPECS}
    records = []
    for name in CATALOGUE:
        world_id = f"{name}-full"
        world = store.world(world_id)
        key = store.answer_key(world_id)
        card = world.card
        fids = list(card.feature_ids())
        public = builder_ids(name, world)

        evidence = welch_log10p(world.x, world.y)
        shortcut = naive_screen(world, key)
        played = across[world_id]  # the same rule, played through the runtime by JourneyAgent
        if played["listed"] != shortcut["ranking"] or played["find"] != shortcut["find"]:
            raise SystemExit(f"naive screen on {world_id} disagrees with the journey agent's run")

        step = max(1, card.n_pool // PREVIEW_ROWS)
        rows = list(range(0, card.n_pool, step))[:PREVIEW_ROWS]
        sd = world.x.std(0)
        z = (world.x - world.x.mean(0)) / np.where(sd == 0, 1, sd)

        recoverable = key.recoverable
        records.append(
            {
                "world": world_id,
                "mechanic": MECHANIC[name],
                "description": specs[name].description,
                "n": card.n_pool,
                "outcome_type": str(card.outcome_type),
                "features": feature_rows(card, price=False),
                "evidence": {fids[j]: round(float(evidence[j]), 3) for j in range(len(fids))},
                "truth": [[p.true_feature for p in g.parts] for g in recoverable],
                "equivalent": [
                    f for g in recoverable for p in g.parts for f in p.equivalence_set if f != p.true_feature
                ],
                "reject": list(key.reject_set),
                "neutral": sorted(scoring.neutral_features(key)),
                "depth": key.depth,
                "is_null": key.is_null,
                "credit_rules": [g.credit_rule.value for g in recoverable],
                "groups": [
                    {
                        "id": g.group_id,
                        "role": g.role,
                        "label": g.label.value,
                        "credit_rule": g.credit_rule.value,
                        "parts": [
                            {
                                "truth": p.true_feature,
                                "equivalence_set": list(p.equivalence_set),
                                "exact_recoverable": p.exact_recoverable,
                                "weight": r(p.weight),
                            }
                            for p in g.parts
                        ],
                    }
                    for g in key.groups
                ],
                "builder_decoys": [
                    {"id": public[col], "role": text} for col, text in BUILDER_DECOYS.get(name, {}).items()
                ],
                "notes": NOTES[name],
                "shortcut": shortcut,
                "shortcut_unfiltered": naive_screen(world, key, baseline_only=False),
                "z_preview": {
                    "rows": rows,
                    "z": [[round(float(v), 1) for v in z[i]] for i in rows],
                    "outcome": [int(world.y[i]) for i in rows],
                    "outcome_note": f"world.y as stored ({card.outcome_type}; 1 = outcome present)",
                },
            }
        )
    return records


# --------------------------------------------------------------------------- what makes it hard


def signed_welch(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(-log10 p, sign of the mean difference) for one column, by Welch t-test."""
    from scipy import stats

    yb = y.astype(bool)
    res = stats.ttest_ind(x[yb], x[~yb], equal_var=False)
    return -math.log10(max(float(res.pvalue), 1e-300)), float(np.sign(res.statistic))


def stratified(x: np.ndarray, y: np.ndarray, strata: np.ndarray) -> tuple[float, float]:
    """Welch within each stratum, combined by Stouffer's method: the association with the
    stratum held fixed. Returns (-log10 p, sign)."""
    from scipy import stats

    zs, ws = [], []
    for s in np.unique(strata):
        m = strata == s
        yb = y[m].astype(bool)
        if yb.all() or not yb.any():
            continue
        res = stats.ttest_ind(x[m][yb], x[m][~yb], equal_var=False)
        zs.append(stats.norm.isf(float(res.pvalue) / 2) * np.sign(res.statistic))
        ws.append(math.sqrt(m.sum()))
    z = float(np.dot(zs, ws) / math.sqrt(np.dot(ws, ws)))
    p = 2 * stats.norm.sf(abs(z))
    return -math.log10(max(p, 1e-300)), float(np.sign(z))


def logistic(rng: np.random.Generator, logit: np.ndarray) -> np.ndarray:
    return (rng.random(len(logit)) < 1 / (1 + np.exp(-logit))).astype(float)


def failures(cat: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Failure cases for the "what makes it hard" gallery: for each mechanism, what a naive
    shortcut does and what the analysis it calls for finds. The leak and no-signal cases are the
    real toy worlds, scored by the real scorer; the others are seeded synthetic demonstrations,
    labelled as such on the page, built so the shortcut visibly fails."""
    out: list[dict[str, Any]] = []
    rnd = lambda v: round(float(v), 3)  # noqa: E731
    by: dict[str, dict[str, Any]] = {}
    for e in cat:
        by.setdefault(e["mechanic"], e)  # the first of each: toy-null-a for no signal

    # 1. Leak (real toy world): the unfiltered screen ranks a post-outcome measurement first.
    leak = by["leak"]
    post = set(leak["reject"])
    out.append(
        {
            "key": "leak",
            "synthetic": False,
            "world": leak["world"],
            "n": leak["n"],
            "cols": [
                {
                    "id": f["id"],
                    "type": f["type"],
                    "post": f["timing"] == "post_outcome",
                    "role": (
                        "truth"
                        if any(f["id"] in group for group in leak["truth"])
                        else (
                            "leak"
                            if f["id"] in [d["id"] for d in leak["builder_decoys"]]
                            else "post" if f["id"] in post else ""
                        )
                    ),
                }
                for f in leak["features"]
            ],
            "evidence": leak["evidence"],
            "line": leak["shortcut_unfiltered"]["threshold"],
            "shortcut": {
                "ranking": leak["shortcut_unfiltered"]["ranking"],
                "find": leak["shortcut_unfiltered"]["find"],
                "leaked": True,
            },
            "proper": {
                "ranking": leak["shortcut"]["ranking"],
                "find": leak["shortcut"]["find"],
                "leaked": False,
            },
        }
    )

    # 2. No signal (real toy world): every measurement stays below the line; naming the
    #    strongest anyway is a false claim, and only returning nothing earns Restraint.
    null = by["no_signal"]
    base = [f["id"] for f in null["features"] if f["timing"] == "baseline"]
    top = max(base, key=lambda f: null["evidence"][f])
    out.append(
        {
            "key": "no_signal",
            "synthetic": False,
            "world": null["world"],
            "n": null["n"],
            "cols": [
                {"id": f["id"], "type": f["type"], "post": f["timing"] == "post_outcome", "role": ""}
                for f in null["features"]
            ],
            "evidence": null["evidence"],
            "line": null["shortcut"]["threshold"],
            "shortcut": {"ranking": [top], "claims": True},
            "proper": {"ranking": [], "restrained": null["shortcut"]["restrained"]},
        }
    )

    # 3. Interaction (synthetic): the outcome follows the product of two measurements, so
    #    neither is associated on its own; a marginal screen sees nothing.
    rng = np.random.default_rng(build_toy.seed_of("site-failure-interaction"))
    n = 400
    x = rng.standard_normal((n, 10))
    y = logistic(rng, 2.6 * x[:, 0] * x[:, 1])
    ids = ["a1", "a2"] + [f"n{k}" for k in range(1, 9)]
    ev = {ids[j]: rnd(signed_welch(x[:, j], y)[0]) for j in range(10)}
    prod = rnd(signed_welch(x[:, 0] * x[:, 1], y)[0])
    k = rng.choice(n, 220, replace=False)
    out.append(
        {
            "key": "interaction",
            "synthetic": True,
            "n": n,
            "cols": [
                {"id": i, "type": "expression", "post": False, "role": "truth" if i in ("a1", "a2") else ""}
                for i in ids
            ],
            "evidence": ev,
            "line": rnd(-math.log10(0.05 / 10)),
            "joint": {"pair": ["a1", "a2"], "evidence": prod},
            "scatter": {
                "x": [rnd(v) for v in x[k, 0]],
                "y": [rnd(v) for v in x[k, 1]],
                "outcome": [int(v) for v in y[k]],
            },
            "shortcut": {
                "ranking": [i for i in sorted(ev, key=lambda i: -ev[i]) if ev[i] > -math.log10(0.05 / 10)]
            },
            "proper": {"ranking": ["a1", "a2"]},
        }
    )

    # 4. Confounding (synthetic): a clinical subtype raises risk and lifts five genes; a weaker
    #    gene acts on its own. The screen ranks the lifted genes first; within each subtype only
    #    the true gene and the subtype remain.
    rng = np.random.default_rng(build_toy.seed_of("site-failure-confounder"))
    n = 360
    s = (rng.random(n) < 0.5).astype(float)
    g = 1.8 * s[:, None] + rng.standard_normal((n, 5))
    xg = rng.standard_normal(n)
    other = rng.standard_normal((n, 6))
    y = logistic(rng, -1.1 + 2.2 * s + 0.55 * xg)
    cols = {
        "subtype": s,
        **{f"g{k + 1}": g[:, k] for k in range(5)},
        "x7": xg,
        **{f"o{k + 1}": other[:, k] for k in range(6)},
    }
    naive = {c: rnd(signed_welch(v, y)[0]) for c, v in cols.items()}
    adj = {c: (rnd(stratified(v, y, s)[0]) if c != "subtype" else naive[c]) for c, v in cols.items()}
    line = rnd(-math.log10(0.05 / len(cols)))
    genes = [c for c in cols if c != "subtype"]
    out.append(
        {
            "key": "confounder",
            "synthetic": True,
            "n": n,
            "cols": [
                {
                    "id": c,
                    "type": "clinical" if c == "subtype" else "expression",
                    "post": False,
                    "role": "truth" if c in ("x7", "subtype") else "lifted" if c.startswith("g") else "",
                }
                for c in cols
            ],
            "evidence": naive,
            "adjusted": adj,
            "line": line,
            "shortcut": {
                "ranking": [c for c in sorted(genes, key=lambda c: -naive[c]) if naive[c] > line][:3]
            },
            "proper": {"ranking": [c for c in sorted(genes, key=lambda c: -adj[c]) if adj[c] > line]},
        }
    )

    # 5. Wrong data type (synthetic): a copy-number change drives the outcome and switches on a
    #    gene; reading only expression finds the switched-on gene, not the cause.
    rng = np.random.default_rng(build_toy.seed_of("site-failure-wrong-type"))
    n = 300
    cn = rng.choice([-1, 0, 0, 1, 2], n).astype(float) + 0.15 * rng.standard_normal(n)
    target = 0.85 * cn + 0.55 * rng.standard_normal(n)
    expr = rng.standard_normal((n, 7))
    cn_other = rng.standard_normal((n, 3))
    y = logistic(rng, -0.6 + 1.1 * cn)
    cols = {
        "c1": cn,
        "c2": cn_other[:, 0],
        "c3": cn_other[:, 1],
        "c4": cn_other[:, 2],
        "e1": target,
        **{f"e{k + 2}": expr[:, k] for k in range(7)},
    }
    types = {c: ("copy_number" if c.startswith("c") else "expression") for c in cols}
    ev = {c: rnd(signed_welch(v, y)[0]) for c, v in cols.items()}
    line = rnd(-math.log10(0.05 / len(cols)))
    out.append(
        {
            "key": "wrong_data_type",
            "synthetic": True,
            "n": n,
            "cols": [
                {
                    "id": c,
                    "type": types[c],
                    "post": False,
                    "role": "truth" if c == "c1" else "downstream" if c == "e1" else "",
                }
                for c in cols
            ],
            "evidence": ev,
            "line": line,
            "shortcut": {
                "ranking": [
                    c for c in sorted(ev, key=lambda c: -ev[c]) if types[c] == "expression" and ev[c] > line
                ][:1]
            },
            "proper": {"ranking": [c for c in sorted(ev, key=lambda c: -ev[c]) if ev[c] > line][:1]},
        }
    )

    # 6. Simpson's paradox (synthetic): within each of two batches the marker raises risk, but
    #    the high-marker batch has fewer events, so pooling reverses the association.
    rng = np.random.default_rng(build_toy.seed_of("site-failure-simpson"))
    n = 400
    batch = (rng.random(n) < 0.5).astype(float)
    m = rng.standard_normal(n) + 2.6 * batch
    y = logistic(rng, -0.2 + 0.9 * (m - 2.6 * batch) - 2.6 * batch)
    pooled = signed_welch(m, y)
    within = stratified(m, y, batch)
    k = rng.choice(n, 240, replace=False)
    out.append(
        {
            "key": "simpson",
            "synthetic": True,
            "n": n,
            "pooled": {"evidence": rnd(pooled[0]), "sign": pooled[1]},
            "within": {"evidence": rnd(within[0]), "sign": within[1]},
            "scatter": {
                "x": [rnd(v) for v in m[k]],
                "batch": [int(v) for v in batch[k]],
                "outcome": [int(v) for v in y[k]],
            },
        }
    )
    return out


HERO_WORLDS = ("toy-driver-full", "toy-null-a-full")


def hero() -> dict[str, Any]:
    """The opening scene's two worlds in full: every patient's standardised measurements.

    One world has a planted cause and one has none; the page draws both as cohort fields.
    """
    store = FileWorldStore(fixture_store())
    out: dict[str, Any] = {}
    for world_id in HERO_WORLDS:
        world = store.world(world_id)
        z = (world.x - world.x.mean(0)) / np.where(world.x.std(0) == 0, 1, world.x.std(0))
        out[world_id] = {
            "features": [f.feature_id for f in world.card.features],
            "z": [[round(float(v), 1) for v in row] for row in z],
            "outcome": [int(v) for v in world.y],
        }
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "smoke.json").write_text(json.dumps(smoke(), indent=1) + "\n")
    (OUT / "oracle.json").write_text(json.dumps(oracle(), indent=1) + "\n")
    trip = journey()
    (OUT / "journey.json").write_text(json.dumps(trip, indent=1) + "\n")
    (OUT / "play.json").write_text(json.dumps(play(trip), indent=1) + "\n")
    cat = catalogue(trip)
    (OUT / "catalogue.json").write_text(json.dumps(cat, indent=1) + "\n")
    (OUT / "failures.json").write_text(json.dumps(failures(cat), indent=1) + "\n")
    (OUT / "hero.json").write_text(json.dumps(hero(), separators=(",", ":")) + "\n")
    print(
        f"wrote {OUT.relative_to(ROOT)}/smoke.json, oracle.json, journey.json, play.json, catalogue.json and hero.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
