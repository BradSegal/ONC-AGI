"""Build the site's data files from the real runtime.

* ``src/data/smoke.json`` - the stdout of the public ``onc-agi smoke`` command, run in
  process on the packaged fixture worlds; the page prints it verbatim as expected output.
* ``src/data/oracle.json`` - the oracle's method demonstrated on a toy world: the
  null maximum statistic over permuted outcomes, its 95th-percentile threshold,
  and detection rates of the planted cause and its stand-in as the sample grows.
* ``src/data/journey.json`` - one recorded sequential episode on a toy world, scored, and
  the same policy played on every toy world.

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


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "smoke.json").write_text(json.dumps(smoke(), indent=1) + "\n")
    (OUT / "oracle.json").write_text(json.dumps(oracle(), indent=1) + "\n")
    (OUT / "journey.json").write_text(json.dumps(journey(), indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}/smoke.json, oracle.json and journey.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
