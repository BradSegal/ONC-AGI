"""Build the ONC-AGI toy fixture worlds.

These are **contract fixtures, not benchmark tasks**. Each one exercises a scoring
rule or a mechanic in the simplest readable way so that an agent or harness can
be developed and tested offline. Benchmark worlds are built differently: from
real cohorts, by a private generator, with answer keys certified by a Monte Carlo
oracle. Nothing here is drawn from, or describes, that generator.

Every toy world:

* has 240 patients and the same 20 columns by data type (10 expression, 3 copy
  number, 2 protein, 2 clinical, 2 derived and 1 baseline lab), plus the same 3
  post-outcome lab columns, so the column layout says nothing about the world's
  mechanism;
* uses fake feature names (lowercase, so they can never collide with HGNC
  symbols) in a per-world random order;
* plants its mechanism with large effects, so the answer key below is correct
  **by construction**; the labels are not oracle-certified;
* exists in full-access and sequential mode as two separate draws, so no two bundles
  share a pool (as for benchmark worlds).

Run from the repository root (deterministic; the output is committed)::

    uv run python tools/build_toy.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from onc_agi.core.schema import (
    AnswerKey,
    CreditRule,
    FeatureMeta,
    GroupLabel,
    Mode,
    NameVisibility,
    PriceList,
    Tier,
    Timing,
    TrueGroup,
    TruthPart,
    WorldCard,
)
from onc_agi.core.world import WorldData
from onc_agi.infra.bundles import write_world
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

BUILDER_VERSION = "toy-1"
STORE = Path(__file__).resolve().parents[1] / "src" / "onc_agi" / "fixtures" / "store"
N = 240
PREVALENCE = 0.35
RECRUIT_PRICE = 40.0
PRICES = {"expression": 2.0, "copy_number": 3.0, "protein": 8.0, "clinical": 0.5, "derived": 1.0, "lab": 5.0}
CLUSTER_R = 0.8  # complete linkage at |r| >= 0.8 (scoring specification)
CLUSTER_CAP = 10
NULL_DRAWS = 200

type Array = NDArray[np.float64]


@dataclass
class Draft:
    """Columns under internal names; ``build`` renames them before writing."""

    rng: np.random.Generator
    columns: dict[str, Array] = field(default_factory=dict)
    types: dict[str, str] = field(default_factory=dict)
    timing: dict[str, Timing] = field(default_factory=dict)

    def add(self, name: str, values: Array, data_type: str, timing: Timing = Timing.BASELINE) -> None:
        self.columns[name] = np.asarray(values, dtype=float)
        self.types[name] = data_type
        self.timing[name] = timing

    def z(self, name: str) -> Array:
        v = self.columns[name]
        return (v - v.mean()) / v.std()


@dataclass(frozen=True)
class Group:
    """One true group under internal names (renamed when the key is written)."""

    group_id: str
    role: str
    label: GroupLabel
    parts: tuple[tuple[str, tuple[str, ...], bool, float], ...]  # (true, equivalence set, exact?, weight)
    credit_rule: CreditRule = CreditRule.SINGLE


@dataclass(frozen=True)
class Spec:
    world: str
    description: str
    tier: int
    plant: Callable[[Draft], tuple[Array, list[Group]]]
    strata: tuple[str, ...] = ("all",)
    n_ref: int = 160


def seed_of(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def base_columns(draft: Draft) -> None:
    """The shared layout: blocks of correlated expression, copy number, protein, clinical, derived, lab."""
    rng = draft.rng
    shared_a = rng.normal(size=N)
    draft.add("e01", shared_a + 0.25 * rng.normal(size=N), "expression")
    draft.add("e02", shared_a + 0.25 * rng.normal(size=N), "expression")  # near-duplicate of e01 (r ~ 0.94)
    shared_b = rng.normal(size=N)
    for name in ("e03", "e04", "e05"):
        draft.add(name, shared_b + 1.0 * rng.normal(size=N), "expression")  # a moderate block (r ~ 0.5)
    for name in ("c01", "c02", "c03"):
        draft.add(name, rng.normal(size=N), "copy_number")
    draft.add("e06", 0.7 * draft.columns["c01"] + 0.7 * rng.normal(size=N), "expression")  # driven by c01
    for name in ("e07", "e08", "e09", "e10"):
        draft.add(name, rng.normal(size=N), "expression")
    draft.add("p01", rng.normal(size=N), "protein")
    draft.add("p02", 0.6 * draft.columns["e10"] + 0.8 * rng.normal(size=N), "protein")
    draft.add("s01", (rng.random(N) < 0.45).astype(float), "clinical")  # subtype indicator
    draft.add("s02", (rng.random(N) < 0.5).astype(float), "clinical")  # site indicator
    draft.add("d01", rng.normal(size=N), "derived")
    draft.add("d02", rng.normal(size=N), "derived")
    draft.add("l01", rng.normal(size=N), "lab")


def post_outcome(draft: Draft, y: NDArray[np.int64], leak: bool) -> list[str]:
    """Three post-outcome columns in every world; in the leak world one tracks the outcome."""
    rng = draft.rng
    names = ["q01", "q02", "q03"]
    for name in names:
        draft.add(name, rng.normal(size=N), "lab", Timing.POST_OUTCOME)
    if leak:
        draft.columns["q01"] = y + 0.5 * rng.normal(size=N)
    return names


def outcome(signal: Array, rng: np.random.Generator) -> NDArray[np.int64]:
    """Binary outcome at the target prevalence: logit = alpha + signal."""
    lo, hi = -20.0, 20.0
    for _ in range(60):
        alpha = (lo + hi) / 2
        if float(np.mean(1 / (1 + np.exp(-(alpha + signal))))) > PREVALENCE:
            hi = alpha
        else:
            lo = alpha
    p = 1 / (1 + np.exp(-(alpha + signal)))
    return (rng.random(N) < p).astype(np.int64)


# --------------------------------------------------------------------------- mechanisms


def single(name: str, role: str, equivalent: tuple[str, ...] = (), exact: bool = True) -> Group:
    return Group(f"g-{role}", role, GroupLabel.RECOVERABLE, ((name, (name, *equivalent), exact, 1.0),))


def plant_null(draft: Draft) -> tuple[Array, list[Group]]:
    return np.zeros(N), []


def plant_driver(draft: Draft) -> tuple[Array, list[Group]]:
    return 1.6 * draft.z("e07"), [single("e07", "generating")]


def plant_stand_in(draft: Draft) -> tuple[Array, list[Group]]:
    # e01 generates; e02 is a near-duplicate the data cannot tell apart, so either earns credit.
    return 1.6 * draft.z("e01"), [single("e01", "stand_in", equivalent=("e02",), exact=False)]


def plant_wrong_type(draft: Draft) -> tuple[Array, list[Group]]:
    # Copy number c01 drives both e06 and the outcome; e06 is downstream and earns nothing.
    return 1.6 * draft.z("c01"), [single("c01", "wrong_data_type")]


def plant_confounder(draft: Draft) -> tuple[Array, list[Group]]:
    # Subtype s01 drives the outcome and shifts e08 and e09; those two are only correlates.
    s = draft.columns["s01"]
    for name in ("e08", "e09"):
        draft.columns[name] = draft.columns[name] + 1.2 * (s - s.mean())
    return 2.2 * draft.z("s01"), [single("s01", "observed_confounder")]


def plant_leak(draft: Draft) -> tuple[Array, list[Group]]:
    # A real driver plus (added later) a post-outcome column that tracks the outcome.
    return 1.6 * draft.z("e03"), [single("e03", "generating")]


def plant_interaction(draft: Draft) -> tuple[Array, list[Group]]:
    # Only the product matters: neither member has a marginal effect.
    a, b = draft.z("e09"), draft.z("e10")
    group = Group(
        "g-interaction",
        "interaction",
        GroupLabel.RECOVERABLE,
        (("e09", ("e09",), True, 1.0), ("e10", ("e10",), True, 1.0)),
        CreditRule.JOINT,
    )
    return 1.8 * a * b, [group]


def plant_module(draft: Draft) -> tuple[Array, list[Group]]:
    # A three-member module credited by weighted coverage of its absolute weights.
    weights = {"p01": 1.2, "c02": 0.9, "e04": 0.7}
    signal = sum(w * draft.z(f) for f, w in weights.items())
    group = Group(
        "g-module",
        "module",
        GroupLabel.RECOVERABLE,
        tuple((f, (f,), True, w) for f, w in weights.items()),
        CreditRule.WEIGHTED_COVERAGE,
    )
    return np.asarray(signal, dtype=float), [group]


def plant_neutral(draft: Draft) -> tuple[Array, list[Group]]:
    # One strong driver and one planted effect too small to recover: listing it neither helps nor hurts.
    neutral = Group("g-neutral", "generating", GroupLabel.NEUTRAL, (("d01", ("d01",), False, 0.05),))
    return 1.6 * draft.z("e05") + 0.05 * draft.z("d01"), [single("e05", "generating"), neutral]


SPECS = (
    Spec("toy-null-a", "No signal: the correct answer is an empty list.", 0, plant_null),
    Spec("toy-null-b", "No signal, in a world with two recruitment strata.", 2, plant_null, ("s-a", "s-b")),
    Spec("toy-driver", "One strong expression driver.", 0, plant_driver, n_ref=120),
    Spec("toy-stand-in", "A driver with a near-duplicate partner; either earns credit.", 1, plant_stand_in),
    Spec(
        "toy-wrong-type",
        "A copy-number cause whose expression target is only a correlate.",
        1,
        plant_wrong_type,
    ),
    Spec(
        "toy-confounder",
        "An observed clinical confounder drives the outcome.",
        0,
        plant_confounder,
        ("s-a", "s-b"),
    ),
    Spec(
        "toy-leak",
        "A real driver plus a post-outcome column that tracks the outcome.",
        0,
        plant_leak,
        n_ref=120,
    ),
    Spec("toy-interaction", "Two features matter only jointly.", 2, plant_interaction, n_ref=240),
    Spec("toy-module", "A three-member module credited by weighted coverage.", 2, plant_module, n_ref=200),
    Spec("toy-neutral", "One driver and one unrecoverable planted effect.", 1, plant_neutral),
)


# --------------------------------------------------------------------------- answer-key helpers


def clusters_of(x: Array, names: list[str]) -> dict[str, int]:
    """Complete-linkage clusters at |r| >= 0.8, capped at 10 members (truth-independent)."""
    corr = np.abs(np.corrcoef(x, rowvar=False))
    np.fill_diagonal(corr, 1.0)
    dist = squareform(np.clip(1.0 - corr, 0.0, None), checks=False)
    labels = fcluster(linkage(dist, method="complete"), t=1.0 - CLUSTER_R, criterion="distance")
    out: dict[str, int] = {}
    counts: Counter[int] = Counter()
    for name, label in zip(names, labels, strict=True):
        index = int(label) * 100 + counts[int(label)] // CLUSTER_CAP
        counts[int(label)] += 1
        out[name] = index
    return out


def strata_of(x: Array, types: list[str], min_size: int = 4) -> dict[int, str]:
    """Truth-independent strata used for matched chance (scoring specification).

    Finest level: data type x max-|r| band (<0.5, 0.5-0.8, >=0.8) x hub-degree band
    (number of |r| > 0.5 partners: 0, 1-2, 3+). Within each data type, levels are
    coarsened (drop the degree band, then the correlation band) until every stratum
    has at least ``min_size`` members; anything still smaller becomes ``mixed``.
    """
    corr = np.abs(np.corrcoef(x, rowvar=False))
    np.fill_diagonal(corr, 0.0)
    cband = np.digitize(corr.max(axis=0), [0.5, 0.8])
    dband = np.digitize((corr > 0.5).sum(axis=0), [1, 3])
    levels = [
        [f"{t}|c{c}|d{d}" for t, c, d in zip(types, cband, dband, strict=True)],
        [f"{t}|c{c}" for t, c in zip(types, cband, strict=True)],
        list(types),
    ]
    labels = ["" for _ in types]
    for data_type in sorted(set(types)):
        idx = [i for i, t in enumerate(types) if t == data_type]
        for level in levels:
            counts = Counter(level[i] for i in idx)
            if all(counts[level[i]] >= min_size for i in idx) or level is levels[-1]:
                for i in idx:
                    labels[i] = level[i]
                break
    counts = Counter(labels)
    return {j: (lab if counts[lab] >= min_size else "mixed") for j, lab in enumerate(labels)}


def null_threshold(x: Array, rng: np.random.Generator) -> float:
    """95th percentile of the largest |z| any feature reaches under permuted outcomes."""
    z = (x - x.mean(axis=0)) / np.where(x.std(axis=0) == 0, 1.0, x.std(axis=0))
    base = (rng.random(N) < PREVALENCE).astype(float)
    maxima = []
    for _ in range(NULL_DRAWS):
        y = rng.permutation(base)
        yc = (y - y.mean()) / y.std()
        maxima.append(float(np.max(np.abs(z.T @ yc)) / np.sqrt(N)))
    return float(np.quantile(maxima, 0.95))


def fake_names(n: int, rng: np.random.Generator) -> list[str]:
    letters = "bcdfghjklmnpqrstvwxz"
    out: list[str] = []
    while len(out) < n:
        name = "".join(rng.choice(list(letters), size=3)) + str(int(rng.integers(1, 10)))
        if name not in out:
            out.append(name)
    return out


# --------------------------------------------------------------------------- world assembly


def build(spec: Spec) -> list[tuple[WorldData, AnswerKey]]:
    """The spec's world in both modes; each mode is its own draw, so no two bundles share a pool."""
    return [build_one(spec, mode) for mode in (Mode.FULL_ACCESS, Mode.SEQUENTIAL)]


def build_one(spec: Spec, mode: Mode) -> tuple[WorldData, AnswerKey]:
    world_id = f"{spec.world}-{'full' if mode is Mode.FULL_ACCESS else 'seq'}"
    rng = np.random.default_rng(seed_of(f"{BUILDER_VERSION}:{world_id}"))
    draft = Draft(rng)
    base_columns(draft)
    signal, groups = spec.plant(draft)
    y = outcome(signal, rng)
    post = post_outcome(draft, y, leak=spec.world == "toy-leak")

    internal = list(draft.columns)
    order = [internal[i] for i in rng.permutation(len(internal))]
    public = dict(zip(order, fake_names(len(order), rng), strict=True))
    x = np.column_stack([draft.columns[f] for f in order])
    types = [draft.types[f] for f in order]
    ids = [public[f] for f in order]
    stratum_label = {j: s for j, s in strata_of(x, types).items()}
    features = tuple(
        FeatureMeta(
            feature_id=public[f],
            data_type=draft.types[f],
            timing=draft.timing[f],
            assay_price=PRICES[draft.types[f]],
        )
        for f in order
    )
    per_patient = RECRUIT_PRICE + sum(f.assay_price for f in features)

    if spec.strata == ("all",):
        stratum = tuple("all" for _ in range(N))
    else:
        s = draft.columns["s01"]
        stratum = tuple(spec.strata[0] if v > 0.5 else spec.strata[1] for v in s)
    patients = tuple(f"p{i:04d}" for i in rng.permutation(N))
    queues = {
        s: tuple(int(i) for i in rng.permutation([i for i, v in enumerate(stratum) if v == s]))
        for s in spec.strata
    }
    key_groups = tuple(
        TrueGroup(
            group_id=g.group_id,
            role=g.role,
            label=g.label,
            credit_rule=g.credit_rule,
            parts=tuple(
                TruthPart(
                    true_feature=public[t],
                    equivalence_set=tuple(public[e] for e in eq),
                    exact_recoverable=exact,
                    weight=w,
                )
                for t, eq, exact, w in g.parts
            ),
        )
        for g in groups
    )
    threshold = null_threshold(x, rng)
    card = WorldCard(
        world_id=world_id,
        tier=Tier.PUBLIC_TRAIN,
        mode=mode,
        n_pool=N,
        features=features,
        strata=spec.strata,
        stratum_sizes={name: len(queue) for name, queue in queues.items()},
        prices=PriceList(recruit_per_patient=RECRUIT_PRICE),
        budget=N * per_patient,
        name_visibility=NameVisibility.FAKE,
    )
    world = WorldData(card=card, patient_ids=patients, x=x, y=y, stratum=stratum, queues=queues)
    key = AnswerKey(
        world_id=world_id,
        difficulty_tier=spec.tier,
        groups=key_groups,
        reject_set=tuple(public[f] for f in post),
        clusters=clusters_of(x, ids),
        strata={ids[j]: label for j, label in stratum_label.items()},
        reference_cost=spec.n_ref * per_patient,
        detection_threshold=threshold,
        oracle_version=f"{BUILDER_VERSION}-by-construction",
    )
    return world, key


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the ONC-AGI toy fixture worlds.")
    parser.add_argument(
        "--out", type=Path, default=STORE, help="store directory (default: the packaged store)"
    )
    out: Path = parser.parse_args(argv).out
    shutil.rmtree(out, ignore_errors=True)
    index = []
    for spec in SPECS:
        for world, key in build(spec):
            write_world(out, world, key)
            index.append(
                {
                    "world_id": world.card.world_id,
                    "mode": world.card.mode.value,
                    "difficulty_tier": key.difficulty_tier,
                    "depth": key.depth,
                    "description": spec.description,
                }
            )
    manifest = {
        "builder": BUILDER_VERSION,
        "kind": "toy contract fixtures (not benchmark tasks; answer keys by construction)",
        "worlds": index,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    nulls = sum(1 for w in index if w["depth"] == 0)
    print(f"wrote {len(index)} toy worlds ({nulls} with no signal) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
