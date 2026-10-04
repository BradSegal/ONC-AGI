"""Seeded stratified world samples: every subset of ``n`` worlds is representative and reproducible.

A world's stratum is ``source/family/mode``. ``source`` is the cohort the world was derived from and
``family`` its role (the mechanism planted in it), both read from the world-set manifest when the
store carries one; null worlds form their own ``null`` family whatever role their twin had. A null is
recognised from the published answer key, so only public-train worlds can be told apart that way.
``mode`` is on every card. What the store does not show is ``unknown``. Family is truth, so it is
never read for a public-eval or private world, even from a manifest the store carries: those worlds
are stratified by source (when listed) and mode only, and their draws never read truth.

Seats are allocated in proportion to size, level by level: the ``n`` seats over sources, each
source's seats over its modes, each source-and-mode's seats over its families. Every allocation
rounds by largest remainder, so each count is within one of its proportional quota. Remainders are
ranked together with the rounding error the same field has carried from earlier allocations of its
level (ties then go by a seeded order), which keeps each mode's and each family's total close to its
share of ``n`` too. A joint stratum of 10 worlds has a quota near one seat in a sample of 240 from
2,000, so rounding the joint strata independently would push whole sources off their share; the
nesting prevents that. Each
stratum's seats are then filled by a seeded draw. The quota and drawn count of every stratum are the
sample's declared mix. A draw that would equal the first ``n`` sorted ids is redrawn, so a sample is
never the first-N selection it replaces unless the mix leaves no other choice.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import count

from onc_agi.core.errors import ArenaError
from onc_agi.core.ports import WorldStore
from onc_agi.core.schema import Tier, WorldCard
from onc_agi.services.scoring import stable_seed

NULL_FAMILY = "null"
UNKNOWN = "unknown"
STRATUM_FIELDS = ("source", "family", "mode")


@dataclass(frozen=True)
class WorldProfile:
    """What can be seen of one world without hidden answer keys: its stratum and its size."""

    world_id: str
    source: str
    family: str
    mode: str
    rows: int
    features: int

    @property
    def stratum(self) -> str:
        return f"{self.source}/{self.family}/{self.mode}"


@dataclass(frozen=True)
class MixRow:
    """One stratum of a declared mix: worlds available, the proportional quota, and worlds drawn.

    ``quota`` is the family's share of its source-and-mode's seats (which are the mode's share of
    its source's seats, which are the source's share of ``n``); ``drawn`` is within one of it.
    """

    source: str
    family: str
    mode: str
    available: int
    quota: Fraction
    drawn: int

    @property
    def stratum(self) -> str:
        return f"{self.source}/{self.family}/{self.mode}"


@dataclass(frozen=True)
class WorldSample:
    """A seeded stratified sample: the drawn ids (sorted) and the declared mix that produced them."""

    world_ids: tuple[str, ...]
    seed: int
    mix: tuple[MixRow, ...]

    def marginal(self, field: str) -> dict[str, tuple[int, int]]:
        """``value -> (available, drawn)`` over one stratum field: ``source``, ``family`` or ``mode``."""
        if field not in STRATUM_FIELDS:
            raise ValueError(f"no stratum field {field!r}")
        totals: dict[str, tuple[int, int]] = {}
        for row in self.mix:
            value = getattr(row, field)
            available, drawn = totals.get(value, (0, 0))
            totals[value] = (available + row.available, drawn + row.drawn)
        return dict(sorted(totals.items()))

    def summary(self) -> str:
        """Short report: the sample size and seed, then drawn/available for each field's values."""
        available = sum(r.available for r in self.mix)
        lines = [
            f"{len(self.world_ids)} of {available} worlds, seed {self.seed}, stratified by"
            f" source/family/mode ({len(self.mix)} strata; drawn/available)"
        ]
        for field in STRATUM_FIELDS:
            counts = self.marginal(field)
            lines.append(f"  {field:7s}" + ", ".join(f"{v} {d}/{a}" for v, (a, d) in counts.items()))
        return "\n".join(lines)

    def table(self) -> str:
        """The full declared mix: one row per stratum, then the total."""
        head = "stratum (source/family/mode)"
        width = max([len(head), *(len(r.stratum) for r in self.mix)])
        lines = [f"{head:{width}s}  available   quota  drawn"]
        lines += [
            f"{r.stratum:{width}s}  {r.available:9d}  {float(r.quota):6.2f}  {r.drawn:5d}" for r in self.mix
        ]
        n, available = len(self.world_ids), sum(r.available for r in self.mix)
        lines.append(f"{'total':{width}s}  {available:9d}  {n:6.2f}  {n:5d}   (seed {self.seed})")
        return "\n".join(lines)


def world_profile(
    card: WorldCard, entry: Mapping[str, object] | None = None, *, is_null: bool | None = None
) -> WorldProfile:
    """Profile of one world from its card, its manifest entry (if any) and whether it is null (if known)."""
    entry = entry or {}
    null = bool(is_null) or entry.get("is_null") is True
    return WorldProfile(
        world_id=card.world_id,
        source=str(entry.get("source") or UNKNOWN),
        family=NULL_FAMILY if null else str(entry.get("role") or UNKNOWN),
        mode=card.mode.value,
        rows=card.n_pool,
        features=len(card.features),
    )


def world_profiles(
    store: WorldStore, tier: Tier, world_ids: Iterable[str] | None = None
) -> tuple[WorldProfile, ...]:
    """Profiles of ``world_ids`` (default: every world of ``tier``) as ``store`` can show them.

    Source and role come from the store's manifests when it has ``manifest_entry`` (the file store
    does). Role and nullness are truth, so they are read for public-train worlds only, nulls from
    their published answer keys; any other world shows its source and mode, never its family.
    """
    entry_of: Callable[[str], Mapping[str, object] | None] | None = getattr(store, "manifest_entry", None)
    profiles = []
    for world_id in store.world_ids(tier) if world_ids is None else world_ids:
        card = store.card(world_id)
        entry = (entry_of(world_id) if entry_of else None) or {}
        if card.tier is not Tier.PUBLIC_TRAIN:
            profiles.append(world_profile(card, {"source": entry.get("source")}))
            continue
        try:
            is_null: bool | None = store.answer_key(world_id).is_null
        except ArenaError:  # a public-train bundle without its key: the manifest decides
            is_null = None
        profiles.append(world_profile(card, entry, is_null=is_null))
    return tuple(profiles)


def allocate(
    sizes: Mapping[str, int],
    n: int,
    seed: int = 0,
    *,
    carry: Mapping[str, Fraction] | None = None,
    label: str = "",
) -> dict[str, int]:
    """Proportional allocation of ``n`` seats over strata of ``sizes``, by largest remainder.

    Each stratum gets the floor of its quota ``n x size / total``. The remaining seats go one each
    to the strata with the largest fractional remainders, so every count is within one of its quota
    and no stratum receives more seats than it has worlds. ``carry`` adds the rounding error a
    stratum's value has accumulated in earlier allocations (quota minus seats) to its remainder when
    ranking; ties then go by an order seeded from ``seed`` and ``label``.
    """
    total = sum(sizes.values())
    if any(size < 0 for size in sizes.values()) or not 0 <= n <= total:
        raise ValueError(f"cannot draw {n} of {total} worlds")
    quotas = {s: Fraction(n * size, total) for s, size in sizes.items()} if total else {}
    seats = {s: q.numerator // q.denominator for s, q in quotas.items()}
    carried = carry or {}
    ranked = sorted(
        (s for s in quotas if quotas[s] > seats[s]),
        key=lambda s: (
            seats[s] - quotas[s] - carried.get(s, Fraction(0)),
            stable_seed("sample-tie", str(seed), label, s),
        ),
    )
    for s in ranked[: n - sum(seats.values())]:
        seats[s] += 1
    return seats


def _nested_allocation(
    sizes: Mapping[tuple[str, ...], int], n: int, seed: int
) -> dict[tuple[str, ...], tuple[Fraction, int]]:
    """Quota and seats of every stratum: ``n`` over its first field, then each value's seats over the next.

    Each field keeps its own carried rounding error across the allocations of its level.
    """
    cells: dict[tuple[str, ...], tuple[Fraction, int]] = {(): (Fraction(n), n)}
    for depth in range(len(next(iter(sizes)))):
        carry: defaultdict[str, Fraction] = defaultdict(Fraction)
        deeper: dict[tuple[str, ...], tuple[Fraction, int]] = {}
        for prefix, (_, seats) in sorted(cells.items()):
            below = Counter[str]()
            for key, size in sizes.items():
                if key[:depth] == prefix:
                    below[key[depth]] += size
            split = allocate(below, seats, seed, carry=carry, label="/".join(prefix))
            total = sum(below.values())
            for value in sorted(below):
                quota = Fraction(seats * below[value], total)
                carry[value] += quota - split[value]
                deeper[(*prefix, value)] = (quota, split[value])
        cells = deeper
    return cells


def stratified_sample(profiles: Sequence[WorldProfile], n: int, seed: int = 0) -> WorldSample:
    """Seeded sample of ``n`` worlds stratified by ``source/family/mode``, with its declared mix."""
    ids = [p.world_id for p in profiles]
    if len(set(ids)) != len(ids):
        raise ValueError("world ids must be distinct")
    if not 0 < n <= len(ids):
        raise ValueError(f"cannot draw {n} of {len(ids)} worlds")
    members: dict[tuple[str, str, str], list[str]] = {}
    for p in sorted(profiles, key=lambda p: p.world_id):
        members.setdefault((p.source, p.family, p.mode), []).append(p.world_id)
    # sources first, then modes (two balanced halves), then the many families
    sizes: dict[tuple[str, ...], int] = {(s, m, f): len(v) for (s, f, m), v in members.items()}
    allocation = {(s, f, m): v for (s, m, f), v in _nested_allocation(sizes, n, seed).items()}
    strata = sorted(members)
    first = sorted(ids)[:n]
    free = any(0 < allocation[k][1] < len(members[k]) for k in strata)  # else the mix fixes the draw
    for attempt in count():
        chosen = sorted(
            w
            for k in strata
            for w in random.Random(stable_seed("sample", str(seed), str(attempt), *k)).sample(
                members[k], allocation[k][1]
            )
        )
        if chosen != first or not free:  # with a free stratum a repeat has probability at most 1/2
            break
    mix = tuple(MixRow(*k, len(members[k]), *allocation[k]) for k in strata)
    return WorldSample(tuple(chosen), seed, mix)
