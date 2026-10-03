"""Scoring: one ordered list in, one :class:`WorldScore` out; many scores in, one scorecard.

Per world:

1. Unknown features are rejected; any listed leak (reject set) zeroes the world.
2. The list is deduplicated to one representative per truth-independent cluster
   (first occurrence wins).
3. Members of *neutral* groups' equivalence sets are removed: they neither earn
   credit nor use depth.
4. The top ``R`` remaining representatives earn credit one-to-one. Under the
   default, ``CREDIT_ORDER = "max_matching"``, the
   assignment of representatives to parts is the one that maximises raw credit, then
   exact credit, so credit depends neither on the order of the top R nor on the
   hidden group order; ``"greedy"`` credits the first uncredited part, in list
   order. An interaction group counts only when both of its parts are credited.
5. Recovery is chance-normalised against a *matched* random list pushed through the
   same pipeline - each listed feature replaced by a random feature of its stratum: ``q = clip((raw - chance) / (1 - chance), 0, 1)``.

Across worlds:

* ``Find = mean over signal worlds of q * efficiency``
* ``Restraint = mean over null worlds of restrained * efficiency - abstention rate on signal worlds``
  (Youden's J, unfloored for statistics)
* ``Discovery Score = Find * max(0, Restraint)`` for display; the unfloored estimand drives intervals.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from onc_agi.core.errors import ArenaError
from onc_agi.core.schema import (
    INTERFACE_VERSION,
    AlignmentDiagnostics,
    AnswerKey,
    CreditRule,
    ErrorCode,
    Interval,
    Scorecard,
    Tier,
    TierSummary,
    WorldScore,
)

CHANCE_DRAWS = 200
BOOTSTRAP_DRAWS = 1000


def _source_label(module: str) -> str:
    """First 8 hex digits of a sibling module's bytes, so any edit to the rules changes the label."""
    return hashlib.sha256((Path(__file__).parent / module).read_bytes()).hexdigest()[:8]


# The base names the interface generation; the suffix pins the exact code, so scorecards from
# different rule sets are never compared as if equal (a hand-maintained label went stale).
SCORER_VERSION = f"scorer-1.0+{_source_label('scoring.py')}"
ENGINE_VERSION = f"engine-1.0+{_source_label('engine.py')}"

# Statistical rules adopted for 1.0. Each alternative stays selectable so that
# earlier scorecards can be reproduced.
CreditOrder = Literal["max_matching", "greedy"]
UnflooredRule = Literal["sign_guard", "product", "min_guard"]
# D12: assignment of the top-R representatives to parts. "max_matching" maximises (raw, exact)
# credit over one-to-one assignments, so neither list order inside the top R nor the hidden order of
# groups in the key moves the score; "greedy" is the earlier first-uncredited-part rule in list order.
CREDIT_ORDER: CreditOrder = "max_matching"
# D21: the statistical estimand from signed Find F and unfloored Restraint J.
# "sign_guard": F*J, but -|F*J| whenever either is negative. "product": F*J.
# "min_guard": F*J when both are non-negative, else min(F, J) (monotone in both components).
UNFLOORED_RULE: UnflooredRule = "sign_guard"
MATCHING_LIMIT = 50_000  # assignments searched before falling back to greedy (never reached in practice)


def stable_seed(*parts: str) -> int:
    """Deterministic 32-bit seed from text parts (no process-dependent hashing)."""
    digest = hashlib.sha256("\x1f".join(parts).encode()).digest()
    return int.from_bytes(digest[:4], "little")


@dataclass(frozen=True)
class _Credit:
    raw: float
    exact: float


def _neutral_features(key: AnswerKey) -> frozenset[str]:
    recoverable = {f for g in key.recoverable for part in g.parts for f in part.equivalence_set}
    neutral = {
        f for g in key.groups if g not in key.recoverable for part in g.parts for f in part.equivalence_set
    }
    return frozenset(neutral - recoverable)


def _representatives(ranking: Sequence[str], clusters: dict[str, int], neutral: frozenset[str]) -> list[str]:
    seen: set[int] = set()
    out: list[str] = []
    for feature in ranking:
        cluster = clusters[feature]
        if cluster in seen:
            continue
        seen.add(cluster)
        if feature not in neutral:
            out.append(feature)
    return out


def _top_representatives(
    ranking: Iterable[str], clusters: dict[str, int], neutral: frozenset[str], depth: int
) -> list[str]:
    """The first ``depth`` representatives of :func:`_representatives`, reading no further than needed."""
    seen: set[int] = set()
    out: list[str] = []
    for feature in ranking:
        if len(out) == depth:
            break
        cluster = clusters[feature]
        if cluster in seen:
            continue
        seen.add(cluster)
        if feature not in neutral:
            out.append(feature)
    return out


Slot = tuple[int, int]  # (recoverable group index, part index)


def _greedy_assignment(top: Sequence[str], key: AnswerKey) -> dict[Slot, bool]:
    """Greedy credit: each representative credits the first uncredited part containing it, in list order."""
    parts = [(gi, pi, part) for gi, g in enumerate(key.recoverable) for pi, part in enumerate(g.parts)]
    credited: dict[Slot, bool] = {}  # (group, part) -> exact?
    for feature in top:
        for gi, pi, part in parts:
            if (gi, pi) in credited or feature not in part.equivalence_set:
                continue
            credited[(gi, pi)] = feature == part.true_feature or not part.exact_recoverable
            break
    return credited


def _matching_assignment(top: Sequence[str], key: AnswerKey) -> dict[Slot, bool]:
    """D12 maximum matching: the one-to-one assignment of representatives to parts with the most raw
    credit, then the most exact credit (group rules applied), independent of list and group order.

    Only representatives eligible for some part branch, and a representative eligible for exactly one
    part that no other representative can take is assigned directly, so the search is trivial unless
    equivalence sets overlap.
    """
    recoverable = key.recoverable
    eligible: list[tuple[str, list[tuple[Slot, bool]]]] = []
    for feature in top:
        options = [
            ((gi, pi), feature == part.true_feature or not part.exact_recoverable)
            for gi, g in enumerate(recoverable)
            for pi, part in enumerate(g.parts)
            if feature in part.equivalence_set
        ]
        if options:
            eligible.append((feature, options))
    claims: dict[Slot, int] = {}
    for _, options in eligible:
        for slot, _exact in options:
            claims[slot] = claims.get(slot, 0) + 1
    fixed: dict[Slot, bool] = {}
    contested: list[list[tuple[Slot, bool]]] = []
    for _, options in eligible:
        if len(options) == 1 and claims[options[0][0]] == 1:
            fixed[options[0][0]] = options[0][1]
        else:
            contested.append(options)
    if not contested:
        return fixed
    best: tuple[float, float] = (-1.0, -1.0)
    best_assignment: dict[Slot, bool] = dict(fixed)
    searched = 0

    def search(i: int, current: dict[Slot, bool]) -> None:
        nonlocal best, best_assignment, searched
        searched += 1
        if searched > MATCHING_LIMIT:
            return
        if i == len(contested):
            value = _group_credit(current, key)
            if value > best:
                best, best_assignment = value, dict(current)
            return
        for slot, exact in contested[i]:
            if slot not in current:
                current[slot] = exact
                search(i + 1, current)
                del current[slot]
        search(i + 1, current)  # this representative credits nothing

    search(0, dict(fixed))
    if searched > MATCHING_LIMIT:
        return _greedy_assignment(top, key)
    return best_assignment


def _group_credit(credited: dict[Slot, bool], key: AnswerKey) -> tuple[float, float]:
    """(raw, exact) credited slots after each group's credit rule (before dividing by R)."""
    raw = 0.0
    exact = 0.0
    for gi, g in enumerate(key.recoverable):
        hits = [credited.get((gi, pi)) for pi in range(len(g.parts))]
        if g.credit_rule is CreditRule.WEIGHTED_COVERAGE:
            total = sum(p.weight for p in g.parts)
            covered = sum(p.weight for p, h in zip(g.parts, hits, strict=True) if h is not None)
            covered_exact = sum(p.weight for p, h in zip(g.parts, hits, strict=True) if h)
            raw += len(g.parts) * covered / total
            exact += len(g.parts) * covered_exact / total
        elif all(h is not None for h in hits):
            raw += len(g.parts)
            exact += sum(1 for h in hits if h)
    return raw, exact


def _credit(top: Sequence[str], key: AnswerKey, order: CreditOrder | None = None) -> _Credit:
    """One-to-one credit (``CREDIT_ORDER``), then each group's credit rule."""
    depth = key.depth
    if depth == 0:
        return _Credit(0.0, 0.0)
    rule = CREDIT_ORDER if order is None else order
    credited = _greedy_assignment(top, key) if rule == "greedy" else _matching_assignment(top, key)
    raw, exact = _group_credit(credited, key)
    return _Credit(raw / depth, exact / depth)


def _signed(raw: float, chance: float) -> float:
    if chance >= 1.0:
        return 0.0
    return float(min(1.0, (raw - chance) / (1.0 - chance)))


def _normalise(raw: float, chance: float) -> float:
    return max(0.0, _signed(raw, chance))


def chance_recovery(
    key: AnswerKey, ranking: Sequence[str] | None = None, draws: int = CHANCE_DRAWS
) -> tuple[float, float]:
    """Expected (raw, exact) recovery of a matched random list.

    Each listed feature is replaced by a random feature from the same
    truth-independent stratum (data type x correlation bin), sampling without
    replacement within strata. An agent that only exploits feature classes - data
    types, hubs, near-duplicates, synthetic columns - therefore gains nothing over
    chance. With ``ranking=None`` (or no strata) the reference is a uniformly
    random complete ordering.

    The Monte Carlo draws depend on the world alone: draw ``i`` fixes one permutation
    of every stratum, and the list's representatives consume each stratum's permutation
    in order. The estimate is therefore a fixed function of the strata sequence of the
    representatives the scorer actually credits, so neither re-ordering or padding the tail nor inserting cluster-mates or neutral features moves it.
    """
    if key.is_null or (ranking is not None and not ranking):
        return 0.0, 0.0
    neutral = _neutral_features(key)
    raws = np.empty(draws)
    exacts = np.empty(draws)
    if ranking is None or not key.strata:
        rng = np.random.default_rng(stable_seed("chance", key.world_id))
        universe = np.array(sorted(key.clusters))
        for i in range(draws):
            order = (str(f) for f in rng.permutation(universe))
            credit = _credit(_top_representatives(order, key.clusters, neutral, key.depth), key)
            raws[i], exacts[i] = credit.raw, credit.exact
        return float(raws.mean()), float(exacts.mean())
    members: dict[str, list[str]] = {}
    for feature in sorted(key.strata):
        members.setdefault(key.strata[feature], []).append(feature)
    # the matched list mirrors the *scored* list: one draw per representative (deduplicated,
    # neutral-removed), so padding with cluster-mates or neutral features cannot move chance
    # scanning stops at depth R, so the tail beyond it cannot either
    representatives = _representatives(ranking, key.clusters, neutral)
    if not representatives:
        return 0.0, 0.0
    listed_strata = [key.strata[f] for f in representatives]
    perms = {
        s: np.random.default_rng(stable_seed("chance", key.world_id, s)).permuted(
            np.tile(np.arange(len(members[s])), (draws, 1)), axis=1
        )
        for s in dict.fromkeys(listed_strata)
    }

    def matched(i: int) -> Iterator[str]:
        cursor = dict.fromkeys(perms, 0)
        for s in listed_strata:
            yield members[s][int(perms[s][i, cursor[s]])]
            cursor[s] += 1

    for i in range(draws):
        credit = _credit(_top_representatives(matched(i), key.clusters, neutral, key.depth), key)
        raws[i], exacts[i] = credit.raw, credit.exact
    return float(raws.mean()), float(exacts.mean())


def score_world(
    ranking: Sequence[str],
    key: AnswerKey,
    *,
    spent: float = 0.0,
    sequential: bool = False,
    chance: tuple[float, float] | None = None,
) -> WorldScore:
    """Score one submission against one answer key.

    Raises
    ------
    ArenaError
        ``UNKNOWN_FEATURE`` if the ranking names a feature outside the world, or
        ``INVALID_PAYLOAD`` if it repeats a feature.
    """
    if len(set(ranking)) != len(ranking):
        raise ArenaError(ErrorCode.INVALID_PAYLOAD, "a ranking may not repeat a feature")
    unknown = [f for f in ranking if f not in key.clusters]
    if unknown:
        raise ArenaError(ErrorCode.UNKNOWN_FEATURE, f"unknown features: {unknown[:5]}")

    leaked = any(f in key.reject_set for f in ranking)
    reps = _representatives(ranking, key.clusters, _neutral_features(key))
    abstained = len(reps) == 0
    efficiency = (1.0 if spent <= 0 else float(min(1.0, key.reference_cost / spent))) if sequential else 1.0

    if key.is_null:
        return WorldScore(
            world_id=key.world_id,
            difficulty_tier=key.difficulty_tier,
            is_null=True,
            find=0.0,
            find_signed=0.0,
            find_exact=0.0,
            raw_recovery=0.0,
            chance_recovery=0.0,
            restrained=abstained and not leaked,
            abstained=abstained,
            leaked=leaked,
            spent=spent,
            efficiency=efficiency,
            listed=len(ranking),
        )

    credit = _credit(reps[: key.depth], key)
    chance_raw, chance_exact = chance if chance is not None else chance_recovery(key, ranking)
    find = 0.0 if leaked else _normalise(credit.raw, chance_raw)
    find_signed = _signed(0.0, chance_raw) if leaked else _signed(credit.raw, chance_raw)
    find_exact = 0.0 if leaked else _normalise(credit.exact, chance_exact)
    return WorldScore(
        world_id=key.world_id,
        difficulty_tier=key.difficulty_tier,
        is_null=False,
        find=find,
        find_signed=find_signed,
        find_exact=find_exact,
        raw_recovery=credit.raw,
        chance_recovery=chance_raw,
        restrained=False,
        abstained=abstained,
        leaked=leaked,
        spent=spent,
        efficiency=efficiency,
        listed=len(ranking),
    )


def unfloored_estimate(find_signed: float, restraint: float, rule: UnflooredRule | None = None) -> float:
    """The statistical estimand from signed Find and unfloored Restraint."""
    chosen = UNFLOORED_RULE if rule is None else rule
    product = find_signed * restraint
    if chosen == "product" or not (find_signed < 0 or restraint < 0):
        return product  # NaN propagates
    if chosen == "min_guard":
        return min(find_signed, restraint)
    return -abs(product)


@dataclass(frozen=True)
class Components:
    find: float
    find_exact: float
    restraint_null: float
    abstention_signal: float
    find_signed: float = 0.0

    @property
    def restraint(self) -> float:
        return self.restraint_null - self.abstention_signal

    @property
    def unfloored(self) -> float:
        """Statistical estimand: signed Find x unfloored Restraint (at most zero in expectation for blind agents).

        Under ``UNFLOORED_RULE = "sign_guard"`` the product is negative whenever either component
        is: two below-chance components must not multiply into an above-floor score.
        """
        return unfloored_estimate(self.find_signed, self.restraint)

    @property
    def displayed(self) -> float:
        restraint = self.restraint
        return self.find * max(0.0, restraint) if restraint == restraint else float("nan")


def components(scores: Sequence[WorldScore]) -> Components:
    signal = [s for s in scores if not s.is_null]
    null = [s for s in scores if s.is_null]
    if not scores:
        raise ValueError("no scores to aggregate")
    if not signal or not null:
        # Restraint (or Find) is undefined without both world types: report NaN, never a guess.
        nan = float("nan")
        find = float(np.mean([s.find * s.efficiency for s in signal])) if signal else nan
        signed = float(np.mean([s.find_signed * s.efficiency for s in signal])) if signal else nan
        exact = float(np.mean([s.find_exact * s.efficiency for s in signal])) if signal else nan
        rest = float(np.mean([float(s.restrained) * s.efficiency for s in null])) if null else nan
        abst = float(np.mean([float(s.abstained) for s in signal])) if signal else nan
        return Components(
            find=find, find_exact=exact, restraint_null=rest, abstention_signal=abst, find_signed=signed
        )
    return Components(
        find=float(np.mean([s.find * s.efficiency for s in signal])),
        find_exact=float(np.mean([s.find_exact * s.efficiency for s in signal])),
        restraint_null=float(np.mean([float(s.restrained) * s.efficiency for s in null])),
        abstention_signal=float(np.mean([float(s.abstained) for s in signal])),
        find_signed=float(np.mean([s.find_signed * s.efficiency for s in signal])),
    )


def bootstrap_interval(
    scores: Sequence[WorldScore], *, draws: int = BOOTSTRAP_DRAWS, seed: int = 0
) -> Interval:
    """Percentile interval of the unfloored score, resampling signal and null worlds separately."""
    signal = [s for s in scores if not s.is_null]
    null = [s for s in scores if s.is_null]
    if not signal or not null:
        return Interval(low=None, high=None)
    rng = np.random.default_rng(seed)
    values = np.empty(draws)
    for i in range(draws):
        resampled = [signal[j] for j in rng.integers(0, len(signal), len(signal))] + [
            null[j] for j in rng.integers(0, len(null), len(null))
        ]
        values[i] = components(resampled).unfloored
    low, high = np.quantile(values, [0.025, 0.975])
    return Interval(low=float(low), high=float(high))


def _defined(value: float) -> float | None:
    """NaN marks an undefined metric internally; the wire contract uses ``None``."""
    return None if value != value else value


def aggregate(
    scores: Sequence[WorldScore],
    *,
    scorecard_id: str,
    agent: str,
    tier: Tier,
    track: str = "open",
    alignment: AlignmentDiagnostics | None = None,
    tags: Sequence[str] = (),
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    model: str | None = None,
    harness: str | None = None,
    pool_commitment: str | None = None,
    tokens: int | None = None,
    cost_usd: float | None = None,
    extra_versions: dict[str, str] | None = None,
) -> Scorecard:
    """Aggregate per-world scores into a scorecard with tier exposure applied."""
    overall = components(scores)
    per_tier: list[TierSummary] = []
    for tier_index in sorted({s.difficulty_tier for s in scores}):
        subset = [s for s in scores if s.difficulty_tier == tier_index]
        n_signal = sum(1 for s in subset if not s.is_null)
        n_null = len(subset) - n_signal
        if n_signal and n_null:
            c = components(subset)
            find, restraint, ds = c.find, c.restraint, c.displayed
        else:
            find = restraint = ds = float("nan")
        per_tier.append(
            TierSummary(
                difficulty_tier=tier_index,
                n_signal=n_signal,
                n_null=n_null,
                find=_defined(find),
                restraint=_defined(restraint),
                discovery_score=_defined(ds),
            )
        )
    return Scorecard(
        scorecard_id=scorecard_id,
        track=track,
        agent=agent,
        model=model,
        harness=harness,
        versions={"interface": INTERFACE_VERSION, "scorer": SCORER_VERSION, "engine": ENGINE_VERSION}
        | dict(extra_versions or {}),
        pool_commitment=pool_commitment,
        tokens=tokens,
        cost_usd=cost_usd,
        tier=tier,
        n_worlds=len(scores),
        discovery_score=_defined(overall.displayed),
        discovery_score_unfloored=_defined(overall.unfloored),
        interval=bootstrap_interval(
            scores, draws=bootstrap_draws, seed=stable_seed("bootstrap", scorecard_id)
        ),
        find=_defined(overall.find),
        find_signed=_defined(overall.find_signed),
        restraint=_defined(overall.restraint),
        strict_discovery_score=_defined(
            overall.find_exact * max(0.0, overall.restraint)
            if overall.restraint == overall.restraint
            else float("nan")
        ),
        leak_rate=float(np.mean([s.leaked for s in scores])),
        abstention_on_signal=_defined(overall.abstention_signal),
        restraint_on_null=_defined(overall.restraint_null),
        mean_data_cost=float(np.mean([s.spent for s in scores])),
        per_tier=tuple(per_tier),
        alignment=alignment,
        worlds=tuple(scores) if tier is Tier.PUBLIC_TRAIN else (),
        tags=tuple(tags),
    )
