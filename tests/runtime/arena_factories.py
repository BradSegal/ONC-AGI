"""Small, explicit builders for worlds, answer keys and stores used across the arena tests.

All construction of contract models goes through here, so an additive change to the
interface contract touches one file rather than every test.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from onc_agi.core.schema import (
    AnswerKey,
    CreditRule,
    FeatureMeta,
    GroupLabel,
    Mode,
    PriceList,
    Tier,
    Timing,
    TrueGroup,
    TruthPart,
    WorldCard,
)
from onc_agi.core.world import WorldData


def fid(j: int) -> str:
    return f"f{j:02d}"


def make_card(
    world_id: str = "w-0",
    *,
    n_pool: int = 60,
    n_features: int = 8,
    tier: Tier = Tier.PUBLIC_TRAIN,
    mode: Mode = Mode.FULL_ACCESS,
    post_outcome: Sequence[str] = (),
    strata: Sequence[str] = ("all",),
    recruit_price: float = 1.0,
    assay_price: float = 0.5,
    budget: float | None = None,
) -> WorldCard:
    features = tuple(
        FeatureMeta(
            feature_id=fid(j),
            data_type="expression",
            timing=Timing.POST_OUTCOME if fid(j) in post_outcome else Timing.BASELINE,
            assay_price=assay_price,
        )
        for j in range(n_features)
    )
    full_pool = n_pool * recruit_price + n_pool * assay_price * n_features
    return WorldCard(
        world_id=world_id,
        tier=tier,
        mode=mode,
        n_pool=n_pool,
        features=features,
        strata=tuple(strata),
        prices=PriceList(recruit_per_patient=recruit_price),
        budget=full_pool if budget is None else budget,
    )


def make_world(
    card: WorldCard,
    *,
    seed: int = 0,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
) -> WorldData:
    rng = np.random.default_rng(seed)
    n, p = card.n_pool, len(card.features)
    x = rng.normal(size=(n, p)) if x is None else x
    y = rng.integers(0, 2, size=n) if y is None else y
    stratum = tuple(card.strata[i % len(card.strata)] for i in range(n))
    queues = {
        s: tuple(int(i) for i in rng.permutation([i for i in range(n) if stratum[i] == s]))
        for s in card.strata
    }
    return WorldData(
        card=card,
        patient_ids=tuple(f"p{i:04d}" for i in range(n)),
        x=np.asarray(x, dtype=np.float64),
        y=np.asarray(y, dtype=np.int64),
        stratum=stratum,
        queues=queues,
    )


def part(true_feature: str, *substitutes: str, exact: bool = True, weight: float = 1.0) -> TruthPart:
    return TruthPart(
        true_feature=true_feature,
        equivalence_set=(true_feature, *substitutes),
        exact_recoverable=exact,
        weight=weight,
    )


def group(
    group_id: str,
    *parts: TruthPart,
    label: GroupLabel = GroupLabel.RECOVERABLE,
    role: str = "generating",
    rule: CreditRule | None = None,
) -> TrueGroup:
    """A true group; the credit rule defaults to single for one part and joint otherwise."""
    if rule is None:
        rule = CreditRule.SINGLE if len(parts) == 1 else CreditRule.JOINT
    return TrueGroup(group_id=group_id, role=role, label=label, credit_rule=rule, parts=tuple(parts))


def make_key(
    card: WorldCard,
    groups: Iterable[TrueGroup] = (),
    *,
    reject: Sequence[str] = (),
    clusters: dict[str, int] | None = None,
    strata: dict[str, str] | None = None,
    difficulty_tier: int = 0,
    reference_cost: float | None = None,
    threshold: float = 3.0,
) -> AnswerKey:
    ids = card.feature_ids()
    return AnswerKey(
        world_id=card.world_id,
        difficulty_tier=difficulty_tier,
        groups=tuple(groups),
        reject_set=tuple(reject),
        clusters=clusters if clusters is not None else {f: j for j, f in enumerate(ids)},
        strata=strata or {},
        reference_cost=card.budget / 2 if reference_cost is None else reference_cost,
        detection_threshold=threshold,
        oracle_version="test",
    )


class InMemoryStore:
    """:class:`onc_agi.core.ports.WorldStore` over prepared worlds and keys."""

    def __init__(self) -> None:
        self._worlds: dict[str, tuple[WorldData, AnswerKey]] = {}

    @classmethod
    def of(cls, *pairs: tuple[WorldData, AnswerKey]) -> InMemoryStore:
        store = cls()
        for world, key in pairs:
            store.add(world, key)
        return store

    def add(self, world: WorldData, key: AnswerKey) -> None:
        if world.card.world_id != key.world_id:
            raise ValueError("world and key disagree on world_id")
        self._worlds[world.card.world_id] = (world, key)

    def world_ids(self, tier: Tier) -> tuple[str, ...]:
        return tuple(sorted(w for w, (data, _) in self._worlds.items() if data.card.tier is tier))

    def card(self, world_id: str) -> WorldCard:
        return self._worlds[world_id][0].card

    def world(self, world_id: str) -> WorldData:
        return self._worlds[world_id][0]

    def answer_key(self, world_id: str) -> AnswerKey:
        return self._worlds[world_id][1]


LEAK = fid(7)


def planted_world(
    world_id: str,
    *,
    signal: bool,
    seed: int,
    n_pool: int = 200,
    n_features: int = 8,
    tier: Tier = Tier.PUBLIC_TRAIN,
    mode: Mode = Mode.FULL_ACCESS,
    effect: float = 2.5,
    difficulty_tier: int = 0,
) -> tuple[WorldData, AnswerKey]:
    """A world whose outcome is driven by ``f00`` (or by nothing) with a post-outcome leak ``f07``.

    With the default effect the truth is detected essentially always, so baselines and
    the oracle analyst recover it; null worlds draw the outcome independently.
    """
    card = make_card(
        world_id, n_pool=n_pool, n_features=n_features, tier=tier, mode=mode, post_outcome=(LEAK,)
    )
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_pool, n_features))
    logit = effect * x[:, 0] if signal else np.zeros(n_pool)
    y = (rng.random(n_pool) < 1 / (1 + np.exp(-logit))).astype(np.int64)
    x[:, n_features - 1] = y + 0.3 * rng.normal(size=n_pool)
    world = make_world(card, seed=seed, x=x, y=y)
    groups = [group("g0", part(fid(0)))] if signal else []
    strata = {f: "expression" for f in card.feature_ids()}  # real keys always carry strata
    return world, make_key(card, groups, reject=(LEAK,), strata=strata, difficulty_tier=difficulty_tier)


def standard_world(
    i: int, *, tier: Tier = Tier.PUBLIC_TRAIN, mode: Mode = Mode.FULL_ACCESS
) -> tuple[WorldData, AnswerKey]:
    """World ``i`` of the standard store: every third world is null."""
    return planted_world(
        f"w-{i:02d}", signal=i % 3 != 2, seed=100 + i, tier=tier, mode=mode, difficulty_tier=i % 2
    )


HORIZON = 1826.25


def survival_world(
    world_id: str,
    *,
    signal: bool,
    seed: int,
    n_pool: int = 240,
    n_features: int = 8,
    tier: Tier = Tier.PUBLIC_TRAIN,
    mode: Mode = Mode.FULL_ACCESS,
    log_hazard_ratio: float = 1.0,
    missing: float = 0.0,
) -> tuple[WorldData, AnswerKey]:
    """A survival world: exponential event times with log-hazard ``log_hazard_ratio * f00`` (or none),
    independent exponential censoring and administrative censoring at the horizon; ``f07`` is a leak.

    ``missing`` blanks that share of cells, completely at random, in the baseline columns other
    than the truth ``f00``.
    """
    base = make_card(
        world_id, n_pool=n_pool, n_features=n_features, tier=tier, mode=mode, post_outcome=(LEAK,)
    )
    card = WorldCard.model_validate(base.model_dump() | {"outcome_type": "survival", "horizon_days": HORIZON})
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_pool, n_features))
    rate = np.exp(log_hazard_ratio * x[:, 0] if signal else np.zeros(n_pool)) / 1500.0
    event_time = rng.exponential(1.0 / rate)
    censor_time = np.minimum(rng.exponential(3000.0, size=n_pool), HORIZON)
    time = np.minimum(event_time, censor_time)
    y = (event_time <= censor_time).astype(np.int64)
    x[:, n_features - 1] = y + 0.3 * rng.normal(size=n_pool)
    if missing > 0:
        inner = x[:, 1 : n_features - 1]
        x[:, 1 : n_features - 1] = np.where(rng.random(inner.shape) < missing, np.nan, inner)
    plain = make_world(base, seed=seed, x=x, y=y)  # the binary twin supplies ids, strata and queues
    world = WorldData(
        card=card,
        patient_ids=plain.patient_ids,
        x=plain.x,
        y=plain.y,
        stratum=plain.stratum,
        queues=plain.queues,
        time=np.asarray(time, dtype=np.float64),
    )
    groups = [group("g0", part(fid(0)))] if signal else []
    strata = {f: "expression" for f in card.feature_ids()}
    return world, make_key(card, groups, reject=(LEAK,), strata=strata)
