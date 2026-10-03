"""Matched chance follows the scored representatives, so padding cannot move it."""

from __future__ import annotations

import numpy as np
from onc_agi.core.schema import AnswerKey, CreditRule, GroupLabel, TrueGroup, TruthPart
from onc_agi.services import scoring


def _key(seed: int) -> AnswerKey:
    """Strata with different hit densities, clusters of mates, a neutral decoy, two true parts."""
    rng = np.random.default_rng(seed)
    features = [f"f{j:02d}" for j in range(24)]
    clusters = {f: j // 2 for j, f in enumerate(features)}  # pairs of cluster-mates
    strata = {f: ("dense" if j < 8 else "sparse") for j, f in enumerate(features)}
    a, b = (str(f) for f in rng.choice(features[:20], size=2, replace=False))
    group = lambda gid, f: TrueGroup(  # noqa: E731
        group_id=gid,
        role="direct",
        label=GroupLabel.RECOVERABLE,
        credit_rule=CreditRule.SINGLE,
        parts=(TruthPart(true_feature=f, equivalence_set=(f,), exact_recoverable=True),),
    )
    neutral = TrueGroup(
        group_id="n",
        role="decoy",
        label=GroupLabel.NEUTRAL,
        credit_rule=CreditRule.SINGLE,
        parts=(TruthPart(true_feature="f23", equivalence_set=("f23",), exact_recoverable=False),),
    )
    groups = (group("g1", a),) if a == b else (group("g1", a), group("g2", b))
    return AnswerKey(
        world_id=f"pad-{seed}",
        difficulty_tier=1,
        groups=(*groups, neutral),
        clusters=clusters,
        strata=strata,
        reference_cost=1.0,
        detection_threshold=3.0,
        oracle_version="test",
    )


def test_cluster_mates_and_neutral_features_do_not_move_chance() -> None:
    key = _key(0)
    honest = ("f02", "f12")
    padded = ("f02", "f03", "f23", "f12", "f13")  # f03 and f13 are cluster-mates; f23 is neutral
    assert scoring.chance_recovery(key, honest) == scoring.chance_recovery(key, padded)


def test_a_padded_outcome_blind_agent_has_zero_signed_find_in_expectation() -> None:
    finds = []
    for seed in range(1500):
        key = _key(seed)
        rng = np.random.default_rng(10_000 + seed)
        first, second = (
            str(f) for f in rng.choice([f"f{j:02d}" for j in range(0, 22, 2)], size=2, replace=False)
        )
        mate = f"f{int(first[1:]) + 1:02d}"  # the first pick's cluster-mate, listed right after it
        finds.append(scoring.score_world((first, mate, "f23", second), key).find_signed)
    mean, se = float(np.mean(finds)), float(np.std(finds) / np.sqrt(len(finds)))
    assert abs(mean) <= 3 * se + 0.005, (mean, se)
