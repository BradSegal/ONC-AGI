"""Scoring-latency envelope: one submission, matched chance included, in at most 100 ms at p = 2,000."""

from __future__ import annotations

import time

import numpy as np
import pytest
from onc_agi.core.schema import AnswerKey, CreditRule, GroupLabel, TrueGroup, TruthPart
from onc_agi.services import scoring

NORMAL_CASE_MS = 100.0
FAILURE_LIMIT_MS = 1000.0


def _world(p: int = 2000, strata: int = 24) -> AnswerKey:
    features = [f"g{j:04d}" for j in range(p)]
    groups = tuple(
        TrueGroup(
            group_id=f"t{i}",
            role="direct",
            label=GroupLabel.RECOVERABLE,
            credit_rule=CreditRule.SINGLE,
            parts=(
                TruthPart(
                    true_feature=features[7 * i],
                    equivalence_set=(features[7 * i], features[7 * i + 1]),
                    exact_recoverable=True,
                ),
            ),
        )
        for i in range(6)
    )
    return AnswerKey(
        world_id="latency",
        difficulty_tier=1,
        groups=groups,
        clusters={f: j // 3 for j, f in enumerate(features)},
        strata={f: f"s{j % strata}" for j, f in enumerate(features)},
        reference_cost=1.0,
        detection_threshold=3.0,
        oracle_version="test",
    )


@pytest.mark.parametrize("listed", [10, 50, 200, 2000])
def test_scoring_one_submission_is_within_the_latency_envelope(listed: int) -> None:
    key = _world()
    ranking = tuple(np.random.default_rng(listed).permutation(sorted(key.clusters))[:listed])
    timings = []
    for _ in range(5):
        start = time.perf_counter()
        scoring.score_world(ranking, key)
        timings.append(1000 * (time.perf_counter() - start))
    median = float(np.median(timings))
    print(f"p=2000 listed={listed}: median {median:.1f} ms, max {max(timings):.1f} ms")
    assert median <= NORMAL_CASE_MS and max(timings) <= FAILURE_LIMIT_MS
