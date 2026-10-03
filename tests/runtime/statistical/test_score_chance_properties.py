"""Chance properties of the Discovery Score.

Strategies that carry no information must score zero *in expectation* on the
unfloored estimand, and the oracle must score one. These are simulated over many
synthetic answer keys with declared tolerances.
"""

from __future__ import annotations

import numpy as np
import pytest
from arena_factories import fid, group, make_card, make_key, part
from onc_agi.core.schema import AnswerKey, WorldScore
from onc_agi.services.scoring import components, score_world

P = 30
CARD = make_card(n_features=P)
STRATA = {fid(j): f"s{j % 3}" for j in range(P)}


def synthetic_keys(n: int, seed: int, null_share: float = 0.2) -> list[AnswerKey]:
    rng = np.random.default_rng(seed)
    keys = []
    for i in range(n):
        if rng.random() < null_share:
            keys.append(make_key(CARD, [], strata=STRATA).model_copy(update={"world_id": f"w-{i:04d}"}))
            continue
        k = int(rng.integers(1, 4))
        truth = rng.choice(P, size=k, replace=False)
        groups = [group(f"g{j}", part(fid(int(t)))) for j, t in enumerate(truth)]
        keys.append(make_key(CARD, groups, strata=STRATA).model_copy(update={"world_id": f"w-{i:04d}"}))
    return keys


def play(keys: list[AnswerKey], policy, seed: int) -> list[WorldScore]:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    return [score_world(policy(key, rng), key) for key in keys]


def random_list(key: AnswerKey, rng: np.random.Generator) -> list[str]:
    return [fid(int(j)) for j in rng.permutation(P)[: int(rng.integers(1, 6))]]


def random_abstainer(p: float):  # type: ignore[no-untyped-def]
    def policy(key: AnswerKey, rng: np.random.Generator) -> list[str]:
        return [] if rng.random() < p else random_list(key, rng)

    return policy


def stratum_exploiter(key: AnswerKey, rng: np.random.Generator) -> list[str]:
    """Lists the whole of one stratum: exploits feature classes, never the outcome."""
    return [f for f, s in STRATA.items() if s == "s0"]


def oracle_policy(key: AnswerKey, rng: np.random.Generator) -> list[str]:
    return [p.true_feature for g in key.recoverable for p in g.parts]


def mean_ci(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(1.96 * arr.std(ddof=1) / np.sqrt(len(arr)))


KEYS = synthetic_keys(600, seed=1)


def test_the_oracle_scores_one() -> None:
    c = components(play(KEYS, oracle_policy, seed=0))
    assert (c.find, c.restraint, c.displayed) == (1.0, 1.0, 1.0)


@pytest.mark.parametrize("policy", [random_list, stratum_exploiter], ids=["random", "stratum"])
def test_uninformed_lists_have_zero_signed_find_in_expectation(policy) -> None:  # type: ignore[no-untyped-def]
    scores = [s for s in play(KEYS, policy, seed=2) if not s.is_null]
    mean, half = mean_ci([s.find_signed for s in scores])
    assert abs(mean) <= max(half, 0.02), (mean, half)


@pytest.mark.parametrize("p", [0.2, 0.5, 0.8])
def test_random_abstention_has_zero_restraint_in_expectation(p: float) -> None:
    scores = play(synthetic_keys(3000, seed=3), random_abstainer(p), seed=4)
    c = components(scores)
    n_null = sum(s.is_null for s in scores)
    n_signal = len(scores) - n_null
    se = np.sqrt(p * (1 - p) / n_null + p * (1 - p) / n_signal)
    assert abs(c.restraint) <= 3 * se, (c.restraint, se)


@pytest.mark.parametrize("abstain", [True, False])
def test_constant_policies_score_exactly_zero(abstain: bool) -> None:
    policy = (lambda key, rng: []) if abstain else random_list
    c = components(play(KEYS, policy, seed=5))
    assert c.restraint == 0.0 and c.displayed == 0.0


def test_giant_lists_earn_nothing_over_matched_chance() -> None:
    scores = [s for s in play(KEYS, lambda key, rng: [fid(j) for j in range(P)], seed=6) if not s.is_null]
    mean, half = mean_ci([s.find_signed for s in scores])
    assert abs(mean) <= max(half, 0.02)
