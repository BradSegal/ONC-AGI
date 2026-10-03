"""The oracle analyst builds product terms as the generator does."""

from __future__ import annotations

import numpy as np
import pytest
from arena_factories import fid, group, make_card, make_key, part
from onc_agi.core.schema import CreditRule
from onc_agi.services.alignment import oracle_analyst_ranking


@pytest.mark.parametrize("role", ["interaction", "effect_modifier", "mixture"])
def test_product_terms_are_found_on_positive_mean_measurements(role: str) -> None:
    rng = np.random.default_rng(3)
    n = 300  # small enough that raw products (the old analyst) fail for every role
    card = make_card("w-prod", n_features=6, n_pool=n)
    x = rng.normal(5.0, 1.0, size=(n, 6))  # expression-like: mean far from zero
    if role != "interaction":
        x[:, 1] = (rng.random(n) < 0.4).astype(float)  # a clinical indicator
    za = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
    zb = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()
    term = {"interaction": za * zb, "effect_modifier": za * x[:, 1], "mixture": za * (2 * x[:, 1] - 1)}[role]
    term = (term - term.mean()) / term.std()
    y = (rng.random(n) < 1 / (1 + np.exp(-(-0.8 + 1.0 * term)))).astype(np.int64)
    key = make_key(card, [group("g", part(fid(0)), part(fid(1)), role=role, rule=CreditRule.JOINT)])
    assert set(oracle_analyst_ranking(key, card.feature_ids(), x, y)) == {fid(0), fid(1)}
