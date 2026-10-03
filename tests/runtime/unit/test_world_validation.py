"""Reject corrupt pools while preserving genuine missing measurements."""

from dataclasses import replace

import numpy as np
import pytest
from arena_factories import make_card, make_world


def test_patient_identifiers_cannot_alias_two_rows() -> None:
    world = make_world(make_card(n_pool=6))
    with pytest.raises(ValueError, match="patient identifiers"):
        replace(world, patient_ids=("p0",) * 6)


def test_missing_measurements_are_valid_but_infinite_values_are_not() -> None:
    world = make_world(make_card(n_pool=6))
    x = world.x.copy()
    x[0, 0] = np.nan
    replace(world, x=x)
    x[0, 0] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        replace(world, x=x)


@pytest.mark.parametrize("bad", [0.5, -1, 2, np.nan])
def test_outcome_values_must_be_binary(bad: float) -> None:
    world = make_world(make_card(n_pool=6))
    y = world.y.astype(float)
    y[0] = bad
    with pytest.raises(ValueError, match="binary"):
        replace(world, y=y)


def test_queues_must_agree_with_each_rows_stratum() -> None:
    world = make_world(make_card(n_pool=6, strata=("a", "b")))
    with pytest.raises(ValueError, match="stratum"):
        replace(world, queues={"a": world.queues["b"], "b": world.queues["a"]})


def test_published_stratum_sizes_must_match_the_pool() -> None:
    world = make_world(make_card(n_pool=6, strata=("a", "b")))
    card = world.card.model_copy(update={"stratum_sizes": {"a": 2, "b": 4}})
    with pytest.raises(ValueError, match="stratum sizes"):
        replace(world, card=card)
