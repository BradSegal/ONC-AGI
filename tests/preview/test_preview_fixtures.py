"""The toy fixture store: shape, keys, both modes, determinism and reference behaviour."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import build_toy
import pytest
from onc_agi.adapters.agents import CHEATERS, make_agent
from onc_agi.adapters.cli import fixture_store
from onc_agi.core.schema import Mode, Tier, Timing
from onc_agi.infra.bundles import FileWorldStore
from onc_agi.services.kit import evaluate

STORE = FileWorldStore(fixture_store())
IDS = STORE.world_ids(Tier.PUBLIC_TRAIN)


def tree_digest(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_store_has_every_spec_in_both_modes() -> None:
    assert len(IDS) == 2 * len(build_toy.SPECS)
    for spec in build_toy.SPECS:
        modes = {STORE.card(f"{spec.world}-{suffix}").mode for suffix in ("full", "seq")}
        assert modes == {Mode.FULL_ACCESS, Mode.SEQUENTIAL}


def test_null_share_is_about_one_in_five() -> None:
    nulls = sum(STORE.answer_key(w).is_null for w in IDS)
    assert 0.15 <= nulls / len(IDS) <= 0.25


def test_every_world_has_the_same_shape_so_shape_reveals_no_role() -> None:
    shapes = set()
    for w in IDS:
        card = STORE.card(w)
        shapes.add(tuple(sorted(Counter((f.data_type, f.timing) for f in card.features).items())))
    assert len(shapes) == 1
    (shape,) = shapes
    assert dict(shape)[("lab", Timing.POST_OUTCOME)] == 3


def test_every_post_outcome_feature_is_a_reject_and_every_feature_is_clustered() -> None:
    for w in IDS:
        card, key = STORE.card(w), STORE.answer_key(w)
        post = {f.feature_id for f in card.features if f.timing is Timing.POST_OUTCOME}
        assert set(key.reject_set) == post
        assert set(key.clusters) == set(card.feature_ids()) == set(key.strata)


def test_budget_is_the_full_pool_at_published_prices() -> None:
    for w in IDS:
        card = STORE.card(w)
        per_patient = card.prices.recruit_per_patient + sum(f.assay_price for f in card.features)
        assert card.budget == pytest.approx(card.n_pool * per_patient)


def test_names_are_fake_and_cannot_collide_with_hgnc_symbols() -> None:
    for w in IDS:
        assert all(f.feature_id == f.feature_id.lower() for f in STORE.card(w).features)


def test_builder_is_deterministic(tmp_path: Path) -> None:
    build_toy.main(["--out", str(tmp_path / "a")])
    rebuilt, committed = tree_digest(tmp_path / "a"), tree_digest(fixture_store())
    differing = sorted(k for k in rebuilt.keys() | committed.keys() if rebuilt.get(k) != committed.get(k))
    assert not differing, differing


@pytest.mark.parametrize("mode", ["full", "seq"])
def test_oracle_scores_one_and_cheaters_are_held_at_the_floor(mode: str) -> None:
    ids = [w for w in IDS if w.endswith(f"-{mode}")]
    oracle, _ = evaluate(
        make_agent("oracle", STORE), STORE, Tier.PUBLIC_TRAIN, world_ids=ids, bootstrap_draws=100
    )
    assert oracle.discovery_score == 1.0
    for name in ("random", *CHEATERS):
        card, _ = evaluate(
            make_agent(name, STORE), STORE, Tier.PUBLIC_TRAIN, world_ids=ids, bootstrap_draws=100
        )
        assert card.discovery_score_unfloored is not None
        assert card.discovery_score_unfloored <= 0.02, name
