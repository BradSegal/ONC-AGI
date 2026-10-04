"""Every catalogue method plays both modes through the standard agent path, on binary and survival worlds.

Binary worlds are the packaged fixture worlds (one with signal and one without, per mode);
survival worlds come from :func:`arena_factories.survival_world`, since the fixture set is
binary. Agents are built by :func:`make_agent` and played by :func:`evaluate`, exactly as
``onc-agi play`` and the live harness run them.
"""

from __future__ import annotations

from functools import cache
from importlib import resources
from pathlib import Path

import pytest
from arena_factories import LEAK, InMemoryStore, fid, survival_world
from onc_agi.adapters.agents import CATALOGUE, make_agent
from onc_agi.core.ports import WorldStore
from onc_agi.core.schema import Mode, Tier, Timing
from onc_agi.infra.bundles import FileWorldStore
from onc_agi.services.kit import evaluate

MODES = (Mode.FULL_ACCESS, Mode.SEQUENTIAL)
# methods with a Cox form, or a tree form on the event indicator, that must find a strong planted hazard
SURVIVAL_FINDERS = ("adjusted", "boruta", "stability_pfer", "penalised_cox", "two_phase")
# Boruta fits up to 100 forests per world; its cases take tens of seconds
METHODS = [pytest.param(n, marks=pytest.mark.slow) if n == "boruta" else n for n in CATALOGUE]


@cache
def fixture_store() -> FileWorldStore:
    root = Path(str(resources.files("onc_agi") / "fixtures" / "store"))
    if not (root / "public_train").exists():
        pytest.skip("packaged fixture worlds are not built")
    return FileWorldStore(root)


def fixture_pair(mode: Mode) -> tuple[str, str]:
    """The first fixture world with signal and the first without, in ``mode``."""
    store = fixture_store()
    ids = [w for w in store.world_ids(Tier.PUBLIC_TRAIN) if store.card(w).mode is mode]
    signal = next(w for w in ids if not store.answer_key(w).is_null)
    null = next(w for w in ids if store.answer_key(w).is_null)
    return signal, null


@cache
def survival_store(mode: Mode) -> InMemoryStore:
    return InMemoryStore.of(
        survival_world(f"surv-{mode.value}-1", signal=True, seed=31, mode=mode),
        survival_world(f"surv-{mode.value}-0", signal=False, seed=32, mode=mode),
    )


def check_played(store: WorldStore, name: str, ids: tuple[str, ...]) -> list[tuple[str, ...]]:
    """Play ``name`` on ``ids`` and check the run is a valid, scored, budget-respecting episode set."""
    if name == "knockoffs":
        pytest.importorskip("knockpy")
    card, results = evaluate(
        make_agent(name, store), store, Tier.PUBLIC_TRAIN, world_ids=ids, bootstrap_draws=50
    )
    assert card.n_worlds == len(ids) and card.discovery_score is not None  # signal and null both scored
    assert card.leak_rate == 0.0
    for result in results:
        world = store.card(result.world_id)
        baseline = {f.feature_id for f in world.features if f.timing is Timing.BASELINE}
        assert set(result.ranking) <= baseline and len(set(result.ranking)) == len(result.ranking)
        assert result.spent <= world.budget + 1e-9
        assert (result.spent > 0) is (world.mode is Mode.SEQUENTIAL)
    return [r.ranking for r in results]


@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.value)
@pytest.mark.parametrize("name", METHODS)
def test_catalogue_methods_play_binary_fixture_worlds_in_both_modes(name: str, mode: Mode) -> None:
    check_played(fixture_store(), name, fixture_pair(mode))


@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.value)
@pytest.mark.parametrize("name", METHODS)
def test_catalogue_methods_play_survival_worlds_in_both_modes(name: str, mode: Mode) -> None:
    store = survival_store(mode)
    signal, null = store.world_ids(Tier.PUBLIC_TRAIN)[::-1]  # "...-1" sorts after "...-0"
    rankings = check_played(store, name, (signal, null))
    assert LEAK not in rankings[0]
    if name in SURVIVAL_FINDERS:
        assert rankings[0][:1] == (fid(0),)
    if name == "icp":
        assert rankings == [(), ()]  # one environment: invariance cannot be assessed, so it abstains
