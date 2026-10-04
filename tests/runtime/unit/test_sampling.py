"""Seeded stratified world samples: the declared mix is reproduced and no sample is the first-N selection."""

from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest
from arena_factories import planted_world
from hypothesis import given, settings
from hypothesis import strategies as st
from onc_agi.adapters.cli import fixture_store
from onc_agi.core.schema import Tier
from onc_agi.infra.bundles import FileWorldStore, parse_id_list, world_id_list, write_world
from onc_agi.services.sampling import (
    NULL_FAMILY,
    UNKNOWN,
    WorldProfile,
    WorldSample,
    allocate,
    stratified_sample,
    world_profiles,
)

FIXTURES = FileWorldStore(fixture_store())
FIXTURE_PROFILES = world_profiles(FIXTURES, Tier.PUBLIC_TRAIN)
SOURCES = ("metabric", "msk_impact", "nhanes", "scanb", "tcga_brca", "tcga_brca+scanb", "tcga_pancan")
ROLES = ("generating", "interaction", "mediator", "confounder", "leak", "shift", "wrong_type", "module")


def synthetic_manifest(n_per_source: int = 40, seed: int = 7) -> list[dict[str, object]]:
    """A multi-source pack whose sorted ids cluster by source, as real shard prefixes do."""
    rng = random.Random(seed)
    worlds: list[dict[str, object]] = []
    for shard, source in enumerate(SOURCES):
        for _ in range(n_per_source + rng.randrange(n_per_source)):
            null = rng.random() < 0.2
            worlds.append(
                {
                    "world_id": f"pk-{shard}-{rng.getrandbits(40):010x}",
                    "source": source,
                    "role": rng.choice(ROLES),  # a null keeps its twin's role, as real manifests do
                    "is_null": null,
                    "mode": rng.choice(("full_access", "sequential")),
                }
            )
    return worlds


def _profiles(manifest: list[dict[str, object]]) -> tuple[WorldProfile, ...]:
    return tuple(
        WorldProfile(
            world_id=str(w["world_id"]),
            source=str(w["source"]),
            family=NULL_FAMILY if w["is_null"] else str(w["role"]),
            mode=str(w["mode"]),
            rows=300,
            features=40,
        )
        for w in manifest
    )


SYNTHETIC = _profiles(synthetic_manifest())


def assert_reproduces_declared_mix(profiles: tuple[WorldProfile, ...], sample: WorldSample, n: int) -> None:
    """The sample is its declared mix, and the mix is proportional allocation rounded level by level."""
    by_id = {p.world_id: p for p in profiles}
    assert len(sample.world_ids) == len(set(sample.world_ids)) == n
    assert list(sample.world_ids) == sorted(sample.world_ids) and set(sample.world_ids) <= set(by_id)
    drawn = Counter(by_id[w].stratum for w in sample.world_ids)
    assert {r.stratum: r.available for r in sample.mix} == dict(Counter(p.stratum for p in profiles))
    assert sum(r.drawn for r in sample.mix) == n
    sources: Counter[str] = Counter()
    modes: Counter[tuple[str, str]] = Counter()
    for row in sample.mix:
        assert drawn[row.stratum] == row.drawn  # the sample is exactly the declared mix
        assert abs(row.drawn - row.quota) < 1  # within rounding of its proportional quota
        sources[row.source] += row.drawn
        modes[row.source, row.mode] += row.drawn
    for source, seats in sources.items():  # sources: shares of n
        assert abs(seats - Fraction(n * sum(p.source == source for p in profiles), len(profiles))) < 1
    for (source, mode), seats in modes.items():  # modes: shares of their source's seats
        size = sum(r.available for r in sample.mix if (r.source, r.mode) == (source, mode))
        share = Fraction(sources[source] * size, sum(r.available for r in sample.mix if r.source == source))
        assert abs(seats - share) < 1
        for row in (r for r in sample.mix if (r.source, r.mode) == (source, mode)):
            assert row.quota == Fraction(seats * row.available, size)  # families: shares of those seats


def marginal_error(profiles: tuple[WorldProfile, ...], sample: WorldSample, field: str) -> Fraction:
    """Largest gap between a field value's drawn count and its share of ``n`` (overall, not nested)."""
    n = len(sample.world_ids)
    return max(
        abs(drawn - Fraction(n * available, len(profiles)))
        for available, drawn in sample.marginal(field).values()
    )


def first_ids(profiles: tuple[WorldProfile, ...], n: int) -> tuple[str, ...]:
    return tuple(sorted(p.world_id for p in profiles)[:n])


# ------------------------------------------------------------------------------------- fixture store


def test_fixture_profiles_separate_nulls_from_their_twins_roles() -> None:
    nulls = {w for w in FIXTURES.world_ids(Tier.PUBLIC_TRAIN) if FIXTURES.answer_key(w).is_null}
    assert nulls and {p.world_id for p in FIXTURE_PROFILES if p.family == NULL_FAMILY} == nulls
    for p in FIXTURE_PROFILES:
        card = FIXTURES.card(p.world_id)
        assert (p.mode, p.rows, p.features) == (card.mode.value, card.n_pool, len(card.features))
        if p.family != NULL_FAMILY:
            assert p.family == (FIXTURES.manifest_entry(p.world_id) or {}).get("role", UNKNOWN)


@pytest.mark.parametrize("seed", [0, 1, 2, 17])
@pytest.mark.parametrize("n", range(1, len(FIXTURE_PROFILES)))
def test_fixture_samples_reproduce_the_declared_mix_and_are_never_first_n(n: int, seed: int) -> None:
    sample = stratified_sample(FIXTURE_PROFILES, n, seed)
    assert_reproduces_declared_mix(FIXTURE_PROFILES, sample, n)
    assert sample.world_ids != first_ids(FIXTURE_PROFILES, n)
    assert stratified_sample(FIXTURE_PROFILES, n, seed) == sample  # seeded: the same draw every time


def test_the_whole_store_is_its_own_only_sample() -> None:
    n = len(FIXTURE_PROFILES)
    assert stratified_sample(FIXTURE_PROFILES, n, 5).world_ids == first_ids(FIXTURE_PROFILES, n)


def test_seeds_give_different_samples_of_the_same_mix() -> None:
    samples = {stratified_sample(SYNTHETIC, 60, seed) for seed in range(8)}
    assert len({s.world_ids for s in samples}) == 8
    assert len({tuple((r.stratum, r.available) for r in s.mix) for s in samples}) == 1


# -------------------------------------------------------------------------- synthetic multi-source pack


def test_first_n_of_a_sharded_pack_is_single_source_but_a_sample_spans_every_source() -> None:
    by_id = {p.world_id: p for p in SYNTHETIC}
    assert {by_id[w].source for w in first_ids(SYNTHETIC, 40)} == {"metabric"}
    sample = stratified_sample(SYNTHETIC, 40, 0)
    assert set(sample.marginal("source")) == set(SOURCES)
    assert all(drawn >= 1 for _, drawn in sample.marginal("source").values())


@settings(max_examples=200, deadline=None)
@given(n=st.integers(1, len(SYNTHETIC) - 1), seed=st.integers(0, 2**32 - 1))
def test_synthetic_samples_reproduce_the_declared_mix_and_are_never_first_n(n: int, seed: int) -> None:
    sample = stratified_sample(SYNTHETIC, n, seed)
    assert_reproduces_declared_mix(SYNTHETIC, sample, n)
    assert sample.world_ids != first_ids(SYNTHETIC, n)
    assert marginal_error(SYNTHETIC, sample, "source") < 1  # guaranteed: sources are allocated first
    # modes and families are allocated within sources; their carried rounding error keeps them close
    assert all(marginal_error(SYNTHETIC, sample, field) <= 1 for field in ("family", "mode"))


@settings(max_examples=200, deadline=None)
@given(
    sizes=st.lists(st.integers(2, 12), min_size=1, max_size=12),
    data=st.data(),
    seed=st.integers(0, 2**32 - 1),
)
def test_any_mix_of_strata_with_two_or_more_worlds_is_reproduced_and_never_first_n(
    sizes: list[int], data: st.DataObject, seed: int
) -> None:
    rng = random.Random(seed)
    profiles = tuple(
        WorldProfile(
            f"w{rng.getrandbits(32):08x}{s:02d}{i:02d}", f"src{s}", "generating", "full_access", 9, 3
        )
        for s, size in enumerate(sizes)
        for i in range(size)
    )
    n = data.draw(st.integers(1, len(profiles)))
    sample = stratified_sample(profiles, n, seed)
    assert_reproduces_declared_mix(profiles, sample, n)
    if n < len(profiles):
        assert sample.world_ids != first_ids(profiles, n)


def test_a_store_manifest_supplies_source_and_role(tmp_path: Path) -> None:
    root = tmp_path / "pack"
    shutil.copytree(fixture_store(), root)
    (root / "manifest.json").unlink(missing_ok=True)
    ids = FIXTURES.world_ids(Tier.PUBLIC_TRAIN)
    worlds = [
        {"world_id": w, "source": SOURCES[i % 3], "role": "generating", "mode": "ignored"}
        for i, w in enumerate(ids)
    ]
    (root / "manifests").mkdir()
    (root / "manifests" / "pack-a.json").write_text(json.dumps({"worlds": worlds[:10]}))
    (root / "manifests" / "pack-b.json").write_text(json.dumps({"worlds": worlds[10:]}))
    profiles = world_profiles(FileWorldStore(root), Tier.PUBLIC_TRAIN)
    assert Counter(p.source for p in profiles) == Counter(SOURCES[i % 3] for i in range(len(ids)))
    assert {p.family for p in profiles} == {"generating", NULL_FAMILY}
    assert [p.mode for p in profiles] == [FIXTURES.card(w).mode.value for w in ids]  # the card decides
    sample = stratified_sample(profiles, 9, 0)
    assert_reproduces_declared_mix(profiles, sample, 9)
    assert {r.source for r in sample.mix} == set(SOURCES[:3])


def test_without_a_manifest_or_answer_keys_only_mode_is_visible() -> None:
    card = FIXTURES.card(FIXTURES.world_ids(Tier.PUBLIC_TRAIN)[0])
    hidden = card.model_copy(update={"tier": Tier.PUBLIC_EVAL})

    class EvalStore:
        def world_ids(self, tier: Tier) -> tuple[str, ...]:
            return (hidden.world_id,) if tier is Tier.PUBLIC_EVAL else ()

        def card(self, world_id: str):  # type: ignore[no-untyped-def]
            return hidden

        def answer_key(self, world_id: str):  # type: ignore[no-untyped-def]
            raise AssertionError("eval strata never read answer keys")

    (profile,) = world_profiles(EvalStore(), Tier.PUBLIC_EVAL)  # type: ignore[arg-type]
    assert (profile.source, profile.family, profile.mode) == (UNKNOWN, UNKNOWN, card.mode.value)


@pytest.mark.parametrize("tier", [Tier.PUBLIC_EVAL, Tier.PRIVATE])
def test_eval_tier_profiles_ignore_role_and_null_even_when_a_manifest_lists_them(
    tmp_path: Path, tier: Tier
) -> None:
    root, keys = tmp_path / "store", tmp_path / "keys"
    worlds = [planted_world(f"ev-{i}", signal=i != 1, seed=i, n_pool=60, tier=tier) for i in range(3)]
    for world, key in worlds:
        write_world(root, world, key, keys_dir=keys)
    entries = [
        {"world_id": "ev-0", "source": "scanb", "role": "leak", "is_null": False},
        {"world_id": "ev-1", "source": "nhanes", "role": "generating", "is_null": True},
    ]  # ev-2 is not listed
    (root / "manifest.json").write_text(json.dumps({"worlds": entries}))
    store = FileWorldStore(root, keys)  # the operator's store: it could read keys, and must not

    def no_keys(world_id: str) -> None:
        raise AssertionError(f"read the answer key of {world_id}")

    store.answer_key = no_keys  # type: ignore[assignment,method-assign]
    profiles = {p.world_id: p for p in world_profiles(store, tier)}
    assert {w: (p.source, p.family) for w, p in profiles.items()} == {
        "ev-0": ("scanb", UNKNOWN),
        "ev-1": ("nhanes", UNKNOWN),
        "ev-2": (UNKNOWN, UNKNOWN),
    }


# ------------------------------------------------------------------------------------- allocation


def test_allocation_is_largest_remainder_with_carried_error_then_seeded_ties() -> None:
    assert allocate({"a": 5, "b": 3, "c": 2}, 4) == {"a": 2, "b": 1, "c": 1}  # quotas 2.0, 1.2, 0.8
    tied = {allocate({"a": 1, "b": 1, "c": 1, "d": 1}, 2, seed)["a"] for seed in range(20)}
    assert tied == {0, 1}  # ties go by a seeded order, not by name
    assert allocate({"a": 1, "b": 1}, 1, 3) == allocate({"a": 1, "b": 1}, 1, 3)
    for seed in range(10):  # a value already short of its share wins the tie
        assert allocate({"a": 1, "b": 1}, 1, seed, carry={"b": Fraction(1, 2)}) == {"a": 0, "b": 1}
        # but a whole-number quota never gains a seat, so every count stays within one of its quota
        assert allocate({"a": 2, "b": 1, "c": 1}, 2, seed, carry={"a": Fraction(5)})["a"] == 1


@pytest.mark.parametrize(("sizes", "n"), [({"a": 2}, 3), ({"a": 2}, -1), ({}, 1)])
def test_allocation_refuses_impossible_draws(sizes: dict[str, int], n: int) -> None:
    with pytest.raises(ValueError, match="cannot draw"):
        allocate(sizes, n)


@pytest.mark.parametrize("n", [0, len(SYNTHETIC) + 1])
def test_samples_refuse_impossible_sizes(n: int) -> None:
    with pytest.raises(ValueError, match="cannot draw"):
        stratified_sample(SYNTHETIC, n)


def test_samples_refuse_duplicate_worlds() -> None:
    with pytest.raises(ValueError, match="distinct"):
        stratified_sample((*SYNTHETIC[:3], SYNTHETIC[0]), 2)


def test_the_mix_reports_render_every_stratum_and_field() -> None:
    sample = stratified_sample(SYNTHETIC, 50, 4)
    table = sample.table().splitlines()
    assert len(table) == len(sample.mix) + 2 and table[-1].split()[:4] == [
        "total",
        str(len(SYNTHETIC)),
        "50.00",
        "50",
    ]
    summary = sample.summary()
    assert summary.startswith(f"50 of {len(SYNTHETIC)} worlds, seed 4")
    assert all(f"  {field}" in summary for field in ("source", "family", "mode"))
    with pytest.raises(ValueError, match="no stratum field"):
        sample.marginal("rows")


def test_id_lists_read_files_or_comma_separated_ids(tmp_path: Path) -> None:
    path = tmp_path / "set.txt"
    path.write_text("b\n\n a \nc\n")
    assert world_id_list(str(path)) == ("b", "a", "c")
    assert world_id_list("x, y,,z") == ("x", "y", "z")


def test_a_long_comma_separated_list_is_never_looked_up_as_a_path() -> None:
    ids = tuple(f"pt-full-{i % 7}-18a88-{i:08x}" for i in range(240))  # about 6,000 characters
    assert world_id_list(",".join(ids)) == ids
    long_id = "w" * 5000  # one overlong id: the filesystem rejects it as a name, so it is an id
    assert world_id_list(long_id) == (long_id,)


def test_id_list_files_take_comments_and_crlf(tmp_path: Path) -> None:
    path = tmp_path / "set.txt"
    path.write_bytes(b"# core-240, seed 0\r\nb\r\n\r\na  # METABRIC\r\n  # trailing note\r\nc")
    assert world_id_list(str(path)) == ("b", "a", "c")
    assert parse_id_list(path.read_text()) == ("b", "a", "c")
