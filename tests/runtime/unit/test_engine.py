"""Episode engine: what an agent sees, what it pays, and how requests are de-duplicated."""

from __future__ import annotations

import numpy as np
import pytest
from arena_factories import fid, make_card, make_world
from onc_agi.core.errors import ArenaError
from onc_agi.core.schema import Assay, EpisodeStatus, ErrorCode, Mode, Recruit, Reset, Submit
from onc_agi.core.world import WorldData
from onc_agi.services.engine import Episode

RECRUIT, ASSAY = 2.0, 0.5


def sequential(
    n_pool: int = 20, n_features: int = 4, strata: tuple[str, ...] = ("all",), budget: float | None = None
) -> Episode:
    card = make_card(
        "w-seq",
        n_pool=n_pool,
        n_features=n_features,
        mode=Mode.SEQUENTIAL,
        strata=strata,
        recruit_price=RECRUIT,
        assay_price=ASSAY,
        budget=budget,
    )
    return Episode(make_world(card, seed=3))


def full() -> Episode:
    return Episode(make_world(make_card("w-full", n_pool=12, n_features=3), seed=5))


def code(excinfo: pytest.ExceptionInfo[ArenaError]) -> ErrorCode:
    return excinfo.value.code


# ---------------------------------------------------------------- full access


def test_full_access_reveals_the_whole_pool_and_only_allows_submit() -> None:
    ep = full()
    view = ep.apply(Reset(request_id="r0", world_id="w-full"))
    assert view.available == ("submit",)
    assert view.rows == tuple(range(12)) and all(view.measured)
    np.testing.assert_array_equal(view.x, ep.world.x)
    np.testing.assert_array_equal(view.outcome, ep.world.y)
    assert view.spent == 0.0


@pytest.mark.parametrize(
    "action", [Recruit(request_id="a", count=1), Assay(request_id="a", feature_ids=("f00",))]
)
def test_full_access_refuses_acquisition(action: Recruit | Assay) -> None:
    with pytest.raises(ArenaError) as err:
        full().apply(action)
    assert code(err) is ErrorCode.ACTION_NOT_AVAILABLE


def test_submit_closes_the_episode_and_refuses_further_actions() -> None:
    ep = full()
    view = ep.apply(Submit(request_id="s", ranking=(fid(1),)))
    assert view.status is EpisodeStatus.SUBMITTED and view.available == ()
    assert ep.submission == (fid(1),)
    with pytest.raises(ArenaError) as err:
        ep.apply(Submit(request_id="s2", ranking=()))
    assert code(err) is ErrorCode.EPISODE_CLOSED


def test_submitting_an_unknown_feature_fails_without_closing() -> None:
    ep = full()
    with pytest.raises(ArenaError) as err:
        ep.apply(Submit(request_id="s", ranking=("ghost",)))
    assert code(err) is ErrorCode.UNKNOWN_FEATURE
    assert ep.status is EpisodeStatus.ACTIVE and ep.submission is None


def test_reset_for_another_world_is_refused() -> None:
    with pytest.raises(ArenaError) as err:
        full().apply(Reset(request_id="r", world_id="w-other"))
    assert code(err) is ErrorCode.UNKNOWN_WORLD


def test_reset_on_a_started_episode_neither_rewinds_nor_advances() -> None:
    ep = sequential()
    ep.apply(Reset(request_id="r0", world_id="w-seq"))
    ep.apply(Recruit(request_id="a", count=3))
    before = ep.view()
    again = ep.apply(Reset(request_id="r1", world_id="w-seq"))
    assert (again.step, again.rows, again.spent) == (before.step, before.rows, before.spent)


# ---------------------------------------------------------------- sequential


def test_sequential_reset_reveals_nothing() -> None:
    view = sequential().apply(Reset(request_id="r0", world_id="w-seq"))
    assert view.rows == () and view.x.shape == (0, 4) and not any(view.measured)
    assert view.available == ("recruit", "assay", "submit")


def test_recruit_pops_the_fixed_queue_and_charges_per_patient() -> None:
    ep = sequential()
    queue = ep.world.queues["all"]
    first = ep.apply(Recruit(request_id="a", count=3))
    second = ep.apply(Recruit(request_id="b", count=2))
    assert first.rows == queue[:3] and second.rows == queue[:5]
    assert second.spent == pytest.approx(5 * RECRUIT)
    np.testing.assert_array_equal(second.outcome, ep.world.y[list(queue[:5])])
    assert np.isnan(second.x).all()


def test_recruiting_beyond_the_pool_takes_what_remains_then_refuses() -> None:
    ep = sequential(n_pool=5)
    view = ep.apply(Recruit(request_id="a", count=50))
    assert len(view.rows) == 5 and view.spent == pytest.approx(5 * RECRUIT)
    with pytest.raises(ArenaError) as err:
        ep.apply(Recruit(request_id="b", count=1))
    assert code(err) is ErrorCode.ACTION_NOT_AVAILABLE


def test_recruiting_an_unknown_stratum_is_refused() -> None:
    with pytest.raises(ArenaError) as err:
        sequential().apply(Recruit(request_id="a", count=1, stratum="nope"))
    assert code(err) is ErrorCode.UNKNOWN_STRATUM


def test_strata_have_independent_queues() -> None:
    ep = sequential(strata=("A", "B"))
    view = ep.apply(Recruit(request_id="a", count=2, stratum="B"))
    assert view.rows == ep.world.queues["B"][:2]
    assert set(view.stratum) == {"B"}


def test_assay_charges_only_unmeasured_cells_and_reveals_values() -> None:
    ep = sequential()
    ep.apply(Recruit(request_id="a", count=4))
    view = ep.apply(Assay(request_id="b", feature_ids=(fid(1), fid(1), fid(2))))
    assert view.spent == pytest.approx(4 * RECRUIT + 4 * 2 * ASSAY)
    assert view.measured == (False, True, True, False)
    rows = list(view.rows)
    np.testing.assert_array_equal(view.x[:, 1], ep.world.x[rows, 1])
    again = ep.apply(Assay(request_id="c", feature_ids=(fid(1),)))
    assert again.spent == view.spent


def test_newly_recruited_patients_are_unmeasured_until_assayed() -> None:
    ep = sequential()
    ep.apply(Recruit(request_id="a", count=2))
    ep.apply(Assay(request_id="b", feature_ids=(fid(0),)))
    view = ep.apply(Recruit(request_id="c", count=2))
    assert not np.isnan(view.x[:2, 0]).any() and np.isnan(view.x[2:, 0]).all()
    topped = ep.apply(Assay(request_id="d", feature_ids=(fid(0),)))
    assert topped.spent == pytest.approx(view.spent + 2 * ASSAY)


def test_assaying_with_nobody_recruited_is_free_and_reveals_nothing() -> None:
    view = sequential().apply(Assay(request_id="a", feature_ids=(fid(0),)))
    assert view.spent == 0.0 and not any(view.measured)


def test_assaying_an_unknown_feature_is_refused() -> None:
    with pytest.raises(ArenaError) as err:
        sequential().apply(Assay(request_id="a", feature_ids=("ghost",)))
    assert code(err) is ErrorCode.UNKNOWN_FEATURE


def test_over_budget_actions_change_nothing() -> None:
    ep = sequential(budget=5 * RECRUIT)
    ep.apply(Recruit(request_id="a", count=4))
    before = ep.view()
    with pytest.raises(ArenaError) as err:
        ep.apply(Recruit(request_id="b", count=2))
    assert code(err) is ErrorCode.OVER_BUDGET
    with pytest.raises(ArenaError):
        ep.apply(Assay(request_id="c", feature_ids=(fid(0), fid(1), fid(2))))
    after = ep.view()
    assert (after.rows, after.spent, after.step, after.measured) == (
        before.rows,
        before.spent,
        before.step,
        before.measured,
    )


def test_the_published_budget_affords_the_whole_pool() -> None:
    ep = sequential(n_pool=10, n_features=3)
    ep.apply(Recruit(request_id="a", count=10))
    view = ep.apply(Assay(request_id="b", feature_ids=(fid(0), fid(1), fid(2))))
    assert view.spent == pytest.approx(ep.world.card.budget)


def test_a_retried_request_returns_the_original_response_without_charging() -> None:
    ep = sequential()
    action = Recruit(request_id="a", count=3)
    once, twice = ep.apply(action), ep.apply(action)
    assert twice is once
    assert ep.spent == pytest.approx(3 * RECRUIT) and ep.step == 1


def test_reusing_a_request_id_for_a_different_action_is_a_conflict() -> None:
    ep = sequential()
    ep.apply(Recruit(request_id="a", count=3))
    with pytest.raises(ArenaError) as err:
        ep.apply(Recruit(request_id="a", count=4))
    assert code(err) is ErrorCode.REQUEST_CONFLICT


def test_a_failed_request_can_be_retried_under_the_same_id() -> None:
    """Refused requests are not recorded, so the same id may carry a corrected action."""
    ep = sequential(budget=3 * RECRUIT)
    with pytest.raises(ArenaError):
        ep.apply(Recruit(request_id="a", count=4))
    with pytest.raises(ArenaError) as err:
        ep.apply(Recruit(request_id="a", count=4))
    assert code(err) is ErrorCode.OVER_BUDGET
    assert len(ep.apply(Recruit(request_id="a", count=3)).rows) == 3


def test_identical_action_sequences_reveal_identical_data() -> None:
    actions = [
        Reset(request_id="r", world_id="w-seq"),
        Recruit(request_id="a", count=7),
        Assay(request_id="b", feature_ids=(fid(0), fid(3))),
        Recruit(request_id="c", count=4),
        Submit(request_id="d", ranking=(fid(3),)),
    ]
    one, two = sequential(), sequential()
    for action in actions:
        a, b = one.apply(action), two.apply(action)
        assert (a.rows, a.spent, a.measured, a.status) == (b.rows, b.spent, b.measured, b.status)
        np.testing.assert_array_equal(a.x, b.x)
        np.testing.assert_array_equal(a.outcome, b.outcome)


def test_wire_observation_carries_only_measured_columns_with_none_for_missing() -> None:
    ep = sequential()
    ep.apply(Recruit(request_id="a", count=2))
    ep.apply(Assay(request_id="b", feature_ids=(fid(2),)))
    view = ep.apply(Recruit(request_id="c", count=1))
    obs = view.to_observation(ep.world.patient_ids)
    assert set(obs.revealed.columns) == {fid(2)}
    assert obs.revealed.columns[fid(2)][2] is None
    assert obs.revealed.patient_ids == tuple(ep.world.patient_ids[i] for i in view.rows)


# ---------------------------------------------------------------- world data


def _world_kwargs() -> dict[str, object]:
    world = make_world(make_card(n_pool=4, n_features=2))
    return {f: getattr(world, f) for f in ("card", "patient_ids", "x", "y", "stratum", "queues")}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("x", np.zeros((4, 3)), "width"),
        ("y", np.zeros(3, dtype=np.int64), "misaligned"),
        ("patient_ids", ("a", "b", "c"), "misaligned"),
        ("queues", {"other": (0, 1, 2, 3)}, "strata"),
        ("queues", {"all": (0, 1, 1, 3)}, "partition"),
        ("queues", {"all": (0, 1, 2)}, "partition"),
    ],
)
def test_world_data_rejects_inconsistent_construction(field: str, value: object, message: str) -> None:
    kwargs = _world_kwargs() | {field: value}
    with pytest.raises(ValueError, match=message):
        WorldData(**kwargs)  # type: ignore[arg-type]


def test_world_column_lookup_follows_card_order() -> None:
    world = make_world(make_card(n_features=5))
    assert [world.column(fid(j)) for j in range(5)] == list(range(5))
