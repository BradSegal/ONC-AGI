"""Agent registry: reference, baseline, catalogue, sequential-policy and cheater agents.

Every agent runs through :class:`onc_agi.services.kit.PipelineAgent` or a
policy subclass, i.e. the same execution path as any external agent.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import replace

import numpy as np
from scipy import stats

from onc_agi.adapters.agents import baselines, catalogue, cheaters
from onc_agi.core.ports import WorldStore
from onc_agi.core.schema import Action, Assay, GroupLabel, Mode, Recruit, Submit, Timing, WorldCard
from onc_agi.services.engine import EpisodeView
from onc_agi.services.kit import Agent, AnalysisInput, Analyst, PipelineAgent, analysis_input

BASELINES: dict[str, Analyst] = {
    "univariate_bh": baselines.univariate_bh,
    "lasso": baselines.lasso,
    "elastic_net": baselines.elastic_net,
    "stability": baselines.stability_selection,
    "random_forest": baselines.random_forest,
    "knockoffs": baselines.knockoffs,
    "forward_score": baselines.forward_score,
    # the literature-grounded catalogue (catalogue.py)
    "adjusted": catalogue.adjusted_analysis,
    "boruta": catalogue.boruta,
    "stability_pfer": catalogue.stability_pfer,
    "penalised_cox": catalogue.penalised_cox,
    "icp": catalogue.invariant_prediction,
}

CHEATERS: dict[str, Analyst] = {
    "giant_list": cheaters.giant_list,
    "always_empty": cheaters.always_empty,
    "random_abstain": cheaters.random_abstain,
    "leak_exploiter": cheaters.leak_exploiter,
    "auc_maximiser": cheaters.auc_maximiser,
    "famous_genes": cheaters.famous_genes,
    "metadata_only": cheaters.metadata_only,
    "variance_ranker": cheaters.variance_ranker,
    "hub_ranker": cheaters.hub_ranker,
    "cluster_size_ranker": cheaters.cluster_size_ranker,
    "synthetic_ranker": cheaters.synthetic_ranker,
}

CHANCE_LEVEL = {
    "metadata_only",
    "variance_ranker",
    "hub_ranker",
    "cluster_size_ranker",
    "synthetic_ranker",
    "learned_ranker",
}


def _stratum_sizes(card: WorldCard) -> dict[str, int]:
    if card.stratum_sizes:
        return dict(card.stratum_sizes)
    share = card.n_pool // len(card.strata)  # older cards: assume equal strata
    return dict.fromkeys(card.strata, share)


def _recruit_towards(agent: Agent, card: WorldCard, view: EpisodeView, fraction: float) -> Recruit | None:
    """Recruit from the first stratum still below ``fraction`` of its published size, if any."""
    recruited = Counter(view.stratum)
    for stratum, size in _stratum_sizes(card).items():
        target = min(size, round(fraction * size))
        if recruited[stratum] < target:
            return Recruit(request_id=agent.request_id(), count=target - recruited[stratum], stratum=stratum)
    return None


class GroupSequentialAgent(Agent):
    """Recruit in stages, assay everything on new patients, stop early when the analyst's
    selection is non-empty and unchanged between two stages (or at the last stage).

    Each stage recruits the same fraction of *every* stratum, up to the stratum's published
    size, so multi-stratum worlds are sampled proportionally and no stratum is over-drawn.
    """

    def __init__(
        self, name: str, analyst: Analyst, stages: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    ) -> None:
        super().__init__()
        self.name = name
        self.analyst = analyst
        self.stages = stages
        self._stage = 0
        self._previous: tuple[str, ...] | None = None
        # rows assayed so far: NaN cannot tell "not yet assayed" from genuine missingness in the
        # source, so the agent tracks its own purchases (terminates on missing data, locally and
        # over HTTP, where both arrive as null)
        self._assayed_rows = 0

    def _next_recruit(self, card: WorldCard, view: EpisodeView) -> Recruit | None:
        """The first stratum still below this stage's target, if any."""
        return _recruit_towards(self, card, view, self.stages[self._stage])

    def choose_action(self, card: WorldCard, view: EpisodeView) -> Action:
        if card.mode is Mode.FULL_ACCESS:
            # nothing to acquire: analyse what is revealed, as PipelineAgent does
            return Submit(
                request_id=self.request_id(), ranking=tuple(self.analyst(analysis_input(card, view)))
            )
        if not view.rows:
            self._stage, self._previous, self._assayed_rows = 0, None, 0
        while True:
            recruit = self._next_recruit(card, view)
            if recruit is not None:
                return recruit
            if len(view.rows) > self._assayed_rows:
                self._assayed_rows = len(view.rows)
                return Assay(request_id=self.request_id(), feature_ids=card.feature_ids())
            current = tuple(self.analyst(analysis_input(card, view)))
            last = self._stage >= len(self.stages) - 1
            if last or (current and current == self._previous):
                return Submit(request_id=self.request_id(), ranking=current)
            self._previous = current
            self._stage += 1


SCREEN_FRACTION = 0.5  # share of every stratum recruited and fully assayed in the screening phase
SCREEN_ALPHA = 0.10  # two-sided screening level: features with p above it are not bought again


def _adjusted_z(data: AnalysisInput) -> dict[str, float]:
    """Clinical- and stratum-adjusted score z per feature, empty when the data cannot support it."""
    d = catalogue.design(data)
    if not baselines.usable(data, d.x):
        return {}
    z = catalogue.adjusted_main_z(data, d)
    return {} if z is None else dict(zip(d.ids, (float(v) for v in z), strict=True))


def _rows(data: AnalysisInput, start: int, features: tuple[str, ...]) -> AnalysisInput:
    """The rows from ``start`` on (later recruits) and only ``features``."""
    cols = [data.feature_ids.index(f) for f in features]
    return replace(
        data,
        feature_ids=features,
        x=data.x[start:][:, cols],
        y=data.y[start:],
        stratum=data.stratum[start:],
        time=None if data.time is None else data.time[start:],
    )


class TwoPhaseAgent(Agent):
    """Two-phase acquisition: screen half the patients on every assay, then buy only the survivors.

    Phase 1 recruits ``SCREEN_FRACTION`` = 50% of every stratum and assays every baseline
    feature (post-outcome assays are never bought). Each feature is screened by its clinical-
    and stratum-adjusted score z (:func:`catalogue.adjusted_main_z`, logistic or Cox); features
    with two-sided p above ``SCREEN_ALPHA`` = 0.10 are dropped, and if none survives the agent
    submits an empty list without buying more. Phase 2 recruits the rest of every stratum and
    assays only the survivors (and the clinical indicators, the adjustment covariates) on the
    new patients. Each survivor's evidence is the weighted inverse-normal combination
    ``sqrt(n1/n) z1 + sqrt(n2/n) z2`` of its independent phase statistics, with weights fixed
    by the design's sample sizes; dropped features get p = 1, which can only lower their
    rejection probability, so Benjamini-Hochberg at 5% over every baseline feature keeps its
    level. This is the joint analysis of a two-stage design (Skol et al. 2006) made valid
    without an interim correction through the combination test (Bauer & Kohne 1994;
    Lehmacher & Wassmer 1999). A screen that the data cannot support keeps every feature.

    In full-access worlds nothing can be acquired, so the policy analyses every revealed row
    with the same adjusted score tests and Benjamini-Hochberg rule.

    References: Bauer & Kohne (1994) Biometrics 50:1029; Lehmacher & Wassmer (1999) Biometrics
    55:1286; Satagopan, Verbel, Venkatraman, Offit & Begg (2002) Biometrics 58:163; Skol,
    Scott, Abecasis & Boehnke (2006) Nat Genet 38:209.
    """

    name = "two_phase"

    def __init__(self) -> None:
        super().__init__()
        self._reset()

    def _reset(self) -> None:
        self._phase = 0
        self._screened = 0
        self._z1: dict[str, float] = {}
        self._survivors: tuple[str, ...] = ()

    @staticmethod
    def _select(p: dict[str, float], candidates: tuple[str, ...]) -> tuple[str, ...]:
        values = np.array([p.get(f, 1.0) for f in candidates])
        return tuple(candidates[j] for j in baselines.bh_select(values, baselines.FDR))

    def choose_action(self, card: WorldCard, view: EpisodeView) -> Action:
        baseline = tuple(f.feature_id for f in card.features if f.timing is Timing.BASELINE)
        if card.mode is Mode.FULL_ACCESS:
            z = _adjusted_z(analysis_input(card, view))
            p = {f: float(2 * stats.norm.sf(abs(v))) for f, v in z.items()}
            return Submit(request_id=self.request_id(), ranking=self._select(p, baseline))
        if not view.rows:
            self._reset()
        clinical = tuple(
            f.feature_id for f in card.features if f.data_type == "clinical" and f.feature_id in baseline
        )
        if self._phase == 0:
            recruit = _recruit_towards(self, card, view, SCREEN_FRACTION)
            if recruit is not None:
                return recruit
            self._phase, self._screened = 1, len(view.rows)
            if view.rows and baseline:
                return Assay(request_id=self.request_id(), feature_ids=baseline)
        if self._phase == 1:
            self._phase = 2
            self._z1 = _adjusted_z(analysis_input(card, view)) if view.rows else {}
            if self._z1:
                self._survivors = tuple(
                    f for f in baseline if 2 * stats.norm.sf(abs(self._z1.get(f, 0.0))) <= SCREEN_ALPHA
                )
            else:
                self._survivors = baseline
            if not self._survivors:
                return Submit(request_id=self.request_id(), ranking=())
        if self._phase == 2:
            recruit = _recruit_towards(self, card, view, 1.0)
            if recruit is not None:
                return recruit
            self._phase = 3
            if len(view.rows) > self._screened:
                wanted = tuple(dict.fromkeys(self._survivors + clinical))
                return Assay(request_id=self.request_id(), feature_ids=wanted)
        n1, n = self._screened, len(view.rows)
        later = tuple(dict.fromkeys(self._survivors + clinical))
        z2 = _adjusted_z(_rows(analysis_input(card, view), n1, later)) if n > n1 else {}
        w1, w2 = np.sqrt(n1 / n), np.sqrt((n - n1) / n)
        p = {
            f: float(2 * stats.norm.sf(abs(w1 * self._z1.get(f, 0.0) + w2 * z2.get(f, 0.0))))
            for f in self._survivors
        }
        return Submit(request_id=self.request_id(), ranking=self._select(p, baseline))


class OracleAgent(Agent):
    """Reference: lists one true feature per recoverable part (scores 100% by construction)."""

    name = "oracle"

    def __init__(self, store: WorldStore) -> None:
        super().__init__()
        self.store = store

    def choose_action(self, card: WorldCard, view: EpisodeView) -> Action:
        """Submit the truth straight away: a clairvoyant reference acquires nothing."""
        key = self.store.answer_key(card.world_id)
        ranking = [p.true_feature for g in key.groups if g.label is GroupLabel.RECOVERABLE for p in g.parts]
        return Submit(request_id=self.request_id(), ranking=tuple(dict.fromkeys(ranking)))


def make_agent(name: str, store: WorldStore | None = None, *, mode: Mode = Mode.FULL_ACCESS) -> Agent:
    """Build a registered agent by name (``seq_<baseline>`` for the group-sequential policy)."""
    if name == "oracle":
        if store is None:
            raise ValueError("the oracle reference agent needs a store with answer keys")
        return OracleAgent(store)
    if name == "random":
        return PipelineAgent("random", cheaters.random_ranking)
    if name == TwoPhaseAgent.name:
        return TwoPhaseAgent()
    if name.startswith("seq_") and name[4:] in BASELINES:
        return GroupSequentialAgent(name, BASELINES[name[4:]])
    table: dict[str, Analyst] = BASELINES | CHEATERS
    if name not in table:
        raise KeyError(
            f"unknown agent {name!r}; known: {[*sorted(table), 'oracle', 'random', TwoPhaseAgent.name]}"
        )
    return PipelineAgent(name, table[name])


AGENT_NAMES: tuple[str, ...] = ("oracle", "random", *BASELINES, TwoPhaseAgent.name, *CHEATERS)
CATALOGUE: tuple[str, ...] = (
    "adjusted",
    "boruta",
    "stability_pfer",
    "knockoffs",
    "penalised_cox",
    "icp",
    TwoPhaseAgent.name,
)  # the literature-grounded methods catalogue (catalogue.py, TwoPhaseAgent and the knockoff rung).
# ``knockoffs`` is a modified-FDR rung, not an error-controlled selector: it claims on many
# no-signal worlds, and Gaussian knockoffs lack model-X validity on binary and sparse columns.
AgentFactory = Callable[[], Agent]
