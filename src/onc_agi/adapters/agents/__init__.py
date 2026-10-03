"""Agent registry: reference, baseline, sequential-policy and cheater agents.

Every agent runs through :class:`onc_agi.services.kit.PipelineAgent` or a
policy subclass, i.e. the same execution path as any external agent.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from onc_agi.adapters.agents import baselines, cheaters
from onc_agi.core.ports import WorldStore
from onc_agi.core.schema import Action, Assay, GroupLabel, Mode, Recruit, Submit, WorldCard
from onc_agi.services.engine import EpisodeView
from onc_agi.services.kit import Agent, Analyst, PipelineAgent, analysis_input

BASELINES: dict[str, Analyst] = {
    "univariate_bh": baselines.univariate_bh,
    "lasso": baselines.lasso,
    "elastic_net": baselines.elastic_net,
    "stability": baselines.stability_selection,
    "random_forest": baselines.random_forest,
    "knockoffs": baselines.knockoffs,
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


class GroupSequentialAgent(Agent):
    """Recruit in stages, assay everything on new patients, stop early when the analyst's
    selection is non-empty and unchanged between two stages (or at the last stage)."""

    def __init__(
        self, name: str, analyst: Analyst, stages: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    ) -> None:
        super().__init__()
        self.name = name
        self.analyst = analyst
        self.stages = stages
        self._stage = 0
        self._previous: tuple[str, ...] | None = None

    def choose_action(self, card: WorldCard, view: EpisodeView) -> Action:
        if card.mode is Mode.FULL_ACCESS:
            # nothing to acquire: analyse what is revealed, as PipelineAgent does
            return Submit(
                request_id=self.request_id(), ranking=tuple(self.analyst(analysis_input(card, view)))
            )
        if not view.rows:
            self._stage, self._previous = 0, None
        elif _has_unmeasured(view):
            return Assay(request_id=self.request_id(), feature_ids=card.feature_ids())
        else:
            current = tuple(self.analyst(analysis_input(card, view)))
            if self._stage >= len(self.stages) or (current and current == self._previous):
                return Submit(request_id=self.request_id(), ranking=current)
            self._previous = current
        target = round(self.stages[self._stage] * card.n_pool)
        self._stage += 1
        return Recruit(
            request_id=self.request_id(), count=max(1, target - len(view.rows)), stratum=card.strata[0]
        )


def _has_unmeasured(view: EpisodeView) -> bool:
    return bool(np.isnan(view.x).any())


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
    if name.startswith("seq_") and name[4:] in BASELINES:
        return GroupSequentialAgent(name, BASELINES[name[4:]])
    table: dict[str, Analyst] = BASELINES | CHEATERS
    if name not in table:
        raise KeyError(f"unknown agent {name!r}; known: {[*sorted(table), 'oracle', 'random']}")
    return PipelineAgent(name, table[name])


AGENT_NAMES: tuple[str, ...] = ("oracle", "random", *BASELINES, *CHEATERS)
AgentFactory = Callable[[], Agent]
