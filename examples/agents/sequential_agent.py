"""A sequential-acquisition policy: buy data in stages and stop when the answer settles.

Each turn the agent returns one action. It recruits a batch, assays every feature
on the new patients, re-analyses, and submits once the same non-empty list appears
twice in a row (or the pool is exhausted). Spending less than the oracle's
reference cost keeps efficiency at 1; spending more scales the world's credit down.

    uv run python examples/agents/sequential_agent.py
"""

from __future__ import annotations

import numpy as np
from onc_agi.adapters.cli import fixture_store
from onc_agi.core.schema import Action, Assay, Mode, Recruit, Scorecard, Submit, Tier, WorldCard
from onc_agi.infra.bundles import FileWorldStore
from onc_agi.services.engine import EpisodeView
from onc_agi.services.kit import Agent, analysis_input, evaluate
from pipeline_agent import bonferroni_t  # type: ignore[import-not-found]

BATCH = 60


class StagedAgent(Agent):
    name = "staged-bonferroni"

    def __init__(self) -> None:
        super().__init__()
        self.previous: tuple[str, ...] | None = None

    def choose_action(self, card: WorldCard, view: EpisodeView) -> Action:
        if view.mode is Mode.FULL_ACCESS:
            return Submit(
                request_id=self.request_id(), ranking=tuple(bonferroni_t(analysis_input(card, view)))
            )
        if not view.rows:
            self.previous = None  # a new episode: the same instance plays many worlds
        elif not all(view.measured) or self._unmeasured_rows(view):
            return Assay(request_id=self.request_id(), feature_ids=card.feature_ids())
        else:
            current = tuple(bonferroni_t(analysis_input(card, view)))
            if len(view.rows) >= card.n_pool or (current and current == self.previous):
                return Submit(request_id=self.request_id(), ranking=current)
            self.previous = current
        # Recruit evenly from strata that still have patients (sizes are published on the card).
        sizes = card.stratum_sizes or {s: card.n_pool for s in card.strata}
        left = {s: sizes[s] - view.stratum.count(s) for s in card.strata}
        stratum = max(left, key=lambda s: left[s])
        count = min(BATCH // len(card.strata), left[stratum])
        return Recruit(request_id=self.request_id(), count=count, stratum=stratum)

    @staticmethod
    def _unmeasured_rows(view: EpisodeView) -> bool:
        """Newly recruited patients have no measurements yet (every value is missing)."""
        return bool(view.x.shape[0]) and bool(np.isnan(view.x).all(axis=1).any())


def main() -> Scorecard:
    store = FileWorldStore(fixture_store())
    worlds = [w for w in store.world_ids(Tier.PUBLIC_TRAIN) if w.endswith("-seq")]
    card, results = evaluate(StagedAgent(), store, Tier.PUBLIC_TRAIN, world_ids=worlds)
    for r in results:
        assert r.score is not None
        print(
            f"{r.world_id:22s} spent={r.spent:8.0f} efficiency={r.score.efficiency:.2f} find={r.score.find:.2f}"
        )
    print(f"Discovery Score {card.discovery_score}  mean data cost {card.mean_data_cost:.0f} USD")
    return card


if __name__ == "__main__":
    main()
