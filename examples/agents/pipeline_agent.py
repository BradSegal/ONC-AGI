"""A full-access agent in a dozen lines: an analysis function wrapped by PipelineAgent.

The analyst sees revealed rows and measured columns and returns an ordered list of
feature ids (most likely driver first), or an empty list to say "nothing here".

    uv run python examples/agents/pipeline_agent.py
"""

from __future__ import annotations

import numpy as np
from onc_agi.adapters.cli import fixture_store
from onc_agi.core.schema import Scorecard, Tier, Timing
from onc_agi.infra.bundles import FileWorldStore
from onc_agi.services.kit import AnalysisInput, PipelineAgent, evaluate
from scipy import stats

ALPHA = 0.05


def bonferroni_t(data: AnalysisInput) -> list[str]:
    """Welch t-test per baseline feature; list those passing Bonferroni, strongest first."""
    timing = {f.feature_id: f.timing for f in data.card.features}
    keep = [j for j, f in enumerate(data.feature_ids) if timing[f] is Timing.BASELINE]  # never list a leak
    y = data.y.astype(bool)
    if len(keep) == 0 or y.all() or not y.any():
        return []
    p = np.array([stats.ttest_ind(data.x[y, j], data.x[~y, j], equal_var=False).pvalue for j in keep])
    order = np.argsort(p)
    return [data.feature_ids[keep[i]] for i in order if p[i] < ALPHA / len(keep)]


def main() -> Scorecard:
    store = FileWorldStore(fixture_store())
    worlds = [w for w in store.world_ids(Tier.PUBLIC_TRAIN) if w.endswith("-full")]
    card, results = evaluate(
        PipelineAgent("bonferroni-t", bonferroni_t), store, Tier.PUBLIC_TRAIN, world_ids=worlds
    )
    for r in results:
        assert r.score is not None
        print(f"{r.world_id:22s} find={r.score.find:.2f} listed={list(r.ranking)[:3]}")
    print(f"Discovery Score {card.discovery_score}  Find {card.find}  Restraint {card.restraint}")
    return card


if __name__ == "__main__":
    main()
