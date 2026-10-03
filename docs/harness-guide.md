# Building an agent or harness

There are four ways to play, and all of them use the same engine and scorer. A result therefore means the same thing whichever path produced it.

| Path | Use it for | Start from |
|---|---|---|
| In-process `Agent` | Fast offline development and autoresearch loops | [`examples/agents/pipeline_agent.py`](../examples/agents/pipeline_agent.py), [`sequential_agent.py`](../examples/agents/sequential_agent.py) |
| HTTP custom harness | Any language or framework (open track) | [`examples/agents/http_agent.py`](../examples/agents/http_agent.py), [`schemas/openapi.json`](../schemas/openapi.json) |
| LLM tool-calling loop | Prototyping a model-driven agent (open track) | [`examples/agents/llm_agent.py`](../examples/agents/llm_agent.py) |
| Inspect standard track | Comparable model evaluations: fixed prompt, fixed tools, no-network sandbox | `onc_agi.adapters.inspect_task` (preview) |

## 1. In-process agents

An analysis function is enough for full-access worlds:

```python
from onc_agi.adapters.cli import fixture_store
from onc_agi.core.schema import Tier, Timing
from onc_agi.infra.bundles import FileWorldStore
from onc_agi.services.kit import AnalysisInput, PipelineAgent, evaluate
import numpy as np

def top_correlate(data: AnalysisInput) -> list[str]:
    """List the single baseline feature most correlated with the outcome, or nothing if it is weak."""
    timing = {f.feature_id: f.timing for f in data.card.features}
    best, best_r = None, 0.0
    for j, fid in enumerate(data.feature_ids):
        if timing[fid] is Timing.BASELINE:
            r = abs(np.corrcoef(np.nan_to_num(data.x[:, j]), data.y)[0, 1])
            if r > best_r:
                best, best_r = fid, r
    return [best] if best is not None and best_r > 0.3 else []

store = FileWorldStore(fixture_store())
worlds = [w for w in store.world_ids(Tier.PUBLIC_TRAIN) if w.endswith("-full")]
card, results = evaluate(PipelineAgent("top-correlate", top_correlate), store, Tier.PUBLIC_TRAIN, world_ids=worlds)
print(card.discovery_score, card.find, card.restraint)
```

For sequential worlds, subclass `Agent` and return one action per call to `choose_action(card, view)`. The same instance plays many worlds, so reset any per-episode state when `view.rows` is empty. [`sequential_agent.py`](../examples/agents/sequential_agent.py) recruits in stages and stops when its answer settles.

Registered reference, baseline and cheater agents are available by name: `onc_agi.adapters.agents.make_agent(name, store)`. Run `uv run onc-agi smoke` to see them all side by side. They are:
- the oracle and random references;
- the baselines `univariate_bh`, `lasso`, `elastic_net`, `stability`, `random_forest` and `knockoffs`;
- the cheaters `giant_list`, `always_empty`, `random_abstain`, `leak_exploiter`, `auc_maximiser`, `metadata_only` and the outcome-blind rankers.

## 2. HTTP custom harnesses

Open a scorecard, play each world by posting actions, then close the scorecard. `ArenaClient.play(agent, scorecard_id, card)` runs any `Agent` remotely. Every request carries a `request_id`, so retries are safe. After a disconnect, `GET /v1/scorecards/{sid}/worlds/{wid}` returns the current state. Check your harness with `uv run onc-agi conformance --url ...`.

## 3. LLM agents

[`llm_agent.py`](../examples/agents/llm_agent.py) exposes the arena actions as four tools: `recruit`, `assay`, `analyse` and `submit`. Each tool has a JSON input schema in the common `name` / `description` / `input_schema` shape. The model makes one tool call per turn. Typed errors go back to the model as tool results, so it can recover.

The file runs offline with a scripted stand-in for the model. To use a real model, implement `ToolPolicy.next_call` with your provider's tool-calling API. Things that matter in practice:
- Show the model the world card, including timing: listing a `post_outcome` feature zeroes the world.
- Let the model analyse data with code rather than reading raw numbers.
- Record `tokens` and `cost_usd` per world. Scorecards report cost beside the score, with no cap.

## 4. The standard track (Inspect, preview)

`onc_agi.adapters.inspect_task` defines `arena_full_access` and `arena_sequential` as [Inspect AI](https://inspect.aisi.org.uk/) tasks (install with `pip install "onc-agi[inspect]"`). Every model gets:
- the same premise and task card;
- `python` and `bash` tools inside a Docker sandbox with **no network**, where the revealed data are written to `/data/revealed.csv`;
- `recruit` and `assay` (sequential mode), which run on the host, where the episode lives;
- an answer submitted as a comma-separated ranking;
- a safety timeout per world. Tokens and cost are reported, not capped.

`log_to_scorecard` converts an Inspect log into a standard-track `Scorecard`. Integrity claims for sequential mode and hidden-variable mechanics hold only on this no-network track; open-track results there are labelled unverified. This harness is a **preview** in 1.0.0-rc1.

## The in-process kit in detail

Agents subclass `onc_agi.services.kit.Agent` and implement `choose_action(card, view)`. `view` is an `EpisodeView`: the same revealed state as an `Observation`, as NumPy arrays. The same agent runs unchanged in process, through `ArenaClient.play` over HTTP, and in the scorers. ARC-AGI-3's agents kit follows the same pattern.

```python
from onc_agi.adapters.cli import fixture_store
from onc_agi.core.schema import Assay, Recruit, Reset, Submit
from onc_agi.infra.bundles import FileWorldStore
from onc_agi.services.engine import Episode

store = FileWorldStore(fixture_store())
episode = Episode(store.world("toy-driver-seq"))
card = episode.world.card
view = episode.apply(Reset(request_id="r0", world_id=card.world_id))
assert view.rows == () and view.available == ("recruit", "assay", "submit")

recruited = episode.apply(Recruit(request_id="r1", count=40, stratum="all"))
view = episode.apply(Assay(request_id="r2", feature_ids=card.feature_ids()[:5]))
assert len(view.rows) == 40 and sum(view.measured) == 5

# Retrying a request returns its original response and charges nothing again.
retry = episode.apply(Recruit(request_id="r1", count=40, stratum="all"))
assert retry == recruited and episode.spent == view.spent

view = episode.apply(Submit(request_id="r3", ranking=()))
print(view.status.value, f"spent {view.spent:.0f} of {view.budget:.0f} USD")
```

`evaluate(agent, store, tier)` runs an agent over a tier's worlds and returns its `Scorecard` and per-episode results. `PipelineAgent(name, analyst)` turns an analysis function `analyst(AnalysisInput) -> ranking` into an agent: in full access it simply submits; in sequential mode it recruits the whole pool, assays everything, then submits. See [`examples/agents/`](../examples/agents).

## Tiers and what they expose

A **scorecard** groups episodes for one agent on one tier. A session is identified by `(scorecard_id, world_id)`. Disconnecting does not lose state: `GET` the world's state and continue.

| Tier | Answer keys | Per-world results | Caps |
|---|---|---|---|
| `public_train` | Shipped with each world | Yes | None |
| `public_eval` | Held by the server | No (aggregates with intervals only) | 5 scorecards per day per key |
| `private` | Held by the server | No | 3 official runs |

Every public-eval and private scorecard scores a fresh, never-reused draw of worlds from a pool whose hash is committed before any result. About 20% of the draw has no signal, and the rest is spread across difficulty tiers. This release ships only public-train toy worlds; evaluation tiers arrive with the benchmark release.

## World bundles

A world on disk is `<store>/<tier>/<world_id>/` containing:
- `card.json`;
- `pool.parquet`, with columns `patient_id`, `stratum`, `outcome`, then one column per feature in card order;
- `queues.json`, the recruitment order per stratum;
- `answer_key.json`, for public train only.

`FileWorldStore(root)` reads a store. Bundles are written once and never overwritten.

## Rules for collaborators

- **Develop on public-train worlds only.** Tuning against evaluation tiers is not allowed, and no evaluation tier ships in this release.
- **Toy fixtures are not benchmark tasks.** They are easier and simpler than benchmark worlds and exist to exercise the interface. Do not report toy-world scores as results.
- **Report interface problems as issues.** Use the *Interface feedback* template.
