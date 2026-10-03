# Agents kit

One command plays any agent against the arena, in-process or against a server, with worlds played in parallel, a recording of everything the agent did, and an explanation of every world's outcome. It follows the pattern of the ARC-AGI-3 agents kit: an `Agent` base class, a swarm runner, recordings, scorecards and an LLM template.

## Two tracks

- **Standard track.** This is how models are compared. It is the Inspect AI task in `onc_agi.adapters.inspect_task`, where every model sees the same prompt, the same tools and the same limits.
- **Open track.** This is the kit. You may bring any agent, prompt, tools or pipeline. Its scorecards are labelled with the agent, model and harness that produced them, so results from the two tracks are never mixed.

The kit's LLM template starts from the standard harness: it uses the same task card, the same `/data/revealed.csv`, the same tools and the same sandbox image. You can then change whatever you want to explore.

## Install and configure

```bash
pip install onc-agi                 # built-in agents, your own agents, recordings
pip install "onc-agi[inspect]"      # also the LLM template (it shows the standard task card)
```

Keys belong only in the environment, or in a git-ignored `.env` in the working directory. Variables that are already set always win over `.env`.

```bash
# .env
OPENROUTER_API_KEY=sk-or-...
ARENA_URL=https://arena.example.org        # only for playing a remote server
ARENA_KEY=your-issued-key
```

## Look at the worlds

```bash
onc-agi worlds                               # public-train ids, modes and sizes (the bundled worlds)
onc-agi worlds --url "$ARENA_URL"            # a server's public-train worlds
```

## Play

```bash
onc-agi play --agent random --n 3                       # three bundled worlds, in-process
onc-agi play --agent seq_univariate_bh --mode sequential --workers 8
onc-agi play --agent univariate_bh --worlds fixture-full-1-08d0kt7j,fixture-seq-2-0pyy8phw --record runs/bh
```

`--agent` takes four kinds of value:

- a built-in name: `random`, the baselines (`univariate_bh`, `lasso` and others), their sequential policies (`seq_<baseline>`), the cheaters, and `oracle`. The oracle reads answer keys, so it plays in-process only.
- `module:Class`;
- `path.py:Class`;
- `llm`.

A class must subclass `Agent` and take no constructor arguments. The kit builds a fresh instance for each world and calls its `close()` afterwards.

Each run opens one scorecard and prints its summary row. `--json OUT` writes the full scorecard. A world in which the agent raises an exception is recorded with its error, and the other worlds carry on; the command then exits with status 1. A world that is never submitted scores as an empty submission.

## An LLM agent

```bash
onc-agi play --agent llm --profile openrouter-luna --n 4 --record runs/luna
onc-agi play --agent llm --profile openrouter-luna --set temperature=0.2 --set extra_body.reasoning='{effort="high"}'
onc-agi play --agent llm --profile local --harness-config harness.toml
```

Profiles are read from `--profiles PATH`, else from `./profiles.toml`, else from the packaged example (`adapters/agents/profiles.example.toml`). The example holds an OpenRouter profile and a template for a local vLLM or Ollama server. Copy it to `./profiles.toml` to edit it. A profile names:

- its endpoint;
- the environment variable that holds its key;
- the model;
- its chat-completions parameters (`[generate]`);
- provider-specific fields (`[extra_body]`);
- optionally `prices_per_mtok`.

If the key is missing, the command fails before any world opens and names the variable it needs.

The agent works like this:

- It runs a tool-calling conversation for each world: `python` and `bash` in a no-network Docker sandbox, plus `recruit`, `assay` and `submit`.
- A refused action, such as an unknown feature or an exhausted budget, returns to the model as a tool error. The model can then try again.
- If it runs out of `max_turns`, the world ends unsubmitted.
- Cost is the provider's own figure when it reports one (OpenRouter's `usage.cost`), otherwise it is computed from the profile's prices.

A `[harness]` table changes the agent's behaviour:

```toml
[harness]
max_turns = 40
tool_timeout = 180
output_limit = 12000
tools = ["python", "bash"]
system_prompt = "..."
continue_prompt = "..."
```

The scorecard records the model and a harness label, `openai-tools-1.0+<hash>`. The hash changes whenever the config, the template or the standard harness changes.

## Write your own agent

Subclass `Agent` and return a `Recruit`, `Assay` or `Submit` from `choose_action`. The view holds what has been revealed so far. The example below runs as written, and the test suite executes it.

```python run
import numpy as np

from onc_agi.adapters.cli import fixture_store
from onc_agi.adapters.swarm import LocalArena, run_swarm
from onc_agi.core.schema import Assay, Mode, Recruit, Submit, Tier, Timing
from onc_agi.infra.bundles import FileWorldStore
from onc_agi.services.kit import Agent, analysis_input


class HalfCohort(Agent):
    """Recruit half of every stratum, assay its baseline features, list strong marginals."""

    name = "half-cohort"

    def choose_action(self, card, view):
        baseline = [f.feature_id for f in card.features if f.timing is Timing.BASELINE]
        for stratum in card.strata:
            if view.mode is Mode.SEQUENTIAL and stratum not in view.stratum:
                half = card.stratum_sizes.get(stratum, card.n_pool) // 2
                return Recruit(request_id=self.request_id(), count=half, stratum=stratum)
        if not any(view.measured):
            return Assay(request_id=self.request_id(), feature_ids=tuple(baseline))
        data = analysis_input(card, view)
        z = {}
        for j, feature in enumerate(data.feature_ids):
            column = np.nan_to_num(data.x[:, j], nan=float(np.nanmean(data.x[:, j])))
            if feature in baseline and column.std() > 0:
                z[feature] = abs(np.corrcoef(column, data.y)[0, 1]) * np.sqrt(len(data.y))
        ranking = tuple(f for f, v in sorted(z.items(), key=lambda kv: -kv[1]) if v > 3.3)
        return Submit(request_id=self.request_id(), ranking=ranking)


arena = LocalArena.over_store(FileWorldStore(fixture_store()))
result = run_swarm(arena, HalfCohort, agent_name="half-cohort", tier=Tier.PUBLIC_TRAIN, n_worlds=3, mode=Mode.SEQUENTIAL)
print(result.scorecard.find, [run.ranking for run in result.runs])
assert len(result.runs) == 3 and not result.errors and all(run.submitted for run in result.runs)
```

From the command line, the same agent runs with `onc-agi play --agent half_cohort.py:HalfCohort`.

Four optional hooks are available:

- `self.record(world_id, kind, content)` writes to the recording.
- `on_refused(action, error)` is called when the arena refuses an action. By default it re-raises, which ends the world. Override it to recover.
- `usage()` reports tokens and cost.
- `close()` releases per-world resources.

Raising `EndEpisode` from `choose_action` stops a world without submitting.

## Recordings and explanations

`--record DIR` writes:

- `recording.jsonl`: one scorecard header, then the agent's events, then one run record per world;
- `scorecard.json`;
- for public-train runs, `explanations.md` and `explanations.json`.

Each event has the fields `{scorecard_id, world_id, turn, kind, content, tool, error, ts}`. Its kind is `assistant`, `reasoning`, `tool_call`, `tool_result`, `action` or `note`. Each run record has the fields `{world_id, ranking, submitted, spent, tokens, cost_usd, model, error}`. A recording is never appended to, so each run needs a new directory.

```bash
onc-agi explain --record runs/luna                         # a kit recording
onc-agi explain --inspect-log logs/arena.eval --store S    # a standard-track Inspect log
```

An explanation recomputes the scorer's own credit, so it never disagrees with the scorecard. It classifies each world as one of:

- `found`;
- `partial`;
- `too_deep`: the right features were listed, but after the point where the scorer stops reading;
- `dropped`: the agent's own analysis surfaced a credited feature, but the list left it out;
- `never_surfaced`;
- `leak_listed`;
- `abstained_on_signal`;
- `correct_abstain`;
- `false_claim`;
- `no_submission`.

A printout of every column does not count as surfacing a feature; heading a ranked table does. Explanations read answer keys, so they cover public-train worlds only. `--operator` with `--keys` covers other tiers, and marks the report operator-only.

## Parallel worlds and budget

`--workers W` plays up to W worlds at once. Each world gets its own agent instance and its own sandbox.

`--budget-usd X` stops new worlds from starting once the agents' reported cost passes X. Worlds already running finish. Worlds that never started stay unsubmitted, score as empty, and the summary says how many there were. Agents that report no cost are not limited by `--budget-usd`.

## Playing a server

```bash
onc-agi play --agent my_agent.py:MyAgent --url "$ARENA_URL" --key "$ARENA_KEY" --n 10
onc-agi play --agent llm --profile openrouter-luna --tier public_eval --n 20 --record runs/eval-1
```

Over HTTP, the agent sees exactly the observations it sees in-process. In both cases they come from the same scorecard service, through the same wire contract (`docs/interface.md`). Eval tiers draw fresh worlds and enforce the caps of `docs/integrity.md`. They expose aggregates only, so no explanations are written for them.

The scorecard the server keeps carries no client claims. The model, harness, tokens and cost that `play` prints annotate only the scorecard returned to you.
