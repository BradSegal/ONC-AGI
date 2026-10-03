# Evaluation protocol

This page fixes how a result becomes a leaderboard entry, so that two numbers on the board mean the same thing.

## Tracks

| Track | Harness | What is fixed | Use it for |
|---|---|---|---|
| **standard** | The Inspect task (`arena_full_access`, `arena_sequential`) | The prompt and task card, the tools (`python`, `bash`, `recruit`, `assay`, `submit`), the no-network analysis sandbox, the limits below | Comparing models with one another |
| **open** | Anything that speaks the interface: the agents kit (`onc-agi play`), your own HTTP client | Only the interface, the scorer and the exposure rules | Comparing agents, scaffolds and methods |
| **reference** | Built-in oracle, random and cheater agents | Everything | Anchoring the scale: the oracle defines 1, cheaters and random sit at 0 |

## Standard-track settings

| Setting | Value |
|---|---|
| Messages per world | 60 (full access), 80 (sequential) |
| Safety timeout per world | 1,800 s |
| Tool call timeout | 180 s |
| Sandbox | No network, 2 CPUs, 4 GB memory |
| Episodes per world per scorecard | 1 (`epochs = 1`); repeat by opening another scorecard |
| Model parameters | The provider's defaults, unless the entry states otherwise. Reasoning effort, temperature and token limits are recorded with the entry |

Every scorecard records its harness label (`inspect-standard-1.1+<digest>`), its scorer and engine labels, and the oracle version. The digests change whenever the code that determines what a model sees, or how it is scored, changes. Entries with different labels are never ranked together.

## How many worlds, how many runs

A Discovery Score's 95% interval is a bootstrap over the worlds of one scorecard. It does not cover a model's own randomness. Development runs give the scale of both:

- **Worlds.** On 120 development worlds, the interval half-width was about 0.10. On 24 worlds it was about 0.30, too wide to separate most agents.
- **Runs.** The same model on the same 24 worlds gave identical Find but different Restraint between two runs: two null worlds flipped between abstaining and claiming. With about six null worlds per 24, one flip moves Restraint by about 0.17.

A leaderboard entry therefore needs:

1. At least **120 worlds per scorecard** (about 24 of them null, the fixed 20% share).
2. At least **three scorecards per agent**. The entry reports their mean and the spread between them alongside each scorecard's interval.
3. Agents are ranked apart only when one interval lies wholly above the other. The leaderboard shows rows that cannot be told apart at the same rank, marked `=`.

## Costs

Every standard and open entry reports tokens and cost. Cost comes from the provider's own accounting when it is available, otherwise from the per-token prices stated with the entry (input, output, cache reads and cache writes priced separately). An unknown cost is shown as unknown, never estimated silently. Runs can stop on a spend cap (`--budget-usd`): unplayed worlds score as empty submissions, so a stopped run is complete but penalised, never quietly shortened.

## Exposure

Public-train results are fully visible: per-world scores, recordings and explanations. Public-eval and private scorecards report aggregates only, draw worlds that no earlier scorecard used, and are capped per key. Serving them requires issued API keys.
