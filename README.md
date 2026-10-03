# ONC-AGI

**ARC-style biomarker discovery worlds.** Each world is a cohort with an outcome and a hidden, planted mechanism (or none). An agent must find the drivers rather than their correlates, and must say "nothing" when nothing can be found. One deterministic score, the Discovery Score, ranks agents against an oracle, standard baselines and cheating strategies, in full-access and sequential-acquisition modes.

> **Status: contract preview `1.0.0-rc2`.** This release contains:
> - the interface specification;
> - the real engine, scorer, Agent kit, HTTP server and conformance suite;
> - toy fixture worlds and template agents.
>
> It is for collaborators to design and test agents offline while the benchmark is being built. Benchmark worlds, built from real cancer cohorts with oracle-certified answer keys, and the evaluation server arrive in a later release. Toy-world scores are not results.
>
> ONC-AGI is inspired by [ARC-AGI](https://arcprize.org/) and follows its conventions. It is not affiliated with the ARC Prize Foundation. It measures discovery competence on planted mechanisms in real correlation structure. It does not validate real biomarkers.

## The premise

> *Here is a cohort with an outcome. Which measurements drive it? Return an ordered list, most likely first, or an empty list if nothing can be found.*

## Quickstart

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/BradSegal/ONC-AGI && cd ONC-AGI
uv sync
uv run onc-agi smoke                            # reference, baseline and cheater agents on the toy worlds, both modes
uv run python examples/agents/pipeline_agent.py # a full-access agent in a dozen lines
```

Without uv: `pip install .` then `onc-agi smoke`.

The smoke command prints one scorecard per agent: the oracle at 1.00, the standard baselines in between, and every cheater at 0.

## What's here

| Path | Contents |
|---|---|
| [`docs/premise.md`](docs/premise.md) | The task card and the rules every agent should know |
| [`docs/interface.md`](docs/interface.md) | Interface specification v1.0: objects, actions, episode semantics, errors, sessions, HTTP |
| [`docs/scoring.md`](docs/scoring.md) | Scoring specification, with executable worked examples |
| [`docs/roles.md`](docs/roles.md) | The catalogue of causal roles and mechanics, and their answer-key rules |
| [`docs/harness-guide.md`](docs/harness-guide.md) | In-process agents, HTTP harnesses, LLM agents, the Inspect standard track |
| [`docs/fixtures.md`](docs/fixtures.md) | The toy fixture worlds |
| [`schemas/`](schemas) | JSON Schemas for every payload, and `openapi.json` |
| [`examples/payloads/`](examples/payloads) | Real captured requests and responses, validated against the schemas |
| [`examples/agents/`](examples/agents) | Template agents: pipeline, sequential policy, HTTP harness, LLM tool loop |
| `src/onc_agi/` | The runtime: contract, engine, scorer, Agent kit, baselines, cheaters, HTTP server and client, conformance, Inspect task |
| [`STABILITY.md`](STABILITY.md) | What is frozen, what is a preview, the change policy and known limitations |

## Two modes

| | Full access (like ARC-AGI-1/2) | Sequential acquisition (like ARC-AGI-3) |
|---|---|---|
| The agent receives | The whole dataset | A budget, a price list and the actions `recruit`, `assay` and `submit` |
| It is tested on | Inference under correlation, confounding and leaks | That, plus what to measure and when to stop |
| Data cost | Fixed | Credit is scaled by efficiency against a reference cost |

## The score

```
Discovery Score = Find × Restraint
Find      = chance-normalised recovery of the planted drivers (top R of the list, correlated substitutes counted once)
Restraint = P(empty list | no signal) − P(empty list | signal)
```

Listing a post-outcome feature anywhere zeroes the world. The oracle scores 1. Always answering, never answering, random lists and random abstention all score 0. See [`docs/scoring.md`](docs/scoring.md).

## Feedback

Interface feedback is open until the `1.0.0` freeze. Open an issue with the *Interface feedback* template. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Credits

- The **harness pattern** (an `Agent` base class with `choose_action`, scorecards, recordings and replay) follows [ARC-AGI-3-Agents](https://github.com/arcprize/ARC-AGI-3-Agents) (MIT). It is reimplemented here; no code is copied.
- **Benchmark worlds** are constructed with [PyPlasmode](https://pypi.org/project/pyplasmode/) (BSD-3-Clause). This package does not depend on it.
- **Libraries:**
  - [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [pandas](https://pandas.pydata.org/) and [PyArrow](https://arrow.apache.org/docs/python/);
  - [pydantic](https://docs.pydantic.dev/);
  - [scikit-learn](https://scikit-learn.org/) and [statsmodels](https://www.statsmodels.org/);
  - [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/) and [HTTPX](https://www.python-httpx.org/);
  - [Hatchling](https://hatch.pypa.io/);
  - optional: [Inspect AI](https://inspect.aisi.org.uk/) for the standard track and [knockpy](https://github.com/amspector100/knockpy) for the knockoffs baseline;
  - development: pytest, Hypothesis, black, Ruff, mypy and jsonschema.
- **Data:** this release contains synthetic toy data only. Real cohorts used by benchmark worlds are credited and cited in the release that ships them.

## Licence

BSD-3-Clause. See [LICENSE](LICENSE).
