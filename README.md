# ONC-AGI

**ARC-style biomarker discovery worlds.** Each world is a cohort with an outcome and a hidden, planted mechanism (or none). An agent must find the drivers rather than their correlates, and must say "nothing" when nothing can be found. One deterministic score, the Discovery Score, ranks agents against an oracle, standard baselines and cheating strategies, in full-access and sequential-acquisition modes.

> **Status: release candidate `1.0.0-rc3`, first-pass results.** This release contains:
> - the interface specification, the engine, the scorer, the Agent kit, the HTTP server and the conformance suite;
> - **2,489 public-train benchmark worlds**, built from seven real cohort sources with oracle-certified answer keys, as release downloads (below);
> - first-pass scorecards for reference, baseline, gaming and LLM agents on those worlds, and the website.
>
> Public-train worlds publish their answer keys, as ARC's training tasks do: use them to develop agents and to run large offline evaluations. The hidden evaluation tiers are scored by a hosted server in a later release. First-pass results are public-train results and are labelled as such.
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

## World sets

Each set is one release download: a world store you can point any command at. Answer keys are included (public train).

| Download | Worlds | Contents |
|---|---|---|
| `onc-agi-public-train-full-access-1.0.0rc3.tar.gz` | 995 | Full-access worlds over every mechanism, 20% with no signal |
| `onc-agi-public-train-sequential-1.0.0rc3.tar.gz` | 996 | The same distribution in sequential-acquisition mode |
| `onc-agi-public-train-expressive-full-access-1.0.0rc3.tar.gz` | 248 | Harder worlds: survival outcomes, nonlinear drivers, missing data, up to 400 features |
| `onc-agi-public-train-expressive-sequential-1.0.0rc3.tar.gz` | 250 | The expressive set in sequential mode |

Sources are mixed across TCGA-BRCA, SCAN-B, METABRIC, TCGA pan-cancer, MSK-IMPACT, NHANES and pooled TCGA-BRCA with SCAN-B. Every world composes up to three mechanisms from [`docs/roles.md`](docs/roles.md). Difficulty tiers 0–2 are balanced within each set.

```bash
gh release download v1.0.0rc3 -R BradSegal/ONC-AGI -p 'onc-agi-public-train-full-access-*'
sha256sum -c --ignore-missing SHA256SUMS            # optional: the checksums are a release asset too
mkdir -p worlds/full && tar -xzf onc-agi-public-train-full-access-1.0.0rc3.tar.gz -C worlds/full
uv run onc-agi evaluate --agent univariate_bh --store worlds/full            # one baseline on every world
uv run onc-agi play --agent llm --profile <name> --store worlds/full --n 120 --workers 16  # any OpenAI-compatible model (docs/agents-kit.md)
```

Each archive holds `public_train/<world>/` bundles (`card.json`, `pool.parquet`, `queues.json`, `answer_key.json`) and `manifests/` (each world's mechanism, source, mode and difficulty tier).

## First-pass results

Discovery Score on the first-pass set: the first 120 worlds of each core public-train download (240 worlds, 20% with no signal). Each row is one scorecard per mode. GPT 6 Luna ran on the standard Inspect track; its two scorecards per mode agree within 0.04. All ten gaming strategies score 0.00 in both modes (the largest is 0.0004).

| Agent | Full access | Sequential | Find | Restraint |
|---|---|---|---|---|
| Oracle | 1.00 | 1.00 | 1.00 | 1.00 |
| Random list | 0.00 | 0.00 | 0.01 | 0.00 |
| GPT 6 Luna | 0.29 | 0.22 | 0.50 | 0.50 |
| Forward score selection | 0.37 | 0.32 | 0.47 | 0.74 |
| Univariate + BH | 0.28 | 0.23 | 0.35 | 0.73 |
| Lasso | 0.22 | 0.29 | 0.46 | 0.55 |
| Elastic net | 0.17 | 0.27 | 0.44 | 0.49 |
| Stability selection | 0.22 | 0.20 | 0.33 | 0.63 |
| Random forest | 0.12 | 0.11 | 0.33 | 0.35 |

These are public-train results. Rows whose 95% intervals overlap are not separated; the intervals and every scorecard are on the website and in `site/results/` and `site/src/data/results.json`.

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
