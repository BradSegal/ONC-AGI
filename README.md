# ONC-AGI

**ARC-style biomarker discovery worlds.** Each world is a cohort with an outcome and a hidden, planted mechanism (or none). An agent must find the drivers rather than their correlates, and must say "nothing" when nothing can be found. One deterministic score, the Discovery Score, ranks agents against an oracle, standard baselines and cheating strategies, in full-access and sequential-acquisition modes.

> **Status: release candidate `1.0.0-rc3`, superseded.** The interface specification, the engine, the scorer, the Agent kit, the HTTP server and the conformance suite are current. The rc3 world sets remain downloadable for reference, but two defects were found after release:
> - 8 of the 2,489 public-train worlds have answer keys that no solver can reach, because of a world-construction defect;
> - the first-pass results were scored on the first 120 worlds of each core download, which all came from one cohort source (METABRIC). Those results are withdrawn.
>
> Public-train worlds publish their answer keys, as ARC's training tasks do: use them to develop agents and to run offline evaluations.
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
- **Data:** the cohorts behind the benchmark worlds are credited under [Data sources and credits](#data-sources-and-credits).

## Data sources and credits


ONC-AGI worlds are derived from real, de-identified cohort data. Each world resamples real
patients' measurements, adds small observation noise, replaces feature names with invented
symbols and patient identifiers with new ones, and generates the outcome from a planted model.
Values remain close to the source records. The source of every world is recorded in the
archive manifest (`manifests/*.json`, field `source`). Code is BSD-3-Clause; world data are
licensed under the ODC Open Database License (ODbL) 1.0.

**The Cancer Genome Atlas (TCGA-BRCA and TCGA PanCanAtlas), via UCSC Xena.** The results
shown here are in whole or part based upon data generated by the TCGA Research Network:
https://www.cancer.gov/tcga.
- Goldman MJ, Craft B, Hastie M, et al. Visualizing and interpreting cancer genomics data via the Xena platform. *Nat Biotechnol* 2020;38:675–678. doi:10.1038/s41587-020-0546-8
- Cancer Genome Atlas Network. Comprehensive molecular portraits of human breast tumours. *Nature* 2012;490:61–70. doi:10.1038/nature11412
- Hoadley KA, Yau C, Hinoue T, et al. Cell-of-origin patterns dominate the molecular classification of 10,000 tumors from 33 types of cancer. *Cell* 2018;173:291–304.e6. doi:10.1016/j.cell.2018.03.022
- Cancer Genome Atlas Research Network, Weinstein JN, et al. The Cancer Genome Atlas Pan-Cancer analysis project. *Nat Genet* 2013;45:1113–1120. doi:10.1038/ng.2764
- Copy number: Broad Institute TCGA Genome Data Analysis Center, Firehose analyses run 2016_01_28 (GISTIC2; Mermel CH et al. *Genome Biol* 2011;12:R41. doi:10.1186/gb-2011-12-4-r41). Protein: TCGA RPPA, MD Anderson (Li J et al. TCPA: a resource for cancer functional proteomics data. *Nat Methods* 2013;10:1046–1047. doi:10.1038/nmeth.2650).

**METABRIC and MSK-IMPACT 2017, via the cBioPortal for Cancer Genomics** (studies
`brca_metabric` and `msk_impact_2017`). This data is made available under the ODC Open
Database License (ODbL) 1.0, https://opendatacommons.org/licenses/odbl/1-0/.
- Curtis C, Shah SP, Chin SF, et al. The genomic and transcriptomic architecture of 2,000 breast tumours reveals novel subgroups. *Nature* 2012;486:346–352. doi:10.1038/nature10983
- Pereira B, Chin SF, Rueda OM, et al. The somatic mutation profiles of 2,433 breast cancers refine their genomic and transcriptomic landscapes. *Nat Commun* 2016;7:11479. doi:10.1038/ncomms11479
- Rueda OM, Sammut SJ, Seoane JA, et al. Dynamics of breast-cancer relapse reveal late-recurring ER-positive genomic subgroups. *Nature* 2019;567:399–404. doi:10.1038/s41586-019-1007-8
- Zehir A, Benayed R, Shah RH, et al. Mutational landscape of metastatic cancer revealed from prospective clinical sequencing of 10,000 patients. *Nat Med* 2017;23:703–713. doi:10.1038/nm.4333
- Cerami E, Gao J, Dogrusoz U, et al. The cBio Cancer Genomics Portal: an open platform for exploring multidimensional cancer genomics data. *Cancer Discov* 2012;2:401–404. doi:10.1158/2159-8290.CD-12-0095
- Gao J, Aksoy BA, Dogrusoz U, et al. Integrative analysis of complex cancer genomics and clinical profiles using the cBioPortal. *Sci Signal* 2013;6:pl1. doi:10.1126/scisignal.2004088
- de Bruijn I, Kundra R, Mastrogiacomo B, et al. Analysis and visualization of longitudinal genomic and clinical data from the AACR Project GENIE Biopharma Collaborative in cBioPortal. *Cancer Res* 2023;83:3861–3867. doi:10.1158/0008-5472.CAN-23-0816

**SCAN-B (Sweden Cancerome Analysis Network – Breast), NCBI GEO accession GSE96058.**
- Brueffer C, Vallon-Christersson J, Grabau D, et al. Clinical value of RNA sequencing-based classifiers for prediction of the five conventional breast cancer biomarkers: a report from the population-based multicenter Sweden Cancerome Analysis Network–Breast Initiative. *JCO Precis Oncol* 2018;2:1–18. doi:10.1200/PO.17.00135
- Saal LH, Vallon-Christersson J, Häkkinen J, et al. The Sweden Cancerome Analysis Network – Breast (SCAN-B) Initiative. *Genome Med* 2015;7:20. doi:10.1186/s13073-015-0131-9

**NHANES 2017–2018** (laboratory and demographic public-use files).
- Centers for Disease Control and Prevention (CDC), National Center for Health Statistics (NCHS). National Health and Nutrition Examination Survey Data. Hyattsville, MD: U.S. Department of Health and Human Services, Centers for Disease Control and Prevention, 2017–2018. https://wwwn.cdc.gov/nchs/nhanes/

**HGNC** (used only so invented feature names avoid real gene symbols; not redistributed).
- Seal RL, Braschi B, Gray K, et al. Genenames.org: the HGNC resources in 2023. *Nucleic Acids Res* 2023;51:D1003–D1009. doi:10.1093/nar/gkac888

We thank the patients and participants who contributed to these studies, and the TCGA
Research Network, the METABRIC, MSK-IMPACT and SCAN-B investigators, the cBioPortal and
UCSC Xena teams, and CDC/NCHS for making the data openly available.

## Licence

Code: BSD-3-Clause, see [LICENSE](LICENSE). World data: [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/); see [Data sources and credits](#data-sources-and-credits).
