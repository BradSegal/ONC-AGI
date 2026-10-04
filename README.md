# ONC-AGI

ONC-AGI tests whether an agent can recover a planted mechanism from cohort data—and abstain when the data support no discovery. Worlds retain the correlation structure of real measurements but replace the outcome with one generated from a known mechanism. This makes recovery assessable without treating a predictive association as evidence of a real biomarker.

An agent returns an ordered feature list or an empty list. It either receives the full dataset or chooses which patients and measurements to acquire within a budget. The [Discovery Score](docs/scoring.md) combines recovery beyond chance with restraint on worlds that contain no recoverable signal.

**Release status: `1.0.0rc4`.** This release candidate updates the runtime: corrected scoring, seeded stratified sampling, one replay-verified run format for both tracks, and a catalogue of literature-grounded reference methods. The world packs are still the `1.0.0rc3` downloads, now with published id lists: 2,465 of their 2,489 worlds pass the current solvability screen, and 24 are excluded. Use the certified lists (`--worlds <pack>-certified.txt`). Packs rebuilt on the corrected generator follow in a later release. No accepted benchmark results or hosted evaluation release are announced here. See [stability and limitations](STABILITY.md).

## Play on the Arena

The hosted **[ONC-AGI Arena](https://onc-agi.com/arena)** runs agents for you. Sign in with GitHub to get a key, then:
- bring a model through your own provider key;
- upload a one-file agent; or
- play from anywhere with `onc-agi play --url`.

Leaderboard entries are verified by replaying the server's own trace before they are listed.

## Run offline

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). From a fresh checkout:

```bash
git clone https://github.com/BradSegal/ONC-AGI
cd ONC-AGI
uv sync
uv run onc-agi smoke
uv run python examples/agents/pipeline_agent.py
```

The smoke test exercises both modes on 20 small, synthetic fixtures. It checks the oracle, baselines and shortcut agents and finishes with `smoke passed`. The pipeline example prints a scorecard. Neither command needs an API key, Docker or a benchmark download; fixture scores are software checks, not evidence of benchmark performance. Without uv, install the checkout with `pip install .` and run `onc-agi smoke`.

## Use the benchmark

| Task | Guide |
|---|---|
| Understand worlds and mechanisms | [Task](docs/premise.md) |
| Build or run an agent | [Agents](docs/agents-kit.md) · [Runnable examples](examples/agents) |
| Implement a client | [Interface](docs/interface.md) · [Schemas](schemas) · [Payloads](examples/payloads) |
| Check how a submission earns credit | [Scoring](docs/scoring.md) |
| Select worlds and compare results | [Evaluation](docs/evaluation-protocol.md) |

This repository contains the runtime, fixtures, schemas and agent examples. The world generator is maintained separately. [CONTRIBUTING.md](CONTRIBUTING.md) identifies which files are maintained here and how to report a problem.

ONC-AGI is inspired by [ARC-AGI](https://arcprize.org/) and is not affiliated with the ARC Prize Foundation. Its claims concern planted mechanisms in benchmark worlds; it does not validate clinical biomarkers.

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
  - development: pytest, pytest-cov, Hypothesis, black, Ruff, mypy, pandas-stubs, jsonschema, Twine, check-wheel-contents and pip-audit.
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
