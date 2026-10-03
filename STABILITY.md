# Stability

**Release:** `1.0.0-rc1`. Interface `1.0`, scorer `scorer-1.0`, engine `engine-1.0`.

## Release plan

| Release | Contents | When |
|---|---|---|
| `1.0.0-rc1` (this) | Interface specification, runtime, toy fixtures, template agents, conformance | Now |
| `1.0.0-rc*` | Interface fixes from collaborator feedback (additive only) | Until the feedback freeze |
| `1.0.0` | Frozen interface | **Feedback freeze: 2026-10-03 23:00 BST** |
| Benchmark release | Public-train benchmark worlds from real cohorts, evaluation server, baseline results | After independent assurance |

## What you can rely on

| Surface | Status |
|---|---|
| Payload models in `onc_agi.core.schema`, `schemas/*.schema.json` | **Release candidate.** Only additive changes until 1.0.0 |
| HTTP endpoints and status codes (`schemas/openapi.json`) | **Release candidate** |
| Error codes | **Release candidate** |
| Episode semantics (reset, recruit, assay, submit, idempotency, budget) | **Release candidate** |
| Scoring semantics (`docs/scoring.md`) | **Release candidate.** The open question on greedy credit order is noted there |
| `Agent`, `PipelineAgent`, `evaluate`, `run_episode`, `ArenaClient` | **Release candidate** |
| Baseline and cheater agents | **Stable names.** Internals may be tuned |
| Inspect standard-track task (`onc_agi.adapters.inspect_task`) | **Preview** |
| Alignment diagnostics | **Preview.** Values may be recalibrated |
| Toy fixture worlds | **Fixtures.** They may be extended. They are not a benchmark |

## Change policy

A MINOR interface change adds optional fields only. A MAJOR change removes, renames or re-types a field, or changes a scoring semantic. Every change is listed in [CHANGELOG.md](CHANGELOG.md).

## Known limitations in this release

- **Binary outcomes only.** Time-to-event outcomes are out of scope for version 1.
- **Every observation resends the full revealed state.** This is simple and resumable, but large for wide worlds.
- **Credit is greedy in list order** (see [docs/scoring.md](docs/scoring.md)).
- **Integrity claims for sequential mode and hidden-variable mechanics** hold only on the no-network standard track. Open-track results for them are labelled unverified.
- **The local server accepts any API key.** Hosted evaluation servers issue keys.
