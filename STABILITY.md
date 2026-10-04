# Stability

**Release:** `1.0.0-rc3`. Interface `1.0`, scorer `scorer-1.0` (labels carry a source digest), engine `engine-1.0`.

## Release plan

| Release | Contents | When |
|---|---|---|
| `1.0.0-rc1` | Interface specification, runtime, toy fixtures, template agents, conformance | 2026-10-03 |
| `1.0.0-rc2` | Security and correctness fixes to the hosted scorecards, matched chance and sequential agents | 2026-10-03 |
| `1.0.0-rc3` (this; superseded) | 2,489 public-train benchmark worlds (8 unsolvable), first-pass results (withdrawn), the agents kit and the website | 2026-10-04 |
| `1.0.0-rc*` | Interface fixes from collaborator feedback (additive only) | Until the feedback freeze |
| `1.0.0` | Frozen interface | **Feedback freeze: 2026-10-03 23:00 BST** |
| Evaluation release | Hosted evaluation server for the hidden tiers | After independent assurance |

## What you can rely on

| Surface | Status |
|---|---|
| Payload models in `onc_agi.core.schema`, `schemas/*.schema.json` | **Release candidate.** Only additive changes until 1.0.0 |
| HTTP endpoints and status codes (`schemas/openapi.json`) | **Release candidate** |
| Error codes | **Release candidate** |
| Episode semantics (reset, recruit, assay, submit, idempotency, budget) | **Release candidate** |
| Scoring semantics (`docs/scoring.md`) | **Release candidate.** Credit uses maximum matching |
| `Agent`, `PipelineAgent`, `evaluate`, `run_episode`, `ArenaClient` | **Release candidate** |
| Baseline and cheater agents | **Stable names.** Internals may be tuned |
| Inspect standard-track task (`onc_agi.adapters.inspect_task`) | **Preview** |
| Alignment diagnostics | **Preview.** Values may be recalibrated |
| Toy fixture worlds | **Fixtures.** They may be extended. They are not a benchmark |
| Public-train world sets (release downloads) | **Versioned.** A set never changes once released; new versions are new downloads |

## Change policy

A MINOR interface change adds optional fields only. A MAJOR change removes, renames or re-types a field, or changes a scoring semantic. Every change is listed in [CHANGELOG.md](CHANGELOG.md).

## Known limitations in this release

- **Survival worlds need rc3 clients.** Their `time` and `horizon_days` fields are additive; a client that rejects unknown fields must update.
- **Every observation resends the full revealed state.** This is simple and resumable, but large for wide worlds.
- **Integrity against public-source re-identification is not yet certified** for the hidden tiers; the public-train sets are published with answers, so it does not affect them.
- **Integrity claims for sequential mode and hidden-variable mechanics** hold only on the no-network standard track. Open-track results for them are labelled unverified.
- **The local server accepts any API key.** Hosted evaluation servers issue keys.
