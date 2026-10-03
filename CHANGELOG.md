# Changelog

## 1.0.0rc3 — 2026-10-04

Benchmark worlds, first-pass results and the website. The payload schemas stay interface `1.0`. Survival worlds add optional fields that binary payloads omit. The scorer label now carries a digest of its source (`scorer-1.0+<digest>`), so scorecards from different rule sets are never compared as equal.

**Added**
- **2,489 public-train benchmark worlds** as release downloads: 995 full-access, 996 sequential and 498 expressive worlds, mixed over seven real cohort sources, each with its oracle certificate and answer key. See *World sets* in the README.
- New world kinds:
  - survival outcomes (time to event with censoring), certified by the Cox score statistic;
  - nonlinear drivers;
  - missing cells, some of them outcome-dependent on non-answer columns;
  - composed worlds with up to three mechanisms.
- Every clinical variable causes a few measurements, and every measurement carries a cohort offset, in every world. Confounder and batch worlds are therefore no longer recognisable by their shape.
- The agents kit: `onc-agi play` runs any agent (built-in, your own class, or the OpenAI-compatible `llm` template) on a scorecard in parallel, with recordings, a spend cap and `onc-agi explain`. Also new: `onc-agi worlds`, `GET /v1/worlds`, named public-train draws, and gzip responses. See `docs/agents-kit.md` and `docs/evaluation-protocol.md`.
- Scorecards are archived and survive a server restart. Closed scorecards are served at `GET /v1/scorecards/{sid}`.
- The website source is in `site/`, with GitHub Pages deployment.

**Changed**
- Credit at depth R uses maximum matching, so list order inside the top R and the private order of groups never move a score.
- `onc-agi serve` refuses a store that holds evaluation worlds unless issued keys are supplied (`--api-keys`).
- Eval-tier scorecards need at least 40 worlds by default.
- The score test refits a nuisance model that separates the outcome under a weak prior. A strong leak beside a tested feature no longer voids that feature's statistic.

**Fixed**
- `evaluate --agent` listed `seq_` names that could not be built.
- The sequential task card now names the strata, prices and refresh rule, and tool schemas are strict-compatible for OpenAI models.

## 1.0.0rc2 — 2026-10-03

Security and correctness fixes found by independent statistical review and adversarial testing. Payload schemas are unchanged (interface `1.0`, scorer `scorer-1.0`).

**Changed — read this if you call the HTTP API directly**
- Scorecards belong to the key that opened them. `POST …/actions`, `GET …/worlds/{wid}` and `POST …/close` now require the `X-Arena-Key` header, and a different key gets the same `invalid_payload` "unknown scorecard" error as an id that does not exist. `ArenaClient` already sent the header, so code using it is unaffected. This closes a hole where anyone who learned a scorecard id could act on it or close it.
- Matched chance follows the representatives the scorer actually credits (deduplicated and with neutral features removed). Padding a list with near-duplicates or neutral features can no longer move chance. This restores the documented property that an outcome-blind list scores zero in expectation. Scores of lists that repeat cluster-mates can change slightly.

**Fixed**
- Concurrent scorecard openings on one key could bypass the daily cap, reuse worlds and corrupt the JSON ledger. Openings are now serialised, and the ledger is file-locked and written atomically.
- Group-sequential agents (`seq_*`) recruited only from the first stratum, so they aborted on multi-stratum worlds. They now recruit every stratum in proportion and never beyond its published size.
- Group-sequential agents treated genuinely missing values as unassayed cells and re-assayed forever. They now track their own purchases.
- Alignment diagnostics built product terms from raw columns. Interaction, effect-modifier and mixture terms are now built the way the world builder builds them, and every analyst uses one shared statistic, the signed Rao score z (`onc_agi.services.score_test`), which stays monotone under separation.
- The learned outcome-blind ranker refuses any world outside `public_train`.

**Other**
- `onc-agi smoke` also runs a group-sequential agent.
- `docs/interface.md` states the scorecard ownership rule.

## 1.0.0rc1 — 2026-10-03

First public contract preview.

- Interface 1.0: `WorldCard` (with published `stratum_sizes`), the `reset`, `recruit`, `assay` and `submit` actions, `Observation`, typed errors, `Scorecard`, `WorldScore`, `TraceEvent` and `AnswerKey`, with JSON Schemas and OpenAPI.
- Runtime:
  - the engine, for full-access and sequential modes;
  - the scorer, `scorer-1.0`;
  - the Agent kit;
  - the HTTP server and client;
  - conformance and replay.
- Agents: reference, baseline and cheater agents.
- Inspect standard-track task (preview).
- Twenty toy fixture worlds (ten mechanisms × two modes), with answer keys set by construction.
- Template agents and captured example payloads.
