# Changelog

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
