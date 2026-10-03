# Changelog

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
