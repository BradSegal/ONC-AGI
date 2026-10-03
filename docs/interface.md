# ONC-AGI interface contract v1.0

This document is for harness authors. Every payload that crosses a process, file or network boundary is a model in `onc_agi.core.schema`, and this page describes those models. Every JSON example tagged `schema=<Model>` is validated against that model by the package's test suite (`tests/runtime/unit/test_interface_doc.py`), so the examples cannot drift from the code.

## Version and change policy

`INTERFACE_VERSION` is `"1.0"` and appears on world cards, observations, answer keys, scorecards and trace events.

- **MINOR** (1.x) may add optional fields only.
- **MAJOR** (2.0) covers anything that removes, renames or re-types a field, or changes a scoring semantic.

Every change is listed in `CHANGELOG.md`. Models reject unknown fields (`extra="forbid"`), so a client that sends a field the server does not know gets a typed error instead of a silent drop.

## The task

The premise is the same in every world:

> Here is a cohort with an outcome. Which measurements drive it? Return an ordered list, most likely first, or an empty list if nothing can be found.

An agent reads a **world card**, takes **actions**, receives **observations**, and finishes with exactly one **submit**.

## World card

The card is everything an agent may know before acting. Prices and budget are truth-blind: the budget always equals the full revealable pool at published prices.

```json schema=WorldCard
{
  "interface_version": "1.0",
  "world_id": "pt-2-k3x9q1ab",
  "tier": "public_train",
  "mode": "sequential",
  "outcome_type": "binary",
  "n_pool": 400,
  "features": [
    {"feature_id": "GXR4", "data_type": "expression", "timing": "baseline", "assay_price": 2.0},
    {"feature_id": "CN_17q12", "data_type": "copy_number", "timing": "baseline", "assay_price": 5.0},
    {"feature_id": "relapse_marker", "data_type": "lab", "timing": "post_outcome", "assay_price": 1.0}
  ],
  "strata": ["site_a", "site_b"],
  "stratum_sizes": {"site_a": 250, "site_b": 150},
  "prices": {"recruit_per_patient": 10.0, "currency": "USD"},
  "budget": 7200.0,
  "name_visibility": "fake"
}
```

`post_outcome` features are measured after the outcome. Listing one in a submission is a **leak** and zeroes that world.

## Actions

Each action carries a client-chosen `request_id`. Retrying the same `request_id` with the same payload returns the same observation (idempotent). Reusing it with a different payload is refused with `request_conflict`. The wire format wraps every action in an envelope:

```json schema=ActionEnvelope
{"action": {"kind": "reset", "request_id": "r-0", "world_id": "pt-2-k3x9q1ab"}}
```

Reset on an episode that has already started is an idempotent no-op that returns the current observation. It never re-draws data.

```json schema=ActionEnvelope
{"action": {"kind": "recruit", "request_id": "r-1", "count": 50, "stratum": "site_a"}}
```

Recruiting reveals outcome and stratum for the next patients in that stratum's fixed queue. Identical action sequences reveal identical data.

```json schema=ActionEnvelope
{"action": {"kind": "assay", "request_id": "r-2", "feature_ids": ["GXR4", "CN_17q12"]}}
```

An assay measures the listed features on every recruited patient who has not yet been measured. Only unmeasured cells are charged.

```json schema=ActionEnvelope
{"action": {"kind": "submit", "request_id": "r-3", "ranking": ["GXR4", "CN_17q12"]}}
```

The ranking is ordered with the most likely driver first. A feature may not appear twice. To abstain, submit an empty list:

```json schema=Submit
{"kind": "submit", "request_id": "r-3", "ranking": []}
```

In **full-access** mode all data are revealed at reset, so only `submit` is available. In **sequential** mode `recruit`, `assay` and `submit` are available until submission, and spend is tracked by the server.

## Observation

Revealed data are columnar. A cell that has not been measured is `null`.

```json schema=Observation
{
  "interface_version": "1.0",
  "world_id": "pt-2-k3x9q1ab",
  "mode": "sequential",
  "step": 2,
  "status": "active",
  "budget": 7200.0,
  "spent": 600.0,
  "available_actions": ["recruit", "assay", "submit"],
  "revealed": {
    "patient_ids": ["p0", "p1"],
    "outcome": [1, 0],
    "stratum": ["site_a", "site_a"],
    "columns": {"GXR4": [0.42, -1.3], "CN_17q12": [null, null]}
  }
}
```

## Errors

Expected failures return an `ArenaErrorPayload`, mapped to an HTTP status:

| Code | HTTP | Meaning |
|---|---|---|
| `unknown_world` | 404 | No such world in this scorecard |
| `unknown_feature` | 422 | A feature id is not on the card |
| `unknown_stratum` | 422 | A stratum is not on the card, or recruiting beyond its size |
| `over_budget` | 402 | The action would exceed the budget |
| `action_not_available` | 409 | For example, recruit in a full-access world |
| `episode_closed` | 409 | The world has already been submitted |
| `request_conflict` | 409 | A `request_id` was reused for a different payload |
| `cap_exceeded` | 429 | Too many scorecards opened for this tier |
| `scorecard_closed` | 409 | The scorecard has already been closed |
| `invalid_payload` | 400 | The request does not validate against the schema |

```json schema=ArenaErrorPayload
{"code": "over_budget", "message": "assay would spend 7400 of a 7200 USD budget"}
```

## HTTP endpoints

All endpoints live under `/v1`. Every scorecard endpoint needs an `X-Arena-Key` header, and only the key that opened a scorecard can act on it, read it or close it. Any other key gets the same `unknown scorecard` error as an id that does not exist. When the server runs with `--api-keys`, only issued keys are accepted.

| Method and path | Body | Returns |
|---|---|---|
| `GET /v1/health` | none | `{"status": "ok", "interface_version": "1.0"}` |
| `POST /v1/scorecards` | `OpenRequest` | `scorecard_id` and the drawn world cards |
| `POST /v1/scorecards/{sid}/worlds/{wid}/actions` | `ActionEnvelope` | `Observation` |
| `GET /v1/scorecards/{sid}/worlds/{wid}` | none | current `Observation` (resume) |
| `POST /v1/scorecards/{sid}/close` | none | `Scorecard` |

```json schema=OpenRequest
{"agent": "my-agent-v3", "track": "open", "tier": "public_eval", "n_worlds": 40}
```

Public-eval and private scorecards are fresh draws from a committed pool, never reused:

- 20% are null worlds, and the rest are spread evenly over difficulty tiers.
- Public-eval scorecards are capped at 5 per key per day, and private scorecards at 3 per key in total.
- Eval tiers give no score feedback until the scorecard is closed.

## Scorecard

A scorecard reports the following:

- **Discovery Score**: `Find × max(0, Restraint)`. This is the headline.
- **Find**: chance-normalised recovery on signal worlds.
- **Restraint**: Youden's J between empty submissions on null worlds and empty submissions on signal worlds.
- The **unfloored** statistical estimand, with a stratified bootstrap interval.
- **Strict** score: exact recovery only.
- **Leak rate**: the share of worlds that listed a post-outcome feature.
- **Data cost**: mean spend.
- A summary per tier.
- Optionally, alignment diagnostics.

Undefined metrics are `null`, never 0. For example, a scorecard without null worlds has no Restraint. Per-world results are published only for `public_train`.

```json schema=Scorecard
{
  "interface_version": "1.0",
  "scorecard_id": "sc-7f3a",
  "track": "open",
  "agent": "my-agent-v3",
  "versions": {"engine": "1.0", "scorer": "1.0"},
  "tier": "public_eval",
  "n_worlds": 40,
  "discovery_score": 0.31,
  "discovery_score_unfloored": 0.29,
  "interval": {"low": 0.18, "high": 0.41},
  "find": 0.52,
  "find_signed": 0.49,
  "restraint": 0.6,
  "strict_discovery_score": 0.22,
  "leak_rate": 0.0,
  "abstention_on_signal": 0.15,
  "restraint_on_null": 0.75,
  "mean_data_cost": 3100.0,
  "per_tier": [
    {"difficulty_tier": 1, "n_signal": 16, "n_null": 4, "find": 0.7, "restraint": 0.65, "discovery_score": 0.46},
    {"difficulty_tier": 2, "n_signal": 16, "n_null": 4, "find": 0.34, "restraint": 0.55, "discovery_score": 0.19}
  ],
  "alignment": {"analysis_regret": 0.12, "acquisition_gap": 0.2}
}
```

## Trace events

Every action applied through the server is recorded with its exact payload, so an episode can be replayed and checked byte for byte (`onc-agi replay`, `onc-agi conformance`).

```json schema=TraceEvent
{
  "interface_version": "1.0",
  "scorecard_id": "sc-7f3a",
  "world_id": "pt-2-k3x9q1ab",
  "step": 1,
  "action_kind": "recruit",
  "request_id": "r-1",
  "request_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
  "response_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
  "spent": 500.0,
  "action_json": "{\"kind\":\"recruit\",\"request_id\":\"r-1\",\"count\":50,\"stratum\":\"site_a\"}"
}
```

## Conformance

`onc-agi conformance --url <server>` plays the fixture worlds through a server and checks the following:

- remote play equals local play;
- server traces replay exactly;
- sessions resume;
- retries are idempotent;
- conflicting retries are refused;
- closed episodes are refused.

A custom harness that passes conformance is scored identically to the standard one.
