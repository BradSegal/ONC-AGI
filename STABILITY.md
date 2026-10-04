# Stability and release status

The checkout exposes interface `1.0`, engine `engine-1.0` and scorer `scorer-1.0`; runtime labels include source digests. Pin the commit and world-store digest when recording a result.

## Current boundaries

| Surface | Status |
|---|---|
| Runtime, schemas, HTTP API and Agent kit | Release candidate; runnable locally |
| Bundled worlds | 20 synthetic contract fixtures; suitable for software checks |
| `1.0.0rc3` benchmark packs | Usable through the `1.0.0rc4` certified id lists: 2,465 of 2,489 worlds pass the solvability screen; 24 excluded |
| First-pass benchmark results | Withdrawn after selecting 120 worlds from one cohort source |
| Inspect standard track and alignment diagnostics | Preview |
| Hosted public-eval and private tiers | Supported by server code; no hosted evaluation release announced here |

The [historical packs](CHANGELOG.md#historical-rc3-packs) remain available for reproduction. Corrections require a new pack version; released archives are not overwritten. `--n` draws a seeded sample stratified by source, family and mode (`--seed`); `onc-agi subset` writes the id list of any named sample.

## Compatibility

A minor interface change adds optional fields; a major change removes, renames or re-types a field, or changes scoring semantics. Changes are recorded in [CHANGELOG.md](CHANGELOG.md). Strict clients must recognise the survival fields `time` and `horizon_days` before opening survival worlds. Unknown request fields are rejected.

The development runtime may contain additions absent from this checkout, including seeded sampling. Use the checked-in [interface](docs/interface.md), schemas and CLI help for this version.

## Limitations

Observations resend the full revealed state, which can be large. Public-train answer keys are distributed for development and must not enter agent observations. Protection against matching derived records back to public source data has not been certified for hidden-tier evaluation. A no-network sandbox limits access; it does not by itself establish that protection.

The local server can run without issued keys for public training. Hidden-tier serving requires issued keys unless an operator explicitly enables the development override. The [evaluation protocol](docs/evaluation-protocol.md) separates server controls from requirements for a comparable result.
