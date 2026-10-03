# Contributing

## Interface feedback

This release exists to collect feedback before the interface freezes at `1.0.0`.
- Open an issue with the **Interface feedback** template.
- Say what you tried, which payload or behaviour got in the way, and what you would change.
- Only additive changes (new optional fields) are made before the freeze.

## Bugs

Open an issue with the **Bug report** template. Include the command, the world id and the full error. Most bugs reproduce on a toy world.

## Code

The runtime in `src/onc_agi/` and `tests/runtime/` is exported from the upstream source, and `EXPORT.json` records each file's hash. A change there is applied upstream and arrives in the next export, so open an issue or a pull request describing it rather than relying on a direct edit here.

The toy builder, the contract generator, docs, examples and `tests/preview/` are maintained in this repository.

```bash
uv sync
uv run black --check . && uv run ruff check . && uv run mypy src
uv run pytest
uv run python tools/build_toy.py && uv run python tools/gen_contract.py && git diff --exit-code
```

Do not commit keys, tokens or absolute paths. `tools/scan_release.py` runs in CI.
