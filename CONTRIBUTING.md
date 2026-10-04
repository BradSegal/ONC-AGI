# Contributing

For a bug, open an issue with the command, commit, world ID and full error. Prefer a reproduction on a bundled fixture. For interface feedback, include the request or response that caused difficulty and the behaviour you need. Do not include credentials or private world data; see [SECURITY.md](SECURITY.md) for security reports.

## Where changes belong

The runtime (`src/onc_agi/`, excluding fixtures), `tests/runtime/`, `site/`, and four guides—`interface.md`, `scoring.md`, `agents-kit.md` and `evaluation-protocol.md`—are exported from the creation repository. `EXPORT.json` records their hashes. Propose changes here with an issue or pull request; maintainers apply them upstream before exporting.

The other repository guides, toy builder, contract generator, examples and `tests/preview/` are maintained here. Keep examples runnable from a checkout and distinguish fixture behaviour from benchmark results.

## Check a change

```bash
uv sync --locked --all-extras --group dev
uv run black --check .
uv run ruff check .
uv run mypy src
uv run pytest --cov --cov-fail-under=80
uv run python tools/build_toy.py
uv run python tools/gen_contract.py
git diff --exit-code
```

Run generation checks on a clean working tree; the final command reports any regenerated differences. Never commit keys, tokens or machine-specific absolute paths. CI also runs `tools/scan_release.py`.

Before a release, run `uv run python tools/check_package.py` to build and install both distributions in fresh environments. The release scan requires Gitleaks; a missing scanner is an error.
