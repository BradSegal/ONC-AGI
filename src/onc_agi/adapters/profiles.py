"""Model profiles: any OpenAI-compatible chat-completions endpoint, configured in TOML.

A profile names an endpoint (``base_url``), the environment variable holding its key
(``api_key_env``; keys live only in the environment or a git-ignored ``.env``, never
in the profile), the provider's model id, and the request parameters to send::

    [profiles.luna]
    base_url = "https://openrouter.ai/api/v1"
    api_key_env = "OPENROUTER_API_KEY"
    model = "openai/gpt-6-luna"
    prices_per_mtok = { input = 0.10, output = 0.50, cache_read = 0.01, cache_write = 0.125 }
    [profiles.luna.generate]        # chat-completions parameters (max_tokens, temperature, ...)
    max_tokens = 16000
    [profiles.luna.extra_body]      # provider-specific fields, merged into the request body
    usage = { include = true }

Run-time overrides use ``--set key=value`` (values parsed as TOML, so ``0.2`` is a float
and bare words are strings): ``model=...`` swaps the model id, ``extra_body.<field>=...``
sets a provider field and any other key sets a chat-completions parameter.

The profile file is ``--profiles PATH``, else ``./profiles.toml``, else the packaged
example (``adapters/agents/profiles.example.toml``).
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

_PACKAGE = resources.files("onc_agi")
EXAMPLE_PROFILES = Path(str(_PACKAGE / "adapters" / "agents" / "profiles.example.toml"))
_KNOWN = {
    "base_url",
    "api_key_env",
    "model",
    "generate",
    "extra_body",
    "extra_headers",
    "prices_per_mtok",
    "description",
}


class ProfileError(ValueError):
    """A profile is missing, malformed or lacks its key (fail fast, before any world is opened)."""


def load_dotenv(path: Path = Path(".env")) -> None:
    """Export ``KEY=value`` lines from ``path`` without overriding variables already set."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().removeprefix("export ")
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def profiles_path(explicit: Path | None = None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise ProfileError(f"profiles file {explicit} does not exist")
        return explicit
    local = Path("profiles.toml")
    return local if local.exists() else EXAMPLE_PROFILES


@dataclass(frozen=True)
class Price:
    """USD per token. Unset cache prices fall back to ``input``."""

    input: float
    output: float
    cache_read: float | None = None
    cache_write: float | None = None


@dataclass(frozen=True)
class Profile:
    name: str
    base_url: str
    model: str
    api_key_env: str | None = None  # None: the endpoint needs no key (a local server)
    generate: dict[str, Any] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    extra_headers: dict[str, str] = field(default_factory=dict)
    prices_per_mtok: dict[str, float] | None = None
    description: str = ""

    @property
    def env_prefix(self) -> str:
        return re.sub(r"[^A-Z0-9]", "_", self.name.upper())

    def api_key(self) -> str | None:
        """The key from the environment; raises :class:`ProfileError` naming the variable if unset."""
        if self.api_key_env is None:
            return None
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise ProfileError(
                f"profile {self.name!r} needs the environment variable {self.api_key_env}"
                " (export it or put it in .env)"
            )
        return key

    def price(self) -> Price | None:
        """Per-token prices (the fallback when the provider reports no cost)."""
        if self.prices_per_mtok is None:
            return None
        p = {k: v / 1e6 for k, v in self.prices_per_mtok.items()}
        return Price(p["input"], p["output"], p.get("cache_read"), p.get("cache_write"))

    def request_body(self) -> dict[str, Any]:
        """Chat-completions body fields besides messages and tools (extra_body merged last)."""
        return {"model": self.model, **self.generate, **self.extra_body}

    def headers(self) -> dict[str, str]:
        key = self.api_key()
        auth = {"Authorization": f"Bearer {key}"} if key else {}
        return {**self.extra_headers, **auth}

    def inspect_model(self) -> str:
        """Inspect model string for the standard track (``openai-api/<profile>/<model>``), with the
        endpoint and key exported as Inspect expects."""
        os.environ[f"{self.env_prefix}_BASE_URL"] = self.base_url
        os.environ[f"{self.env_prefix}_API_KEY"] = self.api_key() or "none"
        return f"openai-api/{self.name}/{self.model}"


def parse_override(item: str) -> tuple[str, Any]:
    if "=" not in item:
        raise ProfileError(f"--set expects key=value, got {item!r}")
    key, raw = item.split("=", 1)
    try:
        value = tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        value = raw  # bare words are strings
    return key.strip(), value


def load_profile(name: str, overrides: list[str] | None = None, path: Path | None = None) -> Profile:
    """Load ``[profiles.<name>]`` and apply ``key=value`` overrides (see the module docstring)."""
    file = profiles_path(path)
    try:
        table = tomllib.loads(file.read_text(encoding="utf-8")).get("profiles", {})
    except tomllib.TOMLDecodeError as exc:
        raise ProfileError(f"{file}: {exc}") from exc
    if name not in table:
        raise ProfileError(f"no profile {name!r} in {file}; known: {', '.join(sorted(table)) or 'none'}")
    spec = dict(table[name])
    unknown = set(spec) - _KNOWN
    if unknown or not {"base_url", "model"} <= set(spec):
        raise ProfileError(
            f"profile {name!r} in {file}: needs base_url and model; unknown keys {sorted(unknown)}"
        )
    generate = dict(spec.pop("generate", {}))
    extra_body = dict(spec.pop("extra_body", {}))
    model = str(spec.pop("model"))
    for key, value in map(parse_override, overrides or []):
        if key == "model":
            model = str(value)
        elif key.startswith("extra_body."):
            extra_body[key.removeprefix("extra_body.")] = value
        else:
            generate[key] = value
    prices = spec.pop("prices_per_mtok", None)
    if prices is not None and not {"input", "output"} <= set(prices):
        raise ProfileError(f"profile {name!r}: prices_per_mtok needs input and output")
    return Profile(
        name=name,
        base_url=str(spec.pop("base_url")).rstrip("/"),
        model=model,
        api_key_env=spec.pop("api_key_env", None) or None,
        generate=generate,
        extra_body=extra_body,
        extra_headers={str(k): str(v) for k, v in dict(spec.pop("extra_headers", {})).items()},
        prices_per_mtok={k: float(v) for k, v in prices.items()} if prices else None,
        description=str(spec.pop("description", "")),
    )
