"""Model profiles: TOML loading, run-time overrides, keys from the environment only, ``.env`` handling."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from onc_agi.adapters.profiles import (
    EXAMPLE_PROFILES,
    ProfileError,
    load_dotenv,
    load_profile,
    profiles_path,
)

PROFILES = """
[profiles.p]
base_url = "http://x/v1/"
api_key_env = "P_KEY"
model = "org/m"
prices_per_mtok = { input = 1.0, output = 2.0 }
extra_headers = { "HTTP-Referer" = "https://example.org" }
[profiles.p.generate]
max_tokens = 100
[profiles.p.extra_body]
usage = { include = true }

[profiles.local]
base_url = "http://localhost:8000/v1"
model = "small"
"""


@pytest.fixture
def path(tmp_path: Path) -> Path:
    file = tmp_path / "profiles.toml"
    file.write_text(PROFILES)
    return file


def test_a_profile_loads_and_overrides_apply(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("P_KEY", "secret")
    p = load_profile(
        "p",
        ["temperature=0.2", "reasoning_effort=high", "extra_body.provider={sort='price'}", "model=org/m2"],
        path,
    )
    assert p.base_url == "http://x/v1" and p.model == "org/m2"
    assert p.generate == {"max_tokens": 100, "temperature": 0.2, "reasoning_effort": "high"}
    assert p.extra_body == {"usage": {"include": True}, "provider": {"sort": "price"}}
    assert p.request_body() == {
        "model": "org/m2",
        "max_tokens": 100,
        "temperature": 0.2,
        "reasoning_effort": "high",
        "usage": {"include": True},
        "provider": {"sort": "price"},
    }
    assert p.headers() == {"HTTP-Referer": "https://example.org", "Authorization": "Bearer secret"}
    price = p.price()
    assert price is not None and price.input == pytest.approx(1e-6) and price.cache_read is None


def test_a_missing_key_fails_fast_naming_the_variable(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("P_KEY", raising=False)
    p = load_profile("p", path=path)
    with pytest.raises(ProfileError, match="P_KEY"):
        p.api_key()


def test_a_keyless_local_profile_sends_no_authorization(path: Path) -> None:
    p = load_profile("local", path=path)
    assert p.api_key() is None and p.headers() == {} and p.price() is None


@pytest.mark.parametrize(
    ("name", "overrides", "message"),
    [
        ("absent", [], "no profile 'absent'.*known: local, p"),
        ("p", ["no-equals-sign"], "key=value"),
    ],
)
def test_bad_profile_requests_fail_fast(path: Path, name: str, overrides: list[str], message: str) -> None:
    with pytest.raises(ProfileError, match=message):
        load_profile(name, overrides, path)


def test_malformed_profiles_fail_fast(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text('[profiles.q]\nbase_url = "u"\nmodel = "m"\napi_key = "sk-inline"\n')
    with pytest.raises(ProfileError, match="unknown keys \\['api_key'\\]"):  # keys never live in profiles
        load_profile("q", path=bad)
    bad.write_text('[profiles.q]\nbase_url = "u"\nmodel = "m"\nprices_per_mtok = { input = 1.0 }\n')
    with pytest.raises(ProfileError, match="input and output"):
        load_profile("q", path=bad)
    bad.write_text("[profiles.q\n")
    with pytest.raises(ProfileError, match=r"bad\.toml"):
        load_profile("q", path=bad)
    with pytest.raises(ProfileError, match="does not exist"):
        profiles_path(tmp_path / "absent.toml")


def test_the_profile_file_lookup_prefers_explicit_then_local_then_packaged(
    path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert profiles_path(path) == path
    elsewhere = tmp_path / "empty-dir"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert profiles_path() == EXAMPLE_PROFILES
    (elsewhere / "profiles.toml").write_text(PROFILES)
    assert profiles_path() == Path("profiles.toml")


def test_the_packaged_example_profiles_load(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    luna = load_profile("openrouter-luna", path=EXAMPLE_PROFILES)
    assert luna.model == "openai/gpt-6-luna" and luna.api_key_env == "OPENROUTER_API_KEY"
    assert luna.prices_per_mtok == {"input": 0.10, "output": 0.50, "cache_read": 0.01, "cache_write": 0.125}
    assert luna.extra_body == {
        "usage": {"include": True},
        "reasoning": {"effort": "medium", "summary": "auto"},
    }
    assert load_profile("local", path=EXAMPLE_PROFILES).api_key() is None


def test_dotenv_never_overrides_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text("# keys\nDOTENV_SET=from-file\nexport DOTENV_NEW='quoted'\nDOTENV_KEEP=from-file\nnoise\n")
    monkeypatch.setenv("DOTENV_KEEP", "from-env")
    monkeypatch.delenv("DOTENV_SET", raising=False)
    monkeypatch.delenv("DOTENV_NEW", raising=False)
    load_dotenv(env)
    assert os.environ["DOTENV_SET"] == "from-file" and os.environ["DOTENV_NEW"] == "quoted"
    assert os.environ["DOTENV_KEEP"] == "from-env"
    load_dotenv(tmp_path / "absent.env")  # a missing .env is fine
    for name in ("DOTENV_SET", "DOTENV_NEW"):
        monkeypatch.delenv(name)


def test_the_inspect_mapping_exports_endpoint_and_key(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("P_KEY", "secret")
    monkeypatch.delenv("P_BASE_URL", raising=False)
    monkeypatch.delenv("P_API_KEY", raising=False)
    profile = load_profile("p", ["temperature=0.2", "extra_body.provider={sort='price'}"], path=path)
    assert profile.inspect_model() == "openai-api/p/org/m"
    assert os.environ["P_BASE_URL"] == "http://x/v1" and os.environ["P_API_KEY"] == "secret"
    config = profile.inspect_config()  # the same request parameters the open track sends
    assert {k: v for k, v in config.items() if not k.startswith("extra_")} == profile.generate
    assert config["extra_body"] == profile.extra_body and config["extra_body"]["provider"] == {
        "sort": "price"
    }
    assert "extra_headers" not in config  # header values never reach the Inspect log
