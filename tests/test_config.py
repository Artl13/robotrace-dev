"""Configuration-resolution tests.

The SDK refuses to construct without an explicit api_key + base_url
(or env equivalents). These tests guarantee we don't accidentally
add a default base_url that silently routes user data to the wrong
deployment.
"""

from __future__ import annotations

import pytest

import robotrace as rt
from robotrace import ConfigurationError


def test_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROBOTRACE_API_KEY", raising=False)
    monkeypatch.delenv("ROBOTRACE_BASE_URL", raising=False)
    with pytest.raises(ConfigurationError, match="API key"):
        rt.Client(base_url="https://example.test")


def test_client_requires_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROBOTRACE_API_KEY", raising=False)
    monkeypatch.delenv("ROBOTRACE_BASE_URL", raising=False)
    with pytest.raises(ConfigurationError, match="base URL"):
        rt.Client(api_key="rt_test")


def test_client_reads_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROBOTRACE_API_KEY", "rt_test")
    monkeypatch.setenv("ROBOTRACE_BASE_URL", "https://example.test")
    # Should not raise
    client = rt.Client()
    assert client.base_url == "https://example.test"
    client.close()


def test_base_url_trailing_slash_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROBOTRACE_API_KEY", "rt_test")
    client = rt.Client(base_url="https://example.test/")
    assert client.base_url == "https://example.test"
    client.close()


def test_init_replaces_default_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROBOTRACE_API_KEY", "rt_test")
    monkeypatch.setenv("ROBOTRACE_BASE_URL", "https://example.test")
    rt.init()
    rt.init(base_url="https://other.test")  # should not raise
    rt.close()
