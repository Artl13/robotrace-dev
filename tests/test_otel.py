"""OpenTelemetry trace correlation — wire-format + soft-import tests.

Covers the three states the SDK can find itself in:

  1. OTel not installed at all (the `[otel]` extra wasn't pulled
     in). `capture_trace_context()` returns None; the create-episode
     payload omits the `otel` key entirely.
  2. OTel installed, no active span. `get_current_span()` returns
     `INVALID_SPAN`; we still return None (don't propagate sentinel
     zero IDs that would render as `0000…0000` in the portal).
  3. OTel installed, active span. We propagate trace_id, span_id,
     traceparent in the W3C-compliant hex format the portal expects.

We mock OTel via a tiny stand-in module installed under
`opentelemetry.trace` for the duration of each test that wants it,
so the tests run without the `[otel]` extra actually being present.
That mirrors how downstream users typically run pytest in CI: the
core SDK matrix doesn't install optional adapters.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

import robotrace as rt
from robotrace import _otel


# ── shared fixture: a Client whose transport captures every request ─


def _make_capturing_client() -> tuple[rt.Client, list[httpx.Request]]:
    """Return (client, captured_requests). Same pattern as test_request_shape.py."""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if request.url.path == "/api/ingest/episode":
            return httpx.Response(
                201,
                json={
                    "episode_id": "ep_otel_test",
                    "status": "recording",
                    "storage": "unconfigured",
                    "upload_urls": [],
                },
            )
        return httpx.Response(500, json={"error": "no stub"})

    client = rt.Client(
        api_key="rt_id123_secret456",
        base_url="https://example.test",
        transport=httpx.MockTransport(handler),
    )
    return client, captured


# ── (1) OTel not installed — the default in CI ─────────────────────


def test_no_otel_installed_omits_otel_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """When `_OTEL_AVAILABLE` is False the field is absent — not
    `None`, not an empty dict. The server contract treats the key
    as truly optional, so omitting keeps backwards compatibility.
    """
    monkeypatch.setattr(_otel, "_OTEL_AVAILABLE", False)
    monkeypatch.setattr(_otel, "_otel_trace", None)

    client, captured = _make_capturing_client()
    client.start_episode(name="no-otel", artifacts=[])

    assert len(captured) == 1
    payload = json.loads(captured[0].content)
    assert "otel" not in payload


# ── (2) OTel installed, but no active span ─────────────────────────


@pytest.fixture
def fake_otel_no_span(monkeypatch: pytest.MonkeyPatch) -> Iterator[types.ModuleType]:
    """Install a fake `opentelemetry.trace` module returning INVALID_SPAN.

    INVALID_SPAN's `get_span_context()` returns sentinel zero IDs;
    `capture_trace_context` must short-circuit those to None so the
    portal doesn't surface unclickable `0000…0000` IDs.
    """
    fake = _install_fake_otel_module(
        monkeypatch,
        trace_id=0,  # OTel's INVALID_TRACE_ID
        span_id=0,   # OTel's INVALID_SPAN_ID
        flags=0,
    )
    yield fake


def test_otel_installed_no_active_span_omits_field(
    fake_otel_no_span: types.ModuleType,  # noqa: ARG001 — fixture installs the mock
) -> None:
    client, captured = _make_capturing_client()
    client.start_episode(name="no-active-span", artifacts=[])

    payload = json.loads(captured[0].content)
    assert "otel" not in payload


# ── (3) OTel installed with an active span — happy path ────────────


@pytest.fixture
def fake_otel_with_span(monkeypatch: pytest.MonkeyPatch) -> Iterator[types.ModuleType]:
    """Install a fake module with a real-looking 128-bit trace + 64-bit span."""
    fake = _install_fake_otel_module(
        monkeypatch,
        # Recognizable hex pattern so failures surface useful context.
        trace_id=0x4BF92F3577B34DA6A3CE929D0E0E4736,
        span_id=0x00F067AA0BA902B7,
        flags=1,  # sampled
    )
    yield fake


def test_otel_active_span_attaches_w3c_trace_context(
    fake_otel_with_span: types.ModuleType,  # noqa: ARG001
) -> None:
    """Happy path: full W3C traceparent + raw IDs, hex-encoded with
    the canonical lengths (32 hex for trace, 16 hex for span)."""
    client, captured = _make_capturing_client()
    client.start_episode(name="with-active-span", artifacts=[])

    payload = json.loads(captured[0].content)
    assert "otel" in payload, "expected otel context to be attached"

    otel = payload["otel"]
    assert otel["trace_id"] == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert otel["span_id"] == "00f067aa0ba902b7"
    assert (
        otel["traceparent"]
        == "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    )
    # Pin the field set — adding fields here is a server-contract
    # change, so the test should fail loudly to force a server-side
    # Zod update too.
    assert set(otel.keys()) == {"trace_id", "span_id", "traceparent"}


def test_otel_unsampled_flag_round_trips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the active span carries a non-default trace_flags value
    (e.g. unsampled tail-based propagation), we must preserve it in
    the traceparent — the user's APM may use it to decide whether
    to render the trace at all."""
    _install_fake_otel_module(
        monkeypatch,
        trace_id=0x000000000000000000000000DEADBEEF,
        span_id=0x00000000CAFEBABE,
        flags=0,  # explicitly unsampled — but capture should *force* 01
    )
    client, captured = _make_capturing_client()
    client.start_episode(name="unsampled", artifacts=[])

    payload = json.loads(captured[0].content)
    # `flags` of 0 is a valid OTel state but rare; the SDK falls back
    # to "sampled" (01) so the portal-side deep-link is renderable.
    # Pinning the policy here so a future change is intentional.
    assert payload["otel"]["traceparent"].endswith("-01")


def test_otel_capture_never_raises_when_module_misbehaves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any exception inside the OTel call path must be swallowed —
    SDK telemetry must not crash the user's training run.
    """
    boom = types.SimpleNamespace(
        get_current_span=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(_otel, "_OTEL_AVAILABLE", True)
    monkeypatch.setattr(_otel, "_otel_trace", boom)

    assert _otel.capture_trace_context() is None


# ── helpers ────────────────────────────────────────────────────────


def _install_fake_otel_module(
    monkeypatch: pytest.MonkeyPatch,
    *,
    trace_id: int,
    span_id: int,
    flags: int,
) -> types.ModuleType:
    """Stand up a fake `opentelemetry.trace` exposing just the surface
    `_otel.capture_trace_context` consumes, then re-import `_otel` so
    its module-level `_otel_trace` rebinds to the fake.
    """
    fake_ctx = types.SimpleNamespace(
        trace_id=trace_id, span_id=span_id, trace_flags=flags
    )
    fake_span = types.SimpleNamespace(get_span_context=lambda: fake_ctx)
    fake_trace = types.SimpleNamespace(get_current_span=lambda: fake_span)

    # Drop a synthetic `opentelemetry` package + `opentelemetry.trace`
    # submodule into sys.modules. We have to set the package first so
    # `from opentelemetry import trace as _otel_trace` resolves.
    pkg = types.ModuleType("opentelemetry")
    pkg.__path__ = []  # mark as package so `from opentelemetry import …` works
    monkeypatch.setitem(sys.modules, "opentelemetry", pkg)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", fake_trace)
    pkg.trace = fake_trace  # type: ignore[attr-defined]

    # Force `_otel` to re-evaluate its top-level soft-import against
    # the fake module so the rest of this test sees `_OTEL_AVAILABLE
    # = True` with our stand-in `_otel_trace`.
    importlib.reload(_otel)
    return fake_trace


@pytest.fixture(autouse=True)
def _reload_otel_after_each_test() -> Iterator[None]:
    """After each test, clear the synthetic `opentelemetry` modules
    and reload `_otel` so the next test starts from a clean slate.

    Without this, a happy-path test would leave `_OTEL_AVAILABLE =
    True` for the no-OTel test that follows.
    """
    yield
    for name in ("opentelemetry.trace", "opentelemetry"):
        sys.modules.pop(name, None)
    importlib.reload(_otel)


# Sanity check: verify the surface area we depend on is exported.
def test_otel_module_exposes_public_api() -> None:
    assert callable(_otel.capture_trace_context)
    assert callable(_otel.is_available)
    # TraceContext is a TypedDict, exported for downstream type hints.
    assert hasattr(_otel, "TraceContext")


def test_episode_payload_otel_field_round_trips_through_log_episode(
    fake_otel_with_span: types.ModuleType,  # noqa: ARG001
) -> None:
    """`log_episode` is the "sacred" entrypoint — same OTel attach
    behavior as `start_episode`. This test guards the path we
    expect 95% of users to take.
    """
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        path = request.url.path
        if path == "/api/ingest/episode":
            return httpx.Response(
                201,
                json={
                    "episode_id": "ep_log_otel",
                    "status": "recording",
                    "storage": "unconfigured",
                    "upload_urls": [],
                },
            )
        if path == "/api/ingest/episode/ep_log_otel/finalize":
            return httpx.Response(
                200, json={"episode_id": "ep_log_otel", "status": "ready"}
            )
        return httpx.Response(500, json={"error": f"no stub for {path}"})

    client = rt.Client(
        api_key="rt_id123_secret456",
        base_url="https://example.test",
        transport=httpx.MockTransport(handler),
    )
    client.log_episode(
        name="log + otel",
        source="sim",
        policy_version="otel-v1",
    )

    create_payload: dict[str, Any] = json.loads(captured[0].content)
    assert "otel" in create_payload
    assert (
        create_payload["otel"]["trace_id"]
        == "4bf92f3577b34da6a3ce929d0e0e4736"
    )
