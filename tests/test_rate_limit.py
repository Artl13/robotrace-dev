"""Behavior of HTTP 429 / `RateLimitError` + auto-retry on safe paths.

Three things being pinned here:

  1. The HTTP wrapper raises a typed ``RateLimitError`` with the
     ``Retry-After`` header parsed into ``retry_after: int``.
  2. ``Client.start_episode`` (a safe-to-retry create) transparently
     retries on 429 until success, honoring the server's
     ``Retry-After`` value.
  3. ``Episode.finalize`` (idempotency-sensitive) does NOT auto-retry
     — the user is expected to own that retry policy because a future
     paid tier could double-bill artifact storage. The existing
     comment in ``errors.ServerError`` documents the same rule.

We monkeypatch ``time.sleep`` so the retry math doesn't actually
block the test run. The runtime is still O(microseconds).
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

import robotrace as rt
from robotrace import _http


def _disable_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace time.sleep in the http module with a recorder.

    Returns the list it appends to so each test can assert on the
    backoff schedule the wrapper actually picked.
    """
    captured: list[float] = []
    monkeypatch.setattr(_http.time, "sleep", lambda s: captured.append(s))
    return captured


def _build_client(handler: Callable[[httpx.Request], httpx.Response]) -> rt.Client:
    return rt.Client(
        api_key="rt_id123_secret456",
        base_url="https://example.test",
        transport=httpx.MockTransport(handler),
    )


# ── retry-after parsing ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "header, expected",
    [
        ("30", 30),
        ("0", 0),
        (" 12 ", 12),
        ("not-a-number", None),
        ("", None),
        (None, None),
        ("-5", None),  # non-digit per str.isdigit (leading minus)
        ("999999999", None),  # past 24h cap
    ],
)
def test_parse_retry_after(header: str | None, expected: int | None) -> None:
    assert _http._parse_retry_after(header) == expected


def test_retry_delay_prefers_retry_after() -> None:
    # Header wins over the exponential backoff schedule.
    assert _http._retry_delay_seconds(7, attempt=0) == 7.0
    # Header is capped so a misconfigured server can't pin a robot.
    assert (
        _http._retry_delay_seconds(_http.MAX_RETRY_AFTER_SECONDS + 9999, attempt=0)
        == float(_http.MAX_RETRY_AFTER_SECONDS)
    )
    # No header → exponential backoff (1s, 2s, 4s).
    assert _http._retry_delay_seconds(None, attempt=0) == 1.0
    assert _http._retry_delay_seconds(None, attempt=1) == 2.0
    assert _http._retry_delay_seconds(None, attempt=2) == 4.0


# ── 429 → RateLimitError on a non-retry-safe call ────────────────────


def test_finalize_raises_rate_limit_error_with_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps = _disable_sleep(monkeypatch)
    calls = {"create": 0, "finalize": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ingest/episode":
            calls["create"] += 1
            return httpx.Response(
                201,
                json={
                    "episode_id": "ep_test_429",
                    "status": "recording",
                    "storage": "unconfigured",
                    "upload_urls": [],
                },
            )
        if request.url.path == "/api/ingest/episode/ep_test_429/finalize":
            calls["finalize"] += 1
            return httpx.Response(
                429,
                headers={"Retry-After": "12"},
                json={"error": "quota exceeded"},
            )
        return httpx.Response(500, json={"error": "unstubbed"})

    client = _build_client(handler)
    ep = client.start_episode(name="429 finalize", artifacts=[])

    with pytest.raises(rt.RateLimitError) as info:
        ep.finalize(status="ready")

    # Typed exception, parsed retry_after, intact APIError fields.
    assert isinstance(info.value, rt.APIError)
    assert info.value.status_code == 429
    assert info.value.retry_after == 12
    assert info.value.response_body == {"error": "quota exceeded"}

    # Finalize is idempotency-sensitive → no auto-retry, no sleep.
    assert calls["finalize"] == 1
    assert sleeps == []


# ── retry-safe path: start_episode auto-retries on 429 ───────────────


def test_start_episode_retries_on_429_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps = _disable_sleep(monkeypatch)
    seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/ingest/episode"
        seen.append(len(seen))
        if len(seen) < 3:
            return httpx.Response(
                429,
                headers={"Retry-After": "2"},
                json={"error": "slow down"},
            )
        return httpx.Response(
            201,
            json={
                "episode_id": "ep_after_retry",
                "status": "recording",
                "storage": "unconfigured",
                "upload_urls": [],
            },
        )

    client = _build_client(handler)
    ep = client.start_episode(name="retry me", artifacts=[])

    assert ep.id == "ep_after_retry"
    # 2x 429 + 1 success = 3 attempts -> 2 sleeps, each honoring
    # the server-supplied Retry-After value.
    assert len(seen) == 3
    assert sleeps == [2.0, 2.0]


def test_start_episode_gives_up_after_max_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps = _disable_sleep(monkeypatch)
    seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(len(seen))
        return httpx.Response(
            429,
            headers={"Retry-After": "1"},
            json={"error": "still throttled"},
        )

    client = _build_client(handler)
    with pytest.raises(rt.RateLimitError) as info:
        client.start_episode(name="never works", artifacts=[])

    # Exhausted all attempts, still surfaces the typed error.
    assert len(seen) == _http.MAX_ATTEMPTS
    assert info.value.retry_after == 1
    # MAX_ATTEMPTS attempts → MAX_ATTEMPTS - 1 sleeps before the
    # final raise.
    assert len(sleeps) == _http.MAX_ATTEMPTS - 1


def test_retry_falls_back_to_exponential_backoff_without_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps = _disable_sleep(monkeypatch)
    seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(len(seen))
        if len(seen) < 4:
            # No Retry-After header — exercise the exponential path.
            return httpx.Response(429, json={"error": "throttled"})
        return httpx.Response(
            201,
            json={
                "episode_id": "ep_backoff",
                "status": "recording",
                "storage": "unconfigured",
                "upload_urls": [],
            },
        )

    client = _build_client(handler)
    ep = client.start_episode(name="exp backoff", artifacts=[])

    assert ep.id == "ep_backoff"
    # 3x 429 + 1 success = 4 attempts -> 3 sleeps following the
    # exponential schedule (1s, 2s, 4s).
    assert sleeps == [1.0, 2.0, 4.0]


# ── upload path also retries on 429 ──────────────────────────────────


def test_upload_file_retries_on_429(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    sleeps = _disable_sleep(monkeypatch)
    attempts = {"n": 0}

    def upload_handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 2:
            return httpx.Response(429, headers={"Retry-After": "3"}, text="slow")
        return httpx.Response(200)

    # Monkeypatch httpx.Client used inside upload_file so we don't
    # talk to the network. The signed-URL host has no auth header,
    # which mirrors production.
    real_client_cls = httpx.Client

    def fake_client(*args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["transport"] = httpx.MockTransport(upload_handler)
        return real_client_cls(*args, **kwargs)

    monkeypatch.setattr(_http.httpx, "Client", fake_client)

    artifact = tmp_path / "actions.bin"
    artifact.write_bytes(b"x" * 32)

    http = _http.HTTPClient(
        api_key="rt_x",
        base_url="https://example.test",
        # Don't pass a transport — upload_file uses its own client.
    )
    written = http.upload_file(
        "https://r2.example/signed-put",
        artifact,
        content_type="application/octet-stream",
    )
    assert written == 32
    assert attempts["n"] == 2
    # One retry → one sleep, honoring Retry-After.
    assert sleeps == [3.0]


def test_upload_file_surfaces_rate_limit_after_max_attempts(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _disable_sleep(monkeypatch)
    attempts = {"n": 0}

    def upload_handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(429, headers={"Retry-After": "5"}, text="quota")

    real_client_cls = httpx.Client

    def fake_client(*args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["transport"] = httpx.MockTransport(upload_handler)
        return real_client_cls(*args, **kwargs)

    monkeypatch.setattr(_http.httpx, "Client", fake_client)

    artifact = tmp_path / "actions.bin"
    artifact.write_bytes(b"x")

    http = _http.HTTPClient(api_key="rt_x", base_url="https://example.test")

    with pytest.raises(rt.RateLimitError) as info:
        http.upload_file(
            "https://r2.example/signed-put",
            artifact,
            content_type="application/octet-stream",
        )
    assert info.value.retry_after == 5
    assert attempts["n"] == _http.MAX_ATTEMPTS


# ── backwards compat: non-429 codes still hit the right subclass ─────


def test_non_429_errors_are_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_sleep(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "key revoked"})

    client = _build_client(handler)
    with pytest.raises(rt.AuthError):
        client.start_episode(name="bad key", artifacts=[])


def test_request_body_unchanged_under_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retried request must re-send the exact same JSON payload —
    callers depend on the wire format being identical across attempts.
    """
    _disable_sleep(monkeypatch)
    bodies: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        if len(bodies) < 2:
            return httpx.Response(429, headers={"Retry-After": "1"}, json={})
        return httpx.Response(
            201,
            json={
                "episode_id": "ep_idem",
                "status": "recording",
                "storage": "unconfigured",
                "upload_urls": [],
            },
        )

    client = _build_client(handler)
    client.start_episode(
        name="idempotent body",
        policy_version="v1",
        env_version="env1",
        git_sha="abc",
        seed=1,
        artifacts=["video"],
    )
    assert len(bodies) == 2
    # Same bytes on the wire across the retry — no jitter, no
    # `attempt` field leaking into the payload.
    assert bodies[0] == bodies[1]
    payload = json.loads(bodies[0])
    assert payload["policy_version"] == "v1"
