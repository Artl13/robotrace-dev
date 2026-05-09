"""Pin the wire format the SDK sends to /api/ingest/episode.

These tests are intentionally low-level: they assert the JSON shape
hitting the server, because the `log_episode` signature is "sacred"
per AGENTS.md and the server contract evolves under it.

If a test here fails, you've changed the wire format — confirm the
server side accepts the new shape (or keeps accepting the old one
during the deprecation window) before merging.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest

import robotrace as rt


@dataclass
class StubFixture:
    """Bundles a Client with the requests its MockTransport saw.

    Avoids reaching into httpx internals for the captured-request
    list — every test should use `fix.requests` instead of poking
    at private attributes.
    """

    client: rt.Client
    requests: list[httpx.Request] = field(default_factory=list)


def _make_fixture(responses: dict[tuple[str, str], dict[str, Any]]) -> StubFixture:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        body = responses.get((request.method, request.url.path))
        if body is None:
            return httpx.Response(500, json={"error": "no stub for this path"})
        return httpx.Response(201 if request.method == "POST" else 200, json=body)

    transport = httpx.MockTransport(handler)
    client = rt.Client(
        api_key="rt_id123_secret456",
        base_url="https://example.test",
        transport=transport,
    )
    return StubFixture(client=client, requests=captured)


@pytest.fixture
def fix() -> StubFixture:
    return _make_fixture(
        {
            ("POST", "/api/ingest/episode"): {
                "episode_id": "ep_test_123",
                "status": "recording",
                "storage": "unconfigured",
                "upload_urls": [],
            },
            ("POST", "/api/ingest/episode/ep_test_123/finalize"): {
                "episode_id": "ep_test_123",
                "status": "ready",
                "updated_at": "2026-05-02T15:00:00Z",
            },
        }
    )


def test_start_episode_sends_reproducibility_fields(fix: StubFixture) -> None:
    episode = fix.client.start_episode(
        name="reproducibility test",
        source="sim",
        robot="test-rig",
        policy_version="v1.2.3",
        env_version="env-rev4",
        git_sha="abc1234",
        seed=8124,
        fps=30,
        metadata={"task": "pick"},
        artifacts=["video"],
    )
    assert episode.id == "ep_test_123"
    assert episode.status == "recording"
    assert episode.storage == "unconfigured"

    assert len(fix.requests) == 1
    req = fix.requests[0]
    assert req.method == "POST"
    assert req.url.path == "/api/ingest/episode"

    payload = json.loads(req.content)
    # Every reproducibility field per AGENTS.md must hit the wire
    # (otherwise the server can't render the detail page correctly).
    assert payload["policy_version"] == "v1.2.3"
    assert payload["env_version"] == "env-rev4"
    assert payload["git_sha"] == "abc1234"
    assert payload["seed"] == 8124
    # And the rest
    assert payload["name"] == "reproducibility test"
    assert payload["source"] == "sim"
    assert payload["robot"] == "test-rig"
    assert payload["fps"] == 30
    assert payload["metadata"] == {"task": "pick"}
    assert payload["request_uploads"] == ["video"]


def test_finalize_sends_minimal_payload(fix: StubFixture) -> None:
    episode = fix.client.start_episode(name="finalize test", artifacts=[])
    episode.finalize(status="ready", duration_s=2.5, fps=24, metadata={"ok": True})

    # 2 requests: create + finalize
    assert len(fix.requests) == 2
    finalize_req = fix.requests[1]
    assert finalize_req.method == "POST"
    assert finalize_req.url.path == "/api/ingest/episode/ep_test_123/finalize"

    payload = json.loads(finalize_req.content)
    assert payload["status"] == "ready"
    assert payload["duration_s"] == 2.5
    assert payload["fps"] == 24
    assert payload["metadata"] == {"ok": True}


def test_context_manager_marks_failed_on_exception(fix: StubFixture) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with fix.client.start_episode(name="ctx fail", artifacts=[]) as ep:
            assert ep.id == "ep_test_123"
            raise RuntimeError("boom")

    finalize_req = fix.requests[-1]
    payload = json.loads(finalize_req.content)
    assert payload["status"] == "failed"
    assert "RuntimeError" in payload["metadata"]["failure_reason"]


def test_auth_header_carries_bearer_token(fix: StubFixture) -> None:
    fix.client.start_episode(name="auth header", artifacts=[])
    assert fix.requests[0].headers["authorization"] == "Bearer rt_id123_secret456"
    assert "robotrace-python/" in fix.requests[0].headers["user-agent"]
