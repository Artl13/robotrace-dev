"""Gymnasium adapter tests.

Uses CartPole-v1 for scan/NPZ tests (no pygame). Video tests use a
tiny fake env so CI does not need `gymnasium[classic-control]`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

import robotrace as rt
from robotrace.errors import ConfigurationError

pytest.importorskip("gymnasium")

import gymnasium as gym

from robotrace.adapters import gymnasium as rt_gym


@dataclass
class _FakeSpec:
    id: str = "FakeBox-v0"
    render_modes: tuple[str, ...] = ("rgb_array",)


class FakeBoxEnv:
    """Minimal Gymnasium-shaped env for video tests without pygame."""

    spec = _FakeSpec()
    render_mode = "rgb_array"

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(2)
        self._step = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._step = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action: int):
        self._step += 1
        obs = np.full(4, float(self._step), dtype=np.float32)
        terminated = self._step >= 8
        return obs, 1.0, terminated, False, {}

    def render(self):
        return np.full((48, 64, 3), self._step, dtype=np.uint8)

    def close(self) -> None:
        return None


@pytest.fixture
def cartpole_env():
    env = gym.make("CartPole-v1")
    try:
        yield env
    finally:
        env.close()


@pytest.fixture
def cartpole_env_no_render():
    env = gym.make("CartPole-v1")
    try:
        yield env
    finally:
        env.close()


@pytest.fixture
def fake_video_env():
    env = FakeBoxEnv()
    try:
        yield env
    finally:
        env.close()


def _always_right_policy(obs: Any, info: dict[str, Any]) -> int:
    return 1


def test_scan_env_reports_spaces(cartpole_env) -> None:
    summary = rt_gym.scan_env(cartpole_env)
    assert summary.env_id == "CartPole-v1"
    assert "Box" in summary.observation_space
    assert "Discrete" in summary.action_space
    report = summary.report()
    assert "CartPole-v1" in report


def test_scan_env_video_env(fake_video_env) -> None:
    summary = rt_gym.scan_env(fake_video_env)
    assert summary.can_record_video is True
    assert summary.active_render_mode == "rgb_array"
    assert "video: yes" in summary.report()


def test_encode_rollout_writes_npz(cartpole_env, tmp_path: Path) -> None:
    encoded = rt_gym.encode_rollout(
        cartpole_env,
        tmp_path / "out",
        policy=_always_right_policy,
        seed=0,
        max_steps=20,
        record_video=False,
    )

    assert encoded.metadata["steps"] > 0
    assert encoded.sensors is not None
    assert encoded.actions is not None

    sensors = np.load(encoded.sensors.path)
    actions = np.load(encoded.actions.path)

    assert "observation/value" in sensors.files
    assert sensors["observation/value"].dtype == np.float32
    assert sensors["observation/_t_ns"].dtype == np.int64
    assert sensors["observation/value"].shape[0] == encoded.metadata["steps"]

    assert "action/value" in actions.files
    assert actions["action/value"].dtype == np.float32
    assert actions["action/value"].shape[0] == encoded.metadata["steps"]


def test_encode_rollout_writes_video(fake_video_env, tmp_path: Path) -> None:
    pytest.importorskip("cv2")

    encoded = rt_gym.encode_rollout(
        fake_video_env,
        tmp_path / "out",
        policy=_always_right_policy,
        seed=0,
        max_steps=8,
        record_video=True,
        fps=30.0,
    )

    assert encoded.video is not None
    assert encoded.video.path.name == "video.mp4"
    assert encoded.video.path.stat().st_size > 0


def test_encode_rollout_requires_render_mode_for_video(
    cartpole_env_no_render, tmp_path: Path
) -> None:
    with pytest.raises(ConfigurationError, match="render_mode='rgb_array'"):
        rt_gym.encode_rollout(
            cartpole_env_no_render,
            tmp_path / "out",
            policy=_always_right_policy,
            seed=0,
            max_steps=5,
            record_video=True,
        )


def test_upload_rollout_runs_full_ingest_flow(
    fake_video_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("cv2")

    captured_json: list[httpx.Request] = []
    uploads: list[tuple[str, str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_json.append(request)
        if request.url.path == "/api/ingest/episode" and request.method == "POST":
            return httpx.Response(
                201,
                json={
                    "episode_id": "ep_gym_test",
                    "status": "recording",
                    "storage": "r2",
                    "upload_urls": [
                        {
                            "kind": "video",
                            "url": "https://r2.test/video?sig=v",
                            "expires_at": "2026-05-08T13:00:00Z",
                            "public_url": "https://cdn.test/video.mp4",
                        },
                        {
                            "kind": "sensors",
                            "url": "https://r2.test/sensors?sig=s",
                            "expires_at": "2026-05-08T13:00:00Z",
                            "public_url": "https://cdn.test/sensors.npz",
                        },
                        {
                            "kind": "actions",
                            "url": "https://r2.test/actions?sig=a",
                            "expires_at": "2026-05-08T13:00:00Z",
                            "public_url": "https://cdn.test/actions.npz",
                        },
                    ],
                },
            )
        if request.url.path.endswith("/finalize"):
            return httpx.Response(
                200,
                json={
                    "episode_id": "ep_gym_test",
                    "status": "ready",
                    "updated_at": "2026-05-08T13:01:00Z",
                },
            )
        return httpx.Response(500, json={"error": f"unexpected {request.method} {request.url}"})

    transport = httpx.MockTransport(handler)
    client = rt.Client(
        api_key="rt_test", base_url="https://example.test", transport=transport
    )

    def fake_upload_file(self: Any, url: str, path: Any, *, content_type: str) -> int:
        p = Path(path)
        size = p.stat().st_size
        uploads.append((url, content_type, p.name))
        return size

    monkeypatch.setattr(
        client._http, "upload_file", fake_upload_file.__get__(client._http)
    )

    episode = rt_gym.upload_rollout(
        fake_video_env,
        policy=_always_right_policy,
        client=client,
        name="fake box smoke",
        policy_version="fake-v1",
        env_version="fake-v1",
        git_sha="abc1234",
        seed=42,
        max_steps=8,
        record_video=True,
    )

    assert episode.id == "ep_gym_test"
    assert len(captured_json) == 2
    create_payload = json.loads(captured_json[0].content)
    assert create_payload["source"] == "sim"
    assert create_payload["metadata"]["adapter"] == "gymnasium"
    assert create_payload["metadata"]["gymnasium_env_id"] == "FakeBox-v0"
    assert len(uploads) == 3
    uploaded_names = {name for _, _, name in uploads}
    assert uploaded_names == {"video.mp4", "sensors.npz", "actions.npz"}

    finalize_payload = json.loads(captured_json[-1].content)
    assert finalize_payload["duration_s"] > 0
    assert finalize_payload["fps"] == 30.0
