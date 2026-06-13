"""HDF5 adapter tests.

Synthesises two tiny imitation-learning files on disk - a robomimic
multi-demo file and an ALOHA-style single-episode file - with h5py, so
the test never needs robomimic, lerobot, or torch.

Coverage:

    test_classify_dataset_routes_into_slots   - pure-function classifier
                                                pins robomimic + ALOHA
                                                naming conventions.
    test_scan_robomimic / test_scan_aloha     - layout detection, demo
                                                enumeration, fps + camera
                                                discovery, clear failure
                                                on an unsupported file.
    test_encode_*                             - sensors/actions NPZ keys
                                                and shapes; video encoded
                                                when opencv is available.
    test_upload_episode_runs_full_ingest_flow - one-shot upload hits the
                                                ingest endpoints in order
                                                with merged metadata.
    test_upload_dataset_walks_demos           - bulk verb uploads every
                                                robomimic demo and reports
                                                progress.

Hard deps are guarded with `pytest.importorskip` so a `[dev]`-only CI
run (no `[hdf5]`) skips the whole file - same pattern as the LeRobot
and ROS 2 adapter tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

import robotrace as rt

pytest.importorskip("h5py")

import h5py

from robotrace.adapters import hdf5

NS_PER_S = 1_000_000_000


# ── synthetic fixtures ────────────────────────────────────────────────


@pytest.fixture
def robomimic_file(tmp_path: Path) -> Path:
    """A tiny robomimic file: two demos, eef-pos + image obs, actions."""
    path = tmp_path / "low_dim.hdf5"
    with h5py.File(str(path), "w") as f:
        data = f.create_group("data")
        data.attrs["total"] = 8
        data.attrs["env_args"] = json.dumps(
            {
                "env_name": "Lift",
                "env_kwargs": {"robots": ["Panda"], "control_freq": 20},
            }
        )
        for demo_idx, length in ((0, 5), (1, 3)):
            grp = data.create_group(f"demo_{demo_idx}")
            grp.attrs["num_samples"] = length
            grp.create_dataset("actions", data=np.zeros((length, 7), np.float32))
            grp.create_dataset("rewards", data=np.ones((length,), np.float32))
            dones = np.zeros((length,), np.int64)
            dones[-1] = 1
            grp.create_dataset("dones", data=dones)
            grp.create_dataset("states", data=np.zeros((length, 32), np.float32))
            obs = grp.create_group("obs")
            obs.create_dataset(
                "robot0_eef_pos", data=np.zeros((length, 3), np.float32)
            )
            obs.create_dataset(
                "agentview_image",
                data=np.zeros((length, 12, 16, 3), np.uint8),
            )
    return path


@pytest.fixture
def aloha_file(tmp_path: Path) -> Path:
    """A tiny ALOHA single-episode file: action + qpos + one camera."""
    path = tmp_path / "episode_0.hdf5"
    length = 4
    with h5py.File(str(path), "w") as f:
        f.attrs["sim"] = False
        f.create_dataset("action", data=np.zeros((length, 14), np.float32))
        obs = f.create_group("observations")
        obs.create_dataset("qpos", data=np.zeros((length, 14), np.float32))
        obs.create_dataset("qvel", data=np.zeros((length, 14), np.float32))
        images = obs.create_group("images")
        images.create_dataset("top", data=np.zeros((length, 12, 16, 3), np.uint8))
    return path


# ── classifier (pure) ─────────────────────────────────────────────────


def test_classify_dataset_routes_into_slots() -> None:
    by_name = {
        n: hdf5.classify_dataset(n)
        for n in [
            "action",
            "actions",
            "action_dict/gripper",
            "obs/robot0_eef_pos",
            "obs/agentview_image",
            "obs/robot0_eye_in_hand_image",
            "observations/qpos",
            "observations/images/top",
            "observations/images/wrist_cam",
            "states",
            "rewards",
            "dones",
            "success",
            "discount",
            "timestamp",
            "frame_index",
            "custom_user_field",
        ]
    }

    assert by_name["action"].slot == "actions"
    assert by_name["actions"].slot == "actions"
    assert by_name["action_dict/gripper"].slot == "actions"

    assert by_name["obs/robot0_eef_pos"].slot == "sensors"
    assert by_name["observations/qpos"].slot == "sensors"
    assert by_name["states"].slot == "sensors"

    assert by_name["obs/agentview_image"].slot == "video"
    assert by_name["obs/robot0_eye_in_hand_image"].slot == "video"
    assert by_name["observations/images/top"].slot == "video"
    assert by_name["observations/images/top"].reason == "segment"
    assert by_name["obs/agentview_image"].reason == "leaf"
    assert by_name["observations/images/wrist_cam"].slot == "video"

    for n in ("rewards", "dones", "success", "discount"):
        assert by_name[n].slot == "episode_meta"

    assert by_name["timestamp"].slot == "internal"
    assert by_name["frame_index"].slot == "internal"

    assert by_name["custom_user_field"].slot == "sensors"
    assert by_name["custom_user_field"].reason == "default"


# ── scan ──────────────────────────────────────────────────────────────


def test_scan_robomimic(robomimic_file: Path) -> None:
    summary = hdf5.scan_file(str(robomimic_file))

    assert summary.layout == "robomimic"
    assert summary.total_episodes == 2
    assert summary.episode(0).key == "demo_0"
    assert summary.episode(0).length == 5
    assert summary.episode(1).length == 3
    # control_freq from env_args → fps, not assumed.
    assert summary.fps == 20.0
    assert summary.fps_assumed is False
    assert summary.env_name == "Lift"
    assert summary.robot_type == "Panda"
    assert "obs/agentview_image" in summary.camera_names
    assert "obs/robot0_eef_pos" in summary.dataset_names

    report = summary.report()
    assert "robomimic" in report
    assert "Lift" in report


def test_scan_aloha(aloha_file: Path) -> None:
    summary = hdf5.scan_file(str(aloha_file), fps=50)

    assert summary.layout == "single"
    assert summary.total_episodes == 1
    assert summary.episode(0).key is None
    assert summary.episode(0).length == 4
    assert summary.fps == 50.0
    assert summary.fps_assumed is False
    assert summary.robot_type == "real"
    assert "observations/images/top" in summary.camera_names


def test_scan_assumes_fps_when_absent(aloha_file: Path) -> None:
    summary = hdf5.scan_file(str(aloha_file))
    assert summary.fps == 30.0
    assert summary.fps_assumed is True


def test_scan_rejects_unsupported_file(tmp_path: Path) -> None:
    path = tmp_path / "mystery.hdf5"
    with h5py.File(str(path), "w") as f:
        f.create_dataset("something", data=np.zeros((3, 3)))
    with pytest.raises(rt.ConfigurationError, match="imitation HDF5"):
        hdf5.scan_file(str(path))


# ── encode ────────────────────────────────────────────────────────────


def test_encode_robomimic_sensors_and_actions(
    robomimic_file: Path, tmp_path: Path
) -> None:
    out = tmp_path / "encoded_demo0"
    encoded = hdf5.encode_episode(str(robomimic_file), out, episode_index=0)

    assert encoded.fps == 20.0
    assert encoded.duration_s == pytest.approx(5 / 20.0)

    assert encoded.actions is not None
    actions = np.load(encoded.actions.path)
    assert "actions/value" in actions.files
    assert actions["actions/value"].shape == (5, 7)
    assert actions["actions/_t_ns"].dtype == np.int64
    assert actions["actions/_t_ns"][1] == int(NS_PER_S / 20.0)

    assert encoded.sensors is not None
    sensors = np.load(encoded.sensors.path)
    assert "obs/robot0_eef_pos/value" in sensors.files
    assert sensors["obs/robot0_eef_pos/value"].shape == (5, 3)
    # `states` routes to sensors (safe default) and is flattened to (T, K).
    assert "states/value" in sensors.files
    assert sensors["states/value"].shape == (5, 32)

    # rewards/dones roll into metadata, not actions.npz.
    outcome = encoded.metadata["hdf5_episode_outcome"]
    assert outcome["dones"] == 1
    assert outcome["reward_sum"] == pytest.approx(5.0)
    assert encoded.metadata["hdf5_layout"] == "robomimic"
    assert encoded.metadata["hdf5_episode_key"] == "demo_0"


def test_encode_aloha_sensors_and_actions(aloha_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "encoded_aloha"
    encoded = hdf5.encode_episode(str(aloha_file), out, fps=50)

    assert encoded.actions is not None
    actions = np.load(encoded.actions.path)
    assert "action/value" in actions.files
    assert actions["action/value"].shape == (4, 14)

    assert encoded.sensors is not None
    sensors = np.load(encoded.sensors.path)
    assert "observations/qpos/value" in sensors.files
    assert "observations/qvel/value" in sensors.files


def test_encode_video_when_opencv_available(
    aloha_file: Path, tmp_path: Path
) -> None:
    pytest.importorskip("cv2")
    out = tmp_path / "encoded_video"
    encoded = hdf5.encode_episode(str(aloha_file), out, fps=50)
    assert encoded.video is not None
    assert encoded.video.path.is_file()
    assert encoded.video.bytes_size > 0
    assert encoded.video.columns == ["top"]


def test_encode_canonical_camera_rejects_unknown(
    robomimic_file: Path, tmp_path: Path
) -> None:
    with pytest.raises(rt.ConfigurationError, match="not a camera"):
        hdf5.encode_episode(
            str(robomimic_file),
            tmp_path / "x",
            episode_index=0,
            canonical_camera="obs/nope_image",
        )


# ── upload ────────────────────────────────────────────────────────────


def test_upload_episode_runs_full_ingest_flow(
    aloha_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[httpx.Request] = []
    uploads: list[tuple[str, str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if request.url.path == "/api/ingest/episode" and request.method == "POST":
            payload = json.loads(request.content)
            urls = [
                {
                    "kind": kind,
                    "url": f"https://r2.test/{kind}?sig=x",
                    "expires_at": "2026-06-13T13:00:00Z",
                    "public_url": f"https://cdn.test/{kind}",
                }
                for kind in payload["request_uploads"]
            ]
            return httpx.Response(
                201,
                json={
                    "episode_id": "ep_hdf5_test",
                    "status": "recording",
                    "storage": "r2",
                    "upload_urls": urls,
                },
            )
        if request.url.path.endswith("/finalize"):
            return httpx.Response(
                200,
                json={
                    "episode_id": "ep_hdf5_test",
                    "status": "ready",
                    "updated_at": "2026-06-13T13:01:00Z",
                },
            )
        return httpx.Response(500, json={"error": f"unexpected {request.url}"})

    transport = httpx.MockTransport(handler)
    client = rt.Client(
        api_key="rt_test", base_url="https://example.test", transport=transport
    )

    def fake_upload_file(self: Any, url: str, path: Any, *, content_type: str) -> int:
        p = Path(path)
        uploads.append((url, content_type, p.name))
        return p.stat().st_size

    monkeypatch.setattr(
        client._http, "upload_file", fake_upload_file.__get__(client._http)
    )

    episode = hdf5.upload_episode(
        str(aloha_file),
        client=client,
        fps=50,
        policy_version="act-v1",
        env_version="aloha-cell-1",
        git_sha="abc1234",
        seed=7,
    )

    assert episode.id == "ep_hdf5_test"
    assert episode.status == "ready"

    methods_paths = [(r.method, r.url.path) for r in captured]
    assert methods_paths[0] == ("POST", "/api/ingest/episode")
    assert methods_paths[-1] == ("POST", "/api/ingest/episode/ep_hdf5_test/finalize")

    create_payload = json.loads(captured[0].content)
    assert create_payload["policy_version"] == "act-v1"
    assert create_payload["seed"] == 7
    assert create_payload["metadata"]["adapter"] == "hdf5"
    assert create_payload["metadata"]["hdf5_layout"] == "single"
    assert "sensors" in create_payload["request_uploads"]
    assert "actions" in create_payload["request_uploads"]

    finalize_payload = json.loads(captured[-1].content)
    assert finalize_payload["status"] == "ready"
    assert finalize_payload["fps"] == 50.0


def test_upload_dataset_walks_demos(
    robomimic_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    progress: list[tuple[int, int, bool]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ingest/episode" and request.method == "POST":
            payload = json.loads(request.content)
            return httpx.Response(
                201,
                json={
                    "episode_id": "ep_x",
                    "status": "recording",
                    "storage": "r2",
                    "upload_urls": [
                        {
                            "kind": k,
                            "url": f"https://r2.test/{k}",
                            "expires_at": "2026-06-13T13:00:00Z",
                            "public_url": f"https://cdn.test/{k}",
                        }
                        for k in payload["request_uploads"]
                    ],
                },
            )
        if request.url.path.endswith("/finalize"):
            return httpx.Response(
                200, json={"episode_id": "ep_x", "status": "ready"}
            )
        return httpx.Response(500, json={"error": "unexpected"})

    transport = httpx.MockTransport(handler)
    client = rt.Client(
        api_key="rt_test", base_url="https://example.test", transport=transport
    )
    monkeypatch.setattr(
        client._http,
        "upload_file",
        (lambda self, url, path, *, content_type: Path(path).stat().st_size).__get__(
            client._http
        ),
    )

    def on_progress(done: int, total: int, episode: Any, error: Any) -> None:
        progress.append((done, total, episode is not None))

    episodes = hdf5.upload_dataset(
        str(robomimic_file),
        client=client,
        policy_version="bc-v3",
        on_progress=on_progress,
    )

    assert len(episodes) == 2
    assert progress == [(1, 2, True), (2, 2, True)]
