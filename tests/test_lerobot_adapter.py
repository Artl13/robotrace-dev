"""LeRobot adapter tests.

Synthesises tiny LeRobot datasets on disk - both the v2.1
one-file-per-episode layout and the v3.0 multi-episode-shard layout -
with two episodes, two cameras, an `observation.state`, an `action`
vector, and a few `next.*` outcome columns. Uses pyarrow to write the
parquet shards and opencv (when available) to write the mp4s - so the
test never talks to the HF Hub and never needs the heavy `lerobot`
package.

Coverage:

    test_classify_columns_routes_into_slots     - pure-function classifier
                                                   pins LeRobot's column
                                                   conventions.
    test_scan_dataset_reads_meta                - v2.1 info.json,
                                                   episodes.jsonl, tasks.jsonl
                                                   all parsed.
    test_scan_dataset_reads_v3_meta             - v3.0 info.json +
                                                   meta/episodes/*.parquet
                                                   parsed into locators.
    test_scan_dataset_rejects_*                 - unknown / future versions
                                                   fail clearly.
    test_encode_episode_writes_artifacts        - encode produces video.mp4 +
                                                   sensors.npz + actions.npz
                                                   with the expected keys
                                                   and shapes (v2.1 + v3.0).
    test_encode_episode_v3_*                    - v3.0 slices the shared data
                                                   shard to one episode and
                                                   trims each camera clip out
                                                   of the shared mp4.
    test_encode_episode_canonical_camera        - picking one camera skips
                                                   the multi-cam tile.
    test_upload_episode_uses_start_episode      - one-shot upload hits the
                                                   ingest endpoints in the
                                                   right order with the
                                                   right per-slot URLs and
                                                   merged metadata.
    test_upload_dataset_walks_episodes          - bulk verb uploads each
                                                   trajectory and surfaces
                                                   per-episode progress
                                                   through the on_progress
                                                   callback.

Modules guard their hard deps via `pytest.importorskip` so a CI run
with only `[dev]` installed (no `[lerobot]`) skips the whole file
gracefully - same pattern as the ROS 2 adapter tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

import robotrace as rt

# Guard the heavy deps. Without them, the adapter module can't be
# imported either - `pytest.importorskip` is the right tool. (cv2 is
# only needed for multi-camera tiling; we still want the single-camera
# tests to run without it, which we handle via a per-test skip.)
pytest.importorskip("pyarrow")
pytest.importorskip("huggingface_hub")

import pyarrow as pa
import pyarrow.parquet as pq

from robotrace.adapters import lerobot

NS_PER_S = 1_000_000_000


# ── synthetic v2.1 dataset fixture ────────────────────────────────────


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    """Build a tiny LeRobot v2.1 dataset.

    Layout:

        my_dataset/
        ├── meta/
        │   ├── info.json
        │   ├── episodes.jsonl
        │   └── tasks.jsonl
        ├── data/
        │   └── chunk-000/
        │       ├── episode_000000.parquet  (5 frames @ 10 fps, 6-DoF state, 3-DoF action)
        │       └── episode_000001.parquet  (3 frames @ 10 fps)
        └── videos/
            ├── observation.images.cam_a/
            │   └── chunk-000/
            │       ├── episode_000000.mp4
            │       └── episode_000001.mp4
            └── observation.images.cam_b/
                └── chunk-000/
                    ├── episode_000000.mp4
                    └── episode_000001.mp4

    The mp4 files are minimal - just a few frames of a deterministic
    color so opencv can decode them. We only generate cam_b mp4s if
    cv2 is importable (i.e. on the CI matrix that has the [video]
    extra installed); otherwise we skip the multi-camera tests.
    """
    root = tmp_path / "synthetic_lerobot"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "videos" / "observation.images.cam_a" / "chunk-000").mkdir(parents=True)
    (root / "videos" / "observation.images.cam_b" / "chunk-000").mkdir(parents=True)

    # info.json
    info = {
        "codebase_version": "v2.1",
        "robot_type": "synthetic",
        "fps": 10,
        "total_episodes": 2,
        "total_frames": 8,
        "chunks_size": 1000,
        "features": {
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "observation.images.cam_a": {"dtype": "video", "shape": [3, 24, 32]},
            "observation.images.cam_b": {"dtype": "video", "shape": [3, 24, 32]},
            "observation.state": {"dtype": "float32", "shape": [6]},
            "action": {"dtype": "float32", "shape": [3]},
            "next.reward": {"dtype": "float32", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))

    # episodes.jsonl + tasks.jsonl
    with (root / "meta" / "episodes.jsonl").open("w") as f:
        f.write(json.dumps({"episode_index": 0, "tasks": ["pick up cup"], "length": 5}) + "\n")
        f.write(json.dumps({"episode_index": 1, "tasks": ["place cup"], "length": 3}) + "\n")
    with (root / "meta" / "tasks.jsonl").open("w") as f:
        f.write(json.dumps({"task_index": 0, "task": "pick up cup"}) + "\n")
        f.write(json.dumps({"task_index": 1, "task": "place cup"}) + "\n")

    _write_episode_parquet(
        root, episode_index=0, length=5, fps=10.0, task_index=0, reward_pattern=0.1
    )
    _write_episode_parquet(
        root, episode_index=1, length=3, fps=10.0, task_index=1, reward_pattern=0.2
    )

    _maybe_write_videos(root, episode_index=0, length=5)
    _maybe_write_videos(root, episode_index=1, length=3)

    return root


def _write_episode_parquet(
    root: Path,
    *,
    episode_index: int,
    length: int,
    fps: float,
    task_index: int,
    reward_pattern: float,
) -> None:
    """Write one episode's parquet shard."""
    timestamps = np.arange(length, dtype=np.float32) / fps
    frame_indices = np.arange(length, dtype=np.int64)
    # `index` is the global frame index across the whole dataset.
    base = 0 if episode_index == 0 else 5
    indices = np.arange(length, dtype=np.int64) + base

    state = np.tile(
        np.linspace(0.0, 1.0, 6, dtype=np.float32), (length, 1)
    ) + episode_index * 0.01
    action = np.tile(
        np.array([0.1, 0.0, -0.05], dtype=np.float32), (length, 1)
    ) * (episode_index + 1)
    rewards = np.full(length, reward_pattern, dtype=np.float32)
    done = np.zeros(length, dtype=bool)
    done[-1] = True

    table = pa.table(
        {
            "timestamp": pa.array(timestamps, type=pa.float32()),
            "frame_index": pa.array(frame_indices, type=pa.int64()),
            "episode_index": pa.array(
                np.full(length, episode_index, dtype=np.int64), type=pa.int64()
            ),
            "index": pa.array(indices, type=pa.int64()),
            "task_index": pa.array(
                np.full(length, task_index, dtype=np.int64), type=pa.int64()
            ),
            "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32())),
            "action": pa.array(action.tolist(), type=pa.list_(pa.float32())),
            "next.reward": pa.array(rewards, type=pa.float32()),
            "next.done": pa.array(done, type=pa.bool_()),
        }
    )
    pq.write_table(
        table,
        root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet",
    )


def _maybe_write_videos(root: Path, *, episode_index: int, length: int) -> None:
    """Write a tiny mp4 per camera if cv2 is importable.

    We use opencv even though `_encode_video` for the single-camera
    case only does `shutil.copyfile`. The point is that *some* readable
    mp4 has to exist on disk so the copy is meaningful - and writing a
    real one with cv2 is the cleanest option that doesn't bake a
    binary fixture into the repo.
    """
    try:
        import cv2
    except ImportError:
        return

    for cam in ("cam_a", "cam_b"):
        path = (
            root
            / "videos"
            / f"observation.images.{cam}"
            / "chunk-000"
            / f"episode_{episode_index:06d}.mp4"
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # cam_a slightly taller than cam_b so the multi-cam tile has
        # to do its height-padding code path.
        h = 24 if cam == "cam_a" else 16
        writer = cv2.VideoWriter(str(path), fourcc, 10.0, (32, h))
        try:
            for i in range(length):
                pixel = np.full((h, 32, 3), 16 * (i + 1), dtype=np.uint8)
                writer.write(pixel)
        finally:
            writer.release()


# ── synthetic v3.0 dataset fixture ────────────────────────────────────

# Episode lengths for the v3 fixture; episode 0 then episode 1 are
# concatenated into a single data parquet shard and a single mp4 shard
# per camera. fps=10, so episode 0 occupies [0.0, 0.5) of the shared
# mp4 and episode 1 occupies [0.5, 0.8).
_V3_EP_LENGTHS = [5, 3]
_V3_FPS = 10.0


@pytest.fixture
def synthetic_v3_dataset(tmp_path: Path) -> Path:
    """Build a tiny LeRobot v3.0 dataset (multi-episode shards).

    Layout:

        my_v3/
        ├── meta/
        │   ├── info.json                       (data_path / video_path templates)
        │   ├── tasks.parquet
        │   └── episodes/chunk-000/file-000.parquet   (2 episode rows + locators)
        ├── data/
        │   └── chunk-000/
        │       └── file-000.parquet            (both episodes concatenated)
        └── videos/
            ├── observation.images.cam_a/chunk-000/file-000.mp4   (8 frames)
            └── observation.images.cam_b/chunk-000/file-000.mp4   (8 frames)
    """
    root = tmp_path / "synthetic_lerobot_v3"
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    cameras = ["observation.images.cam_a", "observation.images.cam_b"]
    for cam in cameras:
        (root / "videos" / cam / "chunk-000").mkdir(parents=True)

    total_frames = sum(_V3_EP_LENGTHS)
    info = {
        "codebase_version": "v3.0",
        "robot_type": "synthetic",
        "fps": int(_V3_FPS),
        "total_episodes": len(_V3_EP_LENGTHS),
        "total_frames": total_frames,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 500,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "observation.images.cam_a": {"dtype": "video", "shape": [24, 32, 3]},
            "observation.images.cam_b": {"dtype": "video", "shape": [16, 32, 3]},
            "observation.state": {"dtype": "float32", "shape": [6]},
            "action": {"dtype": "float32", "shape": [3]},
            "next.reward": {"dtype": "float32", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))

    _write_v3_tasks(root)
    _write_v3_data_shard(root)
    _write_v3_episodes_meta(root, cameras)
    _maybe_write_v3_video_shards(root, cameras, total_frames)
    return root


def _write_v3_tasks(root: Path) -> None:
    table = pa.table(
        {
            "task_index": pa.array([0, 1], type=pa.int64()),
            "task": pa.array(["pick up cup", "place cup"], type=pa.string()),
        }
    )
    pq.write_table(table, root / "meta" / "tasks.parquet")


def _write_v3_data_shard(root: Path) -> None:
    """One parquet holding both episodes, concatenated, with the columns a
    real v3 data shard carries (timestamp resets per episode; index is global)."""
    timestamps: list[float] = []
    frame_index: list[int] = []
    episode_index: list[int] = []
    index: list[int] = []
    task_index: list[int] = []
    state: list[list[float]] = []
    action: list[list[float]] = []
    reward: list[float] = []
    done: list[bool] = []

    global_i = 0
    for ep, length in enumerate(_V3_EP_LENGTHS):
        for f in range(length):
            timestamps.append(f / _V3_FPS)  # per-episode reset
            frame_index.append(f)
            episode_index.append(ep)
            index.append(global_i)
            task_index.append(ep)
            state.append([float(ep) + 0.1 * k for k in range(6)])
            action.append([0.1 * (ep + 1), 0.0, -0.05 * (ep + 1)])
            reward.append(0.1 * (ep + 1))
            done.append(f == length - 1)
            global_i += 1

    table = pa.table(
        {
            "timestamp": pa.array(timestamps, type=pa.float32()),
            "frame_index": pa.array(frame_index, type=pa.int64()),
            "episode_index": pa.array(episode_index, type=pa.int64()),
            "index": pa.array(index, type=pa.int64()),
            "task_index": pa.array(task_index, type=pa.int64()),
            "observation.state": pa.array(state, type=pa.list_(pa.float32())),
            "action": pa.array(action, type=pa.list_(pa.float32())),
            "next.reward": pa.array(reward, type=pa.float32()),
            "next.done": pa.array(done, type=pa.bool_()),
        }
    )
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")


def _write_v3_episodes_meta(root: Path, cameras: list[str]) -> None:
    """One episode-metadata parquet row per episode, with data + video locators."""
    rows: dict[str, list[Any]] = {
        "episode_index": [],
        "tasks": [],
        "length": [],
        "data/chunk_index": [],
        "data/file_index": [],
        "dataset_from_index": [],
        "dataset_to_index": [],
    }
    for cam in cameras:
        rows[f"videos/{cam}/chunk_index"] = []
        rows[f"videos/{cam}/file_index"] = []
        rows[f"videos/{cam}/from_timestamp"] = []
        rows[f"videos/{cam}/to_timestamp"] = []

    cumulative = 0.0
    global_from = 0
    task_names = ["pick up cup", "place cup"]
    for ep, length in enumerate(_V3_EP_LENGTHS):
        rows["episode_index"].append(ep)
        rows["tasks"].append([task_names[ep]])
        rows["length"].append(length)
        rows["data/chunk_index"].append(0)
        rows["data/file_index"].append(0)
        rows["dataset_from_index"].append(global_from)
        rows["dataset_to_index"].append(global_from + length)
        duration = length / _V3_FPS
        for cam in cameras:
            rows[f"videos/{cam}/chunk_index"].append(0)
            rows[f"videos/{cam}/file_index"].append(0)
            rows[f"videos/{cam}/from_timestamp"].append(cumulative)
            rows[f"videos/{cam}/to_timestamp"].append(cumulative + duration)
        cumulative += duration
        global_from += length

    table = pa.table(
        {
            "episode_index": pa.array(rows["episode_index"], type=pa.int64()),
            "tasks": pa.array(rows["tasks"], type=pa.list_(pa.string())),
            "length": pa.array(rows["length"], type=pa.int64()),
            "data/chunk_index": pa.array(rows["data/chunk_index"], type=pa.int64()),
            "data/file_index": pa.array(rows["data/file_index"], type=pa.int64()),
            "dataset_from_index": pa.array(
                rows["dataset_from_index"], type=pa.int64()
            ),
            "dataset_to_index": pa.array(rows["dataset_to_index"], type=pa.int64()),
            **{
                k: pa.array(
                    v,
                    type=pa.float64()
                    if k.endswith("timestamp")
                    else pa.int64(),
                )
                for k, v in rows.items()
                if k.startswith("videos/")
            },
        }
    )
    pq.write_table(
        table, root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )


def _maybe_write_v3_video_shards(
    root: Path, cameras: list[str], total_frames: int
) -> None:
    """Write one shared mp4 per camera holding ALL episodes' frames."""
    try:
        import cv2
    except ImportError:
        return
    for cam in cameras:
        path = root / "videos" / cam / "chunk-000" / "file-000.mp4"
        h = 24 if cam.endswith("cam_a") else 16
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, _V3_FPS, (32, h))
        try:
            for i in range(total_frames):
                pixel = np.full((h, 32, 3), 16 * (i + 1) % 256, dtype=np.uint8)
                writer.write(pixel)
        finally:
            writer.release()


# ── classifier ───────────────────────────────────────────────────────


def test_classify_columns_routes_into_slots() -> None:
    """Pure-function classifier - pins LeRobot's column conventions."""
    by_column = {
        col: lerobot.classify_column(col)
        for col in [
            "observation.images.laptop",
            "observation.images.phone",
            "observation.state",
            "observation.environment_state",
            "observation.imu.angular_velocity",
            "action",
            "action.gripper",
            "next.reward",
            "next.done",
            "next.success",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
            "custom_user_field",
        ]
    }

    assert by_column["observation.images.laptop"].slot == "video"
    assert by_column["observation.images.phone"].slot == "video"
    assert by_column["observation.images.laptop"].reason == "prefix"

    assert by_column["observation.state"].slot == "sensors"
    assert by_column["observation.environment_state"].slot == "sensors"
    assert by_column["observation.imu.angular_velocity"].slot == "sensors"
    assert by_column["observation.state"].reason == "prefix"

    assert by_column["action"].slot == "actions"
    assert by_column["action.gripper"].slot == "actions"

    assert by_column["next.reward"].slot == "episode_meta"
    assert by_column["next.done"].slot == "episode_meta"
    assert by_column["next.success"].slot == "episode_meta"

    for c in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        assert by_column[c].slot == "internal"
        assert by_column[c].reason == "exact"

    # Unknown columns fall through to sensors (safe default).
    assert by_column["custom_user_field"].slot == "sensors"
    assert by_column["custom_user_field"].reason == "default"


# ── scan_dataset ─────────────────────────────────────────────────────


def test_scan_dataset_reads_meta(synthetic_dataset: Path) -> None:
    summary = lerobot.scan_dataset(str(synthetic_dataset))

    assert summary.is_local is True
    assert summary.codebase_version == "v2.1"
    assert summary.fps == 10.0
    assert summary.total_episodes == 2
    assert summary.total_frames == 8
    assert "observation.images.cam_a" in summary.camera_keys
    assert "observation.images.cam_b" in summary.camera_keys
    assert {ep.episode_index for ep in summary.episodes} == {0, 1}
    assert summary.episode(0).length == 5
    assert summary.episode(0).tasks == ["pick up cup"]

    report = summary.report()
    assert "v2.1" in report
    assert "10 fps" in report or "10 fps".replace(" ", "") in report.replace(" ", "")


def test_scan_dataset_rejects_future_version(tmp_path: Path) -> None:
    """A v4+ codebase_version is newer than we know how to read - fail clearly."""
    root = tmp_path / "v4_dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v4.0", "fps": 30, "total_episodes": 0})
    )
    with pytest.raises(rt.ConfigurationError, match=r"newer than"):
        lerobot.scan_dataset(str(root))


def test_scan_dataset_rejects_unknown_version(tmp_path: Path) -> None:
    root = tmp_path / "weird_dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "potato", "fps": 30, "total_episodes": 0})
    )
    with pytest.raises(rt.ConfigurationError, match=r"doesn't recognise"):
        lerobot.scan_dataset(str(root))


def test_scan_dataset_rejects_missing_meta(tmp_path: Path) -> None:
    root = tmp_path / "broken"
    root.mkdir()
    with pytest.raises(rt.ConfigurationError, match="meta/"):
        lerobot.scan_dataset(str(root))


# ── encode_episode ───────────────────────────────────────────────────


def test_encode_episode_writes_sensors_and_actions(
    synthetic_dataset: Path, tmp_path: Path
) -> None:
    """Sensors + actions NPZ produced even without cv2 (single-cam path)."""
    out = tmp_path / "encoded_ep0"
    encoded = lerobot.encode_episode(
        str(synthetic_dataset),
        episode_index=0,
        output_dir=out,
        canonical_camera="observation.images.cam_a",
    )

    assert encoded.sensors is not None
    assert encoded.actions is not None
    assert encoded.fps == 10.0
    assert encoded.duration_s == 0.5  # 5 frames / 10 fps

    sensors = np.load(encoded.sensors.path)
    assert "observation.state/_t_ns" in sensors.files
    assert "observation.state/value" in sensors.files
    assert sensors["observation.state/value"].shape == (5, 6)
    assert sensors["observation.state/_t_ns"].dtype == np.int64

    actions = np.load(encoded.actions.path)
    assert "action/value" in actions.files
    assert actions["action/value"].shape == (5, 3)

    # Episode outcome rolls into metadata, not actions.npz.
    assert "lerobot_episode_outcome" in encoded.metadata
    outcome = encoded.metadata["lerobot_episode_outcome"]
    assert "next.done" in outcome
    assert outcome["next.done"] is True
    assert "next.reward_sum" in outcome
    assert outcome["next.reward_sum"] == pytest.approx(0.5, rel=1e-3)


def test_encode_episode_single_camera_passthrough(
    synthetic_dataset: Path, tmp_path: Path
) -> None:
    """Single camera → mp4 is copied byte-for-byte from the source."""
    pytest.importorskip("cv2")  # we wrote the mp4 with cv2; need it to exist
    out = tmp_path / "encoded_solo"
    encoded = lerobot.encode_episode(
        str(synthetic_dataset),
        episode_index=0,
        output_dir=out,
        canonical_camera="observation.images.cam_a",
    )
    assert encoded.video is not None
    assert encoded.video.path.is_file()
    assert encoded.video.columns == ["observation.images.cam_a"]
    # Source mp4 size on disk → copied identically.
    src = (
        synthetic_dataset
        / "videos"
        / "observation.images.cam_a"
        / "chunk-000"
        / "episode_000000.mp4"
    )
    assert encoded.video.bytes_size == src.stat().st_size


def test_encode_episode_multi_camera_tiles(
    synthetic_dataset: Path, tmp_path: Path
) -> None:
    """Multi-camera tile produces a single mp4 spanning both cams horizontally."""
    pytest.importorskip("cv2")
    out = tmp_path / "encoded_tiled"
    encoded = lerobot.encode_episode(
        str(synthetic_dataset), episode_index=0, output_dir=out
    )
    assert encoded.video is not None
    assert encoded.video.columns == [
        "observation.images.cam_a",
        "observation.images.cam_b",
    ]
    # Tiled output exists and has a non-zero size; full pixel
    # validation belongs in an opencv-internal test, not here.
    assert encoded.video.bytes_size > 0


# ── v3.0 scan + encode ───────────────────────────────────────────────


def _count_frames(path: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        n = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            n += 1
        return n
    finally:
        cap.release()


def test_scan_dataset_reads_v3_meta(synthetic_v3_dataset: Path) -> None:
    summary = lerobot.scan_dataset(str(synthetic_v3_dataset))

    assert summary.is_local is True
    assert summary.is_v3 is True
    assert summary.codebase_version == "v3.0"
    assert summary.fps == 10.0
    assert summary.total_episodes == 2
    assert summary.total_frames == 8
    assert summary.data_path is not None and "file-" in summary.data_path
    assert {"observation.images.cam_a", "observation.images.cam_b"} <= set(
        summary.camera_keys
    )

    assert {ep.episode_index for ep in summary.episodes} == {0, 1}
    ep1 = summary.episode(1)
    assert ep1.length == 3
    assert ep1.tasks == ["place cup"]
    # Data locator: both episodes share chunk 0 / file 0.
    assert ep1.data_chunk_index == 0
    assert ep1.data_file_index == 0
    assert ep1.dataset_from_index == 5
    assert ep1.dataset_to_index == 8
    # Video locator: episode 1 occupies [0.5, 0.8) of the shared mp4.
    loc = ep1.video_locators["observation.images.cam_a"]
    assert loc.chunk_index == 0
    assert loc.file_index == 0
    assert loc.from_timestamp == pytest.approx(0.5)
    assert loc.to_timestamp == pytest.approx(0.8)


def test_encode_episode_v3_slices_data_shard(
    synthetic_v3_dataset: Path, tmp_path: Path
) -> None:
    """v3 encode pulls just episode 1's rows out of the shared data shard."""
    out = tmp_path / "v3_ep1"
    encoded = lerobot.encode_episode(
        str(synthetic_v3_dataset),
        episode_index=1,
        output_dir=out,
        canonical_camera="observation.images.cam_a",
    )

    assert encoded.fps == 10.0
    assert encoded.duration_s == pytest.approx(0.3)  # 3 frames / 10 fps

    assert encoded.sensors is not None
    sensors = np.load(encoded.sensors.path)
    # Only episode 1's 3 rows, not all 8 in the shard.
    assert sensors["observation.state/value"].shape == (3, 6)
    # episode 1 state = [1.0, 1.1, ... ] (ep offset 1.0)
    assert sensors["observation.state/value"][0][0] == pytest.approx(1.0)
    # timestamps reset to per-episode 0, 0.1, 0.2 → ns (float32 source, so
    # the 0.1s frame lands within a few ns of 1e8).
    assert sensors["observation.state/_t_ns"][0] == 0
    assert sensors["observation.state/_t_ns"][1] == pytest.approx(
        100_000_000, abs=1000
    )

    assert encoded.actions is not None
    actions = np.load(encoded.actions.path)
    assert actions["action/value"].shape == (3, 3)

    outcome = encoded.metadata["lerobot_episode_outcome"]
    assert outcome["next.done"] is True
    assert encoded.metadata["lerobot_codebase_version"] == "v3.0"
    assert encoded.metadata["lerobot_episode_index"] == 1


def test_encode_episode_v3_trims_single_camera(
    synthetic_v3_dataset: Path, tmp_path: Path
) -> None:
    """v3 single camera → episode is trimmed out of the shared mp4 shard."""
    pytest.importorskip("cv2")
    out = tmp_path / "v3_trim"
    encoded = lerobot.encode_episode(
        str(synthetic_v3_dataset),
        episode_index=1,
        output_dir=out,
        canonical_camera="observation.images.cam_a",
    )
    assert encoded.video is not None
    assert encoded.video.columns == ["observation.images.cam_a"]
    # Episode 1 is 3 frames - the trim must NOT include all 8 shard frames.
    assert _count_frames(encoded.video.path) == 3


def test_encode_episode_v3_multi_camera_tiles(
    synthetic_v3_dataset: Path, tmp_path: Path
) -> None:
    """v3 multi-camera → both cams trimmed to the episode, then tiled."""
    pytest.importorskip("cv2")
    out = tmp_path / "v3_tile"
    encoded = lerobot.encode_episode(
        str(synthetic_v3_dataset), episode_index=0, output_dir=out
    )
    assert encoded.video is not None
    assert encoded.video.columns == [
        "observation.images.cam_a",
        "observation.images.cam_b",
    ]
    # Episode 0 is 5 frames; the tiled clip carries exactly those.
    assert _count_frames(encoded.video.path) == 5


# ── upload_episode (full ingest flow) ────────────────────────────────


def test_upload_episode_runs_full_ingest_flow(
    synthetic_dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """upload_episode opens an episode, uploads three artifacts, finalizes."""
    pytest.importorskip("cv2")
    captured: list[httpx.Request] = []
    uploads: list[tuple[str, str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if request.url.path == "/api/ingest/episode" and request.method == "POST":
            return httpx.Response(
                201,
                json={
                    "episode_id": "ep_lerobot_test",
                    "status": "recording",
                    "storage": "r2",
                    "upload_urls": [
                        {
                            "kind": "video",
                            "url": "https://r2.test/video?sig=v",
                            "expires_at": "2026-05-10T13:00:00Z",
                            "public_url": "https://cdn.test/video.mp4",
                        },
                        {
                            "kind": "sensors",
                            "url": "https://r2.test/sensors?sig=s",
                            "expires_at": "2026-05-10T13:00:00Z",
                            "public_url": "https://cdn.test/sensors.npz",
                        },
                        {
                            "kind": "actions",
                            "url": "https://r2.test/actions?sig=a",
                            "expires_at": "2026-05-10T13:00:00Z",
                            "public_url": "https://cdn.test/actions.npz",
                        },
                    ],
                },
            )
        if request.url.path.endswith("/finalize"):
            return httpx.Response(
                200,
                json={
                    "episode_id": "ep_lerobot_test",
                    "status": "ready",
                    "updated_at": "2026-05-10T13:01:00Z",
                },
            )
        return httpx.Response(
            500, json={"error": f"unexpected {request.method} {request.url}"}
        )

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

    episode = lerobot.upload_episode(
        str(synthetic_dataset),
        episode_index=0,
        client=client,
        canonical_camera="observation.images.cam_a",
        policy_version="aloha-v1",
        env_version="aloha-cell-1",
        git_sha="abc1234",
        seed=42,
    )

    assert episode.id == "ep_lerobot_test"
    assert episode.status == "ready"

    methods_paths = [(r.method, r.url.path) for r in captured]
    assert methods_paths[0] == ("POST", "/api/ingest/episode")
    assert methods_paths[-1] == ("POST", "/api/ingest/episode/ep_lerobot_test/finalize")

    create_payload = json.loads(captured[0].content)
    assert create_payload["policy_version"] == "aloha-v1"
    assert create_payload["env_version"] == "aloha-cell-1"
    assert create_payload["git_sha"] == "abc1234"
    assert create_payload["seed"] == 42
    assert create_payload["request_uploads"] == ["video", "sensors", "actions"]
    assert create_payload["metadata"]["adapter"] == "lerobot"
    assert create_payload["metadata"]["lerobot_episode_index"] == 0
    assert create_payload["metadata"]["lerobot_codebase_version"] == "v2.1"
    assert create_payload["metadata"]["lerobot_tasks"] == ["pick up cup"]
    assert "fps" in create_payload

    upload_by_url = {url: (ct, name) for url, ct, name in uploads}
    assert upload_by_url["https://r2.test/video?sig=v"] == ("video/mp4", "video.mp4")
    assert upload_by_url["https://r2.test/sensors?sig=s"] == (
        "application/octet-stream",
        "sensors.npz",
    )
    assert upload_by_url["https://r2.test/actions?sig=a"] == (
        "application/octet-stream",
        "actions.npz",
    )

    finalize_payload = json.loads(captured[-1].content)
    assert finalize_payload["status"] == "ready"
    assert finalize_payload["fps"] == 10.0
    assert finalize_payload["duration_s"] == pytest.approx(0.5, rel=1e-3)


def test_upload_dataset_walks_episodes(
    synthetic_dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bulk verb uploads each trajectory and reports per-episode progress."""
    pytest.importorskip("cv2")
    progress: list[tuple[int, int, str | None, type | None]] = []
    episode_counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ingest/episode" and request.method == "POST":
            episode_counter["n"] += 1
            ep_id = f"ep_{episode_counter['n']}"
            return httpx.Response(
                201,
                json={
                    "episode_id": ep_id,
                    "status": "recording",
                    "storage": "r2",
                    "upload_urls": [
                        {
                            "kind": "video",
                            "url": f"https://r2.test/{ep_id}/video?s=v",
                            "expires_at": "2026-05-10T13:00:00Z",
                        },
                        {
                            "kind": "sensors",
                            "url": f"https://r2.test/{ep_id}/sensors?s=s",
                            "expires_at": "2026-05-10T13:00:00Z",
                        },
                        {
                            "kind": "actions",
                            "url": f"https://r2.test/{ep_id}/actions?s=a",
                            "expires_at": "2026-05-10T13:00:00Z",
                        },
                    ],
                },
            )
        if request.url.path.endswith("/finalize"):
            ep_id = request.url.path.split("/")[-2]
            return httpx.Response(
                200,
                json={
                    "episode_id": ep_id,
                    "status": "ready",
                    "updated_at": "2026-05-10T13:01:00Z",
                },
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

    def cb(done: int, total: int, episode: Any, error: Any) -> None:
        progress.append(
            (done, total, episode.id if episode else None, type(error) if error else None)
        )

    episodes = lerobot.upload_dataset(
        str(synthetic_dataset),
        client=client,
        canonical_camera="observation.images.cam_a",
        policy_version="aloha-v1",
        env_version="aloha-cell-1",
        on_progress=cb,
    )

    assert [e.id for e in episodes] == ["ep_1", "ep_2"]
    assert [e.status for e in episodes] == ["ready", "ready"]
    # Exactly two progress callbacks, both successful, total=2.
    assert progress == [(1, 2, "ep_1", None), (2, 2, "ep_2", None)]
