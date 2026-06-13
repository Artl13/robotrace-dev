"""`encode_episode(...)` - write artifacts for one HDF5 trajectory.

Reads one trajectory (a robomimic ``demo_*`` group, or the root of a
single-episode ALOHA file) and produces the standard RoboTrace artifact
shape:

    video.mp4    one camera passthrough, OR multi-camera horizontally
                 tiled (needs the [video] extra / opencv)
    sensors.npz  every non-image observation dataset packed under
                 ``<name>/value`` (float32, (T, K)) + ``<name>/_t_ns``
    actions.npz  the action dataset(s) packed the same way
    metadata     episode outcome (reward sum, final done/success) +
                 provenance (source path, layout, episode key, robot)

Frame timestamps are synthesised from ``fps`` because imitation HDF5
files almost never carry a per-step clock - the spacing is uniform by
construction. Pass the true capture rate to `scan_file` / the upload
verbs for accurate replay timing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..._version import install_command
from ...errors import ConfigurationError
from ._classify import Slot, camera_label_from_name, classify_dataset
from ._scan import FileSummary, scan_file

ImageColor = Literal["rgb", "bgr"]


@dataclass
class EncodedArtifact:
    """One encoded file ready to upload."""

    slot: Slot
    path: Path
    bytes_size: int
    columns: list[str] = field(default_factory=list)


@dataclass
class EncodedEpisode:
    """The product of `encode_episode(...)`.

    Mirrors the LeRobot adapter's `EncodedEpisode` so callers that
    composed against LeRobot can reuse the same upload pattern with no
    branching.
    """

    output_dir: Path
    summary: FileSummary
    episode_key: str | None
    video: EncodedArtifact | None = None
    sensors: EncodedArtifact | None = None
    actions: EncodedArtifact | None = None
    duration_s: float | None = None
    fps: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def encode_episode(
    path: str | Path,
    output_dir: str | Path,
    *,
    episode_index: int = 0,
    fps: float | None = None,
    canonical_camera: str | None = None,
    image_color: ImageColor = "rgb",
    summary: FileSummary | None = None,
) -> EncodedEpisode:
    """Encode one HDF5 trajectory to ``video.mp4 / sensors.npz / actions.npz``.

    Parameters
    ----------
    path
        Path to the ``.hdf5`` / ``.h5`` file.
    output_dir
        Directory to write artifacts into. Created if missing. Three
        filenames are reserved: ``video.mp4``, ``sensors.npz``,
        ``actions.npz``.
    episode_index
        Which trajectory to encode (0-based). Always ``0`` for the
        single-episode ALOHA layout.
    fps
        Capture rate used to synthesise per-step timestamps. Forwarded
        to `scan_file` when ``summary`` isn't supplied.
    canonical_camera
        Pick one camera dataset as the only video output, skipping the
        multi-camera tile. Pass the dataset path as it appears in the
        file (e.g. ``"observations/images/top"``).
    image_color
        Channel order of the stored image arrays. ``"rgb"`` (default,
        robomimic / most ALOHA dumps) is converted to BGR for the mp4
        writer; pass ``"bgr"`` to write frames untouched.
    summary
        Pre-computed `FileSummary` from `scan_file(...)`. Skips the
        re-scan when the caller already inspected the file.
    """
    if summary is None:
        summary = scan_file(path, fps=fps)

    ep_ref = summary.episode(episode_index)
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    h5py = _import_h5py()
    np = _import_numpy()

    resolved_fps = summary.fps if summary.fps and summary.fps > 0 else None
    skipped: list[dict[str, Any]] = []

    metadata: dict[str, Any] = {
        "adapter": "hdf5",
        "hdf5_layout": summary.layout,
        "hdf5_source": Path(path).name if not isinstance(path, Path) else path.name,
        "hdf5_episode_index": episode_index,
        "hdf5_trajectory_length": ep_ref.length,
    }
    if ep_ref.key is not None:
        metadata["hdf5_episode_key"] = ep_ref.key
    if summary.robot_type:
        metadata["hdf5_robot_type"] = summary.robot_type
    if summary.env_name:
        metadata["hdf5_env"] = summary.env_name
    if summary.fps_assumed and resolved_fps is not None:
        metadata["fps_assumed"] = True

    with h5py.File(summary.path, "r") as f:
        group = _resolve_group(f, ep_ref.key, h5py)
        datasets = _list_datasets(group, h5py)

        decisions = {name: classify_dataset(name) for name in datasets}
        video_names = [
            n for n, d in decisions.items() if d.slot == "video" and _is_video(group[n])
        ]
        # A name that classified as video but isn't a (T,H,W,C) uint8
        # stack falls back to sensors (e.g. a compressed-blob dataset).
        sensors_names = [
            n
            for n, d in decisions.items()
            if d.slot == "sensors"
            or (d.slot == "video" and n not in video_names)
        ]
        actions_names = [n for n, d in decisions.items() if d.slot == "actions"]
        episode_meta_names = [
            n for n, d in decisions.items() if d.slot == "episode_meta"
        ]

        if canonical_camera is not None:
            if canonical_camera not in video_names:
                raise ConfigurationError(
                    f"canonical_camera={canonical_camera!r} is not a camera in "
                    f"this trajectory. Available cameras: {video_names}."
                )
            video_names = [canonical_camera]

        encoded_video = _encode_video(
            group=group,
            camera_names=video_names,
            output_path=out_dir / "video.mp4",
            image_color=image_color,
            np=np,
            skipped=skipped,
        )
        encoded_sensors = _encode_arrays(
            group=group,
            names=sensors_names,
            slot="sensors",
            output_path=out_dir / "sensors.npz",
            fps=resolved_fps,
            np=np,
            skipped=skipped,
        )
        encoded_actions = _encode_arrays(
            group=group,
            names=actions_names,
            slot="actions",
            output_path=out_dir / "actions.npz",
            fps=resolved_fps,
            np=np,
            skipped=skipped,
        )

        outcome = _extract_outcome(group, episode_meta_names, np)
        if outcome:
            metadata["hdf5_episode_outcome"] = outcome

    if skipped:
        metadata["skipped_datasets"] = skipped

    duration_s = (
        ep_ref.length / resolved_fps
        if resolved_fps is not None and ep_ref.length > 0
        else None
    )

    return EncodedEpisode(
        output_dir=out_dir,
        summary=summary,
        episode_key=ep_ref.key,
        video=encoded_video,
        sensors=encoded_sensors,
        actions=encoded_actions,
        duration_s=duration_s,
        fps=resolved_fps,
        metadata=metadata,
    )


# ── group / dataset resolution ────────────────────────────────────────


def _resolve_group(f: Any, episode_key: str | None, h5py: Any) -> Any:
    """Return the HDF5 group that holds one trajectory's datasets."""
    if episode_key is None:
        return f
    data = f.get("data")
    if not isinstance(data, h5py.Group) or episode_key not in data:
        raise ConfigurationError(
            f"trajectory group {episode_key!r} not found under `data` in "
            f"{f.filename!r}."
        )
    return data[episode_key]


def _list_datasets(group: Any, h5py: Any) -> list[str]:
    """All dataset paths relative to ``group``, sorted, internal dropped."""
    names: list[str] = []

    def _visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            names.append(name)

    group.visititems(_visit)
    out = [n for n in names if classify_dataset(n).slot != "internal"]
    out.sort()
    return out


def _is_video(ds: Any) -> bool:
    shape: tuple[int, ...] = getattr(ds, "shape", ())
    if len(shape) != 4 or shape[-1] not in (1, 3, 4):
        return False
    dtype = getattr(ds, "dtype", None)
    return getattr(dtype, "kind", "") == "u" or getattr(dtype, "itemsize", 0) == 1


# ── sensors / actions encoder ─────────────────────────────────────────


def _encode_arrays(
    *,
    group: Any,
    names: Sequence[str],
    slot: Slot,
    output_path: Path,
    fps: float | None,
    np: Any,
    skipped: list[dict[str, Any]],
) -> EncodedArtifact | None:
    """Pack the requested datasets into one NPZ.

    Layout (mirrors the LeRobot / ROS 2 adapters for consistent
    server-side parsing)::

        {
          "<name>/_t_ns":   int64[T]        - timestamp nanoseconds
          "<name>/value":   float32[T, K]   - per-step values, flattened
        }

    Each dataset is read as ``(T, ...)`` and reshaped to ``(T, K)``.
    Timestamps are synthesised from ``fps`` (uniform spacing).
    """
    if not names:
        return None

    arrays: dict[str, Any] = {}
    used: list[str] = []
    t_ns_cache: dict[int, Any] = {}

    for name in names:
        ds = group[name]
        try:
            data = np.asarray(ds[()])
        except (ValueError, TypeError, OSError) as exc:
            skipped.append({"dataset": name, "reason": f"unreadable: {exc}"})
            continue
        if data.ndim == 0 or data.shape[0] == 0:
            skipped.append({"dataset": name, "reason": "empty or scalar dataset"})
            continue
        steps = int(data.shape[0])
        flat = data.reshape(steps, -1).astype(np.float32, copy=False)

        if steps not in t_ns_cache:
            t_ns_cache[steps] = _timestamps_ns(steps, fps, np)

        arrays[f"{name}/value"] = flat
        arrays[f"{name}/_t_ns"] = t_ns_cache[steps]
        used.append(name)

    if not arrays:
        return None

    np.savez(output_path, **arrays)
    return EncodedArtifact(
        slot=slot,
        path=output_path,
        bytes_size=output_path.stat().st_size,
        columns=used,
    )


def _timestamps_ns(steps: int, fps: float | None, np: Any) -> Any:
    """Uniform per-step nanosecond timestamps (zeros if fps unknown)."""
    if fps is None or fps <= 0:
        return np.zeros(steps, dtype=np.int64)
    step_ns = int(1_000_000_000 / fps)
    return (np.arange(steps, dtype=np.int64) * step_ns).astype(np.int64)


# ── video encoder ─────────────────────────────────────────────────────


def _encode_video(
    *,
    group: Any,
    camera_names: Sequence[str],
    output_path: Path,
    image_color: ImageColor,
    np: Any,
    skipped: list[dict[str, Any]],
) -> EncodedArtifact | None:
    """Encode one or more ``(T,H,W,C)`` image stacks into one mp4.

    Single camera → straight encode. Multiple → horizontally tiled,
    frames aligned by index (shorter streams pad with black). Needs
    opencv from the ``[video]`` extra; without it we skip video and
    leave a note rather than failing the whole upload.
    """
    if not camera_names:
        return None

    try:
        cv2 = _import_cv2()
    except ConfigurationError as exc:
        skipped.append(
            {
                "reason": (
                    "image streams found but opencv isn't installed - "
                    f"install `{install_command('hdf5', 'video')}` to encode "
                    "video. Sensors/actions were still uploaded."
                ),
                "exc": str(exc),
                "cameras": list(camera_names),
            }
        )
        return None

    stacks: list[Any] = []
    used: list[str] = []
    for name in camera_names:
        arr = np.asarray(group[name][()])
        if arr.ndim != 4:
            skipped.append({"dataset": name, "reason": "not a (T,H,W,C) image stack"})
            continue
        stacks.append(arr)
        used.append(name)

    if not stacks:
        return None

    # Fps for the writer comes from the same clock as the NPZ
    # timestamps - the summary's resolved fps - but we read it off the
    # caller via a uniform default here to keep the writer self-
    # contained. The episode's authoritative fps is set at finalize().
    n_frames = min(s.shape[0] for s in stacks)
    height = max(s.shape[1] for s in stacks)
    width = sum(s.shape[2] for s in stacks)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, 30.0, (width, height))
    if not writer.isOpened():
        skipped.append(
            {
                "reason": (
                    f"opencv VideoWriter failed to open {output_path}. "
                    "Try `pip install --upgrade opencv-python`."
                ),
                "cameras": used,
            }
        )
        return None

    try:
        for i in range(n_frames):
            cells: list[Any] = []
            for stack in stacks:
                frame = _prepare_frame(stack[i], height, image_color, cv2, np)
                cells.append(frame)
            writer.write(np.hstack(cells) if len(cells) > 1 else cells[0])
    finally:
        writer.release()

    return EncodedArtifact(
        slot="video",
        path=output_path,
        bytes_size=output_path.stat().st_size,
        columns=[camera_label_from_name(n) for n in used],
    )


def _prepare_frame(
    frame: Any, target_height: int, image_color: ImageColor, cv2: Any, np: Any
) -> Any:
    """Coerce one frame to ``(target_height, W, 3)`` uint8 BGR."""
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.shape[2] == 3 and image_color == "rgb":
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    h, w = arr.shape[:2]
    if h < target_height:
        pad = np.zeros((target_height - h, w, 3), dtype=np.uint8)
        arr = np.vstack([arr, pad])
    return arr


# ── episode outcome ───────────────────────────────────────────────────


def _extract_outcome(
    group: Any, names: Sequence[str], np: Any
) -> dict[str, Any]:
    """Roll up per-step ``rewards`` / ``dones`` / ``success`` for the run.

    Returns the *last* value for each (the "did it finish / succeed?"
    signal) plus a summed reward across the trajectory.
    """
    out: dict[str, Any] = {}
    for name in names:
        try:
            arr = np.asarray(group[name][()])
        except (ValueError, TypeError, OSError):
            continue
        if arr.size == 0:
            continue
        flat = arr.reshape(-1) if arr.ndim > 1 else arr
        last = flat[-1]
        if hasattr(last, "item"):
            last = last.item()
        leaf = name.rsplit("/", 1)[-1]
        out[leaf] = last
        if leaf in ("reward", "rewards"):
            try:
                out["reward_sum"] = float(np.asarray(flat, dtype=np.float64).sum())
            except (TypeError, ValueError):
                pass
    return out


# ── lazy imports ──────────────────────────────────────────────────────


def _import_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise ConfigurationError(
            "the HDF5 adapter needs `h5py`. Install with "
            f"`{install_command('hdf5')}`."
        ) from exc
    return h5py


def _import_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise ConfigurationError(
            "the HDF5 adapter needs `numpy`. Install with "
            f"`{install_command('hdf5')}` (which pulls it in)."
        ) from exc
    return np


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise ConfigurationError(
            "encoding HDF5 image streams to mp4 needs OpenCV. "
            f"Install with `{install_command('hdf5', 'video')}`, or pass "
            "only sensor/action data (no opencv needed in that path)."
        ) from exc
    return cv2
