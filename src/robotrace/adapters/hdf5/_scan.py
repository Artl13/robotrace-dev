"""`scan_file(...)` - read-only introspection of an imitation HDF5 file.

Detects which of the two supported layouts a file uses and enumerates
the trajectories it contains without reading any frame bytes:

* **robomimic** - a top-level ``data`` group whose children are
  per-trajectory groups (``demo_0``, ``demo_1``, …). Each child group
  holds ``actions``, an ``obs`` (and optionally ``next_obs``) group,
  ``rewards``, ``dones``, ``states``. Episode count = number of demo
  groups. fps is read from ``data.attrs["env_args"]`` JSON when present.

* **single** (ALOHA / ACT and friends) - the whole file is one
  trajectory: ``/action`` (or ``/actions``) at the root plus an
  ``observations`` (or ``obs``) group. Episode count = 1, addressed by
  ``episode_key=None``.

The scan only opens the file, reads the group/dataset *structure* and
small attributes, and closes it. No frames are decoded - safe to run
against a multi-GB file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..._version import install_command
from ...errors import ConfigurationError

# Names we accept for the root-level action dataset in the single-file
# layout. ALOHA uses ``action``; some exporters pluralize.
_ROOT_ACTION_NAMES: tuple[str, ...] = ("action", "actions")
# Names we accept for the per-trajectory observation group.
_OBS_GROUP_NAMES: tuple[str, ...] = ("obs", "observations")

Layout = str  # "robomimic" | "single"


@dataclass
class EpisodeRef:
    """A pointer to one trajectory inside the file.

    ``key`` is the HDF5 group name to address it (``"demo_0"`` for
    robomimic) or ``None`` for the single-file layout where the
    trajectory lives at the root.
    """

    key: str | None
    length: int
    index: int


@dataclass
class FileSummary:
    """What an imitation HDF5 file contains and how the adapter reads it."""

    path: str
    layout: Layout
    episodes: list[EpisodeRef] = field(default_factory=list)
    # Dataset paths (relative to the trajectory group) seen in the
    # first trajectory - the schema the encoder will classify. Cameras
    # are listed separately for a quick "does this need [video]?" check.
    dataset_names: list[str] = field(default_factory=list)
    camera_names: list[str] = field(default_factory=list)
    fps: float | None = None
    fps_assumed: bool = True
    robot_type: str | None = None
    env_name: str | None = None

    @property
    def total_episodes(self) -> int:
        return len(self.episodes)

    def episode(self, index: int) -> EpisodeRef:
        """Look up one trajectory by 0-based index."""
        for ep in self.episodes:
            if ep.index == index:
                return ep
        raise ConfigurationError(
            f"episode index {index} out of range - file has "
            f"{self.total_episodes} trajectories (0..{self.total_episodes - 1})."
        )

    def report(self) -> str:
        """Human-readable summary for dry-runs before an upload."""
        lines = [
            f"{self.path}",
            f"  layout: {self.layout}",
            f"  trajectories: {self.total_episodes}",
        ]
        if self.fps is not None:
            suffix = " (assumed)" if self.fps_assumed else ""
            lines.append(f"  fps: {self.fps:g}{suffix}")
        if self.robot_type:
            lines.append(f"  robot_type: {self.robot_type}")
        if self.env_name:
            lines.append(f"  env: {self.env_name}")
        if self.camera_names:
            lines.append(f"  cameras: {', '.join(self.camera_names)}")
        if self.dataset_names:
            lines.append(f"  datasets: {', '.join(self.dataset_names)}")
        return "\n".join(lines)


def scan_file(path: str | Path, *, fps: float | None = None) -> FileSummary:
    """Describe an imitation HDF5 file without decoding any frames.

    Parameters
    ----------
    path
        Path to the ``.hdf5`` / ``.h5`` file.
    fps
        Frame rate to record on the episode. HDF5 imitation files
        rarely store a clock, so when this is omitted the adapter
        falls back to a default and marks ``fps_assumed=True`` in the
        summary (and in episode metadata downstream). Pass the true
        capture rate (ALOHA is typically 50, robomimic 20) for
        accurate replay timing.
    """
    h5py = _import_h5py()
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise ConfigurationError(f"no HDF5 file at {file_path}.")

    with h5py.File(str(file_path), "r") as f:
        layout = _detect_layout(f, h5py)
        if layout == "robomimic":
            return _scan_robomimic(f, h5py, file_path, fps)
        return _scan_single(f, h5py, file_path, fps)


# ── layout detection ──────────────────────────────────────────────────


def _detect_layout(f: Any, h5py: Any) -> Layout:
    data = f.get("data")
    if isinstance(data, h5py.Group):
        demo_keys = [k for k in data.keys() if isinstance(data[k], h5py.Group)]
        if demo_keys:
            return "robomimic"

    has_root_action = any(
        isinstance(f.get(n), h5py.Dataset) for n in _ROOT_ACTION_NAMES
    )
    has_obs_group = any(isinstance(f.get(n), h5py.Group) for n in _OBS_GROUP_NAMES)
    if has_root_action or has_obs_group:
        return "single"

    raise ConfigurationError(
        f"{f.filename!r} doesn't look like a supported imitation HDF5 file. "
        "Expected either a robomimic `data/demo_*` layout or an "
        "ALOHA-style root `action` + `observations` layout. Supported "
        "layouts are documented at https://robotrace.dev/docs/sdk/hdf5."
    )


# ── robomimic ─────────────────────────────────────────────────────────


def _scan_robomimic(
    f: Any, h5py: Any, file_path: Path, fps: float | None
) -> FileSummary:
    data = f["data"]
    demo_keys = sorted(
        (k for k in data.keys() if isinstance(data[k], h5py.Group)),
        key=_demo_sort_key,
    )

    episodes: list[EpisodeRef] = []
    for index, key in enumerate(demo_keys):
        grp = data[key]
        length = _trajectory_length(grp, h5py)
        episodes.append(EpisodeRef(key=key, length=length, index=index))

    dataset_names, camera_names = _enumerate_schema(data[demo_keys[0]], h5py)

    env_name, robot_type, file_fps = _robomimic_env_meta(data)
    resolved_fps, assumed = _resolve_fps(fps, file_fps)

    return FileSummary(
        path=str(file_path),
        layout="robomimic",
        episodes=episodes,
        dataset_names=dataset_names,
        camera_names=camera_names,
        fps=resolved_fps,
        fps_assumed=assumed,
        robot_type=robot_type,
        env_name=env_name,
    )


def _robomimic_env_meta(data: Any) -> tuple[str | None, str | None, float | None]:
    """Pull env name / robot / control fps out of ``data.attrs['env_args']``."""
    raw = data.attrs.get("env_args")
    if raw is None:
        return None, None, None
    try:
        env_args = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None, None, None
    env_name = env_args.get("env_name")
    env_kwargs = env_args.get("env_kwargs") or {}
    robot_type = None
    robots = env_kwargs.get("robots")
    if isinstance(robots, (list, tuple)) and robots:
        robot_type = str(robots[0])
    elif isinstance(robots, str):
        robot_type = robots
    fps: float | None = None
    control_freq = env_kwargs.get("control_freq")
    if isinstance(control_freq, (int, float)) and control_freq > 0:
        fps = float(control_freq)
    return env_name, robot_type, fps


def _demo_sort_key(key: str) -> tuple[int, str]:
    """Sort ``demo_10`` after ``demo_2`` (numeric suffix when present)."""
    if "_" in key:
        _, _, tail = key.rpartition("_")
        if tail.isdigit():
            return (int(tail), key)
    return (0, key)


# ── single-file (ALOHA / ACT) ─────────────────────────────────────────


def _scan_single(
    f: Any, h5py: Any, file_path: Path, fps: float | None
) -> FileSummary:
    length = _trajectory_length(f, h5py)
    dataset_names, camera_names = _enumerate_schema(f, h5py)

    robot_type = None
    sim_flag = f.attrs.get("sim")
    if sim_flag is not None:
        robot_type = "sim" if bool(sim_flag) else "real"

    file_fps = f.attrs.get("fps")
    file_fps_val = (
        float(file_fps) if isinstance(file_fps, (int, float)) and file_fps > 0 else None
    )
    resolved_fps, assumed = _resolve_fps(fps, file_fps_val)

    return FileSummary(
        path=str(file_path),
        layout="single",
        episodes=[EpisodeRef(key=None, length=length, index=0)],
        dataset_names=dataset_names,
        camera_names=camera_names,
        fps=resolved_fps,
        fps_assumed=assumed,
        robot_type=robot_type,
    )


# ── shared helpers ────────────────────────────────────────────────────


# Sensible default when neither the caller nor the file declares a clock.
# 30 fps is a neutral choice; ALOHA (50) and robomimic (20) users should
# pass the real value via `fps=`.
_DEFAULT_FPS = 30.0


def _resolve_fps(
    user_fps: float | None, file_fps: float | None
) -> tuple[float, bool]:
    """Pick the fps and report whether it was assumed.

    Priority: explicit caller value > value read from the file >
    module default. ``assumed`` is True only for the default fallback.
    """
    if user_fps is not None and user_fps > 0:
        return float(user_fps), False
    if file_fps is not None and file_fps > 0:
        return file_fps, False
    return _DEFAULT_FPS, True


def _trajectory_length(group: Any, h5py: Any) -> int:
    """Number of timesteps in a trajectory group.

    robomimic stamps ``num_samples`` on the demo group; otherwise we
    fall back to the first dimension of an ``action`` dataset, then any
    dataset under the obs group.
    """
    num_samples = group.attrs.get("num_samples")
    if isinstance(num_samples, (int, float)) and num_samples > 0:
        return int(num_samples)

    for name in _ROOT_ACTION_NAMES:
        ds = group.get(name)
        if isinstance(ds, h5py.Dataset) and ds.ndim >= 1:
            return int(ds.shape[0])

    for obs_name in _OBS_GROUP_NAMES:
        obs = group.get(obs_name)
        if isinstance(obs, h5py.Group):
            for key in obs.keys():
                ds = obs[key]
                if isinstance(ds, h5py.Dataset) and ds.ndim >= 1:
                    return int(ds.shape[0])
    return 0


def _enumerate_schema(group: Any, h5py: Any) -> tuple[list[str], list[str]]:
    """List dataset paths (relative to ``group``) and the camera subset.

    A dataset is treated as a camera when its name classifies to
    ``video`` *and* its array is 4-D ``(T, H, W, C)`` with a plausible
    channel count - the same check the encoder repeats before writing
    mp4.
    """
    from ._classify import classify_dataset

    names: list[str] = []
    cameras: list[str] = []

    def _visit(name: str, obj: Any) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        names.append(name)
        decision = classify_dataset(name)
        if decision.slot == "video" and _looks_like_video(obj):
            cameras.append(name)

    group.visititems(_visit)
    names.sort()
    cameras.sort()
    return names, cameras


def _looks_like_video(ds: Any) -> bool:
    """True for a ``(T, H, W, C)`` uint8-ish image stack."""
    shape: tuple[int, ...] = getattr(ds, "shape", ())
    if len(shape) != 4:
        return False
    channels = shape[-1]
    if channels not in (1, 3, 4):
        return False
    dtype = getattr(ds, "dtype", None)
    return getattr(dtype, "kind", "") == "u" or getattr(dtype, "itemsize", 0) == 1


def _import_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise ConfigurationError(
            "the HDF5 adapter needs `h5py`. Install with "
            f"`{install_command('hdf5')}`."
        ) from exc
    return h5py
