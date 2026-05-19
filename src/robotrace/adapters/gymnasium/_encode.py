"""`encode_rollout(...)` - run a Gymnasium episode and write artifacts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...errors import ConfigurationError
from ._flatten import (
    flatten_action,
    flatten_observation,
    stack_action_series,
    stack_observation_series,
)
from ._scan import EnvSummary, scan_env

Policy = Callable[[Any, dict[str, Any]], Any]

_DEFAULT_FPS = 30.0
_DEFAULT_MAX_STEPS = 10_000


@dataclass
class EncodedArtifact:
    """One encoded file ready to upload."""

    slot: str
    path: Path
    bytes_size: int


@dataclass
class EncodedRollout:
    """The product of `encode_rollout(...)`."""

    output_dir: Path
    summary: EnvSummary
    video: EncodedArtifact | None = None
    sensors: EncodedArtifact | None = None
    actions: EncodedArtifact | None = None
    duration_s: float | None = None
    fps: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def encode_rollout(
    env: Any,
    output_dir: str | Path,
    *,
    policy: Policy,
    seed: int | None = None,
    max_steps: int = _DEFAULT_MAX_STEPS,
    record_video: bool | None = None,
    fps: float = _DEFAULT_FPS,
    summary: EnvSummary | None = None,
) -> EncodedRollout:
    """Run one rollout and write `video.mp4`, `sensors.npz`, `actions.npz`."""
    _import_gymnasium()
    np = _import_numpy()

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if summary is None:
        summary = scan_env(env)

    want_video = _resolve_record_video(env, summary, record_video)
    if want_video:
        _require_cv2()

    obs_space = env.observation_space
    act_space = env.action_space

    observations: list[dict[str, Any]] = []
    actions: list[Any] = []
    frames: list[Any] = []
    total_reward = 0.0
    terminated = False
    truncated = False

    reset_kwargs: dict[str, Any] = {}
    if seed is not None:
        reset_kwargs["seed"] = seed
    obs, info = env.reset(**reset_kwargs)

    steps = 0
    while steps < max_steps:
        flat_obs = flatten_observation(obs, obs_space)
        observations.append(flat_obs)

        action = policy(obs, info)
        action_arr = flatten_action(action, act_space)
        actions.append(action_arr)

        if want_video:
            frame = env.render()
            if frame is None:
                raise ConfigurationError(
                    "env.render() returned None. Create the env with "
                    "render_mode='rgb_array', e.g. "
                    "gymnasium.make('CartPole-v1', render_mode='rgb_array')."
                )
            frames.append(np.asarray(frame, dtype=np.uint8))

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        if terminated or truncated:
            break

    if steps == 0:
        raise ConfigurationError(
            "Rollout produced zero steps - nothing to encode. Check env.reset() "
            "and that max_steps > 0."
        )

    timestamps_ns = (np.arange(steps, dtype=np.int64) * int(1_000_000_000 / fps)).astype(
        np.int64
    )

    sensor_arrays = stack_observation_series(observations, timestamps_ns=timestamps_ns)
    action_arrays = stack_action_series(actions, timestamps_ns=timestamps_ns)

    encoded_sensors: EncodedArtifact | None = None
    encoded_actions: EncodedArtifact | None = None
    sensors_path = out_dir / "sensors.npz"
    actions_path = out_dir / "actions.npz"

    if sensor_arrays:
        np.savez(sensors_path, **sensor_arrays)
        encoded_sensors = EncodedArtifact(
            slot="sensors",
            path=sensors_path,
            bytes_size=sensors_path.stat().st_size,
        )
    if action_arrays:
        np.savez(actions_path, **action_arrays)
        encoded_actions = EncodedArtifact(
            slot="actions",
            path=actions_path,
            bytes_size=actions_path.stat().st_size,
        )

    encoded_video: EncodedArtifact | None = None
    if want_video and frames:
        video_path = out_dir / "video.mp4"
        _write_video(frames, video_path, fps=fps)
        encoded_video = EncodedArtifact(
            slot="video",
            path=video_path,
            bytes_size=video_path.stat().st_size,
        )

    duration_s = steps / fps
    metadata: dict[str, Any] = {
        "adapter": "gymnasium",
        "gymnasium_env_id": summary.env_id,
        "total_reward": total_reward,
        "steps": steps,
        "terminated": terminated,
        "truncated": truncated,
    }
    if summary.active_render_mode is not None:
        metadata["render_mode"] = summary.active_render_mode

    return EncodedRollout(
        output_dir=out_dir,
        summary=summary,
        video=encoded_video,
        sensors=encoded_sensors,
        actions=encoded_actions,
        duration_s=duration_s,
        fps=fps,
        metadata=metadata,
    )


def _resolve_record_video(
    env: Any,
    summary: EnvSummary,
    record_video: bool | None,
) -> bool:
    if record_video is False:
        return False
    active = getattr(env, "render_mode", None)
    if record_video is True:
        if active != "rgb_array":
            raise ConfigurationError(
                "record_video=True requires render_mode='rgb_array'. "
                f"Current render_mode={active!r}. Pass "
                "render_mode='rgb_array' to gymnasium.make(...)."
            )
        return True
    return active == "rgb_array"


def _write_video(frames: list[Any], output_path: Path, *, fps: float) -> None:
    cv2 = _require_cv2()
    np = _import_numpy()

    first = np.asarray(frames[0], dtype=np.uint8)
    if first.ndim != 3 or first.shape[2] not in (3, 4):
        raise ConfigurationError(
            f"env.render() must return an HxWx3 (or HxWx4) uint8 array; got shape {first.shape}."
        )
    if first.shape[2] == 4:
        first = first[:, :, :3]

    height, width = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise ConfigurationError(
            f"opencv VideoWriter failed to open {output_path}. "
            "Install the video extra: `pip install 'robotrace-dev[gymnasium,video]'`."
        )
    try:
        writer.write(first)
        for frame in frames[1:]:
            arr = np.asarray(frame, dtype=np.uint8)
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            writer.write(arr)
    finally:
        writer.release()


def _require_cv2() -> Any:
    from ...errors import ConfigurationError

    try:
        import cv2
    except ImportError as exc:
        raise ConfigurationError(
            "Recording video from env.render() needs opencv. "
            "Install with `pip install 'robotrace-dev[gymnasium,video]'`, "
            "or pass record_video=False for sensor-only logging."
        ) from exc
    return cv2


def _import_gymnasium() -> Any:
    from ...errors import ConfigurationError

    try:
        import gymnasium as gym
    except ImportError as exc:
        raise ConfigurationError(
            "The Gymnasium adapter needs the `gymnasium` package. "
            "Install with `pip install 'robotrace-dev[gymnasium]'`."
        ) from exc
    return gym


def _import_numpy() -> Any:
    from ...errors import ConfigurationError

    try:
        import numpy as np
    except ImportError as exc:
        raise ConfigurationError(
            "The Gymnasium adapter needs `numpy`. Install with "
            "`pip install 'robotrace-dev[gymnasium]'`."
        ) from exc
    return np
