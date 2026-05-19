"""`scan_env(...)` - read-only introspection of a Gymnasium environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EnvSummary:
    """What an env exposes and how the adapter would treat it."""

    env_id: str
    observation_space: str
    action_space: str
    render_modes: tuple[str, ...] = field(default_factory=tuple)
    active_render_mode: str | None = None
    can_record_video: bool = False

    def report(self) -> str:
        """Human-readable summary for dry-runs before a rollout."""
        lines = [
            f"{self.env_id}",
            f"  observation_space: {self.observation_space}",
            f"  action_space: {self.action_space}",
        ]
        if self.render_modes:
            lines.append(f"  render_modes: {', '.join(self.render_modes)}")
        if self.active_render_mode is not None:
            lines.append(f"  active_render_mode: {self.active_render_mode}")
        lines.append(
            "  video: "
            + (
                "yes (env.render() → mp4, needs [video] extra)"
                if self.can_record_video
                else "no (pass render_mode='rgb_array' to gymnasium.make)"
            )
        )
        return "\n".join(lines)


def scan_env(env: Any) -> EnvSummary:
    """Describe a Gymnasium env without running a rollout or opening the network."""
    gym = _import_gymnasium()

    env_id = _env_id(env)
    obs_space = env.observation_space
    act_space = env.action_space
    spec = getattr(env, "spec", None)
    render_modes: tuple[str, ...] = tuple(getattr(spec, "render_modes", None) or ())
    active_render_mode = getattr(env, "render_mode", None)

    can_record_video = active_render_mode == "rgb_array" or "rgb_array" in render_modes

    return EnvSummary(
        env_id=env_id,
        observation_space=_format_space(obs_space, gym),
        action_space=_format_space(act_space, gym),
        render_modes=render_modes,
        active_render_mode=active_render_mode,
        can_record_video=can_record_video,
    )


def _env_id(env: Any) -> str:
    spec = getattr(env, "spec", None)
    if spec is not None and getattr(spec, "id", None):
        return str(spec.id)
    unwrapped = getattr(env, "unwrapped", env)
    return type(unwrapped).__name__


def _format_space(space: Any, gym: Any) -> str:
    name = type(space).__name__
    if hasattr(space, "shape") and getattr(space, "shape", None) is not None:
        return f"{name}{space.shape}"
    if hasattr(space, "n"):
        return f"{name}(n={space.n})"
    if isinstance(space, gym.spaces.Dict):
        keys = ", ".join(sorted(space.spaces.keys()))
        return f"Dict({keys})"
    if isinstance(space, gym.spaces.Tuple):
        return f"Tuple({len(space.spaces)} spaces)"
    return name


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
