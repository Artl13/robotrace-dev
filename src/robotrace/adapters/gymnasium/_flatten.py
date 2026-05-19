"""Flatten Gymnasium observations and actions into NPZ-friendly arrays."""

from __future__ import annotations

from typing import Any

from ...errors import ConfigurationError


def flatten_observation(obs: Any, space: Any, *, prefix: str = "observation") -> dict[str, Any]:
    """Map one observation to `{prefix}/.../value` keys (no timestamps)."""
    gym = _import_gymnasium()
    return _flatten_value(obs, space, prefix, gym)


def flatten_action(action: Any, space: Any) -> Any:
    """Map one action to a float32 ndarray (1-D or 2-D batch of one)."""
    gym = _import_gymnasium()
    np = _import_numpy()
    flat = _flatten_value(action, space, "action", gym)
    if len(flat) != 1:
        raise ConfigurationError(
            f"Expected a single action tensor after flattening, got keys: {sorted(flat)}"
        )
    value = flat["action"]
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def stack_observation_series(
    steps: list[dict[str, Any]],
    *,
    timestamps_ns: Any,
) -> dict[str, Any]:
    """Stack per-step observation dicts into NPZ arrays with `_t_ns` columns."""
    np = _import_numpy()
    if not steps:
        return {}

    keys = sorted({k for step in steps for k in step})
    out: dict[str, Any] = {}
    for key in keys:
        rows = [step[key] for step in steps if key in step]
        if len(rows) != len(steps):
            continue
        out[f"{key}/value"] = np.stack(rows, axis=0).astype(np.float32, copy=False)
        out[f"{key}/_t_ns"] = timestamps_ns
    return out


def stack_action_series(actions: list[Any], *, timestamps_ns: Any) -> dict[str, Any]:
    """Stack per-step actions into `action/value` + `action/_t_ns`."""
    np = _import_numpy()
    if not actions:
        return {}
    stacked = np.stack(actions, axis=0).astype(np.float32, copy=False)
    if stacked.ndim == 1:
        stacked = stacked.reshape(-1, 1)
    return {
        "action/value": stacked,
        "action/_t_ns": timestamps_ns,
    }


def _flatten_value(value: Any, space: Any, prefix: str, gym: Any) -> dict[str, Any]:
    np = _import_numpy()

    if isinstance(space, gym.spaces.Box):
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        return {prefix: arr}

    if isinstance(space, gym.spaces.Discrete):
        return {prefix: np.asarray([float(value)], dtype=np.float32)}

    if isinstance(space, gym.spaces.MultiDiscrete):
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        return {prefix: arr}

    if isinstance(space, gym.spaces.MultiBinary):
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        return {prefix: arr}

    if isinstance(space, gym.spaces.Dict):
        out: dict[str, Any] = {}
        if not isinstance(value, dict):
            raise ConfigurationError(
                f"Dict observation space expects a dict value, got {type(value).__name__}."
            )
        for key, subspace in space.spaces.items():
            if key not in value:
                raise ConfigurationError(f"Observation dict missing key {key!r}.")
            nested = _flatten_value(value[key], subspace, f"{prefix}.{key}", gym)
            out.update(nested)
        return out

    if isinstance(space, gym.spaces.Tuple):
        out = {}
        if not isinstance(value, (tuple, list)):
            raise ConfigurationError(
                f"Tuple observation space expects a tuple/list value, got {type(value).__name__}."
            )
        if len(value) != len(space.spaces):
            raise ConfigurationError(
                f"Tuple observation length mismatch: got {len(value)}, "
                f"expected {len(space.spaces)}."
            )
        for idx, (item, subspace) in enumerate(zip(value, space.spaces, strict=True)):
            nested = _flatten_value(item, subspace, f"{prefix}.{idx}", gym)
            out.update(nested)
        return out

    raise ConfigurationError(
        f"Unsupported Gymnasium space type {type(space).__name__!r}. "
        "v1 supports Box, Discrete, MultiDiscrete, MultiBinary, Dict, and Tuple."
    )


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
