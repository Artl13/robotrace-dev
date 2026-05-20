"""Typed metadata classes for `log_episode` / `start_episode`.

Robotics teams ship the same handful of shapes in metadata over and
over: joint states, Cartesian poses, twists, IMU readings, battery
state, run outcomes. The empirical evidence shows up across the
ROS 2, LeRobot, and Gymnasium adapters - same five or six payloads,
different field names every time.

This module promotes them to first-class types. Construct them like
ordinary dataclasses, pass them straight into `metadata=`, and the
SDK serializes each one with a ``__type`` discriminator that the
portal recognizes and renders with a per-type widget. Plain dicts
continue to work; this is a pure superset.

Wire format
-----------

Each typed value serializes to a JSON object with a ``__type``
sentinel:

.. code-block:: json

    {
      "__type": "robotrace.JointState",
      "positions": [0.1, 0.2, 0.3],
      "velocities": null,
      "efforts": null,
      "names": null
    }

The double-underscore prefix mirrors the Python ``__dunder__``
convention - the key is framework-owned and users should not collide
with it for their own keys.

Conventions (all locked)
------------------------

- Distances in meters; angles in radians.
- Quaternions ordered ``[x, y, z, w]`` (ROS 2 / Eigen convention).
  Not ``[w, x, y, z]`` (PyTorch3D / numpy-quaternion). We pick one
  and document it loudly; cross-frame conversion is the caller's
  problem.
- ``JointState`` field order follows ``sensor_msgs/JointState``:
  positions, velocities, efforts, names. Names parallel-array with
  positions when present.
- Forward-compat: the server validates the envelope but passes
  unknown ``__type`` values through. Newer SDKs can ship new types
  against older servers without a release coordination dance.

Encoding
--------

``robotrace.types.encode(value)`` walks any value recursively:

* Typed dataclass instances become ``{"__type": ..., ...}`` dicts.
* Mappings / lists / tuples are descended into.
* Anything else is returned unchanged.

The metadata mapping passed to ``log_episode`` /
``start_episode`` is run through ``encode(...)`` automatically -
no manual call needed unless you want to round-trip a value through
a non-SDK path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "Battery",
    "EpisodeOutcome",
    "Imu",
    "JointState",
    "Pose3D",
    "Twist",
    "encode",
]


# ── helpers ─────────────────────────────────────────────────────────


def _vec3(name: str, values: Sequence[float]) -> list[float]:
    """Validate + freeze a 3-vector. Returns a fresh list."""
    try:
        out = [float(v) for v in values]
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be an iterable of three numbers, got {values!r}"
        ) from exc
    if len(out) != 3:
        raise ValueError(f"{name} must have exactly 3 values, got {len(out)}")
    return out


def _quat(name: str, values: Sequence[float]) -> list[float]:
    """Validate + freeze a quaternion as ``[x, y, z, w]``."""
    try:
        out = [float(v) for v in values]
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be an iterable of four numbers, got {values!r}"
        ) from exc
    if len(out) != 4:
        raise ValueError(
            f"{name} must have exactly 4 values (quaternion [x, y, z, w]), "
            f"got {len(out)}"
        )
    return out


def _floats(name: str, values: Sequence[float] | None) -> list[float] | None:
    if values is None:
        return None
    try:
        return [float(v) for v in values]
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be an iterable of numbers, got {values!r}"
        ) from exc


def _strs(name: str, values: Sequence[str] | None) -> list[str] | None:
    if values is None:
        return None
    try:
        return [str(v) for v in values]
    except TypeError as exc:
        raise TypeError(
            f"{name} must be an iterable of strings, got {values!r}"
        ) from exc


# ── typed classes ───────────────────────────────────────────────────


@dataclass(frozen=True)
class JointState:
    """Joint-space snapshot - mirrors ``sensor_msgs/JointState``.

    ``positions`` is required. ``velocities`` / ``efforts`` /
    ``names`` are optional and, when supplied, must be the same length
    as ``positions`` so the parallel-array contract holds.
    """

    positions: Sequence[float]
    velocities: Sequence[float] | None = None
    efforts: Sequence[float] | None = None
    names: Sequence[str] | None = None

    def __post_init__(self) -> None:
        # Frozen dataclasses can't use plain assignment in __post_init__,
        # so we go through object.__setattr__ to coerce / freeze the
        # iterables into immutable lists. Net effect: callers can pass
        # numpy arrays, tuples, generators - we always store lists.
        positions = _floats("positions", self.positions)
        if positions is None or len(positions) == 0:
            raise ValueError("JointState.positions must have at least one value.")
        n = len(positions)

        velocities = _floats("velocities", self.velocities)
        if velocities is not None and len(velocities) != n:
            raise ValueError(
                "JointState.velocities must be the same length as positions "
                f"({len(velocities)} vs {n})."
            )

        efforts = _floats("efforts", self.efforts)
        if efforts is not None and len(efforts) != n:
            raise ValueError(
                "JointState.efforts must be the same length as positions "
                f"({len(efforts)} vs {n})."
            )

        names = _strs("names", self.names)
        if names is not None and len(names) != n:
            raise ValueError(
                "JointState.names must be the same length as positions "
                f"({len(names)} vs {n})."
            )

        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "velocities", velocities)
        object.__setattr__(self, "efforts", efforts)
        object.__setattr__(self, "names", names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "__type": "robotrace.JointState",
            "positions": list(self.positions),
            "velocities": list(self.velocities) if self.velocities is not None else None,
            "efforts": list(self.efforts) if self.efforts is not None else None,
            "names": list(self.names) if self.names is not None else None,
        }


@dataclass(frozen=True)
class Pose3D:
    """Cartesian pose - ``translation`` (m) + ``rotation`` (quaternion).

    Quaternion order is ``[x, y, z, w]`` (ROS 2 / Eigen). If you have
    a ``[w, x, y, z]`` quaternion (numpy-quaternion, PyTorch3D),
    re-order before constructing.
    """

    translation: Sequence[float]
    rotation: Sequence[float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "translation", _vec3("translation", self.translation))
        object.__setattr__(self, "rotation", _quat("rotation", self.rotation))

    def to_dict(self) -> dict[str, Any]:
        return {
            "__type": "robotrace.Pose3D",
            "translation": list(self.translation),
            "rotation": list(self.rotation),
        }


@dataclass(frozen=True)
class Twist:
    """Linear + angular velocity - mirrors ``geometry_msgs/Twist``.

    Linear in m/s, angular in rad/s.
    """

    linear: Sequence[float]
    angular: Sequence[float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "linear", _vec3("linear", self.linear))
        object.__setattr__(self, "angular", _vec3("angular", self.angular))

    def to_dict(self) -> dict[str, Any]:
        return {
            "__type": "robotrace.Twist",
            "linear": list(self.linear),
            "angular": list(self.angular),
        }


@dataclass(frozen=True)
class Imu:
    """IMU sample - mirrors ``sensor_msgs/Imu``.

    Linear acceleration in m/s², angular velocity in rad/s,
    orientation as a ``[x, y, z, w]`` quaternion (optional - many
    IMUs publish only accel + gyro).
    """

    linear_acceleration: Sequence[float]
    angular_velocity: Sequence[float]
    orientation: Sequence[float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "linear_acceleration",
            _vec3("linear_acceleration", self.linear_acceleration),
        )
        object.__setattr__(
            self,
            "angular_velocity",
            _vec3("angular_velocity", self.angular_velocity),
        )
        if self.orientation is not None:
            object.__setattr__(self, "orientation", _quat("orientation", self.orientation))

    def to_dict(self) -> dict[str, Any]:
        return {
            "__type": "robotrace.Imu",
            "linear_acceleration": list(self.linear_acceleration),
            "angular_velocity": list(self.angular_velocity),
            "orientation": list(self.orientation) if self.orientation is not None else None,
        }


@dataclass(frozen=True)
class Battery:
    """Battery state. All fields optional - sensors vary.

    ``percent`` in [0, 100] (not [0, 1]). ``current_a`` is positive
    when discharging by convention - flip the sign on your side if
    you record it the other way.
    """

    percent: float | None = None
    voltage_v: float | None = None
    current_a: float | None = None
    charging: bool | None = None

    def __post_init__(self) -> None:
        if self.percent is not None:
            pct = float(self.percent)
            if pct < 0 or pct > 100:
                raise ValueError(
                    f"Battery.percent must be in [0, 100], got {pct}."
                )
            object.__setattr__(self, "percent", pct)
        if self.voltage_v is not None:
            object.__setattr__(self, "voltage_v", float(self.voltage_v))
        if self.current_a is not None:
            object.__setattr__(self, "current_a", float(self.current_a))
        if self.charging is not None:
            object.__setattr__(self, "charging", bool(self.charging))

    def to_dict(self) -> dict[str, Any]:
        return {
            "__type": "robotrace.Battery",
            "percent": self.percent,
            "voltage_v": self.voltage_v,
            "current_a": self.current_a,
            "charging": self.charging,
        }


@dataclass(frozen=True)
class EpisodeOutcome:
    """Episode-level outcome - mirrors the eval harness ``_outcome``.

    All fields optional. The eval-run finalize rollup already reads
    ``success``, ``reward_total``, ``collision_count``, and
    ``time_to_goal_s`` from the per-step ``_outcome`` sentinel on
    replay actions; this class lets non-replay episodes report the
    same values.
    """

    success: bool | None = None
    reward_total: float | None = None
    collision_count: int | None = None
    time_to_goal_s: float | None = None

    def __post_init__(self) -> None:
        if self.success is not None:
            object.__setattr__(self, "success", bool(self.success))
        if self.reward_total is not None:
            object.__setattr__(self, "reward_total", float(self.reward_total))
        if self.collision_count is not None:
            cc = int(self.collision_count)
            if cc < 0:
                raise ValueError(
                    f"EpisodeOutcome.collision_count must be ≥ 0, got {cc}."
                )
            object.__setattr__(self, "collision_count", cc)
        if self.time_to_goal_s is not None:
            tt = float(self.time_to_goal_s)
            if tt < 0:
                raise ValueError(
                    f"EpisodeOutcome.time_to_goal_s must be ≥ 0, got {tt}."
                )
            object.__setattr__(self, "time_to_goal_s", tt)

    def to_dict(self) -> dict[str, Any]:
        return {
            "__type": "robotrace.EpisodeOutcome",
            "success": self.success,
            "reward_total": self.reward_total,
            "collision_count": self.collision_count,
            "time_to_goal_s": self.time_to_goal_s,
        }


# ── encoder ─────────────────────────────────────────────────────────


# Single place where we know how to flatten our own typed values into
# the wire format. Sibling-free: any class with a ``to_dict()`` method
# would also work, but we restrict to the known set to avoid acciden-
# tally serializing user dataclasses (some of which may have private
# fields the user doesn't want shipped).
_TYPED_CLASSES: tuple[type, ...] = (
    JointState,
    Pose3D,
    Twist,
    Imu,
    Battery,
    EpisodeOutcome,
)


def encode(value: Any) -> Any:
    """Recursively encode typed values into wire-format dicts.

    * Typed dataclass instances → ``{"__type": ..., ...}``.
    * Mappings / lists / tuples → descended into (a new container is
      returned; the input is not mutated).
    * Anything else → returned unchanged.

    Cyclic references are NOT supported - the metadata bag is
    expected to be a small, finite JSON-serializable structure.
    """
    if isinstance(value, _TYPED_CLASSES):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    return value
