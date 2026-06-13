"""Heuristics for sorting HDF5 demonstration datasets into RoboTrace slots.

Imitation-learning HDF5 files don't share a single schema the way
LeRobot parquet does, but the two dominant layouts - **robomimic**
(``data/demo_*`` groups) and **ALOHA / ACT** (``/action`` +
``/observations/{qpos,qvel,images/*}``) - use stable, recognisable
dataset names *within* a trajectory. This module maps a dataset's
path (relative to its trajectory group) to a RoboTrace slot.

Routing rules (ordered, first match wins). ``name`` is the path of
the dataset relative to the episode group, e.g. ``"actions"``,
``"obs/robot0_eef_pos"``, ``"observations/images/top"``,
``"rewards"``:

    1. Bookkeeping leaves (``timestamp``, ``frame_index``,
       ``episode_index``, ``index``) → ``internal``. Not shipped -
       recoverable from fps + shape.

    2. ``action`` / ``actions`` (leaf), or any leaf starting with
       ``action`` (``action_dict`` etc.) → ``actions``.

    3. Episode-outcome leaves (``reward(s)``, ``done(s)``,
       ``success``, ``discount``, ``terminated``, ``truncated``) →
       ``episode_meta`` - per-step signals rolled into the episode's
       ``metadata`` rather than shipped as an artifact stream.

    4. Image streams → ``video``. Detected by name: a path segment
       named ``images`` (ALOHA ``observations/images/<cam>``) or a
       leaf containing ``image`` / ``rgb`` / ``depth`` (robomimic
       ``agentview_image``, ``robot0_eye_in_hand_image``). The encoder
       still verifies the array is ``(T, H, W, C)`` uint8 before
       writing video and re-routes to ``sensors`` otherwise.

    5. Everything else (``obs/*``, ``observations/qpos``, ``states``,
       ``observations/effort``, custom keys) → ``sensors`` (the safe
       default, same as the ROS 2 unknown-msgtype fallback).

Pure function. No IO, no numpy, no h5py - tests can call it with
arbitrary names to pin the conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# ``video`` / ``sensors`` / ``actions`` mirror the SDK's `ArtifactKind`
# and the LeRobot / ROS 2 adapters exactly. ``episode_meta`` carries
# per-step reward/done/success into the episode's `metadata` jsonb;
# ``internal`` datasets are bookkeeping and not shipped.
Slot = Literal["video", "sensors", "actions", "episode_meta", "internal"]

# Bookkeeping datasets - recoverable from fps + frame count, so we
# don't ship them. Matched on the *leaf* name only (full segment),
# never a prefix, so a real proprio key like ``index_finger`` isn't
# swallowed.
_INTERNAL_LEAVES: frozenset[str] = frozenset(
    {
        "timestamp",
        "timestamps",
        "frame_index",
        "episode_index",
        "index",
    }
)

# Per-step outcome signals. robomimic stores ``rewards`` / ``dones``;
# ALOHA-derived eval dumps sometimes add ``success``. These describe
# the trajectory outcome, not a sensor stream.
_EPISODE_META_LEAVES: frozenset[str] = frozenset(
    {
        "reward",
        "rewards",
        "done",
        "dones",
        "success",
        "successes",
        "discount",
        "terminated",
        "truncated",
    }
)

# Leaf-name hints that mark an image stream. ``images`` is also matched
# as a path *segment* (ALOHA nests cameras under ``observations/images/``).
_IMAGE_LEAF_HINTS: tuple[str, ...] = ("image", "rgb", "depth")


@dataclass(frozen=True)
class DatasetClass:
    """One classification decision for one HDF5 dataset path."""

    name: str
    slot: Slot
    # ``exact`` - leaf matched a bookkeeping / outcome set.
    # ``segment`` - an ``images/`` path segment marked it video.
    # ``leaf`` - the leaf name matched (action*, *image*, …).
    # ``default`` - nothing matched, routed to sensors.
    reason: Literal["exact", "segment", "leaf", "default"]


def classify_dataset(name: str) -> DatasetClass:
    """Return the slot decision for one HDF5 dataset path.

    ``name`` is the dataset path relative to its trajectory group
    (``"actions"``, ``"obs/agentview_image"``, ``"observations/images/top"``).
    Pure function - no file access.
    """
    segments = [s for s in name.split("/") if s]
    leaf = segments[-1].lower() if segments else name.lower()
    lower_segments = [s.lower() for s in segments]

    if leaf in _INTERNAL_LEAVES:
        return DatasetClass(name=name, slot="internal", reason="exact")

    # Actions: the leaf is an action vector, OR the dataset lives under
    # an action-rooted group (robomimic's ``action_dict/{abs_pos,
    # gripper,…}`` components in newer exports).
    root = lower_segments[0] if lower_segments else ""
    if leaf.startswith("action") or root == "action" or root.startswith("action_"):
        return DatasetClass(name=name, slot="actions", reason="leaf")

    if leaf in _EPISODE_META_LEAVES:
        return DatasetClass(name=name, slot="episode_meta", reason="exact")

    if "images" in lower_segments:
        return DatasetClass(name=name, slot="video", reason="segment")
    if any(hint in leaf for hint in _IMAGE_LEAF_HINTS):
        return DatasetClass(name=name, slot="video", reason="leaf")

    return DatasetClass(name=name, slot="sensors", reason="default")


def camera_label_from_name(name: str) -> str:
    """Human-friendly camera label from a video dataset path.

    ``observations/images/top`` → ``top``;
    ``obs/agentview_image`` → ``agentview_image``. Used for the
    per-camera ordering note in episode metadata.
    """
    leaf = name.rsplit("/", 1)[-1]
    return leaf
