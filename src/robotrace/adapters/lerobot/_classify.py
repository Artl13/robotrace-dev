"""Heuristics for sorting LeRobot dataset columns into RoboTrace slots.

LeRobot v2.1 episode parquet schemas use dotted column names that
encode the slot. The mapping is mechanical because LeRobot has a
strong convention — the only ambiguity is custom user fields, which
we route to ``sensors`` as the safe default (same call as ROS 2's
unknown-msgtype fallback).

Routing rules (ordered, first match wins):

    1. ``observation.images.<camera_key>`` → ``video``.
       The parquet column itself holds frame metadata (timestamp,
       file path); the actual pixels live in
       ``videos/observation.images.<camera_key>/chunk-XXX/episode_YYYYYY.mp4``.
       The encoder copies (or tiles) those mp4s into one video.mp4.

    2. ``action`` (the canonical action vector) or ``action.<x>``
       → ``actions``.

    3. ``next.reward`` / ``next.done`` / ``next.success`` /
       ``next.<x>`` → ``episode_meta`` — these describe the episode
       outcome, not a per-frame stream. The encoder rolls them into
       the episode-level metadata block instead of into actions.npz.

    4. ``timestamp`` / ``frame_index`` / ``episode_index`` /
       ``index`` / ``task_index`` → ``internal``. These are LeRobot
       bookkeeping; we don't ship them as artifacts because they're
       implicit in the episode shape (timestamp is recoverable from
       fps + frame_index, episode_index is the call argument, etc.).
       Kept available to the encoder so it can compute duration.

    5. ``observation.state`` → ``sensors``. The canonical robot
       proprioception vector — joint positions for an arm, end-
       effector pose for a teleop dataset.

    6. ``observation.<x>`` (anything else: ``observation.environment_state``,
       ``observation.force``, ``observation.imu.*``) → ``sensors``.

    7. Anything else → ``sensors`` (the safe default — same logic
       as the ROS 2 unknown-msgtype fallback).

Pure function. No IO, no numpy, no pyarrow. Tests can call it with
arbitrary column names to pin behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Slot labels. ``video`` / ``sensors`` / ``actions`` mirror the SDK's
# `ArtifactKind` and the ROS 2 adapter exactly. ``episode_meta``
# carries trajectory-level outcome (reward, done, success) into the
# episode's `metadata` jsonb. ``internal`` columns are LeRobot
# bookkeeping and not shipped.
Slot = Literal["video", "sensors", "actions", "episode_meta", "internal"]


# Internal columns — LeRobot v2.1 default features per
# `src/lerobot/utils/constants.py`. Listed here verbatim so a custom
# user feature accidentally named `timestamp` doesn't quietly route
# to internal. We match the *full* column name, not a prefix.
_INTERNAL_COLUMNS: frozenset[str] = frozenset(
    {
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    }
)


@dataclass(frozen=True)
class ColumnClass:
    """One classification decision for one parquet column."""

    column: str
    slot: Slot
    # Reason the slot was picked. ``prefix`` means we matched
    # ``observation.images.``, ``observation.``, ``action``, or
    # ``next.``; ``exact`` means the column name was in the internal
    # set; ``default`` means none matched and we routed to sensors.
    reason: Literal["prefix", "exact", "default"]


def classify_column(column: str) -> ColumnClass:
    """Return the slot decision for one parquet column name.

    Pure function. Tests can call this with arbitrary names without
    setting up a parquet file or HF mock.
    """
    if column in _INTERNAL_COLUMNS:
        return ColumnClass(column=column, slot="internal", reason="exact")
    if column.startswith("observation.images."):
        return ColumnClass(column=column, slot="video", reason="prefix")
    if column == "action" or column.startswith("action."):
        return ColumnClass(column=column, slot="actions", reason="prefix")
    if column.startswith("next."):
        return ColumnClass(column=column, slot="episode_meta", reason="prefix")
    if column.startswith("observation."):
        return ColumnClass(column=column, slot="sensors", reason="prefix")
    return ColumnClass(column=column, slot="sensors", reason="default")


def camera_key_from_column(column: str) -> str:
    """Pull the camera identifier out of an ``observation.images.<key>`` column.

    LeRobot uses the suffix as the directory name under ``videos/``,
    so ``observation.images.laptop`` → ``observation.images.laptop``
    (the directory name is the *full* column, not just the suffix).
    Returning the full column saves the encoder from re-stitching it.
    """
    if not column.startswith("observation.images."):
        raise ValueError(
            f"camera_key_from_column expects an observation.images.<key> "
            f"column; got {column!r}"
        )
    return column
