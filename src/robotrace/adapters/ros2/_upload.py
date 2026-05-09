"""`upload_bag(...)` — the one-shot rosbag2 → episode pipeline.

Composes `scan_bag` + `encode_bag` + `Client.start_episode` + the
per-slot `upload_*` helpers + `finalize`. Encoded artifacts land in a
temporary directory that's cleaned up on return — pass
``keep_artifacts=True`` to debug a problematic encode.

Why not `log_episode`?
----------------------

The "sacred" `log_episode(...)` call validates that file extensions
match their slot, e.g. a `.npz` file in `actions=` is rejected
because `kind_from_extension` guesses `sensors`. The adapter knows
exactly what it wrote, so it bypasses that check by going directly
through the lower-level `start_episode + upload_* + finalize` API.
This is the right place for that bypass — `log_episode` stays strict
for human callers, the adapter speaks to a friendlier interface.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ...client import Client, EpisodeSource
from ...episode import ArtifactKind, Episode, EpisodeFinalStatus
from ._encode import EncodedBag, encode_bag
from ._scan import BagSummary, scan_bag


def upload_bag(
    path: str | Path,
    *,
    # Auth — pass an explicit Client for tests / multi-deployment, or
    # leave None to use the module-level default (configured by
    # `robotrace.init(...)` or env vars).
    client: Client | None = None,
    # Episode identification + reproducibility — same shape as the
    # core `start_episode` call.
    name: str | None = None,
    source: EpisodeSource = "real",
    robot: str | None = None,
    policy_version: str | None = None,
    env_version: str | None = None,
    git_sha: str | None = None,
    seed: int | None = None,
    metadata: Mapping[str, Any] | None = None,
    status: EpisodeFinalStatus = "ready",
    # Adapter-specific topic overrides. Forwarded to encode_bag.
    video_topics: Sequence[str] | None = None,
    sensor_topics: Sequence[str] | None = None,
    action_topics: Sequence[str] | None = None,
    canonical_video_topic: str | None = None,
    # Debugging / inspection escape hatch.
    output_dir: str | Path | None = None,
    keep_artifacts: bool = False,
) -> Episode:
    """Open a bag, encode every classified topic, upload, finalize.

    Returns the finalized `Episode`. The episode URL is printed by the
    underlying Client on `start_episode` so the user sees a clickable
    link as soon as the row exists, before bytes finish uploading.

    `metadata` is merged with the encoder's own metadata
    (`{"adapter": "ros2", "bag": "...", "skipped_topics": [...]}`).
    User keys win on conflict.
    """
    bag_path = Path(path).expanduser().resolve()
    summary = scan_bag(bag_path)

    if output_dir is not None:
        work_dir = Path(output_dir).expanduser().resolve()
        using_caller_dir = True
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="robotrace-ros2-"))
        using_caller_dir = False
    try:
        encoded = encode_bag(
            bag_path,
            work_dir,
            video_topics=video_topics,
            sensor_topics=sensor_topics,
            action_topics=action_topics,
            canonical_video_topic=canonical_video_topic,
            summary=summary,
        )
        episode = _upload_encoded(
            encoded=encoded,
            client=client,
            name=name or _default_episode_name(bag_path),
            source=source,
            robot=robot,
            policy_version=policy_version,
            env_version=env_version,
            git_sha=git_sha,
            seed=seed,
            metadata=metadata,
            status=status,
        )
        return episode
    finally:
        # Tempdirs we created always get cleaned. Caller-supplied
        # output_dir is the user's responsibility unless they opted
        # in to keep_artifacts=False (the default for that path).
        if not using_caller_dir and not keep_artifacts:
            shutil.rmtree(work_dir, ignore_errors=True)


def _upload_encoded(
    *,
    encoded: EncodedBag,
    client: Client | None,
    name: str | None,
    source: EpisodeSource,
    robot: str | None,
    policy_version: str | None,
    env_version: str | None,
    git_sha: str | None,
    seed: int | None,
    metadata: Mapping[str, Any] | None,
    status: EpisodeFinalStatus,
) -> Episode:
    """Take an `EncodedBag` and run it through the standard upload path."""
    resolved_client = client if client is not None else _get_default_client()

    # Compute the slot list before opening the episode so the server
    # only mints signed URLs for slots we actually have data for.
    slots: list[ArtifactKind] = []
    if encoded.video is not None:
        slots.append("video")
    if encoded.sensors is not None:
        slots.append("sensors")
    if encoded.actions is not None:
        slots.append("actions")

    episode = resolved_client.start_episode(
        name=name,
        source=source,
        robot=robot,
        policy_version=policy_version,
        env_version=env_version,
        git_sha=git_sha,
        seed=seed,
        fps=encoded.fps,
        metadata=_merge_metadata(encoded.metadata, metadata),
        artifacts=slots,
    )

    # All-or-nothing upload. On any failure mark the episode failed so
    # the admin UI doesn't show a ghost "recording" row, then re-raise.
    try:
        if encoded.video is not None:
            episode.upload_video(encoded.video.path)
        if encoded.sensors is not None:
            episode.upload_sensors(encoded.sensors.path)
        if encoded.actions is not None:
            episode.upload_actions(encoded.actions.path)
    except Exception as exc:
        try:
            episode.finalize(
                status="failed",
                metadata={"failure_reason": f"ros2 adapter upload error: {exc}"},
            )
        except Exception:
            pass
        raise

    episode.finalize(
        status=status,
        duration_s=encoded.duration_s,
        fps=encoded.fps,
        metadata=_merge_metadata(encoded.metadata, metadata),
    )
    return episode


def _merge_metadata(
    encoder_meta: Mapping[str, Any],
    user_meta: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """User keys win, encoder keys provide context."""
    out: dict[str, Any] = dict(encoder_meta)
    if user_meta:
        out.update(user_meta)
    return out


def _default_episode_name(bag_path: Path) -> str:
    """`{parent_or_name} (ros2 bag)` — recognisable in the portal list."""
    label = bag_path.name or str(bag_path)
    return f"{label} (ros2 bag)"


def _get_default_client() -> Client:
    """Resolve the module-level default client lazily.

    Importing inside the function avoids a circular import at module
    load (`robotrace.adapters.ros2` is imported eagerly when the user
    runs `from robotrace.adapters import ros2`, before
    `robotrace.__init__` is fully evaluated in some shells).
    """
    from ... import _ensure_default_client  # local — see docstring

    return _ensure_default_client()


# Re-exported for users who call `scan_bag` themselves.
__all__ = ["BagSummary", "scan_bag", "upload_bag"]
