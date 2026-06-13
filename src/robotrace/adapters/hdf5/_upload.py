"""`upload_episode(...)` and `upload_dataset(...)` - HDF5 → RoboTrace.

Composes `scan_file` + `encode_episode` + `Client.start_episode` + the
per-slot `upload_*` helpers + `finalize`. Each HDF5 trajectory (a
robomimic ``demo_*`` group, or the whole single-episode ALOHA file)
becomes one RoboTrace episode.

`upload_dataset(...)` is the bulk entry point for a multi-demo
robomimic file - it walks every trajectory and uploads each in turn,
returning the finalized `Episode` objects. Sequential, not parallel:
a flaky network only loses one trajectory's worth of progress.

Like the LeRobot adapter, the upload path goes through the lower-level
`start_episode + upload_* + finalize` API instead of `log_episode` -
the adapter knows exactly which artifacts it wrote and bypasses the
extension-sniffing that `log_episode` does for convenience callers.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ...client import Client, EpisodeSource
from ...episode import ArtifactKind, Episode, EpisodeFinalStatus
from ._encode import EncodedEpisode, ImageColor, encode_episode
from ._scan import FileSummary, scan_file


def upload_episode(
    path: str | Path,
    *,
    episode_index: int = 0,
    client: Client | None = None,
    name: str | None = None,
    source: EpisodeSource = "replay",
    robot: str | None = None,
    policy_version: str | None = None,
    env_version: str | None = None,
    git_sha: str | None = None,
    seed: int | None = None,
    metadata: Mapping[str, Any] | None = None,
    status: EpisodeFinalStatus = "ready",
    fps: float | None = None,
    canonical_camera: str | None = None,
    image_color: ImageColor = "rgb",
    summary: FileSummary | None = None,
    output_dir: str | Path | None = None,
    keep_artifacts: bool = False,
) -> Episode:
    """Upload one HDF5 trajectory as a RoboTrace episode.

    Returns the finalized `Episode`. The episode URL is printed by the
    underlying Client on `start_episode`, so the user sees a clickable
    link before bytes finish uploading.

    The default ``source="replay"`` reflects the HDF5 context: these
    files are recorded demonstrations being replayed against new
    policies. Override to ``"real"`` for freshly-captured teleop data
    or ``"sim"`` for a simulator dump.

    `metadata` is merged with the encoder's own keys (``adapter``,
    ``hdf5_layout``, ``hdf5_source``, ``hdf5_episode_index``,
    ``hdf5_episode_key``, ``hdf5_trajectory_length``,
    ``hdf5_episode_outcome``, …). User keys win on conflict.
    """
    if summary is None:
        summary = scan_file(path, fps=fps)

    if output_dir is not None:
        work_dir = Path(output_dir).expanduser().resolve()
        using_caller_dir = True
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="robotrace-hdf5-"))
        using_caller_dir = False

    try:
        encoded = encode_episode(
            path,
            work_dir,
            episode_index=episode_index,
            fps=fps,
            canonical_camera=canonical_camera,
            image_color=image_color,
            summary=summary,
        )
        return _upload_encoded(
            encoded=encoded,
            client=client,
            name=name or _default_episode_name(summary, episode_index),
            source=source,
            robot=robot or summary.robot_type,
            policy_version=policy_version,
            env_version=env_version,
            git_sha=git_sha,
            seed=seed,
            metadata=metadata,
            status=status,
        )
    finally:
        if not using_caller_dir and not keep_artifacts:
            shutil.rmtree(work_dir, ignore_errors=True)


def upload_dataset(
    path: str | Path,
    *,
    client: Client | None = None,
    name_template: str | None = None,
    source: EpisodeSource = "replay",
    robot: str | None = None,
    policy_version: str | None = None,
    env_version: str | None = None,
    git_sha: str | None = None,
    seed: int | None = None,
    metadata: Mapping[str, Any] | None = None,
    status: EpisodeFinalStatus = "ready",
    fps: float | None = None,
    canonical_camera: str | None = None,
    image_color: ImageColor = "rgb",
    episode_indices: Iterable[int] | None = None,
    on_progress: Any | None = None,
    stop_on_error: bool = False,
) -> list[Episode]:
    """Walk every (or a subset of) HDF5 trajectory and upload each.

    Sequential - one trajectory at a time, each encoded into a fresh
    tempdir that's cleaned up after upload, so peak disk use stays at
    one trajectory's worth.

    Parameters
    ----------
    name_template
        Optional ``str.format(...)``-style template for the per-episode
        name. Receives ``source`` (the file stem), ``episode_index``,
        and ``episode_key`` keywords. Defaults to
        ``"<file stem> #<episode_index>"``.
    episode_indices
        Iterable of ints to upload. Defaults to every trajectory.
    on_progress
        Optional callback ``fn(done, total, episode, error)`` invoked
        after each attempt. ``episode`` is None on failure; ``error``
        is None on success.
    stop_on_error
        If True, the first failure aborts and re-raises. If False
        (default), per-trajectory errors are reported via
        ``on_progress`` and the loop continues.
    """
    summary = scan_file(path, fps=fps)

    if episode_indices is None:
        indices = list(range(summary.total_episodes))
    else:
        indices = list(episode_indices)

    total = len(indices)
    out: list[Episode] = []

    for done, idx in enumerate(indices, start=1):
        ep_name = _format_episode_name(name_template, summary, idx)
        try:
            episode = upload_episode(
                path,
                episode_index=idx,
                client=client,
                name=ep_name,
                source=source,
                robot=robot,
                policy_version=policy_version,
                env_version=env_version,
                git_sha=git_sha,
                seed=seed,
                metadata=metadata,
                status=status,
                fps=fps,
                canonical_camera=canonical_camera,
                image_color=image_color,
                summary=summary,
            )
            out.append(episode)
            if on_progress is not None:
                on_progress(done, total, episode, None)
        except Exception as exc:
            if on_progress is not None:
                on_progress(done, total, None, exc)
            if stop_on_error:
                raise

    return out


# ── internals ─────────────────────────────────────────────────────────


def _upload_encoded(
    *,
    encoded: EncodedEpisode,
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
    resolved_client = client if client is not None else _get_default_client()

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

    try:
        if encoded.video is not None:
            episode.upload("video", encoded.video.path)
        if encoded.sensors is not None:
            episode.upload("sensors", encoded.sensors.path)
        if encoded.actions is not None:
            episode.upload("actions", encoded.actions.path)
    except Exception as exc:
        try:
            episode.finalize(
                status="failed",
                metadata={"failure_reason": f"hdf5 adapter upload error: {exc}"},
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


def _default_episode_name(summary: FileSummary, episode_index: int) -> str:
    stem = Path(summary.path).stem
    if summary.layout == "single":
        return stem
    return f"{stem} #{episode_index}"


def _format_episode_name(
    template: str | None, summary: FileSummary, episode_index: int
) -> str:
    if template is None:
        return _default_episode_name(summary, episode_index)
    ep_ref = summary.episode(episode_index)
    try:
        return template.format(
            source=Path(summary.path).stem,
            episode_index=episode_index,
            episode_key=ep_ref.key,
        )
    except (KeyError, IndexError, ValueError):
        return _default_episode_name(summary, episode_index)


def _get_default_client() -> Client:
    """Resolve the module-level default client lazily.

    Same circular-import dance as the LeRobot / ROS 2 adapters - see
    their `_get_default_client` for the rationale.
    """
    from ... import _ensure_default_client

    return _ensure_default_client()


__all__ = ["upload_dataset", "upload_episode"]
