"""`upload_episode(...)` and `upload_dataset(...)` — LeRobot → RoboTrace.

Composes `scan_dataset` + `encode_episode` + `Client.start_episode` +
the per-slot `upload_*` helpers + `finalize`. Each LeRobot trajectory
becomes one RoboTrace episode (the natural mapping — a LeRobot
"episode" is a single robot trajectory, which is exactly what
`log_episode` expects).

`upload_dataset(...)` is the bulk entry point — iterates episodes,
uploads each in turn, returns a list of finalized `Episode` objects.
Sequential, not parallel: a flaky network only loses one episode's
worth of progress, never a full dataset, and we don't surprise the
user with N concurrent HF Hub downloads.

Like the ROS 2 adapter, the upload path goes through the lower-level
`start_episode + upload_* + finalize` API instead of `log_episode` —
the high-level helper validates that file extensions match their
slots (a `.npz` in `actions=` is rejected because `kind_from_extension`
guesses `sensors`). The adapter knows what it wrote, so it bypasses
that check.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ...client import Client, EpisodeSource
from ...episode import ArtifactKind, Episode, EpisodeFinalStatus
from ._encode import EncodedEpisode, encode_episode
from ._meta import DatasetSummary, scan_dataset


def upload_episode(
    repo_id_or_path: str,
    episode_index: int,
    *,
    # Auth — explicit Client for tests / multi-deployment, or None to
    # use the module-level default (configured by `robotrace.init` /
    # env vars).
    client: Client | None = None,
    # Episode identification + reproducibility — same shape as the
    # core `start_episode` call. `name` defaults to a recognisable
    # combination of repo id + episode index.
    name: str | None = None,
    source: EpisodeSource = "replay",
    robot: str | None = None,
    policy_version: str | None = None,
    env_version: str | None = None,
    git_sha: str | None = None,
    seed: int | None = None,
    metadata: Mapping[str, Any] | None = None,
    status: EpisodeFinalStatus = "ready",
    # Adapter knobs.
    revision: str | None = None,
    canonical_camera: str | None = None,
    summary: DatasetSummary | None = None,
    # Inspection escape hatch.
    output_dir: str | Path | None = None,
    keep_artifacts: bool = False,
) -> Episode:
    """Upload one LeRobot trajectory as a RoboTrace episode.

    Returns the finalized `Episode`. The episode URL is printed by
    the underlying Client on `start_episode`, so the user sees a
    clickable link as soon as the row exists, before bytes finish
    uploading.

    The default ``source="replay"`` reflects the LeRobot context:
    public datasets are typically logged trajectories being replayed
    against new policies, not live robot rollouts. Override to
    ``"real"`` if you're uploading a freshly-recorded LeRobot dataset
    of real-robot data.

    `metadata` is merged with the encoder's own metadata (``adapter``,
    ``lerobot_repo_id``, ``lerobot_codebase_version``,
    ``lerobot_episode_index``, ``lerobot_episode_length``,
    ``lerobot_tasks``, ``lerobot_episode_outcome``, ``skipped_columns``).
    User keys win on conflict.
    """
    if summary is None:
        summary = scan_dataset(repo_id_or_path, revision=revision)

    if output_dir is not None:
        work_dir = Path(output_dir).expanduser().resolve()
        using_caller_dir = True
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="robotrace-lerobot-"))
        using_caller_dir = False
    try:
        encoded = encode_episode(
            repo_id_or_path,
            episode_index,
            work_dir,
            revision=revision,
            canonical_camera=canonical_camera,
            summary=summary,
        )
        episode = _upload_encoded(
            encoded=encoded,
            client=client,
            name=name or _default_episode_name(repo_id_or_path, episode_index),
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
        if not using_caller_dir and not keep_artifacts:
            shutil.rmtree(work_dir, ignore_errors=True)


def upload_dataset(
    repo_id_or_path: str,
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
    revision: str | None = None,
    canonical_camera: str | None = None,
    episode_indices: Iterable[int] | None = None,
    # Callback signature: fn(done, total, episode, error). `episode` is
    # None on failure; `error` is None on success. Loosely typed so we
    # don't pull tqdm or callable protocol baggage into the SDK surface.
    on_progress: Any | None = None,
    stop_on_error: bool = False,
) -> list[Episode]:
    """Walk every (or a subset of) LeRobot trajectory and upload each.

    Sequential — one episode at a time. Each episode is encoded into
    a fresh tempdir that's cleaned up after upload; total disk use
    stays at one episode's worth at any moment.

    Parameters
    ----------
    name_template
        Optional ``str.format(...)``-style template for the per-episode
        name. Receives ``repo_id`` and ``episode_index`` keywords. If
        unset, defaults to ``"<repo_id> #<episode_index>"``.
    episode_indices
        Iterable of ints to upload. Defaults to ``range(0, total_episodes)``.
    on_progress
        Optional callback ``fn(done, total, episode, error)`` invoked
        after each upload attempt. ``episode`` is None on failure;
        ``error`` is None on success. Useful for tqdm-style progress
        reporting in user code without us depending on tqdm.
    stop_on_error
        If True, the first failure aborts the loop and re-raises. If
        False (default), per-episode errors are reported via
        ``on_progress`` and the loop continues — large dataset uploads
        shouldn't die on a single corrupted parquet.
    """
    summary = scan_dataset(repo_id_or_path, revision=revision)

    if episode_indices is None:
        indices = list(range(summary.total_episodes))
    else:
        indices = list(episode_indices)

    total = len(indices)
    out: list[Episode] = []

    for done, idx in enumerate(indices, start=1):
        ep_name = _format_episode_name(name_template, repo_id_or_path, idx)
        try:
            episode = upload_episode(
                repo_id_or_path,
                idx,
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
                revision=revision,
                canonical_camera=canonical_camera,
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
    """Take an `EncodedEpisode` and run it through the standard upload path."""
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
            episode.upload_video(encoded.video.path)
        if encoded.sensors is not None:
            episode.upload_sensors(encoded.sensors.path)
        if encoded.actions is not None:
            episode.upload_actions(encoded.actions.path)
    except Exception as exc:
        try:
            episode.finalize(
                status="failed",
                metadata={"failure_reason": f"lerobot adapter upload error: {exc}"},
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


def _default_episode_name(repo_id_or_path: str, episode_index: int) -> str:
    """``<repo_id> #<episode_index>`` — recognisable in the portal list."""
    return f"{repo_id_or_path} #{episode_index}"


def _format_episode_name(
    template: str | None, repo_id: str, episode_index: int
) -> str:
    if template is None:
        return _default_episode_name(repo_id, episode_index)
    try:
        return template.format(repo_id=repo_id, episode_index=episode_index)
    except (KeyError, IndexError, ValueError):
        # Fall back rather than crashing the whole bulk upload on a
        # malformed template — the user almost certainly mistyped.
        return _default_episode_name(repo_id, episode_index)


def _get_default_client() -> Client:
    """Resolve the module-level default client lazily.

    Same circular-import dance as the ROS 2 adapter — see its
    `_get_default_client` for the rationale.
    """
    from ... import _ensure_default_client

    return _ensure_default_client()


def _resolve_episode_indices(
    summary: DatasetSummary,
    episode_indices: Sequence[int] | None,
) -> list[int]:
    """Validate user-supplied indices against the dataset's range."""
    if episode_indices is None:
        return list(range(summary.total_episodes))
    return list(episode_indices)


__all__ = ["upload_dataset", "upload_episode"]
