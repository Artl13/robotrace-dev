"""`upload_rollout(...)` - one-shot Gymnasium rollout → episode pipeline."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...client import Client, EpisodeSource
from ...episode import ArtifactKind, Episode, EpisodeFinalStatus
from ._encode import EncodedRollout, Policy, encode_rollout
from ._scan import EnvSummary, scan_env


def upload_rollout(
    env: Any,
    *,
    policy: Policy,
    client: Client | None = None,
    name: str | None = None,
    source: EpisodeSource = "sim",
    robot: str | None = None,
    policy_version: str | None = None,
    env_version: str | None = None,
    git_sha: str | None = None,
    seed: int | None = None,
    metadata: Mapping[str, Any] | None = None,
    status: EpisodeFinalStatus = "ready",
    max_steps: int = 10_000,
    record_video: bool | None = None,
    fps: float = 30.0,
    output_dir: str | Path | None = None,
    keep_artifacts: bool = False,
    summary: EnvSummary | None = None,
) -> Episode:
    """Run one rollout, encode artifacts, upload, and finalize."""
    if summary is None:
        summary = scan_env(env)

    if output_dir is not None:
        work_dir = Path(output_dir).expanduser().resolve()
        using_caller_dir = True
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="robotrace-gymnasium-"))
        using_caller_dir = False

    try:
        encoded = encode_rollout(
            env,
            work_dir,
            policy=policy,
            seed=seed,
            max_steps=max_steps,
            record_video=record_video,
            fps=fps,
            summary=summary,
        )
        return _upload_encoded(
            encoded=encoded,
            client=client,
            name=name or _default_episode_name(summary),
            source=source,
            robot=robot,
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


def _upload_encoded(
    *,
    encoded: EncodedRollout,
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
                metadata={"failure_reason": f"gymnasium adapter upload error: {exc}"},
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
    out: dict[str, Any] = dict(encoder_meta)
    if user_meta:
        out.update(user_meta)
    return out


def _default_episode_name(summary: EnvSummary) -> str:
    return f"{summary.env_id} (gymnasium rollout)"


def _get_default_client() -> Client:
    from ... import _ensure_default_client

    return _ensure_default_client()
