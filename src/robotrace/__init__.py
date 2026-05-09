"""RoboTrace — observability and evals for AI robots.

The public API in 30 seconds:

    import robotrace as rt

    # 1. Quickstart — one call per run.
    rt.init(api_key="rt_…", base_url="https://app.robotrace.dev")
    rt.log_episode(
        name="pick_and_place v3 morning warmup",
        policy_version="pap-v3.2.1",
        env_version="halcyon-cell-rev4",
        git_sha="abc1234",
        seed=8124,
        video="/tmp/run.mp4",
        sensors="/tmp/sensors.bin",
        actions="/tmp/actions.parquet",
        duration_s=47.2,
        fps=30,
        metadata={"task": "pick_and_place"},
    )

    # 2. Streaming — explicit control of the lifecycle.
    with rt.start_episode(name="…", policy_version="…") as ep:
        ep.upload_video("/tmp/run.mp4")
        # auto-finalize: ready on clean exit, failed on exception.

    # 3. Multiple deployments at once — explicit Client.
    with rt.Client(api_key="…", base_url="…") as client:
        client.log_episode(...)

`log_episode` is the **sacred** signature per AGENTS.md — once 1.0
ships, breakages require a major bump and at least one minor of
deprecation warnings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ._version import __version__
from .client import (
    ENV_API_KEY,
    ENV_BASE_URL,
    Client,
    EpisodeSource,
)
from .episode import (
    ArtifactKind,
    Episode,
    EpisodeFinalStatus,
    UploadUrl,
)
from .errors import (
    APIError,
    AuthError,
    ConfigurationError,
    ConflictError,
    NotFoundError,
    RobotraceError,
    ServerError,
    TransportError,
    ValidationError,
)

__all__ = [
    "__version__",
    # Public types
    "Client",
    "Episode",
    "UploadUrl",
    "ArtifactKind",
    "EpisodeSource",
    "EpisodeFinalStatus",
    # Errors
    "RobotraceError",
    "ConfigurationError",
    "TransportError",
    "APIError",
    "AuthError",
    "NotFoundError",
    "ConflictError",
    "ValidationError",
    "ServerError",
    # Top-level convenience (backed by the default client)
    "init",
    "close",
    "start_episode",
    "log_episode",
    # Env-var names exposed for tooling that wants to validate them
    "ENV_API_KEY",
    "ENV_BASE_URL",
]

# ── module-level "default client" plumbing ──────────────────────────
#
# Mirrors what `requests.get(...)` does — convenient for scripts, but
# we explicitly support and document `Client(...)` for tests and any
# multi-deployment setup. The default client is constructed lazily on
# first use, so importing `robotrace` never hits the network or
# requires env vars.

_default_client: Client | None = None


def init(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
) -> None:
    """Configure the module-level default Client.

    Calling this twice rebuilds the client; the previous one (if any)
    is closed first. Safe to call from process startup.
    """
    global _default_client
    if _default_client is not None:
        try:
            _default_client.close()
        except Exception:
            pass
    _default_client = Client(api_key=api_key, base_url=base_url, timeout=timeout)


def close() -> None:
    """Close the module-level default client. Safe to call twice."""
    global _default_client
    if _default_client is not None:
        try:
            _default_client.close()
        finally:
            _default_client = None


def _ensure_default_client() -> Client:
    """Lazy-construct the default client from env vars on first use.

    Lets users skip the explicit `init(...)` call when they've set
    `ROBOTRACE_API_KEY` / `ROBOTRACE_BASE_URL` (the common case for
    CI-driven episode logging from a robot rig).
    """
    global _default_client
    if _default_client is None:
        _default_client = Client()
    return _default_client


def start_episode(
    *,
    name: str | None = None,
    source: EpisodeSource = "real",
    robot: str | None = None,
    policy_version: str | None = None,
    env_version: str | None = None,
    git_sha: str | None = None,
    seed: int | None = None,
    fps: float | None = None,
    metadata: Mapping[str, Any] | None = None,
    artifacts: Sequence[ArtifactKind] = ("video", "sensors", "actions"),
) -> Episode:
    """Open a new run on the configured deployment. See `Client.start_episode`."""
    return _ensure_default_client().start_episode(
        name=name,
        source=source,
        robot=robot,
        policy_version=policy_version,
        env_version=env_version,
        git_sha=git_sha,
        seed=seed,
        fps=fps,
        metadata=metadata,
        artifacts=artifacts,
    )


def log_episode(
    *,
    name: str | None = None,
    source: EpisodeSource = "real",
    robot: str | None = None,
    policy_version: str | None = None,
    env_version: str | None = None,
    git_sha: str | None = None,
    seed: int | None = None,
    video: str | Path | None = None,
    sensors: str | Path | None = None,
    actions: str | Path | None = None,
    duration_s: float | None = None,
    fps: float | None = None,
    metadata: Mapping[str, Any] | None = None,
    status: EpisodeFinalStatus = "ready",
) -> Episode:
    """Log a complete episode in one call. See `Client.log_episode`.

    This is the **sacred** signature per AGENTS.md. Don't change it
    without bumping a major version and shipping deprecation warnings
    for at least one minor first.
    """
    return _ensure_default_client().log_episode(
        name=name,
        source=source,
        robot=robot,
        policy_version=policy_version,
        env_version=env_version,
        git_sha=git_sha,
        seed=seed,
        video=video,
        sensors=sensors,
        actions=actions,
        duration_s=duration_s,
        fps=fps,
        metadata=metadata,
        status=status,
    )


