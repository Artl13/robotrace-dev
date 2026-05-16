"""The `Client` class — owns transport config and exposes the public API.

`Client` is the explicit, dependency-injection-friendly entry point.
For one-off scripts the top-level `robotrace.init(...)` /
`robotrace.start_episode(...)` / `robotrace.log_episode(...)` helpers
in `__init__.py` provide the same surface backed by a module-level
default Client instance.

Auth + base_url resolve in this order:

  1. Explicit kwarg passed to `Client(api_key=..., base_url=...)`
  2. Environment variables `ROBOTRACE_API_KEY` / `ROBOTRACE_BASE_URL`
  3. ``~/.robotrace/credentials`` (written by ``robotrace login``)
  4. Raise `ConfigurationError` (no silent default)

We never default `base_url` to a hardcoded production URL — the SDK
ships before the URL is locked in, and silently routing user data to
the wrong place is a worse failure mode than refusing to start.

Episode-create hook
-------------------

Every call to ``start_episode`` (and therefore ``log_episode``) prints
a clickable link to the new episode's portal page. This is the
"feels alive" loop — engineers see the URL the moment the run is
created, not after the upload finishes. Suppress with
``ROBOTRACE_QUIET=1`` (or pass ``verbose=False`` to ``Client``).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

import httpx

from ._credentials import read_credentials
from ._http import HTTPClient
from ._otel import capture_trace_context
from .episode import (
    ArtifactKind,
    Episode,
    EpisodeFinalStatus,
    UploadUrl,
    kind_from_extension,
)
from .errors import ConfigurationError

EpisodeSource = Literal["real", "sim", "replay"]

# Environment-variable knobs. Documented here once so the names
# don't drift across modules.
ENV_API_KEY = "ROBOTRACE_API_KEY"
ENV_BASE_URL = "ROBOTRACE_BASE_URL"
ENV_QUIET = "ROBOTRACE_QUIET"


class Client:
    """Stateful handle to a RoboTrace deployment.

    Construct once at process startup, reuse across many episodes.
    Holds an httpx connection pool that's recycled between calls.
    Use `client.close()` (or `with Client(...) as client:`) to release
    the pool cleanly on shutdown.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | httpx.Timeout | None = None,
        verbose: bool | None = None,
        # Test hook — production callers leave this None.
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        # 1. Explicit kwarg → 2. env var → 3. ~/.robotrace/credentials.
        # We deliberately keep order strict: a kwarg never gets silently
        # overridden by a stale credentials file, and an env var still
        # wins over the file (handy in CI where the key is injected
        # via the runner's secrets store).
        resolved_key = api_key or os.environ.get(ENV_API_KEY)
        resolved_base = base_url or os.environ.get(ENV_BASE_URL)

        if not resolved_key or not resolved_base:
            stored = read_credentials()
            if stored is not None:
                if not resolved_key:
                    resolved_key = stored.api_key
                if not resolved_base:
                    resolved_base = stored.base_url

        if not resolved_key:
            raise ConfigurationError(
                "RoboTrace API key not provided. Run `robotrace login`, "
                f"pass `api_key=...` to Client(...), or set the {ENV_API_KEY} "
                "environment variable."
            )
        if not resolved_base:
            raise ConfigurationError(
                "RoboTrace base URL not provided. Run `robotrace login` "
                f"or set the {ENV_BASE_URL} environment variable "
                "(e.g. https://app.robotrace.dev or http://localhost:3000 "
                "in dev)."
            )

        self._http = HTTPClient(
            api_key=resolved_key,
            base_url=resolved_base,
            timeout=timeout,
            transport=transport,
        )

        # `verbose` controls whether we print the episode URL after
        # `start_episode` succeeds. Default: yes when stdout is a TTY
        # and ROBOTRACE_QUIET isn't set. Production callers set it
        # explicitly to lock the behaviour.
        if verbose is None:
            verbose = (
                os.environ.get(ENV_QUIET, "").strip() not in {"1", "true", "yes"}
                and sys.stdout.isatty()
            )
        self._verbose: bool = bool(verbose)

    # ── lifecycle ───────────────────────────────────────────────────

    def close(self) -> None:
        """Release the underlying connection pool. Safe to call twice."""
        self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        self.close()
        return False

    @property
    def base_url(self) -> str:
        return self._http.base_url

    # ── public API ──────────────────────────────────────────────────

    def start_episode(
        self,
        *,
        # Identification
        name: str | None = None,
        source: EpisodeSource = "real",
        robot: str | None = None,
        # Reproducibility — load-bearing per AGENTS.md
        policy_version: str | None = None,
        env_version: str | None = None,
        git_sha: str | None = None,
        seed: int | None = None,
        # Run details
        fps: float | None = None,
        metadata: Mapping[str, Any] | None = None,
        # Which artifact slots to mint signed URLs for. Empty list =
        # metadata-only run. Defaults to "all three" because that's
        # the common case for a real robot run.
        artifacts: Sequence[ArtifactKind] = ("video", "sensors", "actions"),
    ) -> Episode:
        """Open a new run on the server.

        Returns an `Episode` you can either use as a context manager
        or finalize explicitly. The server assigns the episode id
        and (when R2 is configured) returns short-lived signed PUT
        URLs for the artifacts you requested.
        """
        payload: dict[str, Any] = {"source": source, "request_uploads": list(artifacts)}
        if name is not None:
            payload["name"] = name
        if robot is not None:
            payload["robot"] = robot
        if policy_version is not None:
            payload["policy_version"] = policy_version
        if env_version is not None:
            payload["env_version"] = env_version
        if git_sha is not None:
            payload["git_sha"] = git_sha
        if seed is not None:
            payload["seed"] = seed
        if fps is not None:
            payload["fps"] = fps
        if metadata is not None:
            payload["metadata"] = dict(metadata)

        # OTel trace correlation. `capture_trace_context()` is a soft
        # import — returns None when (a) `[otel]` extra isn't
        # installed, (b) no active span, or (c) any unexpected error.
        # Never raises; never logs; never affects the request body
        # outside of attaching its own `otel` key. If the customer
        # already instruments their training script with OTel they get
        # episode↔trace correlation for free; if they don't, this is
        # a zero-cost no-op.
        otel_ctx = capture_trace_context()
        if otel_ctx is not None:
            payload["otel"] = dict(otel_ctx)

        # `retry_safe=True`: a 429 here means the server *rejected*
        # the create before any row was written. Retrying with
        # backoff (honoring Retry-After) is safe and friendlier to a
        # robot rig that just bumped a quota.
        body = self._http.request(
            "POST",
            "/api/ingest/episode",
            json=payload,
            retry_safe=True,
        )

        upload_urls: dict[ArtifactKind, UploadUrl] = {}
        for raw in body.get("upload_urls", []) or []:
            if not isinstance(raw, dict):
                continue
            kind = raw.get("kind")
            if kind not in ("video", "sensors", "actions"):
                continue
            upload_urls[kind] = UploadUrl(
                kind=kind,
                url=str(raw["url"]),
                expires_at=str(raw.get("expires_at", "")),
                public_url=raw.get("public_url"),
            )

        episode = Episode(
            id=str(body["episode_id"]),
            status=str(body.get("status", "recording")),
            storage=body.get("storage", "unconfigured"),
            upload_urls=upload_urls,
        )
        episode._client = self

        # The "feels alive" loop — the engineer running their first
        # script sees a clickable link to the just-created episode
        # before bytes finish uploading. Suppress with verbose=False
        # or ROBOTRACE_QUIET=1.
        if self._verbose:
            self._print_episode_link(episode.id)

        return episode

    def _print_episode_link(self, episode_id: str) -> None:
        """Best-effort `[robotrace] → <url>` line on episode create.

        Never raises — printing failures (closed stdout, weird TTYs)
        must not break the user's training run.
        """
        try:
            url = f"{self.base_url.rstrip('/')}/portal/episodes/{episode_id}"
            label = url
            if _stdout_supports_osc8():
                label = f"\x1b]8;;{url}\x1b\\{url}\x1b]8;;\x1b\\"
            sys.stdout.write(f"[robotrace] \u2192 {label}\n")
            sys.stdout.flush()
        except Exception:
            pass

    # ── one-shot convenience — the "sacred" log_episode ─────────────

    def log_episode(
        self,
        *,
        # Identification
        name: str | None = None,
        source: EpisodeSource = "real",
        robot: str | None = None,
        # Reproducibility
        policy_version: str | None = None,
        env_version: str | None = None,
        git_sha: str | None = None,
        seed: int | None = None,
        # Artifacts — local file paths, uploaded inline.
        video: str | Path | None = None,
        sensors: str | Path | None = None,
        actions: str | Path | None = None,
        # Run details
        duration_s: float | None = None,
        fps: float | None = None,
        metadata: Mapping[str, Any] | None = None,
        # Final state — defaults to "ready". Pass "failed" when the
        # run errored before producing usable data.
        status: EpisodeFinalStatus = "ready",
    ) -> Episode:
        """Log a complete episode in one call.

        This is the **sacred** entrypoint per AGENTS.md — keep the
        signature stable. New params land as keyword-only with a
        backward-compatible default; old params get deprecation
        warnings for at least one minor before removal.

        Equivalent to:

            ep = client.start_episode(
                name=..., source=..., ...,
                artifacts=[k for k, p in [("video", video), ...] if p],
            )
            if video: ep.upload_video(video)
            if sensors: ep.upload_sensors(sensors)
            if actions: ep.upload_actions(actions)
            ep.finalize(status=..., duration_s=..., fps=..., metadata=...)
            return ep
        """
        # Translate "what files did you give me" into "which slots to
        # request signed URLs for". This keeps the metadata-only path
        # (no files passed) free of unnecessary R2 round-trips.
        slots: list[ArtifactKind] = []
        for slot, path in (("video", video), ("sensors", sensors), ("actions", actions)):
            if path is not None:
                # Sanity check: warn the user if the file extension
                # doesn't match the slot they put it in. Doesn't
                # block — we trust the explicit kwarg.
                guessed = kind_from_extension(path)
                if guessed is not None and guessed != slot:
                    raise ConfigurationError(
                        f"File {path!r} looks like a {guessed!r} artifact but "
                        f"was passed as `{slot}=`. Re-check the kwarg or pass "
                        "it through start_episode(...) to override."
                    )
                slots.append(slot)  # type: ignore[arg-type]

        episode = self.start_episode(
            name=name,
            source=source,
            robot=robot,
            policy_version=policy_version,
            env_version=env_version,
            git_sha=git_sha,
            seed=seed,
            fps=fps,
            metadata=metadata,
            artifacts=slots,
        )

        # Upload everything the caller handed us. Any failure here
        # propagates to the caller — log_episode is "all or nothing"
        # by design, finer-grained recovery requires start_episode.
        try:
            if video is not None:
                episode.upload_video(video)
            if sensors is not None:
                episode.upload_sensors(sensors)
            if actions is not None:
                episode.upload_actions(actions)
        except Exception as exc:
            # Mark the episode failed so the admin UI doesn't show a
            # ghostly "recording" run forever. Re-raise so the caller
            # can see what went wrong.
            try:
                episode.finalize(
                    status="failed",
                    metadata={"failure_reason": f"upload error: {exc}"},
                )
            except Exception:
                pass
            raise

        episode.finalize(
            status=status,
            duration_s=duration_s,
            fps=fps,
            metadata=metadata,
        )
        return episode


# ── module-level helpers ───────────────────────────────────────────────


def _stdout_supports_osc8() -> bool:
    """Best-effort check for OSC 8 hyperlink support on stdout.

    Returns True only when stdout is a real TTY *and* the terminal
    looks like one of the modern emulators that render OSC 8.
    Mirror the helper in ``cli.py`` — kept duplicated to keep the
    Client import surface minimal (no circular dep on the CLI).
    """
    try:
        if not sys.stdout.isatty():
            return False
    except Exception:
        return False
    term = os.environ.get("TERM", "").lower()
    if term in {"dumb", ""}:
        return False
    if "xterm" in term or "screen" in term or "tmux" in term:
        return True
    if os.environ.get("TERM_PROGRAM"):
        return True
    return False
