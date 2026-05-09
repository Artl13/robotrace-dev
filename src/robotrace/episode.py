"""The `Episode` handle.

Returned by `Client.start_episode(...)`. Wraps:

  • the episode id assigned by the server,
  • the signed PUT URLs (when R2 is configured), and
  • the lifecycle methods (`upload`, `finalize`, context-manager exit).

Designed for two usage shapes:

  1. Explicit:                     ep = client.start_episode(...)
                                   ep.upload_video("./run.mp4")
                                   ep.finalize(status="ready")

  2. Context-managed:              with client.start_episode(...) as ep:
                                       ep.upload_video("./run.mp4")
                                   # auto-finalize: ready on clean exit,
                                   # failed on exception (with the
                                   # exception type recorded in metadata).

The context-managed shape is the recommended default — it guarantees
the episode is always finalized exactly once, even if an unrelated
piece of robot code raises mid-run.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal

from .errors import ConfigurationError

if TYPE_CHECKING:
    from .client import Client

ArtifactKind = Literal["video", "sensors", "actions"]
EpisodeFinalStatus = Literal["ready", "failed"]

# Keep in sync with `lib/storage/r2.ts` on the server. If a value
# diverges the signed PUT URL won't validate Content-Type and R2
# will return 403.
_CONTENT_TYPE: dict[ArtifactKind, str] = {
    "video": "video/mp4",
    "sensors": "application/octet-stream",
    "actions": "application/octet-stream",
}


@dataclass
class UploadUrl:
    """One signed PUT URL returned by the create endpoint.

    `public_url` is optional: it is set only when the server has the
    legacy ``R2_PUBLIC_URL`` env var configured (i.e. the bucket is
    publicly readable). For private buckets — the production default
    after the May 2026 storage refactor — it is ``None``. The portal
    and admin UI do not depend on this field; they fetch artifacts
    through ``/api/episodes/<id>/artifact/<kind>``, which mints a
    short-lived signed GET URL on every request.
    """

    kind: ArtifactKind
    url: str
    expires_at: str  # ISO 8601, server-clock
    public_url: str | None


@dataclass
class Episode:
    """A run that has been opened on the server.

    `id` is assigned by the server and is the only identity that
    matters. `upload_urls` is empty when the deployment hasn't
    configured R2 — in that case the metadata flow still works,
    `upload(...)` simply raises `ConfigurationError` to make the
    misconfiguration loud.
    """

    id: str
    status: str  # starts as "recording"; flips to ready/failed at finalize
    storage: Literal["r2", "unconfigured"]
    upload_urls: dict[ArtifactKind, UploadUrl] = field(default_factory=dict)

    # Internals — populated by the client at construction time.
    _client: Client | None = field(default=None, repr=False, compare=False)
    _finalized: bool = field(default=False, repr=False, compare=False)
    _bytes_uploaded: int = field(default=0, repr=False, compare=False)

    # ── upload helpers ──────────────────────────────────────────────

    def upload(self, kind: ArtifactKind, path: str | Path) -> int:
        """Upload one artifact via its signed PUT URL.

        Returns the number of bytes uploaded. Updates an internal
        running total so `finalize()` can default `bytes_total` for
        callers that don't track sizes themselves.

        Raises `ConfigurationError` when the deployment hasn't
        wired R2 (the create response had `storage="unconfigured"`).
        See `docs/PRODUCTION-SETUP.md` → §1 for setup steps.
        """
        client = self._require_client()
        url = self.upload_urls.get(kind)
        if url is None:
            if self.storage == "unconfigured":
                raise ConfigurationError(
                    "This RoboTrace deployment hasn't wired Cloudflare R2 yet, "
                    "so artifact uploads are disabled. The metadata-only path "
                    "(start + finalize without upload) still works."
                )
            raise ConfigurationError(
                f"No signed URL for kind={kind!r}. Did you list it in "
                f"`request_uploads` when calling start_episode(...)?"
            )

        bytes_uploaded = client._http.upload_file(
            url.url, path, content_type=_CONTENT_TYPE[kind]
        )
        self._bytes_uploaded += bytes_uploaded
        return bytes_uploaded

    # Convenience wrappers — keep the call sites readable for the
    # 95% case where users have one file per artifact kind.
    def upload_video(self, path: str | Path) -> int:
        return self.upload("video", path)

    def upload_sensors(self, path: str | Path) -> int:
        return self.upload("sensors", path)

    def upload_actions(self, path: str | Path) -> int:
        return self.upload("actions", path)

    # ── finalize ────────────────────────────────────────────────────

    def finalize(
        self,
        *,
        status: EpisodeFinalStatus = "ready",
        duration_s: float | None = None,
        fps: float | None = None,
        bytes_total: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Flip the run to `ready` or `failed` and roll up the stats.

        Idempotent at the server: re-finalizing returns the same
        payload but does not roll back state. The SDK still guards
        against double-calls so accidental re-finalize doesn't
        clobber the bytes_total computed from upload_video / etc.
        """
        if self._finalized:
            return
        client = self._require_client()
        payload: dict[str, Any] = {"status": status}
        if duration_s is not None:
            payload["duration_s"] = duration_s
        if fps is not None:
            payload["fps"] = fps
        # If the caller didn't pass an explicit bytes_total but used
        # the upload helpers, prefer our running total. Saves the
        # caller from sending a stat() call's worth of extra code.
        rolled_up = bytes_total if bytes_total is not None else self._bytes_uploaded
        if rolled_up > 0:
            payload["bytes_total"] = int(rolled_up)
        if metadata is not None:
            payload["metadata"] = dict(metadata)

        response = client._http.request(
            "POST",
            f"/api/ingest/episode/{self.id}/finalize",
            json=payload,
        )
        self.status = str(response.get("status", status))
        self._finalized = True

    # ── context manager ────────────────────────────────────────────

    def __enter__(self) -> Episode:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        if self._finalized:
            return False

        if exc_type is None:
            # Clean exit — assume the run succeeded.
            try:
                self.finalize(status="ready")
            except Exception:
                # Don't mask the original (absent) flow with a
                # finalize failure; the user's main code already
                # succeeded, so log silently. The server-side row
                # will sit in `recording` until they call finalize
                # explicitly or it gets reaped.
                pass
            return False

        # Failure path: try to record the failure reason in metadata
        # so the admin can triage from the episode detail page.
        try:
            self.finalize(
                status="failed",
                metadata={
                    "failure_reason": f"{exc_type.__name__}: {exc_val}"
                    if exc_val is not None
                    else exc_type.__name__,
                },
            )
        except Exception:
            # Same reasoning as above — never mask the user's
            # exception with a network failure during cleanup.
            pass
        # Don't swallow — the user's exception keeps propagating.
        return False

    # ── internals ───────────────────────────────────────────────────

    def _require_client(self) -> Client:
        if self._client is None:
            raise ConfigurationError(
                "Episode is detached from its Client (was the client closed?)."
            )
        return self._client


def kind_from_extension(path: str | Path) -> ArtifactKind | None:
    """Heuristic: guess the artifact kind from a file extension.

    Used by `log_episode` so callers can pass `video="./run.mp4"`
    without spelling out which slot it goes in. Returns None when
    the extension doesn't map cleanly — caller should fall back to
    asking the user explicitly.
    """
    ext = os.path.splitext(str(path))[1].lower()
    if ext in {".mp4", ".webm", ".mov", ".m4v"}:
        return "video"
    if ext in {".bin", ".npy", ".npz", ".h5", ".hdf5"}:
        return "sensors"
    if ext in {".parquet", ".feather", ".arrow"}:
        return "actions"
    return None
