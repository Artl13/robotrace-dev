"""Internal HTTP wrapper.

Centralizes:
  • auth header construction
  • base_url normalization
  • status-code → exception mapping
  • Retry-After parsing + bounded auto-retry on 429
  • redaction of the API key in error messages

Public API is intentionally minimal — call sites use only `request()`,
`upload_file()`, and the dataclass shapes. Avoids spreading httpx
specifics through `client.py` / `episode.py`.

NEVER log the value of the `Authorization` header or any request body
to the ingest endpoint. Both can carry secrets per AGENTS.md.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

from . import _version
from .errors import (
    APIError,
    AuthError,
    ConflictError,
    NotFoundError,
    RateLimitError,
    ServerError,
    TransportError,
    ValidationError,
)

USER_AGENT = f"robotrace-python/{_version.__version__}"

# Default request timeout. Generous on read because the create call
# may block briefly on R2 signing; the upload PUT to R2 has its own
# (longer) timeout in `upload_file`.
DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0)

# Object-storage uploads can run minutes for multi-GB videos. We let
# httpx stream the body so memory stays flat regardless of file size,
# and bump the read/write timeout accordingly.
UPLOAD_TIMEOUT = httpx.Timeout(connect=15.0, read=600.0, write=600.0, pool=15.0)

# ── retry policy ────────────────────────────────────────────────────
#
# Tight bounds on purpose. A robot rig hitting a 429 deserves a
# short, well-defined backoff window — long enough to not pound the
# server, short enough that a busy queue still completes a training
# episode rather than wedging on quota math.
#
#   • Total attempts cap = MAX_ATTEMPTS (initial try + retries).
#   • Per-attempt delay   = honors Retry-After when present (capped
#                           at MAX_RETRY_AFTER_SECONDS so a
#                           misconfigured server can't pin a robot
#                           for an hour), otherwise exponential
#                           backoff (1s, 2s, 4s) keyed off attempt #.
#   • Retry surface       = 429 only. 5xx stays user-driven for the
#                           same reason ServerError already docs:
#                           transparently retrying a finalize could
#                           double-bill artifact storage in a future
#                           paid tier.
MAX_ATTEMPTS = 4
MAX_RETRY_AFTER_SECONDS = 30
RETRY_AFTER_HARD_CAP_SECONDS = 24 * 60 * 60  # sanity bound on a parsed header


def _parse_retry_after(value: str | None) -> int | None:
    """Parse the ``Retry-After`` response header into whole seconds.

    Per RFC 9110 §10.2.3 the value is either a non-negative decimal
    integer (delta-seconds) or an ``HTTP-date``. We support the
    integer form only — clock skew between the robot rig and the
    server makes HTTP-date parsing a footgun, and the server's
    rate-limiter already speaks delta-seconds.

    Returns ``None`` for missing, empty, non-numeric, negative, or
    absurdly-large (> 24h) values so the caller can fall back to its
    own exponential-backoff schedule.
    """
    if not value:
        return None
    stripped = value.strip()
    if not stripped or not stripped.isdigit():
        return None
    seconds = int(stripped)
    if 0 <= seconds <= RETRY_AFTER_HARD_CAP_SECONDS:
        return seconds
    return None


def _retry_delay_seconds(retry_after: int | None, attempt: int) -> float:
    """How long to sleep before the next attempt.

    `attempt` is the 0-indexed *failed* attempt count — the very
    first call uses attempt=0 when computing the delay before
    retrying. Honors ``Retry-After`` when present (capped) and falls
    back to exponential backoff (1s, 2s, 4s, …) otherwise.
    """
    if retry_after is not None and retry_after > 0:
        return float(min(retry_after, MAX_RETRY_AFTER_SECONDS))
    return float(2 ** max(0, attempt))


class HTTPClient:
    """Thin wrapper around httpx.Client with RoboTrace defaults."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout: httpx.Timeout | float | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        # Trim trailing slash so we can join paths with a leading slash
        # without ending up with `//api/...`.
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=timeout if timeout is not None else DEFAULT_TIMEOUT,
            headers={
                "User-Agent": USER_AGENT,
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            },
            # `transport` is a hook for tests (httpx.MockTransport).
            # Production callers leave it None and let httpx pick.
            transport=transport,
        )

    @property
    def base_url(self) -> str:
        return self._base_url

    def close(self) -> None:
        self._client.close()

    # ── core request ────────────────────────────────────────────────

    def request(
        self,
        method: str,
        path: str,
        *,
        json: Mapping[str, Any] | None = None,
        retry_safe: bool = False,
    ) -> dict[str, Any]:
        """Send a JSON request and return the parsed JSON response.

        Maps HTTP errors to the typed exception hierarchy. The raised
        exception's `response_body` always carries the parsed body
        (or raw text) so callers can introspect server-supplied
        details without re-parsing.

        When ``retry_safe=True`` and the server returns 429, the
        wrapper transparently retries (up to ``MAX_ATTEMPTS`` total)
        using ``Retry-After`` when present and exponential backoff
        otherwise. Only pass ``retry_safe=True`` for endpoints where
        re-issuing the same request can never cause double-billing or
        observable double-mutation — currently the create half of
        ``start_episode``. Finalize and any other state-mutating call
        keeps the default (``retry_safe=False``) so the user owns the
        retry policy.
        """
        attempts = MAX_ATTEMPTS if retry_safe else 1
        last_rate_limit: RateLimitError | None = None
        for attempt in range(attempts):
            try:
                response = self._client.request(method, path, json=json)
            except httpx.TimeoutException as exc:
                raise TransportError(f"timeout calling {path}: {exc}") from exc
            except httpx.HTTPError as exc:
                raise TransportError(
                    f"transport error calling {path}: {exc}"
                ) from exc

            try:
                return self._parse_response(response, path)
            except RateLimitError as exc:
                last_rate_limit = exc
                if attempt == attempts - 1:
                    raise
                time.sleep(_retry_delay_seconds(exc.retry_after, attempt))

        # Unreachable: the loop either returns the parsed body or
        # re-raises on the final attempt. Kept for type-checker
        # exhaustiveness without an `assert False`.
        assert last_rate_limit is not None
        raise last_rate_limit

    # ── streaming upload to a signed PUT URL ────────────────────────

    def upload_file(
        self,
        url: str,
        path: str | Path,
        *,
        content_type: str,
    ) -> int:
        """PUT `path` to `url` with `content_type`. Returns bytes uploaded.

        Streams the file from disk so we don't blow up on multi-GB
        videos. Uses a fresh httpx client (no auth header — the
        signed URL carries the credentials) and a long timeout.

        Auto-retries on 429 up to ``MAX_ATTEMPTS`` total. Signed-PUT
        uploads target a stable R2 object key, so re-issuing the
        request just overwrites the same blob — safe by construction.
        """
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"artifact not found: {file_path}")
        size = file_path.stat().st_size

        last_rate_limit: RateLimitError | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                with file_path.open("rb") as fh, httpx.Client(
                    timeout=UPLOAD_TIMEOUT,
                    # Don't inherit our auth header here — the signed
                    # URL already carries the auth as query params.
                ) as client:
                    response = client.put(
                        url,
                        content=fh,
                        headers={
                            "Content-Type": content_type,
                            "Content-Length": str(size),
                            "User-Agent": USER_AGENT,
                        },
                    )
            except httpx.TimeoutException as exc:
                raise TransportError(
                    f"upload timeout for {file_path.name}: {exc}"
                ) from exc
            except httpx.HTTPError as exc:
                raise TransportError(
                    f"upload transport error for {file_path.name}: {exc}"
                ) from exc

            if response.status_code == 429:
                retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                last_rate_limit = RateLimitError(
                    f"upload rate-limited for {file_path.name} (429)",
                    status_code=429,
                    response_body=response.text[:1000],
                    retry_after=retry_after,
                )
                if attempt == MAX_ATTEMPTS - 1:
                    raise last_rate_limit
                time.sleep(_retry_delay_seconds(retry_after, attempt))
                continue

            if response.status_code >= 400:
                # Object storage error bodies are XML, not JSON.
                # Surface the raw text in the exception so the user
                # can debug bucket / CORS / signature mismatches.
                raise APIError(
                    f"upload failed for {file_path.name} ({response.status_code})",
                    status_code=response.status_code,
                    response_body=response.text[:1000],
                )
            return size

        # Unreachable — the loop either returns or raises.
        assert last_rate_limit is not None
        raise last_rate_limit

    # ── internals ───────────────────────────────────────────────────

    def _parse_response(self, response: httpx.Response, path: str) -> dict[str, Any]:
        # Try JSON first; fall back to raw text so 5xx HTML pages
        # don't crash error reporting.
        try:
            body: object = response.json()
        except ValueError:
            body = response.text

        if response.is_success:
            if isinstance(body, dict):
                return body
            # The ingest endpoints always return JSON objects; anything
            # else is a server contract violation.
            raise ServerError(
                f"unexpected non-JSON success body from {path}",
                status_code=response.status_code,
                response_body=body,
            )

        message = self._error_message(body, response.status_code)

        # 429 has its own typed subclass + a Retry-After int. Build
        # it inline so the header parse happens once at the boundary
        # rather than getting re-derived by every caller.
        if response.status_code == 429:
            retry_after = _parse_retry_after(response.headers.get("Retry-After"))
            raise RateLimitError(
                message,
                status_code=429,
                response_body=body,
                retry_after=retry_after,
            )

        cls = _STATUS_TO_ERROR.get(response.status_code, APIError)
        # 5xx falls through to ServerError. 401/404/409/4xx route to
        # their typed subclasses for ergonomic catch blocks.
        if response.status_code >= 500:
            cls = ServerError
        raise cls(message, status_code=response.status_code, response_body=body)

    @staticmethod
    def _error_message(body: object, status_code: int) -> str:
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, str) and err:
                return err
        return f"HTTP {status_code}"


# Note: 429 is handled directly in `_parse_response` so the
# Retry-After header can be threaded into `RateLimitError` without
# every caller re-parsing it. Keep this dict in sync with the
# exception hierarchy in errors.py for the remaining 4xx codes.
_STATUS_TO_ERROR: dict[int, type[APIError]] = {
    400: ValidationError,
    401: AuthError,
    404: NotFoundError,
    409: ConflictError,
}
