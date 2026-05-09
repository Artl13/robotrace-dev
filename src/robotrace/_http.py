"""Internal HTTP wrapper.

Centralizes:
  • auth header construction
  • base_url normalization
  • status-code → exception mapping
  • redaction of the API key in error messages

Public API is intentionally minimal — call sites use only `request()`,
`upload_file()`, and the dataclass shapes. Avoids spreading httpx
specifics through `client.py` / `episode.py`.

NEVER log the value of the `Authorization` header or any request body
to the ingest endpoint. Both can carry secrets per AGENTS.md.
"""

from __future__ import annotations

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
    ) -> dict[str, Any]:
        """Send a JSON request and return the parsed JSON response.

        Maps HTTP errors to the typed exception hierarchy. The raised
        exception's `response_body` always carries the parsed body
        (or raw text) so callers can introspect server-supplied
        details without re-parsing.
        """
        try:
            response = self._client.request(method, path, json=json)
        except httpx.TimeoutException as exc:
            raise TransportError(f"timeout calling {path}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise TransportError(f"transport error calling {path}: {exc}") from exc

        return self._parse_response(response, path)

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
        """
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"artifact not found: {file_path}")
        size = file_path.stat().st_size

        try:
            with file_path.open("rb") as fh, httpx.Client(
                timeout=UPLOAD_TIMEOUT,
                # Don't inherit our auth header here — the signed URL
                # already carries the auth as query params.
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
            raise TransportError(f"upload timeout for {file_path.name}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise TransportError(f"upload transport error for {file_path.name}: {exc}") from exc

        if response.status_code >= 400:
            # Object storage error bodies are XML, not JSON. Surface
            # the raw text in the exception so the user can debug
            # bucket / CORS / signature mismatches.
            raise APIError(
                f"upload failed for {file_path.name} ({response.status_code})",
                status_code=response.status_code,
                response_body=response.text[:1000],
            )
        return size

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


_STATUS_TO_ERROR: dict[int, type[APIError]] = {
    400: ValidationError,
    401: AuthError,
    404: NotFoundError,
    409: ConflictError,
}
