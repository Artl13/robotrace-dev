"""Public exception hierarchy.

Every error this SDK can raise inherits from `RobotraceError`. Catching
`RobotraceError` is the recommended way to handle "anything went wrong
talking to RoboTrace" without swallowing user code bugs.

The hierarchy mirrors HTTP status families so callers can pattern-match
on the type rather than parsing message strings:

    RobotraceError
    ├── ConfigurationError       # missing api_key / base_url, bad path, etc.
    ├── TransportError           # network / timeout / DNS
    └── APIError                 # the server responded with an error
        ├── AuthError            # 401 — bad / missing / revoked key
        ├── NotFoundError        # 404 — episode id doesn't exist (or cross-tenant)
        ├── ConflictError        # 409 — episode is archived, etc.
        ├── ValidationError      # 400 — payload didn't match schema
        └── ServerError          # 5xx — flag for retries
"""

from __future__ import annotations


class RobotraceError(Exception):
    """Base class for every error this SDK raises."""


class ConfigurationError(RobotraceError):
    """SDK is missing or has invalid configuration (api_key, base_url, file path)."""


class TransportError(RobotraceError):
    """The HTTP request failed before the server could respond.

    Covers DNS errors, connection resets, timeouts, and TLS failures.
    Worth retrying with backoff; the request is not known to have
    landed on the server.
    """


class APIError(RobotraceError):
    """The server responded but indicated failure.

    `status_code` is the HTTP status. `response_body` is the parsed
    JSON body when available (otherwise the raw text). `message` is
    the server-supplied human-readable reason, or a fallback derived
    from the status code.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        response_body: object | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class AuthError(APIError):
    """401 — the API key is missing, malformed, or revoked.

    Don't retry. The user needs to mint a fresh key in the admin UI
    (Admin → Clients → <client> → API access).
    """


class NotFoundError(APIError):
    """404 — the episode id doesn't exist, or belongs to a different client.

    The two cases are intentionally indistinguishable server-side to
    avoid a UUID-enumeration oracle. Don't retry.
    """


class ConflictError(APIError):
    """409 — the request is well-formed but conflicts with current state.

    Most common case: trying to finalize an archived episode. Restore
    it from the admin UI before retrying.
    """


class ValidationError(APIError):
    """400 — the payload didn't pass server-side validation.

    Don't retry without changing the inputs. The server's `error`
    field carries which field was wrong.
    """


class ServerError(APIError):
    """5xx — the server hit an unexpected error.

    Worth retrying with exponential backoff; the SDK doesn't retry
    automatically because doing so on a finalize call could
    double-bill artifact uploads in future tiers.
    """
