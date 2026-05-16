"""Behaviour of ``robotrace logout`` and ``robotrace logout --revoke``.

Pins two contracts:

  1. Plain ``logout`` removes only the local credentials file and
     prints a hint about server-side revoke. The API key is left
     alive on the server because the user may have moved it to
     another machine — this is the same trade-off documented at
     ``/docs/sdk/cli-login``.

  2. ``logout --revoke`` hits ``POST /api/cli/auth/revoke`` with the
     stored Bearer key, prints the safe-to-log key_prefix the server
     echoes back, then deletes the local file regardless of the
     server outcome. A network failure or 5xx still wipes the file
     (the local guarantee) but exits non-zero so CI scripts notice.

The HTTP boundary is exercised via ``httpx.MockTransport``-style
``monkeypatch`` of ``httpx.post`` — the CLI uses a one-shot httpx
call there instead of the long-lived ``HTTPClient`` wrapper, so the
monkey-patch is the simplest seam.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from robotrace import cli
from robotrace._credentials import (
    DEFAULT_PROFILE,
    StoredCredentials,
    credentials_path,
    write_credentials,
)

# ── fixture: isolated credentials home ───────────────────────────────


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Sandbox `~/.robotrace` into tmp_path so the test never touches
    the developer's real credentials file.
    """
    monkeypatch.setenv("ROBOTRACE_HOME", str(tmp_path))
    return tmp_path


def _seed_credentials(profile: str = DEFAULT_PROFILE) -> Path:
    creds = StoredCredentials(
        api_key="rt_id123example_abcdefghijklmnopqrstuvwxyz012345",
        base_url="https://example.test",
        client_id="00000000-0000-0000-0000-000000000001",
        user_email="user@example.test",
    )
    return write_credentials(creds, profile=profile)


# ── plain logout (no --revoke) ───────────────────────────────────────


def test_logout_without_revoke_only_touches_local_file(
    isolated_home: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_credentials()
    # If anyone tries to call out to the network, blow up the test.
    def fail_post(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("logout without --revoke must not hit the network")

    monkeypatch.setattr(cli.httpx, "post", fail_post)

    rc = cli.cli_main(["logout"])
    out = capsys.readouterr()

    assert rc == 0
    assert "Removed profile 'default'" in out.out
    # The hint mentioning the new --revoke flag is what tells the user
    # the key is still live; pin it so we don't lose the affordance.
    assert "--revoke" in out.out
    # Local file is gone.
    assert not credentials_path().exists()


def test_logout_without_credentials_returns_nonzero(
    isolated_home: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = cli.cli_main(["logout"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "No saved credentials" in err


# ── --revoke happy path ──────────────────────────────────────────────


def test_logout_revoke_calls_endpoint_with_bearer_key(
    isolated_home: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _seed_credentials()
    seen_urls: list[str] = []
    seen_headers: list[dict[str, str]] = []

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        seen_urls.append(url)
        seen_headers.append(dict(kwargs.get("headers") or {}))
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            request=request,
            json={
                "status": "revoked",
                "key_id": "key-uuid",
                "key_prefix": "rt_id123example",
                "revoked_at": "2026-05-16T12:00:00Z",
            },
        )

    monkeypatch.setattr(cli.httpx, "post", fake_post)

    rc = cli.cli_main(["logout", "--revoke"])
    out = capsys.readouterr()

    assert rc == 0
    # One round-trip, to the right path, against the saved base_url.
    assert seen_urls == ["https://example.test/api/cli/auth/revoke"]
    auth = seen_headers[0].get("Authorization", "")
    # Bearer prefix, real key threaded through, never logged to stdout.
    assert auth.startswith("Bearer rt_id123example_")
    assert "rt_id123example_" not in out.out  # secret half stays out of stdout

    # User-visible confirmation includes the safe prefix the server echoed.
    assert "rt_id123example" in out.out
    assert "Revoked" in out.out
    # The "still valid" hint is suppressed on --revoke (don't lie).
    assert "still valid" not in out.out

    # Local file deleted regardless of revoke outcome.
    assert not path.exists()


def test_logout_revoke_treats_401_as_soft_success(
    isolated_home: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Server says 401 → the key already doesn't authenticate, so the
    user's goal ("kill the key") is already met. Surface the situation
    without flagging the exit code red."""
    _seed_credentials()

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            401,
            request=httpx.Request("POST", url),
            json={"error": "API key missing, malformed, revoked, or unknown."},
        )

    monkeypatch.setattr(cli.httpx, "post", fake_post)

    rc = cli.cli_main(["logout", "--revoke"])
    out = capsys.readouterr()

    assert rc == 0
    # The user sees the situation framed honestly — the prefix may
    # be None on 401 so we just look for the explanation string.
    assert "already revoked" in out.out
    assert not credentials_path().exists()


# ── --revoke failure paths ───────────────────────────────────────────


def test_logout_revoke_network_failure_still_deletes_local(
    isolated_home: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Server unreachable → exit non-zero so CI notices the key is
    possibly still live, but DO wipe the local file because the user
    asked to end this machine's trust in the key."""
    path = _seed_credentials()

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(cli.httpx, "post", fake_post)

    rc = cli.cli_main(["logout", "--revoke"])
    cap = capsys.readouterr()

    assert rc == 1
    # The error path warns and tells the user how to recover manually.
    assert "Could not revoke" in cap.err
    assert "Portal" in cap.err
    # Local file gone — local guarantee still holds.
    assert not path.exists()


def test_logout_revoke_5xx_is_a_hard_failure(
    isolated_home: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_credentials()

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            500,
            request=httpx.Request("POST", url),
            json={"error": "boom"},
        )

    monkeypatch.setattr(cli.httpx, "post", fake_post)

    rc = cli.cli_main(["logout", "--revoke"])
    cap = capsys.readouterr()

    assert rc == 1
    assert "Could not revoke" in cap.err
    # Local file gone.
    assert not credentials_path().exists()


# ── _print_summary / _fmt regression ─────────────────────────────────


def test_fmt_signed_prepends_plus_for_non_negative_numbers() -> None:
    """`_fmt(value, signed=True)` must yield `+0.500` / `-0.045` /
    `+0.000`. The previous CLI used a format spec like ``:>+10`` in
    the caller, which is illegal on strings (f-strings only allow
    ``+`` on numeric types) and crashed `_print_summary` the first
    time it ran against a real summary dict."""
    assert cli._fmt(0.5, signed=True) == "+0.500"
    assert cli._fmt(-0.045, signed=True) == "-0.045"
    assert cli._fmt(0.0, signed=True) == "+0.000"
    # Default `signed=False` keeps the original behaviour.
    assert cli._fmt(0.5) == "0.500"
    # Non-numeric values are unaffected by `signed`.
    assert cli._fmt(None, signed=True) == "—"
    assert cli._fmt(True, signed=True) == "yes"


def test_print_summary_renders_full_diffcard_shape_without_crashing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end check: feed `_print_summary` the exact response
    shape `rollupEvalSummary` returns and assert the table prints
    cleanly. Catches the `ValueError: Sign not allowed in string
    format specifier` regression introduced when the formatter and
    the format spec disagreed about whose job the `+` was.
    """
    summary = {
        "success_rate": {
            "baseline": 0.5,
            "candidate": 1.0,
            "delta": 0.5,
            "delta_is_better": True,
        },
        "reward_mean": {
            "baseline": 1.2,
            "candidate": 9.2,
            "delta": 8.0,
            "delta_is_better": True,
        },
        "collision_rate": {
            "baseline": 0.1,
            "candidate": 0.0,
            "delta": -0.1,
            "delta_is_better": True,
        },
        "recommend": "ship",
        "better_count": 3,
        "metric_total": 3,
    }

    # No raise == success here. The format-spec bug was on the very
    # first metric row.
    cli._print_summary(summary)

    out = capsys.readouterr().out
    # Pin the visual contract callers depend on: signed deltas, the
    # recommendation line, both better/worse marks.
    assert "Summary:" in out
    assert "success_rate" in out
    assert "+0.500" in out  # success delta, positive → leading +
    assert "-0.100" in out  # collision delta, negative → leading -
    assert "Recommend: ship" in out
    assert "(3/3 metrics better)" in out


def test_logout_revoke_without_credentials_skips_network(
    isolated_home: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No file to read the key from → no point hitting the network."""
    called = {"n": 0}

    def fake_post(*args: Any, **kwargs: Any) -> httpx.Response:
        called["n"] += 1
        raise AssertionError("must not call revoke without a stored key")

    monkeypatch.setattr(cli.httpx, "post", fake_post)

    rc = cli.cli_main(["logout", "--revoke"])
    cap = capsys.readouterr()
    assert called["n"] == 0
    assert rc == 1
    assert "nothing to revoke" in cap.err
