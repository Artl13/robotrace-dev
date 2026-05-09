"""`robotrace` command-line interface.

Three subcommands today, all built on the same device-code login
flow served by ``/api/cli/auth/start`` + ``/api/cli/auth/poll`` on
the RoboTrace web app:

    robotrace login   [--base-url URL] [--profile NAME] [--no-browser]
    robotrace whoami  [--profile NAME]
    robotrace logout  [--profile NAME]

Design choices
--------------

* **No third-party CLI library.** The SDK's hard-dep list is
  intentionally kept to ``httpx`` per ``pyproject.toml``. Adding
  ``click`` for three subcommands would balloon the install
  footprint robotics CI machines pay; ``argparse`` is fine here.

* **OSC 8 hyperlinks.** When stdout is a TTY and the terminal looks
  like it supports the hyperlink escape, we emit clickable links.
  Plain text fallback otherwise.

* **Friendly polling output.** A small spinner-style status line
  updates in place during ``robotrace login`` so the user knows
  the CLI is alive, then collapses to a single ``✓ Logged in``
  on completion.

* **No interactive prompts.** ``--base-url`` defaults to
  ``ROBOTRACE_BASE_URL`` if set, else ``https://app.robotrace.dev``.
  Anything that looks like a TTY surprise (e.g. asking the user to
  pick a profile) is replaced by an explicit flag.
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
import time
import webbrowser
from typing import NoReturn

import httpx

from . import _version
from ._credentials import (
    DEFAULT_PROFILE,
    StoredCredentials,
    credentials_path,
    delete_credentials,
    read_credentials,
    write_credentials,
)
from .errors import RobotraceError

# ── constants ──────────────────────────────────────────────────────────

DEFAULT_BASE_URL = "https://app.robotrace.dev"

# Total wall-clock budget for the login flow (matches the server-
# side TTL). Past this point the CLI bows out and tells the user to
# run `robotrace login` again.
LOGIN_TOTAL_TIMEOUT_S = 600

# Fallback poll interval when the server doesn't override it.
DEFAULT_POLL_INTERVAL_S = 2.0


# ── public entrypoint ──────────────────────────────────────────────────


def cli_main(argv: list[str] | None = None) -> int:
    """Argparse-driven entrypoint wired into ``[project.scripts]``.

    Returns an exit code. The wrapper installed by setuptools
    converts that to a real process exit; we factor it out here so
    the same function is testable from Python.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "login":
        return _cmd_login(args)
    if args.command == "whoami":
        return _cmd_whoami(args)
    if args.command == "logout":
        return _cmd_logout(args)
    if args.command == "version":
        print(f"robotrace {_version.__version__}")
        return 0

    parser.print_help()
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robotrace",
        description=(
            "RoboTrace command-line interface — log in, check who "
            "you are, log out."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"robotrace {_version.__version__}",
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # login
    p_login = sub.add_parser(
        "login",
        help="Authorize this machine via the browser and save credentials.",
    )
    p_login.add_argument(
        "--base-url",
        default=None,
        help=(
            "RoboTrace deployment URL. Defaults to $ROBOTRACE_BASE_URL "
            f"or {DEFAULT_BASE_URL}."
        ),
    )
    p_login.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help="Profile name to write credentials under (default: 'default').",
    )
    p_login.add_argument(
        "--no-browser",
        action="store_true",
        help="Print the URL but don't try to open a browser automatically.",
    )

    # whoami
    p_whoami = sub.add_parser(
        "whoami",
        help="Print the email and base URL of the currently saved login.",
    )
    p_whoami.add_argument("--profile", default=DEFAULT_PROFILE)
    p_whoami.add_argument(
        "--json",
        action="store_true",
        help="Emit the credentials profile as JSON (no plaintext key).",
    )

    # logout
    p_logout = sub.add_parser(
        "logout",
        help="Forget the saved credentials on this machine.",
    )
    p_logout.add_argument("--profile", default=DEFAULT_PROFILE)

    # version (subcommand alias for `--version`)
    sub.add_parser("version", help="Print the SDK version and exit.")

    return parser


# ── login ──────────────────────────────────────────────────────────────


def _cmd_login(args: argparse.Namespace) -> int:
    base_url = _resolve_base_url(args.base_url)
    profile = args.profile

    print(f"\nWelcome to RoboTrace ({base_url}).")
    print("Logging in this machine via your browser…\n")

    try:
        start = _start_device_session(base_url)
    except _CliError as exc:
        return _bail(str(exc))

    user_code = start["user_code"]
    device_code = start["device_code"]
    verification_full = start.get("verification_uri_complete") or start["verification_uri"]
    interval = float(start.get("interval", DEFAULT_POLL_INTERVAL_S))

    print("To authorize this device, open:")
    print(f"  {_hyperlink(verification_full)}\n")
    print(f"Confirmation code: {user_code}")
    print(
        "Make sure the same code shows up on the page before you click "
        "Authorize.\n"
    )

    if not args.no_browser:
        try:
            webbrowser.open(verification_full)
        except Exception:
            # Best-effort; a headless box may have no browser at all.
            pass

    # Poll until approved / denied / expired / timeout.
    try:
        approved = _poll_until_resolved(
            base_url=base_url,
            device_code=device_code,
            interval_s=interval,
            total_timeout_s=LOGIN_TOTAL_TIMEOUT_S,
        )
    except _CliError as exc:
        return _bail(str(exc))

    creds = StoredCredentials(
        api_key=approved["api_key"],
        base_url=approved.get("base_url", base_url),
        client_id=approved.get("client_id"),
        user_email=approved.get("user_email"),
    )
    saved_path = write_credentials(creds, profile=profile)

    email = creds.user_email or "your account"
    print(f"\n✓ Logged in as {email}.")
    print(f"  Credentials saved to {saved_path} (profile: {profile}).")
    portal = approved.get("portal_url") or f"{creds.base_url.rstrip('/')}/portal"
    print(f"  Portal: {_hyperlink(portal)}")
    return 0


def _start_device_session(base_url: str) -> dict[str, object]:
    """Hit /api/cli/auth/start and return the parsed response.

    Sends a small device fingerprint (UA + hostname) so the user
    can verify on the browser side that the request came from the
    machine they expected.
    """
    body = {
        "user_agent": _user_agent(),
        "hostname": _hostname(),
    }
    try:
        resp = httpx.post(
            f"{base_url.rstrip('/')}/api/cli/auth/start",
            json=body,
            timeout=httpx.Timeout(connect=10.0, read=15.0, write=10.0, pool=10.0),
            headers={"User-Agent": _user_agent()},
        )
    except httpx.HTTPError as exc:
        raise _CliError(
            f"Could not reach {base_url}: {exc}.\nIs the deployment URL correct?"
        ) from exc
    if resp.status_code != 200:
        raise _CliError(_describe_http_error(resp, "start a login session"))
    try:
        data = resp.json()
    except Exception as exc:
        raise _CliError("Server returned a malformed response to /api/cli/auth/start.") from exc
    for key in ("device_code", "user_code", "verification_uri"):
        if key not in data:
            raise _CliError(f"Server omitted `{key}` in the start response.")
    return data


def _poll_until_resolved(
    *,
    base_url: str,
    device_code: str,
    interval_s: float,
    total_timeout_s: float,
) -> dict[str, object]:
    """Spin on /api/cli/auth/poll until we get a terminal response.

    Returns the parsed JSON of the approved response on success.
    Raises ``_CliError`` for ``denied``, ``expired``, transport
    errors that exceed our retry budget, or wall-clock timeout.
    """
    deadline = time.monotonic() + total_timeout_s
    last_print = 0.0
    interval = max(0.5, float(interval_s))

    # Tiny in-place "spinner" — three dots cycling — to make it
    # obvious the CLI is alive while the user authorizes.
    frames = ["", ".", "..", "..."]
    frame_idx = 0

    use_carriage = sys.stdout.isatty()

    while True:
        now = time.monotonic()
        if now >= deadline:
            raise _CliError(
                "Login window expired before the device was authorized. "
                "Run `robotrace login` again."
            )

        try:
            resp = httpx.post(
                f"{base_url.rstrip('/')}/api/cli/auth/poll",
                json={"device_code": device_code},
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=15.0,
                    write=10.0,
                    pool=10.0,
                ),
                headers={"User-Agent": _user_agent()},
            )
        except httpx.HTTPError:
            # Transient network blip — back off a bit and try again.
            time.sleep(min(interval * 2, 5.0))
            continue

        if resp.status_code == 200:
            try:
                data = resp.json()
            except Exception as exc:
                raise _CliError("Server returned a malformed poll response.") from exc
            status = data.get("status")
            if status == "approved":
                if use_carriage:
                    sys.stdout.write("\r")
                    sys.stdout.flush()
                return data
            if status == "denied":
                raise _CliError(
                    "Device login was refused. Run `robotrace login` again "
                    "if you want to retry."
                )
            # status == "pending" → fall through to sleep.
            new_interval = data.get("interval")
            if isinstance(new_interval, (int, float)) and new_interval > 0:
                interval = max(0.5, float(new_interval))
        elif resp.status_code == 410:
            try:
                data = resp.json()
            except Exception:
                data = {}
            status = data.get("status") if isinstance(data, dict) else None
            if status == "expired":
                raise _CliError(
                    "Login window expired. Run `robotrace login` again."
                )
            if status == "consumed":
                raise _CliError(
                    "This device login was already used. Run `robotrace login` to start over."
                )
            raise _CliError(_describe_http_error(resp, "poll for approval"))
        else:
            raise _CliError(_describe_http_error(resp, "poll for approval"))

        # Cosmetic live-status; keep updates infrequent so we don't
        # peg the terminal redraw on slow links.
        if use_carriage and (now - last_print) >= 0.6:
            sys.stdout.write(f"\rWaiting for browser approval{frames[frame_idx]:<4}")
            sys.stdout.flush()
            frame_idx = (frame_idx + 1) % len(frames)
            last_print = now

        time.sleep(interval)


# ── whoami ─────────────────────────────────────────────────────────────


def _cmd_whoami(args: argparse.Namespace) -> int:
    creds = read_credentials(profile=args.profile)
    if not creds:
        print(
            f"Not logged in (profile: {args.profile}).\n"
            f"Run `robotrace login` to authorize this machine.",
            file=sys.stderr,
        )
        return 1

    if args.json:
        # Never expose the plaintext key over the JSON path. The
        # shape mirrors what's safe to dump into a CI log line.
        payload = {
            "profile": args.profile,
            "base_url": creds.base_url,
            "user_email": creds.user_email,
            "client_id": creds.client_id,
            "written_at": creds.written_at,
            "credentials_path": str(credentials_path()),
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(f"Profile:    {args.profile}")
    print(f"User:       {creds.user_email or 'unknown'}")
    print(f"Client ID:  {creds.client_id or 'unknown'}")
    print(f"Base URL:   {creds.base_url}")
    if creds.written_at:
        print(f"Saved at:   {creds.written_at}")
    print(f"File:       {credentials_path()}")
    return 0


# ── logout ─────────────────────────────────────────────────────────────


def _cmd_logout(args: argparse.Namespace) -> int:
    removed = delete_credentials(profile=args.profile)
    if removed:
        print(f"✓ Removed profile '{args.profile}' from {credentials_path()}.")
        print(
            "Note: the API key minted for that login is still valid "
            "until you revoke it from the portal → API keys."
        )
        return 0
    print(
        f"No saved credentials found for profile '{args.profile}'.",
        file=sys.stderr,
    )
    return 1


# ── helpers ────────────────────────────────────────────────────────────


class _CliError(RobotraceError):
    """Internal control-flow exception for CLI failure paths."""


def _bail(message: str) -> int:
    print(f"\nError: {message}", file=sys.stderr)
    return 1


def _resolve_base_url(explicit: str | None) -> str:
    import os
    if explicit:
        return explicit
    env = os.environ.get("ROBOTRACE_BASE_URL")
    if env:
        return env
    return DEFAULT_BASE_URL


def _user_agent() -> str:
    return (
        f"robotrace-python/{_version.__version__} "
        f"cpython/{platform.python_version()} "
        f"{platform.system().lower()}/{platform.machine().lower()}"
    )


def _hostname() -> str | None:
    try:
        return socket.gethostname()
    except OSError:
        return None


def _describe_http_error(resp: httpx.Response, intent: str) -> str:
    """Human-friendly summary of a non-2xx response.

    Pulls the JSON ``error`` field if the server sent one. Otherwise
    falls back to the status code + first slice of the body.
    """
    try:
        data = resp.json()
        if isinstance(data, dict) and isinstance(data.get("error"), str):
            return f"Could not {intent}: {data['error']} (HTTP {resp.status_code})."
    except Exception:
        pass
    body = (resp.text or "").strip()
    if len(body) > 240:
        body = body[:240] + "…"
    if body:
        return f"Could not {intent}: HTTP {resp.status_code} — {body}"
    return f"Could not {intent}: HTTP {resp.status_code}."


def _hyperlink(url: str, label: str | None = None) -> str:
    """Render an OSC-8 hyperlink when the terminal supports it.

    Falls back to the bare URL on dumb terminals, in CI logs, or when
    stdout isn't a tty. Detection is best-effort: we look for a
    ``$TERM`` that's known to be hyperlink-capable, or assume a
    modern macOS / Linux terminal does.
    """
    label = label or url
    if not _supports_osc8():
        return label if label != url else url
    return f"\x1b]8;;{url}\x1b\\{label}\x1b]8;;\x1b\\"


def _supports_osc8() -> bool:
    if not sys.stdout.isatty():
        return False
    import os
    term = os.environ.get("TERM", "").lower()
    if term in {"dumb", ""}:
        return False
    # Conservative allowlist — every other modern terminal renders
    # OSC 8 the same way.
    if "xterm" in term or "screen" in term or "tmux" in term:
        return True
    if os.environ.get("TERM_PROGRAM"):
        return True
    return False


# ── module-level run hook ──────────────────────────────────────────────


def main() -> NoReturn:  # pragma: no cover — thin wrapper
    sys.exit(cli_main())


if __name__ == "__main__":  # pragma: no cover
    main()
