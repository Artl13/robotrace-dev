"""`robotrace` command-line interface.

Subcommands today, all built on the same device-code login flow
served by ``/api/cli/auth/start`` + ``/api/cli/auth/poll`` on the
RoboTrace web app:

    robotrace login   [--base-url URL] [--profile NAME] [--no-browser]
    robotrace whoami  [--profile NAME]
    robotrace logout  [--profile NAME] [--revoke]
    robotrace replay run --policy <module:fn>
                         --candidate-version <vN>
                         --baseline-episodes <id...|@file>
                         [--name <label>]
                         [--baseline-version <vN>]
                         [--dry-run]

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
from typing import NoReturn, cast

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
    if args.command == "replay":
        if args.replay_command == "run":
            return _cmd_replay_run(args)
        parser.print_help()
        return 2

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
    p_logout.add_argument(
        "--revoke",
        action="store_true",
        help=(
            "Also revoke the API key server-side before deleting the local "
            "credentials file. Use this when decommissioning a machine, "
            "rotating a leaked key, or after a stolen-laptop incident."
        ),
    )

    # version (subcommand alias for `--version`)
    sub.add_parser("version", help="Print the SDK version and exit.")

    # replay — group for the regression harness verbs. Only `run`
    # ships in 0.1.0a4; reserving the `replay` namespace now so
    # future verbs (`replay status`, `replay cancel`) slot in
    # without flipping the existing CLI surface.
    p_replay = sub.add_parser(
        "replay",
        help="Replay regression harness — re-roll a candidate policy against history.",
    )
    p_replay_sub = p_replay.add_subparsers(dest="replay_command", metavar="<verb>")
    p_replay_sub.required = True

    p_replay_run = p_replay_sub.add_parser(
        "run",
        help="Re-roll a candidate policy against a set of baseline episodes.",
    )
    p_replay_run.add_argument(
        "--policy",
        required=True,
        metavar="MODULE:FN",
        help=(
            "Dotted import path to the policy callable, e.g. "
            "`my_pkg.policies:candidate_v13`. Must accept a single "
            "Observation dict and return an Action dict."
        ),
    )
    p_replay_run.add_argument(
        "--candidate-version",
        required=True,
        metavar="VERSION",
        help="Free-form identifier for the candidate policy (e.g. `pap-v13`).",
    )
    p_replay_run.add_argument(
        "--baseline-episodes",
        required=True,
        nargs="+",
        metavar="ID",
        help=(
            "Baseline episode ids to replay against. Pass them inline "
            "(`--baseline-episodes uuid-1 uuid-2`) or use `@file` to "
            "read newline-separated ids from a file."
        ),
    )
    p_replay_run.add_argument(
        "--baseline-version",
        default=None,
        metavar="VERSION",
        help="Baseline policy version label, surfaced in the portal DiffCard.",
    )
    p_replay_run.add_argument(
        "--name",
        default=None,
        metavar="LABEL",
        help="Human label for the campaign (defaults to the candidate version).",
    )
    p_replay_run.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run the policy locally and print metrics without uploading "
            "the per-episode results. Useful when developing the policy."
        ),
    )
    p_replay_run.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help="Credentials profile to authenticate with (default: 'default').",
    )

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


# Tight budget — logout should never hang a wrap-up shell on a
# slow API. If we can't reach the server within this window we
# surface the failure and still let the user delete locally.
LOGOUT_REVOKE_TIMEOUT_S = 15.0


def _cmd_logout(args: argparse.Namespace) -> int:
    """`robotrace logout [--profile NAME] [--revoke]`.

    Without `--revoke` this is a local-only operation — drop the
    credentials file and remind the user that the underlying key
    still authenticates until revoked from the portal.

    With `--revoke` we hit the server first to kill the key, then
    delete the local file regardless of the server outcome. The
    "delete the local file regardless" rule is deliberate: the
    point of `logout` is "this machine no longer trusts the key,"
    and that's a local guarantee even when the server is unreachable.
    """
    revoke = getattr(args, "revoke", False)
    revoke_succeeded: bool | None = None
    revoke_prefix: str | None = None
    revoke_message: str | None = None

    if revoke:
        creds = read_credentials(profile=args.profile)
        if not creds:
            # No file to read the key from → nothing to revoke. Skip
            # the network call and fall through to the local-delete
            # branch (which will tell the user nothing was there).
            print(
                f"No saved credentials for profile '{args.profile}' — "
                "nothing to revoke on the server. Removing local file "
                "if present…",
                file=sys.stderr,
            )
        else:
            revoke_succeeded, revoke_prefix, revoke_message = _revoke_key_server_side(
                api_key=creds.api_key,
                base_url=creds.base_url,
            )
            if revoke_succeeded:
                label = revoke_prefix or "the saved key"
                # `revoke_message` is non-None on the 401 soft-
                # success path (key was already revoked); surface
                # that to the user so they don't think this CLI
                # invocation is what killed it.
                if revoke_message:
                    print(f"✓ {label}: {revoke_message}.")
                else:
                    print(f"✓ Revoked {label} server-side.")
            else:
                print(
                    f"⚠ Could not revoke the key server-side: {revoke_message}",
                    file=sys.stderr,
                )
                print(
                    "  Local credentials will still be removed. If you "
                    "suspect the key has leaked, revoke it manually from "
                    "Portal → API keys.",
                    file=sys.stderr,
                )

    removed = delete_credentials(profile=args.profile)
    if removed:
        print(f"✓ Removed profile '{args.profile}' from {credentials_path()}.")
        if not revoke:
            print(
                "Note: the API key minted for that login is still valid "
                "until you revoke it from the portal → API keys, or run "
                "`robotrace logout --revoke` next time."
            )
        # Exit status:
        #   • No --revoke           → 0 if the local delete worked.
        #   • --revoke + server OK  → 0.
        #   • --revoke + server bad → 1 so CI scripts notice (the
        #                              key may still be live!), even
        #                              though the local file is gone.
        if revoke and revoke_succeeded is False:
            return 1
        return 0

    print(
        f"No saved credentials found for profile '{args.profile}'.",
        file=sys.stderr,
    )
    # If the user explicitly asked to revoke and we couldn't, that's
    # a user-visible failure even though there's nothing local to
    # clean up.
    if revoke and revoke_succeeded is False:
        return 1
    return 1


def _revoke_key_server_side(
    *,
    api_key: str,
    base_url: str,
) -> tuple[bool, str | None, str | None]:
    """POST `/api/cli/auth/revoke` with the stored Bearer key.

    Returns ``(ok, key_prefix, error_message)``. ``ok=False`` carries
    a human-readable error message; ``key_prefix`` is the safe-to-log
    prefix the server echoes back on success (e.g. ``rt_aBcDeF…``).

    Per AGENTS.md the function never logs the candidate key or the
    full Authorization header. The only string we surface to the
    user is the public prefix from the server response.
    """
    url = f"{base_url.rstrip('/')}/api/cli/auth/revoke"
    try:
        resp = httpx.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": _user_agent(),
                "Accept": "application/json",
            },
            timeout=httpx.Timeout(
                connect=10.0,
                read=LOGOUT_REVOKE_TIMEOUT_S,
                write=10.0,
                pool=10.0,
            ),
        )
    except httpx.HTTPError as exc:
        return False, None, f"could not reach {base_url}: {exc}"

    if resp.status_code == 200:
        try:
            body = resp.json()
        except Exception:
            # Server accepted but spoke gibberish; treat as success
            # since the side effect (revoke) already happened.
            return True, None, None
        prefix = body.get("key_prefix") if isinstance(body, dict) else None
        return True, prefix if isinstance(prefix, str) else None, None

    if resp.status_code == 401:
        # Most likely cause: the local key was already revoked from
        # the portal. The user's intent (kill that key) is satisfied
        # either way, so treat it as a soft-success — print a hint
        # and still nuke the local file.
        return (
            True,
            None,
            "key was already revoked or no longer authenticates",
        )

    return False, None, _describe_http_error(resp, "revoke API key")


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


# ── replay regression harness ──────────────────────────────────────────


def _cmd_replay_run(args: argparse.Namespace) -> int:
    """`robotrace replay run …` — drive the replay regression harness.

    Wires the customer's policy callable into ``robotrace.evals`` and
    prints per-episode progress as the runner walks the baseline set.
    The hot loop lives in :mod:`robotrace.evals`; this command is
    100% I/O orchestration — argparse → import callable → call
    ``create_run`` / ``run_against`` / ``complete_run`` → render
    summary.
    """
    # Late imports keep the CLI startup lean — `numpy` is the heavy
    # one, only paid when the user actually runs the harness.
    from . import evals
    from .client import Client
    from .errors import RobotraceError

    try:
        policy_callable = _import_callable(args.policy)
    except _CliError as exc:
        return _bail(str(exc))

    try:
        baseline_ids = _resolve_baseline_ids(args.baseline_episodes)
    except _CliError as exc:
        return _bail(str(exc))
    if not baseline_ids:
        return _bail("--baseline-episodes resolved to an empty list.")

    creds = read_credentials(profile=args.profile)
    if not creds:
        return _bail(
            f"Not logged in (profile: {args.profile}).\n"
            "Run `robotrace login` first."
        )

    print(
        f"Replay run: candidate={args.candidate_version} "
        f"baseline={args.baseline_version or 'mixed'} "
        f"episodes={len(baseline_ids)}"
    )
    if args.dry_run:
        print("Dry-run mode: per-episode results will NOT be uploaded.")

    client = Client(api_key=creds.api_key, base_url=creds.base_url, verbose=False)
    try:
        try:
            run = evals.create_run(
                candidate_policy_version=args.candidate_version,
                baseline_policy_version=args.baseline_version,
                baseline_episode_ids=baseline_ids,
                name=args.name,
                client=client,
            )
        except RobotraceError as exc:
            return _bail(f"Failed to open eval run: {exc}")

        print(f"✓ Eval run created: {run.id}")
        portal = f"{creds.base_url.rstrip('/')}/portal/evals/{run.id}"
        print(f"  Portal: {_hyperlink(portal)}\n")

        def on_episode(result: evals.EvalResult) -> None:
            idx = run.episodes_completed + run.episodes_failed
            tag = "✓" if result.status == "complete" else "✗"
            cand_label = (
                f"→ {result.candidate_episode_id[:8]}"
                if result.candidate_episode_id
                else ""
            )
            print(
                f"  {tag} [{idx}/{len(baseline_ids)}] "
                f"{result.baseline_episode_id[:8]} {cand_label} "
                f"{result.error or ''}".rstrip()
            )

        try:
            results = evals.run_against(
                run,
                policy_callable=cast(evals.PolicyCallable, policy_callable),
                on_episode=on_episode,
                dry_run=args.dry_run,
            )
        except RobotraceError as exc:
            return _bail(f"Replay loop failed: {exc}")

        if args.dry_run:
            print(
                f"\nDry-run complete: "
                f"{run.episodes_completed} complete, "
                f"{run.episodes_failed} failed."
            )
            return 0 if run.episodes_failed == 0 else 1

        try:
            summary_body = evals.complete_run(run)
        except RobotraceError as exc:
            return _bail(f"Finalize failed: {exc}")

        _print_summary(summary_body.get("summary"))
        print(f"\nView in portal: {_hyperlink(portal)}")
        return 0 if all(r.status == "complete" for r in results) else 1
    finally:
        client.close()


def _import_callable(spec: str) -> object:
    """Resolve `module.path:fn` into a callable.

    Mirrors the convention used by gunicorn / hypercorn — `module:attr`,
    where `attr` may itself be dotted (e.g. `pkg.mod:cls.method`).
    """
    if ":" not in spec:
        raise _CliError(
            f"--policy must be of the form 'module:fn', got {spec!r}."
        )
    module_path, attr_path = spec.split(":", 1)
    if not module_path or not attr_path:
        raise _CliError(
            f"--policy must be 'module:fn'; both halves required, got {spec!r}."
        )

    import importlib

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise _CliError(
            f"Could not import module {module_path!r}: {exc}. "
            "Is it on PYTHONPATH for this shell?"
        ) from exc

    target: object = module
    for piece in attr_path.split("."):
        try:
            target = getattr(target, piece)
        except AttributeError as exc:
            raise _CliError(
                f"Module {module_path!r} has no attribute path {attr_path!r}: {exc}"
            ) from exc

    if not callable(target):
        raise _CliError(
            f"Policy {spec!r} resolved to a non-callable ({type(target).__name__})."
        )
    return target


def _resolve_baseline_ids(raw: list[str]) -> list[str]:
    """Expand any `@file` arguments into the lines they contain.

    Keeps the CLI ergonomic for both small ad-hoc sweeps (paste a
    handful of ids inline) and CI runs (read a pinned list from
    `baseline_episodes.txt`).
    """
    out: list[str] = []
    for token in raw:
        if token.startswith("@"):
            path = token[1:]
            try:
                with open(path, encoding="utf-8") as fh:
                    for line in fh:
                        s = line.strip()
                        if s and not s.startswith("#"):
                            out.append(s)
            except OSError as exc:
                raise _CliError(
                    f"--baseline-episodes {token}: could not read {path!r}: {exc}"
                ) from exc
        else:
            s = token.strip()
            if s:
                out.append(s)
    return out


def _print_summary(summary: object) -> None:
    """Render the finalize-response summary as a small ASCII table.

    Mirrors the 5-metric layout the portal DiffCard shows so the CLI
    output reads the same way as the web UI — "Recommend: ship vN"
    in both places, no cognitive translation step.
    """
    if not isinstance(summary, dict):
        print("\n(no summary returned)")
        return
    print("\nSummary:")
    for key in (
        "success_rate",
        "reward_mean",
        "collision_rate",
        "time_to_goal_s",
        "ood_action_share",
    ):
        metric = summary.get(key)
        if not isinstance(metric, dict):
            continue
        b = metric.get("baseline")
        c = metric.get("candidate")
        d = metric.get("delta")
        better = metric.get("delta_is_better")
        sign = " " if better is None else ("✓" if better else "✗")
        print(
            f"  {sign} {key:<20} baseline={_fmt(b):>10}  "
            f"candidate={_fmt(c):>10}  Δ={_fmt(d, signed=True):>10}"
        )
    rec = summary.get("recommend")
    better_count = summary.get("better_count")
    metric_total = summary.get("metric_total")
    if rec:
        print(
            f"\nRecommend: {rec}  "
            f"({better_count}/{metric_total} metrics better)"
        )


def _fmt(value: object, *, signed: bool = False) -> str:
    """Format a metric value for the summary table.

    `signed=True` prepends `+` to non-negative numerics so deltas
    line up visually (``+0.123`` / ``-0.045`` / ``+0.000``). We
    can't get the same effect with an f-string spec like ``:>+10``
    in the caller — that's only legal on numeric inputs, and the
    helper already coerces None / bool / strings to text earlier.
    """
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        try:
            v = float(value)
            return f"{v:+.3f}" if signed else f"{v:.3f}"
        except (TypeError, ValueError):
            return str(value)
    return str(value)


# ── module-level run hook ──────────────────────────────────────────────


def main() -> NoReturn:  # pragma: no cover — thin wrapper
    sys.exit(cli_main())


if __name__ == "__main__":  # pragma: no cover
    main()
