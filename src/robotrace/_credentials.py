"""On-disk credentials store for the `robotrace` CLI.

The CLI's `login` command writes a small TOML file at
``~/.robotrace/credentials`` that the Python SDK auto-loads when
neither an explicit `api_key=` kwarg nor the `ROBOTRACE_API_KEY`
environment variable is set. The same file backs `whoami` and
`logout`.

Format
------

::

    [default]
    api_key   = "rt_…"
    base_url  = "https://app.robotrace.dev"
    client_id = "<uuid>"
    user_email = "art@robotrace.dev"
    written_at = "2026-05-04T20:08:14Z"

We support **profiles** in case a single workstation talks to more
than one deployment (production + a staging cell). The MVP only
reads/writes ``[default]``; the file format leaves room to extend
without breaking older SDKs that ignore unknown profiles.

Security
--------

The file holds a long-lived API key, so the writer enforces
``chmod 0600`` after writing. We never log the key value. Reads
fail loudly if the file is world-readable on systems that ship a
strict ``UMASK`` policy — the user can re-run ``robotrace login``
and the file will be re-created with the right perms.

Cross-platform notes
--------------------

* On POSIX we ``os.chmod`` to 0600 after a fresh write.
* On Windows ``os.chmod`` is largely a no-op for the read/write/
  execute bits we care about, so we settle for storing in the
  user's profile directory and rely on filesystem ACLs there.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# tomllib only landed in 3.11; we still support 3.10 per pyproject.
if sys.version_info >= (3, 11):
    import tomllib as _tomllib  # noqa: PLC0415
else:  # pragma: no cover — exercised on 3.10
    _tomllib = None  # type: ignore[assignment]


CREDENTIALS_DIR_NAME = ".robotrace"
CREDENTIALS_FILE_NAME = "credentials"
DEFAULT_PROFILE = "default"


@dataclass
class StoredCredentials:
    """One profile's worth of credentials."""

    api_key: str
    base_url: str
    client_id: str | None = None
    user_email: str | None = None
    written_at: str | None = None


def credentials_path() -> Path:
    """Resolve the path to the credentials file.

    Uses ``$ROBOTRACE_HOME`` when set (handy for tests and CI), then
    falls back to ``~/.robotrace`` on every platform — matches the
    convention popular among devtools (``~/.aws``, ``~/.docker``,
    ``~/.kube``).
    """
    override = os.environ.get("ROBOTRACE_HOME")
    if override:
        return Path(override) / CREDENTIALS_FILE_NAME
    return Path.home() / CREDENTIALS_DIR_NAME / CREDENTIALS_FILE_NAME


def write_credentials(creds: StoredCredentials, *, profile: str = DEFAULT_PROFILE) -> Path:
    """Persist `creds` under `profile` in the credentials file.

    Returns the absolute path written. Creates the parent dir with
    mode 0700, then writes the file with mode 0600. Raises on any
    filesystem error — login is the last step of the flow, so a
    failure to persist is worth surfacing loudly.
    """
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    existing = _read_all_profiles(path)
    existing[profile] = {
        "api_key": creds.api_key,
        "base_url": creds.base_url,
        "client_id": creds.client_id,
        "user_email": creds.user_email,
        "written_at": creds.written_at or _now_iso(),
    }

    _atomic_write_toml(path, existing)
    try:
        os.chmod(path, 0o600)
    except OSError:
        # On Windows / odd filesystems chmod can fail; the dir
        # permission and user-profile location are our backstop.
        pass
    return path


def read_credentials(*, profile: str = DEFAULT_PROFILE) -> StoredCredentials | None:
    """Load `profile` from the credentials file, or `None` if absent.

    Returns `None` for any of: missing dir, missing file, profile
    not present, malformed file. The SDK falls back to env-var auth
    in those cases — we never want a corrupt creds file to surface
    as a confusing "API key not provided" error.
    """
    path = credentials_path()
    if not path.is_file():
        return None
    try:
        all_profiles = _read_all_profiles(path)
    except Exception:
        return None
    raw = all_profiles.get(profile)
    if not isinstance(raw, dict):
        return None
    api_key = raw.get("api_key")
    base_url = raw.get("base_url")
    if not isinstance(api_key, str) or not isinstance(base_url, str):
        return None
    if not api_key or not base_url:
        return None
    client_id = raw.get("client_id")
    user_email = raw.get("user_email")
    written_at = raw.get("written_at")
    return StoredCredentials(
        api_key=api_key,
        base_url=base_url,
        client_id=client_id if isinstance(client_id, str) else None,
        user_email=user_email if isinstance(user_email, str) else None,
        written_at=written_at if isinstance(written_at, str) else None,
    )


def delete_credentials(*, profile: str = DEFAULT_PROFILE) -> bool:
    """Remove `profile` from the credentials file. Returns True if
    something was removed, False otherwise.

    If the resulting file has no profiles left, the file (and its
    parent dir, if empty) are deleted to leave a clean filesystem.
    """
    path = credentials_path()
    if not path.is_file():
        return False
    try:
        existing = _read_all_profiles(path)
    except Exception:
        return False
    if profile not in existing:
        return False
    existing.pop(profile)
    if existing:
        _atomic_write_toml(path, existing)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    else:
        try:
            path.unlink()
        except OSError:
            return False
        try:
            path.parent.rmdir()
        except OSError:
            # Dir not empty (other tools' creds in there) — fine.
            pass
    return True


# ── internals ────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_all_profiles(path: Path) -> dict[str, object]:
    """Parse the whole credentials file into a {profile: {...}} dict.

    Tolerant of either canonical TOML (which `_atomic_write_toml`
    produces) or a fallback JSON-encoded body that older SDKs
    might have written. Returns ``{}`` on parse errors so the
    caller can treat it as a fresh write.
    """
    if _tomllib is not None:
        try:
            with path.open("rb") as fh:
                return dict(_tomllib.load(fh))
        except Exception:
            # fall through to JSON
            pass

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}

    # Last-resort JSON parser for environments without tomllib that
    # somehow ended up with a JSON-formatted creds file.
    try:
        loaded = json.loads(text)
    except Exception:
        return {}
    return dict(loaded) if isinstance(loaded, dict) else {}


def _atomic_write_toml(path: Path, data: dict[str, object]) -> None:
    """Write `data` to `path` as TOML, atomically.

    We don't take a hard dep on a TOML *writer* (the stdlib gained
    one in 3.11 only as ``tomllib`` for *reading*). Hand-rolling the
    minimal subset we use — ``[section]`` headers and ``key = "str"``
    pairs — avoids a third-party dep on the SDK install path.
    """
    lines: list[str] = ["# robotrace credentials\n# managed by `robotrace login` — do not commit.\n\n"]
    # Stable ordering keeps diffs readable when the user opens the
    # file out of curiosity.
    for profile in sorted(data.keys()):
        section = data[profile]
        if not isinstance(section, dict):
            continue
        lines.append(f"[{profile}]\n")
        for key in sorted(section.keys()):
            value = section.get(key)
            if value is None:
                continue
            if not isinstance(value, str):
                # Force-stringify; the schema is all-strings today.
                value = str(value)
            lines.append(f'{key} = "{_escape_toml(value)}"\n')
        lines.append("\n")

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(lines), encoding="utf-8")
    os.replace(tmp, path)


def _escape_toml(value: str) -> str:
    """Escape the bare-minimum characters that break a basic TOML string."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
