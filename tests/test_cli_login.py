"""``robotrace login`` skips the browser flow when already authorized."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from robotrace import cli
from robotrace._credentials import DEFAULT_PROFILE, StoredCredentials, write_credentials


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ROBOTRACE_HOME", str(tmp_path))
    return tmp_path


def _seed_credentials(
    *,
    base_url: str = "http://localhost:3000",
    profile: str = DEFAULT_PROFILE,
) -> None:
    write_credentials(
        StoredCredentials(
            api_key="rt_id123example_abcdefghijklmnopqrstuvwxyz012345",
            base_url=base_url,
            client_id="00000000-0000-0000-0000-000000000001",
            user_email="user@example.test",
        ),
        profile=profile,
    )


def test_login_when_already_signed_in_skips_browser(
    isolated_home: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_credentials()
    monkeypatch.setenv("ROBOTRACE_BASE_URL", "http://localhost:3000")

    def fail_start(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("login should not start a device session when already signed in")

    monkeypatch.setattr(cli, "_start_device_session", fail_start)

    rc = cli.cli_main(["login"])
    out = capsys.readouterr()

    assert rc == 0
    assert "Already signed in" in out.out
    assert "user@example.test" in out.out
    assert "http://localhost:3000" in out.out
    assert "login --force" in out.out


def test_login_force_starts_browser_flow(
    isolated_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_credentials()
    monkeypatch.setenv("ROBOTRACE_BASE_URL", "http://localhost:3000")

    called = {"start": False}

    def fake_start(base_url: str) -> dict[str, object]:
        called["start"] = True
        raise cli._CliError("stop test after start")

    monkeypatch.setattr(cli, "_start_device_session", fake_start)

    rc = cli.cli_main(["login", "--force"])
    assert called["start"] is True
    assert rc != 0
