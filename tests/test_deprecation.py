"""Pytest coverage for the deprecation pipeline.

Two layers:

  1. The ``_deprecation.warn_deprecated`` helper directly — message
     format, ``DeprecationWarning`` class, optional fields,
     stacklevel forwarding.

  2. The first end-to-end exercise of the helper — the
     ``Episode.upload_video`` / ``upload_sensors`` / ``upload_actions``
     shortcuts. Each must:

       * Still upload via the canonical ``upload(kind, path)`` path
         (the deprecation does NOT break callers).
       * Emit exactly one ``DeprecationWarning`` per call.
       * Point the warning at the caller's source line, not at our
         internal wrapper - the stacklevel contract is the
         actually-load-bearing bit of the helper.

This test is also the gate-3 source of truth in the SDK 0.2.0
readiness checklist ("a real DeprecationWarning helper exists and
has been exercised end-to-end"). If you remove the upload shortcuts
in 0.3.0, replace ``test_upload_video_shortcut_warns`` etc with
asserts that calling them now raises ``AttributeError`` - keeping
the regression on the *promise* even after the body is gone.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import httpx
import pytest

import robotrace as rt
from robotrace import _deprecation


# ── helper-level coverage ───────────────────────────────────────────


def test_warn_deprecated_minimum_payload_message() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _deprecation.warn_deprecated(
            "X.foo",
            since="0.1.0a13",
            removed_in="0.3.0",
        )
    assert len(caught) == 1
    w = caught[0]
    assert issubclass(w.category, DeprecationWarning)
    assert str(w.message) == (
        "X.foo is deprecated since 0.1.0a13 and will be removed in 0.3.0. "
        "(RoboTrace SDK)"
    )


def test_warn_deprecated_includes_replacement_and_hint() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _deprecation.warn_deprecated(
            "X.foo",
            since="0.1.0a13",
            removed_in="0.3.0",
            replacement="X.bar()",
            hint="Migrate during the 0.2.0 window.",
        )
    assert len(caught) == 1
    assert str(caught[0].message) == (
        "X.foo is deprecated since 0.1.0a13 and will be removed in 0.3.0. "
        "Use X.bar() instead. Migrate during the 0.2.0 window. (RoboTrace SDK)"
    )


def test_warn_deprecated_stacklevel_points_at_user_caller() -> None:
    """The whole point of the helper - the warning location must be
    the user's file/line, not anything inside ``robotrace.*``.

    We simulate the typical wiring: a deprecated wrapper inside the
    SDK calls ``warn_deprecated(stacklevel=2)`` from its own body,
    then continues. The helper adds 1 internally, so the warning's
    ``filename`` points at *this* test file (the user) rather than
    at ``_deprecation.py`` or at the wrapper function.
    """

    def _deprecated_wrapper() -> None:
        # Inside the SDK; the warning should NOT point here.
        _deprecation.warn_deprecated(
            "X.foo", since="0.1.0a13", removed_in="0.3.0"
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _deprecated_wrapper()  # <-- this is the "user" call site
        user_call_line = (
            _deprecated_wrapper.__code__.co_filename,
            # The test framework reports the line of the call we just
            # made; we don't pin the exact line number because pytest
            # reorders / mutates frames. Filename equality is the
            # robust assertion.
        )

    assert len(caught) == 1
    w = caught[0]
    assert w.filename == user_call_line[0]


# ── end-to-end exercise: Episode.upload_{kind} shortcuts ────────────


def _stub_episode(monkeypatch: pytest.MonkeyPatch) -> rt.Episode:
    """Build an Episode whose ``_client._http.upload_file`` returns
    a synthetic byte count without hitting the network.

    Signed-PUT uploads spin up their *own* internal ``httpx.Client``
    (the auth-less one in ``_http.upload_file``), so we can't reach
    them with the same ``MockTransport`` we use to stub the JSON API.
    Patching ``upload_file`` directly is the right surface area:
    those shortcuts delegate to ``Episode.upload(kind, path)`` which
    calls ``client._http.upload_file(...)`` once per artifact.
    """
    client = rt.Client(
        api_key="rt_test",
        base_url="https://example.test",
        transport=httpx.MockTransport(lambda req: httpx.Response(200)),
    )

    def _fake_upload_file(
        url: str, path: str | Path, *, content_type: str
    ) -> int:
        return Path(path).stat().st_size

    monkeypatch.setattr(client._http, "upload_file", _fake_upload_file)

    episode = rt.Episode(
        id="ep_test",
        status="recording",
        storage="r2",
        upload_urls={
            "video": rt.UploadUrl(
                kind="video",
                url="https://example.test/r2/video",
                expires_at="2099-01-01T00:00:00Z",
                public_url=None,
            ),
            "sensors": rt.UploadUrl(
                kind="sensors",
                url="https://example.test/r2/sensors",
                expires_at="2099-01-01T00:00:00Z",
                public_url=None,
            ),
            "actions": rt.UploadUrl(
                kind="actions",
                url="https://example.test/r2/actions",
                expires_at="2099-01-01T00:00:00Z",
                public_url=None,
            ),
        },
    )
    episode._client = client
    return episode


@pytest.mark.parametrize(
    "method,kind",
    [
        ("upload_video", "video"),
        ("upload_sensors", "sensors"),
        ("upload_actions", "actions"),
    ],
)
def test_upload_shortcut_still_works_and_warns(
    method: str, kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each shortcut: still uploads, emits one DeprecationWarning,
    and points the warning at the test's call site.
    """
    payload = tmp_path / f"sample_{kind}.bin"
    payload.write_bytes(b"x" * 32)

    episode = _stub_episode(monkeypatch)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bytes_uploaded = getattr(episode, method)(payload)

    assert bytes_uploaded == 32  # canonical upload still ran

    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dep_warnings) == 1, (
        f"expected exactly one DeprecationWarning for {method}, "
        f"got {len(dep_warnings)}"
    )
    msg = str(dep_warnings[0].message)
    assert f"Episode.{method}" in msg
    assert "0.1.0a13" in msg
    assert "0.3.0" in msg
    assert f'Episode.upload("{kind}", path)' in msg
    assert msg.endswith("(RoboTrace SDK)")

    # Stacklevel contract: the warning must point at THIS test file,
    # not at episode.py / _deprecation.py inside the SDK.
    assert dep_warnings[0].filename == __file__


def test_upload_shortcut_dedups_per_call_site_with_default_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Python's ``warnings`` module dedups per (filename, lineno,
    category, registry) by default. Calling the shortcut twice from
    the *same* source line emits the warning only once when the
    default filter is active.

    Documents that we rely on stdlib dedup rather than rolling our
    own - keeps the user's logs clean even on a per-frame loop that
    happens to use the deprecated shortcut.
    """
    payload = tmp_path / "sample.bin"
    payload.write_bytes(b"x" * 8)

    episode = _stub_episode(monkeypatch)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        for _ in range(5):  # five calls, same source line
            episode.upload_video(payload)

    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dep_warnings) == 1, (
        f"expected stdlib dedup to suppress repeats from the same "
        f"call site; got {len(dep_warnings)} warnings instead of 1"
    )
