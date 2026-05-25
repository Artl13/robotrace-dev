"""``DeprecationWarning`` helper used by the public API surface.

The SDK 0.2.0 readiness checklist (gate 3 of 4) requires that "a real
``DeprecationWarning`` helper exists and has been exercised
end-to-end by removing one already-deprecated thing through it, so
we know the path works before promising to use it." This module is
that helper.

Design notes
------------

* The helper is **private** (underscore-prefixed module + leading-
  underscore convention).  Public API stays free of incidental
  ``import`` noise; we never want to commit to keeping a deprecation
  helper API stable forever.
* It emits a single ``DeprecationWarning`` instance per call site
  per process. ``warnings`` machinery already handles this via the
  default ``__warningregistry__`` cache, so we don't need to roll
  our own dedup.
* It uses ``warnings.warn(..., stacklevel=N)`` to point the warning
  at the **user's** call site, not at our wrapper. ``stacklevel=2``
  matches the stdlib convention when the helper is called *from
  inside* the deprecated function (e.g. inside
  :meth:`Episode.upload_video`), since we add one extra frame.
* Message format is stable:

      "<name> is deprecated since <since> and will be removed in
      <removed_in>. Use <replacement> instead. <hint> (RoboTrace
      SDK)"

  The trailing ``(RoboTrace SDK)`` tag lets callers grep / filter
  warnings.warn output deterministically when they're triaging
  noisy logs.
"""

from __future__ import annotations

import warnings

__all__ = ["warn_deprecated"]

_TAG = "(RoboTrace SDK)"


def warn_deprecated(
    name: str,
    *,
    since: str,
    removed_in: str,
    replacement: str | None = None,
    hint: str | None = None,
    stacklevel: int = 2,
) -> None:
    """Emit a ``DeprecationWarning`` for a public-API element.

    Call this from *inside* the deprecated function or method so the
    warning points at the caller's source line, not at the SDK.

    Parameters
    ----------
    name
        The fully-qualified user-visible name of the deprecated
        thing, e.g. ``"Episode.upload_video"``. Shown verbatim in the
        warning message.
    since
        Version in which the deprecation took effect, e.g.
        ``"0.1.0a13"``. Use the version string that ships the
        deprecation, not the version that introduced the feature.
    removed_in
        Version in which the deprecated thing will stop working
        entirely, e.g. ``"0.3.0"``. The promise to users; bump
        a major (or, while in 0.x, a minor) to keep it honest.
    replacement
        Optional canonical replacement, e.g.
        ``"Episode.upload(kind, path)"``. Rendered as
        ``"Use <replacement> instead."`` when present.
    hint
        Optional extra sentence explaining motivation or a migration
        gotcha. Rendered verbatim before the trailing ``(RoboTrace
        SDK)`` tag.
    stacklevel
        Same semantics as ``warnings.warn``'s ``stacklevel``: 1
        points at this helper itself, 2 points at the caller of
        this helper (the *typical* deprecated function), 3 points
        at the user that called the deprecated function. Defaults
        to 2 because that's the right value when the helper is
        invoked from *inside* the deprecated function.

        We internally add 1 so callers can think in stdlib terms
        (``stacklevel=2`` => "point at my caller"), matching the
        convention every other stdlib deprecation helper follows
        (e.g. :func:`functools.wraps` chains, ``typing.deprecated``
        in 3.13, etc.).
    """
    parts: list[str] = [
        f"{name} is deprecated since {since} and will be removed in {removed_in}."
    ]
    if replacement:
        parts.append(f"Use {replacement} instead.")
    if hint:
        parts.append(hint)
    parts.append(_TAG)
    message = " ".join(parts)

    # Add 1 because the user's stacklevel value is "frames above my
    # caller"; warnings.warn measures from itself.
    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel + 1)
