"""Best-effort OpenTelemetry trace correlation.

Public surface — one function: ``capture_trace_context()``. It returns
the active OTel trace+span as a small dict the SDK attaches to the
``start_episode`` payload, or ``None`` when there's nothing to attach.

Why this lives in its own module
--------------------------------

OpenTelemetry is **optional**. Most teams running policy training jobs
don't have an OTel pipeline (yet); the ones that do shouldn't pay the
cost of maintaining a parallel "RoboTrace SDK is on a different timeline
than my APM" mental model. Sticking the integration behind a soft
import keeps three guarantees:

1.  ``import robotrace`` never raises because OTel isn't installed.
2.  Calling ``log_episode(...)`` outside an OTel span behaves exactly
    the same as before — the field is simply absent from the payload.
3.  Calling ``log_episode(...)`` inside an active span attaches
    ``trace_id`` / ``span_id`` / ``traceparent`` automatically, with
    zero new kwargs (and therefore zero risk of breaking the
    "sacred" signature locked by AGENTS.md).

W3C Trace Context format
------------------------

We emit both the raw IDs *and* the W3C ``traceparent`` header value:

    00-<32-hex-trace-id>-<16-hex-span-id>-01

The IDs are friendlier in the portal UI (deep-linkable into Datadog /
Honeycomb / Tempo / Jaeger via a template URL). The traceparent string
is what every other OTel-aware system on the team's network expects to
see, so propagating it makes the episode row a first-class member of
the trace graph — e.g. you can paste it into a curl call to replay a
downstream service with the same trace context.

Sampling note
-------------

We deliberately do **not** check ``span.get_span_context().trace_flags``
for the sampled bit. If OTel was sampled out, ``get_current_span()``
returns ``INVALID_SPAN`` and we already short-circuit to ``None`` —
honoring user sampling. But if the span is recorded *and not exported*
(local trace context only), we still want to pin it on the episode so
the user can replay the episode to find what their policy was doing
at that exact ``trace_id`` later. The episode row is its own retention
domain.
"""

from __future__ import annotations

from typing import TypedDict

# OpenTelemetry is an optional dep, gated behind the `[otel]` extra.
# Soft-import so the SDK keeps working when the extra isn't installed.
try:
    from opentelemetry import trace as _otel_trace  # type: ignore[import-not-found]

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover — exercised by tests via monkeypatch
    _otel_trace = None  # type: ignore[assignment]
    _OTEL_AVAILABLE = False


class TraceContext(TypedDict):
    """Subset of the W3C Trace Context we attach to an episode.

    Field names match what the server-side Zod schema expects (snake
    case, since the rest of the wire format is snake case). Format
    follows OTel's ID conventions:

    - ``trace_id`` — 32 hex chars, no dashes (16 raw bytes)
    - ``span_id``  — 16 hex chars, no dashes (8 raw bytes)
    - ``traceparent`` — full W3C header value (``00-trace-span-flags``)

    Two extra fields hint at the user's vendor + service so the portal
    can show "from datadog · my-policy-server" without a second
    round-trip. Both are best-effort; missing values render as muted
    secondary text.
    """

    trace_id: str
    span_id: str
    traceparent: str


def is_available() -> bool:
    """``True`` when the ``opentelemetry-api`` package is importable.

    Exposed for the docs page and for tests that want to assert the
    soft-import didn't degrade silently. Production callers should
    never need to branch on this — ``capture_trace_context()`` is the
    one thing to call.
    """
    return _OTEL_AVAILABLE


def capture_trace_context() -> TraceContext | None:
    """Return the active OTel trace context, or ``None``.

    Returns ``None`` when:

    - OpenTelemetry isn't installed (``[otel]`` extra missing); or
    - OTel is installed but there's no active span (``INVALID_SPAN``);
      or
    - OTel raises any exception while resolving the span (defensive —
      we never want SDK telemetry to fail a customer's training run).

    Otherwise returns a ``TraceContext`` dict ready to drop into a
    ``start_episode`` payload as ``payload["otel"] = ctx``.
    """
    if not _OTEL_AVAILABLE or _otel_trace is None:
        return None

    try:
        span = _otel_trace.get_current_span()
    except Exception:
        return None
    if span is None:
        return None

    try:
        ctx = span.get_span_context()
    except Exception:
        return None

    # OTel uses sentinel zero IDs (``INVALID_TRACE_ID`` /
    # ``INVALID_SPAN_ID``) for "no active span". Don't propagate
    # those — the portal would render an unclickable string of
    # zeroes, and an APM deep-link would 404.
    trace_id_int = getattr(ctx, "trace_id", 0)
    span_id_int = getattr(ctx, "span_id", 0)
    if not trace_id_int or not span_id_int:
        return None

    trace_hex = f"{trace_id_int:032x}"
    span_hex = f"{span_id_int:016x}"

    # W3C traceparent: version-trace-span-flags. We always emit
    # version 00 (the only version defined today). `trace_flags` is
    # 8 bits — the low bit is "sampled". Default to 01 ("sampled")
    # because the user is explicitly opting in by passing through
    # log_episode; if they want the unsampled bit they can post-edit
    # the metadata.
    flags_int = getattr(ctx, "trace_flags", 1) or 1
    traceparent = f"00-{trace_hex}-{span_hex}-{flags_int:02x}"

    return TraceContext(
        trace_id=trace_hex,
        span_id=span_hex,
        traceparent=traceparent,
    )
