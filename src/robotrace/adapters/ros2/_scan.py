"""`scan_bag(...)` — read-only introspection of a rosbag2 directory.

Returns a `BagSummary` describing every connection, the auto-classified
artifact slot, message counts, and the bag's wall-clock duration. No
files are written.

Used by:
  • users who want to see what's in a bag before deciding to upload
  • `encode_bag(...)` and `upload_bag(...)` as the first step

Backed by `rosbags.highlevel.AnyReader` so both sqlite3 and mcap
storage backends work without per-call branching.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ...errors import ConfigurationError
from ._classify import Slot, TopicClass, classify_topic


@dataclass(frozen=True)
class TopicInfo:
    """Per-topic facts surfaced from the bag."""

    topic: str
    msgtype: str
    msgcount: int
    slot: Slot
    # Carries through the `TopicClass.reason` so `BagSummary.report()`
    # can show why a slot was picked.
    classified_by: str


@dataclass
class BagSummary:
    """What a bag contains and how the adapter would treat it.

    Returned by `scan_bag`. Everything here is computed before any
    encoding, so users can dry-run (`scan_bag(path).report()`) before
    paying the cost of a full encode + upload.
    """

    path: Path
    # Wall-clock seconds between the first and last message across
    # all connections. None if the bag is empty (no messages at all).
    duration_s: float | None
    message_count: int
    topics: list[TopicInfo] = field(default_factory=list)

    def topics_by_slot(self, slot: Slot) -> list[TopicInfo]:
        """All topics the auto-classifier routed to one slot."""
        return [t for t in self.topics if t.slot == slot]

    def report(self) -> str:
        """Human-readable summary, one topic per line.

        Used by docs and by users who do a `print(scan_bag(p).report())`
        before invoking `upload_bag(...)`. Stable enough to assert in
        tests but not promised as a parser-friendly format.
        """
        lines: list[str] = []
        path_repr = str(self.path)
        if self.duration_s is None:
            lines.append(f"{path_repr}  (empty bag)")
        else:
            lines.append(
                f"{path_repr}  {self.duration_s:.2f}s, "
                f"{self.message_count} message{'s' if self.message_count != 1 else ''}"
            )
        for slot in ("video", "sensors", "actions"):
            topics = self.topics_by_slot(slot)
            if not topics:
                continue
            lines.append(f"  {slot}:")
            for t in topics:
                lines.append(
                    f"    {t.topic}  ({t.msgtype}, {t.msgcount} msg, "
                    f"via {t.classified_by})"
                )
        return "\n".join(lines)


def scan_bag(path: str | Path) -> BagSummary:
    """Open a rosbag2 directory and describe its contents.

    `path` is the bag *directory* (the one containing `metadata.yaml`),
    not an individual `.db3` / `.mcap` file. Both sqlite3 and mcap
    backends are accepted — `rosbags` picks the right reader.

    Raises `ConfigurationError` if `rosbags` isn't installed or `path`
    doesn't look like a rosbag2 directory; both errors carry actionable
    install / usage hints.
    """
    bag_path = _resolve_bag_path(path)
    AnyReader = _import_anyreader()

    with AnyReader([bag_path]) as reader:
        connections = list(reader.connections)
        topics = [_topic_info(c) for c in connections]
        # `rosbags` exposes start/end timestamps in nanoseconds on the
        # AnyReader — they're ints since the rosbag2 metadata format
        # is nanoseconds-since-epoch. Empty bag → start == end == 0,
        # which we surface as duration_s=None instead of 0.0 so the
        # caller can branch on "no data" cleanly.
        message_count = sum(t.msgcount for t in topics)
        if message_count == 0:
            duration_s: float | None = None
        else:
            duration_ns = max(reader.duration, 0)
            duration_s = duration_ns / 1_000_000_000

    return BagSummary(
        path=bag_path,
        duration_s=duration_s,
        message_count=message_count,
        topics=topics,
    )


# ── internals ─────────────────────────────────────────────────────────


def _topic_info(connection: object) -> TopicInfo:
    """Adapt one rosbags `Connection` into our public dataclass.

    `connection` is intentionally typed as `object` — we don't import
    rosbags' types into the public surface, so we pull `.topic`,
    `.msgtype`, `.msgcount` defensively via `getattr` and let the
    underlying library evolve its internals freely.
    """
    topic = str(getattr(connection, "topic", ""))
    msgtype = str(getattr(connection, "msgtype", ""))
    msgcount = int(getattr(connection, "msgcount", 0) or 0)
    decision: TopicClass = classify_topic(topic, msgtype)
    return TopicInfo(
        topic=topic,
        msgtype=msgtype,
        msgcount=msgcount,
        slot=decision.slot,
        classified_by=decision.reason,
    )


def _resolve_bag_path(path: str | Path) -> Path:
    """Validate `path` looks like a rosbag2 directory.

    rosbag2 stores a `metadata.yaml` next to the storage file(s), which
    is the most reliable signal that the directory is a bag (the
    storage extension differs between sqlite3 and mcap, the directory
    name carries no convention).
    """
    bag_path = Path(path).expanduser().resolve()
    if not bag_path.exists():
        raise ConfigurationError(f"rosbag2 path does not exist: {bag_path}")
    if not bag_path.is_dir():
        raise ConfigurationError(
            f"rosbag2 path must be the bag directory, not a file: {bag_path}. "
            "Pass the directory containing metadata.yaml."
        )
    if not (bag_path / "metadata.yaml").is_file():
        raise ConfigurationError(
            f"no metadata.yaml in {bag_path} — is this really a rosbag2 directory? "
            "If your bag is a single .mcap file, place it in a directory with "
            "metadata.yaml alongside it (the standard rosbag2 layout)."
        )
    return bag_path


def _import_anyreader() -> type:
    """Lazy-import `rosbags` so the SDK base install isn't bloated.

    Surfaces a friendly error pointing at the right extras pin if the
    user installed plain `robotrace` without `[ros2]`.
    """
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as exc:
        raise ConfigurationError(
            "the ROS 2 adapter needs the `rosbags` library. "
            "Install with `pip install 'robotrace-dev[ros2]==0.1.0a3'`."
        ) from exc
    return AnyReader


def iter_connections_for_topics(reader: object, topics: Iterable[str]) -> list[object]:
    """Helper used by the encoder — the connections matching `topics`.

    Kept here (rather than in `_encode.py`) so both `_scan` and `_encode`
    use the same defensive `getattr` pattern when they pull `.topic`
    off a rosbags `Connection`.
    """
    wanted = set(topics)
    out: list[object] = []
    for connection in getattr(reader, "connections", []) or []:
        if str(getattr(connection, "topic", "")) in wanted:
            out.append(connection)
    return out
