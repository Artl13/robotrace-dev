"""`encode_bag(...)` — write artifacts to disk from a rosbag2.

Walks the bag once, encoding every classified topic into the shape the
RoboTrace ingest endpoints expect:

    video    →  one MP4 (single-camera passthrough, or horizontal
                tile when multiple Image topics are present)
    sensors  →  one NPZ holding every sensor topic's flattened
                fields (`<topic>/<field>` keys, plus `<topic>/_t_ns`
                timestamps)
    actions  →  one NPZ in the same shape as sensors

The encoder never opens the network. Use `upload_bag(...)` for that.

Dependency strategy:
  * `rosbags` is required (pulled in by `[ros2]`).
  * `numpy` is required for the NPZ writers.
  * `cv2` (opencv) is required *only* if at least one Image /
    CompressedImage topic actually gets encoded — sensor-only bags
    don't pay the cost. Missing-opencv error points at the
    `[ros2,video]` extras combination.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...errors import ConfigurationError
from ._classify import Slot
from ._scan import BagSummary, scan_bag

# Default frame rate when an Image topic has no usable timestamps
# (rare, but happens with synthesised bags that all share one stamp).
# 10 fps is slow enough that the resulting MP4 is obviously a fallback
# and not silently wrong — production bags with real stamps override.
_FALLBACK_FPS: float = 10.0


@dataclass
class EncodedArtifact:
    """One encoded file ready to upload."""

    slot: Slot
    path: Path
    bytes_size: int
    # Topics that contributed data to this file. For `video` this is
    # the camera ordering used in a tiled mosaic; for sensors / actions
    # it's the topics whose fields were packed into the NPZ.
    topics: list[str] = field(default_factory=list)


@dataclass
class EncodedBag:
    """The product of `encode_bag(...)`.

    The `video` / `sensors` / `actions` keys mirror the SDK's
    artifact slots so callers can splat into `start_episode` +
    `upload_*` directly:

        encoded = ros2.encode_bag(bag, output_dir)
        with client.start_episode(...) as ep:
            if encoded.video:    ep.upload_video(encoded.video.path)
            if encoded.sensors:  ep.upload_sensors(encoded.sensors.path)
            if encoded.actions:  ep.upload_actions(encoded.actions.path)
            ep.finalize(duration_s=encoded.duration_s, fps=encoded.fps)

    `metadata` carries any per-topic notes the encoder wants the
    server to see (skipped messages, fallback fps, …). Callers
    typically merge it into their own `metadata=` dict on finalize.
    """

    output_dir: Path
    summary: BagSummary
    video: EncodedArtifact | None = None
    sensors: EncodedArtifact | None = None
    actions: EncodedArtifact | None = None
    duration_s: float | None = None
    fps: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def artifacts(self) -> list[EncodedArtifact]:
        """Every non-None artifact, in canonical slot order."""
        out: list[EncodedArtifact] = []
        for art in (self.video, self.sensors, self.actions):
            if art is not None:
                out.append(art)
        return out


def encode_bag(
    path: str | Path,
    output_dir: str | Path,
    *,
    video_topics: Sequence[str] | None = None,
    sensor_topics: Sequence[str] | None = None,
    action_topics: Sequence[str] | None = None,
    canonical_video_topic: str | None = None,
    summary: BagSummary | None = None,
) -> EncodedBag:
    """Encode every classified topic in `path` to files under `output_dir`.

    `output_dir` is created if it doesn't exist. Three filenames are
    reserved inside it: ``video.mp4``, ``sensors.npz``, ``actions.npz``.

    Parameters mirror the auto-classifier — pass an explicit topic list
    to override the heuristic for one slot. Pass an empty list (`[]`)
    to deliberately *exclude* a slot. ``canonical_video_topic`` picks
    one camera as the only video output (skipping the multi-cam tile)
    when more than one image topic exists.
    """
    bag_path = Path(path).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if summary is None:
        summary = scan_bag(bag_path)

    # Resolve which topics actually go into each slot. Explicit kwargs
    # (including empty lists) win over the auto-classifier.
    video_list = _resolve_slot(summary, "video", video_topics)
    sensor_list = _resolve_slot(summary, "sensors", sensor_topics)
    action_list = _resolve_slot(summary, "actions", action_topics)

    if canonical_video_topic is not None:
        if canonical_video_topic not in video_list:
            raise ConfigurationError(
                f"canonical_video_topic={canonical_video_topic!r} is not in "
                f"the resolved video topic list {video_list}."
            )
        video_list = [canonical_video_topic]

    metadata: dict[str, Any] = {"adapter": "ros2", "bag": str(bag_path)}
    skipped: list[dict[str, Any]] = []

    AnyReader = _import_anyreader()
    with AnyReader([bag_path]) as reader:
        # rosbags exposes `connections` once per topic+msgtype pair.
        # We index by topic so the encoders below can grab the
        # connection without re-walking the list.
        conn_by_topic = {
            str(getattr(c, "topic", "")): c for c in (reader.connections or [])
        }

        # --- video -----------------------------------------------------
        encoded_video: EncodedArtifact | None = None
        encoded_fps: float | None = None
        if video_list:
            encoded_video, encoded_fps = _encode_video(
                reader=reader,
                conn_by_topic=conn_by_topic,
                topics=video_list,
                output_path=out_dir / "video.mp4",
                skipped=skipped,
            )

        # --- sensors / actions ----------------------------------------
        encoded_sensors = _encode_messages(
            reader=reader,
            conn_by_topic=conn_by_topic,
            topics=sensor_list,
            slot="sensors",
            output_path=out_dir / "sensors.npz",
            skipped=skipped,
        )
        encoded_actions = _encode_messages(
            reader=reader,
            conn_by_topic=conn_by_topic,
            topics=action_list,
            slot="actions",
            output_path=out_dir / "actions.npz",
            skipped=skipped,
        )

    if skipped:
        metadata["skipped_topics"] = skipped

    return EncodedBag(
        output_dir=out_dir,
        summary=summary,
        video=encoded_video,
        sensors=encoded_sensors,
        actions=encoded_actions,
        duration_s=summary.duration_s,
        fps=encoded_fps,
        metadata=metadata,
    )


# ── slot resolution ───────────────────────────────────────────────────


def _resolve_slot(
    summary: BagSummary,
    slot: Slot,
    explicit: Sequence[str] | None,
) -> list[str]:
    """Pick the topics that go into one slot.

    `explicit=None` means "use the auto-classified set". `explicit=[]`
    is the explicit "none" — caller wants to skip this slot. Any other
    list overrides the classifier entirely (including topics the
    classifier put in a different slot — useful for projects that use
    non-standard message types for commands).
    """
    if explicit is not None:
        return list(explicit)
    return [t.topic for t in summary.topics_by_slot(slot)]


# ── video encoder ─────────────────────────────────────────────────────


def _encode_video(
    *,
    reader: object,
    conn_by_topic: dict[str, object],
    topics: Sequence[str],
    output_path: Path,
    skipped: list[dict[str, Any]],
) -> tuple[EncodedArtifact | None, float | None]:
    """Walk Image / CompressedImage messages and write one MP4.

    Single topic → frames are passed through.
    Multi-topic   → frames are aligned by index (not by timestamp — bags
                    rarely have synced cameras at the message level) and
                    tiled horizontally, padded with black if heights
                    differ. The resulting MP4's fps is computed from the
                    *first* topic's median inter-frame delta.
    """
    np = _import_numpy()
    cv2 = _import_cv2()  # raises ConfigurationError pointing at [video]

    selected: list[object] = []
    for topic in topics:
        connection = conn_by_topic.get(topic)
        if connection is None:
            skipped.append({"topic": topic, "reason": "not in bag"})
            continue
        msgtype = str(getattr(connection, "msgtype", ""))
        if msgtype not in {"sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"}:
            skipped.append(
                {"topic": topic, "reason": f"video slot expects Image, got {msgtype}"}
            )
            continue
        selected.append(connection)

    if not selected:
        return None, None

    # Per-topic ordered (timestamp, BGR-frame) lists. We materialize
    # all frames in memory — rosbag2s rarely exceed a few minutes per
    # camera and the alternative (two-pass + temp files) is markedly
    # more complex for first cut. Document the limit; punt larger bags
    # to the live-record path planned for 0.2.
    per_topic_frames: dict[str, list[tuple[int, Any]]] = {}
    for connection in selected:
        topic = str(getattr(connection, "topic", ""))
        msgtype = str(getattr(connection, "msgtype", ""))
        frames: list[tuple[int, Any]] = []
        for _conn, t_ns, rawdata in reader.messages(connections=[connection]):  # type: ignore[attr-defined]
            msg = reader.deserialize(rawdata, msgtype)  # type: ignore[attr-defined]
            try:
                frame = _decode_image(msg, msgtype, np=np, cv2=cv2)
            except ValueError as exc:
                skipped.append(
                    {"topic": topic, "reason": f"image decode failed: {exc}"}
                )
                frames = []
                break
            frames.append((int(t_ns), frame))
        if frames:
            per_topic_frames[topic] = frames

    if not per_topic_frames:
        return None, None

    # Compute fps from the topic with the most frames.
    primary_topic = max(per_topic_frames, key=lambda t: len(per_topic_frames[t]))
    primary_stamps = [t for t, _f in per_topic_frames[primary_topic]]
    fps = _infer_fps(primary_stamps, np=np)

    # Build frame iterator.
    if len(per_topic_frames) == 1:
        ordered_frames: Iterable[Any] = (f for _t, f in per_topic_frames[primary_topic])
        out_topics = [primary_topic]
        out_w, out_h = per_topic_frames[primary_topic][0][1].shape[1::-1]
    else:
        # Multi-topic: align by index, tile horizontally, pad heights.
        topics_in_order = sorted(per_topic_frames.keys())
        n_frames = min(len(per_topic_frames[t]) for t in topics_in_order)
        max_h = max(per_topic_frames[t][0][1].shape[0] for t in topics_in_order)
        widths = [per_topic_frames[t][0][1].shape[1] for t in topics_in_order]
        out_w, out_h = sum(widths), max_h

        def gen() -> Iterable[Any]:
            for i in range(n_frames):
                cells: list[Any] = []
                for t in topics_in_order:
                    frame = per_topic_frames[t][i][1]
                    h, w = frame.shape[:2]
                    if h < max_h:
                        pad = np.zeros((max_h - h, w, 3), dtype=np.uint8)
                        frame = np.vstack([frame, pad])
                    cells.append(frame)
                yield np.hstack(cells)

        ordered_frames = gen()
        out_topics = topics_in_order

    # Write MP4 with the mp4v fourcc — most portable codec opencv ships
    # everywhere (avc1 / h264 isn't always linked into the wheel). The
    # server transcodes for browser playback if needed.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise ConfigurationError(
            f"opencv VideoWriter failed to open {output_path}. "
            "Most often: the output codec mp4v isn't compiled into your "
            "opencv wheel — try `pip install --upgrade opencv-python`."
        )
    try:
        for frame in ordered_frames:
            writer.write(frame)
    finally:
        writer.release()

    return EncodedArtifact(
        slot="video",
        path=output_path,
        bytes_size=output_path.stat().st_size,
        topics=out_topics,
    ), fps


def _decode_image(msg: Any, msgtype: str, *, np: Any, cv2: Any) -> Any:
    """Turn one ROS Image / CompressedImage message into a BGR ndarray.

    Returns a `np.uint8` (H, W, 3) array — opencv's `VideoWriter`
    contract. Raises `ValueError` for encodings we can't (or won't)
    handle so the caller can skip the topic with a useful reason.
    """
    if msgtype == "sensor_msgs/msg/CompressedImage":
        # `data` is a complete encoded image (jpeg/png) — opencv
        # recognises both via imdecode.
        buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"imdecode returned None for format={msg.format!r}")
        return frame

    # Raw `sensor_msgs/Image` — interpret bytes per-encoding.
    encoding = str(getattr(msg, "encoding", "")).lower()
    height = int(msg.height)
    width = int(msg.width)
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)

    if encoding in {"bgr8", "8uc3"}:
        return raw.reshape(height, width, 3)
    if encoding == "rgb8":
        return cv2.cvtColor(raw.reshape(height, width, 3), cv2.COLOR_RGB2BGR)
    if encoding in {"mono8", "8uc1"}:
        return cv2.cvtColor(raw.reshape(height, width), cv2.COLOR_GRAY2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(raw.reshape(height, width, 4), cv2.COLOR_BGRA2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(raw.reshape(height, width, 4), cv2.COLOR_RGBA2BGR)
    if encoding in {"mono16", "16uc1"}:
        # Promote 16-bit grayscale to 8-bit by truncating the high byte.
        # Lossy but acceptable for a video stream — preserves what the
        # human eye picks out without producing a 16bpp mp4.
        arr16 = np.frombuffer(bytes(msg.data), dtype="<u2").reshape(height, width)
        arr8 = (arr16 >> 8).astype(np.uint8)
        return cv2.cvtColor(arr8, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"unsupported Image.encoding={encoding!r}")


def _infer_fps(timestamps_ns: list[int], *, np: Any) -> float:
    """Compute fps from message timestamps (nanoseconds)."""
    if len(timestamps_ns) < 2:
        return _FALLBACK_FPS
    deltas = np.diff(np.asarray(timestamps_ns, dtype=np.int64))
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        return _FALLBACK_FPS
    median_dt_s = float(np.median(deltas)) / 1_000_000_000
    if median_dt_s <= 0 or not math.isfinite(median_dt_s):
        return _FALLBACK_FPS
    fps = 1.0 / median_dt_s
    # Round to a reasonable precision; opencv accepts floats but bag
    # producers usually mean a whole number.
    return round(fps, 2)


# ── sensors / actions encoder ─────────────────────────────────────────


def _encode_messages(
    *,
    reader: object,
    conn_by_topic: dict[str, object],
    topics: Sequence[str],
    slot: Slot,
    output_path: Path,
    skipped: list[dict[str, Any]],
) -> EncodedArtifact | None:
    """Pack the sensor / action topics into one NPZ.

    Each topic contributes a set of arrays, namespaced by topic so a
    single NPZ can hold many heterogeneous streams without clobbering
    keys. Layout:

        {
          "/joint_states/_t_ns":     int64[N]
          "/joint_states/position":  float32[N, K]
          "/joint_states/velocity":  float32[N, K]
          "/cmd_vel/_t_ns":          int64[M]
          "/cmd_vel/linear":         float32[M, 3]
          ...
        }
    """
    np = _import_numpy()

    selected: list[tuple[str, object, str]] = []
    for topic in topics:
        connection = conn_by_topic.get(topic)
        if connection is None:
            skipped.append({"topic": topic, "reason": "not in bag"})
            continue
        msgtype = str(getattr(connection, "msgtype", ""))
        selected.append((topic, connection, msgtype))

    if not selected:
        return None

    # Per-topic accumulator: list of (timestamp_ns, flat-dict).
    per_topic: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for topic, connection, msgtype in selected:
        rows: list[tuple[int, dict[str, Any]]] = []
        flatten = _flattener_for(msgtype)
        for _conn, t_ns, rawdata in reader.messages(connections=[connection]):  # type: ignore[attr-defined]
            msg = reader.deserialize(rawdata, msgtype)  # type: ignore[attr-defined]
            try:
                row = flatten(msg, np=np)
            except Exception as exc:
                skipped.append(
                    {"topic": topic, "reason": f"flatten failed for {msgtype}: {exc}"}
                )
                rows = []
                break
            if row:
                rows.append((int(t_ns), row))
        if rows:
            per_topic[topic] = rows
        elif topic not in {s["topic"] for s in skipped}:
            skipped.append({"topic": topic, "reason": "no usable fields after flatten"})

    if not per_topic:
        return None

    arrays: dict[str, Any] = {}
    for topic, rows in per_topic.items():
        timestamps = np.asarray([r[0] for r in rows], dtype=np.int64)
        arrays[f"{topic}/_t_ns"] = timestamps

        # Union of keys across rows. Variable-shape rows are fine for
        # the per-row dict, but at NPZ stack time every row has to
        # agree. We enforce same-shape by failing loud — silently
        # padding would hide a real bag-recording bug.
        keys = sorted({k for _t, row in rows for k in row.keys()})
        for key in keys:
            stack: list[Any] = []
            shape: tuple[int, ...] | None = None
            ok = True
            for _t, row in rows:
                if key not in row:
                    skipped.append(
                        {
                            "topic": topic,
                            "reason": f"field {key!r} missing in some messages",
                        }
                    )
                    ok = False
                    break
                value = row[key]
                arr = np.asarray(value)
                if shape is None:
                    shape = arr.shape
                elif arr.shape != shape:
                    skipped.append(
                        {
                            "topic": topic,
                            "reason": (
                                f"field {key!r} shape changed mid-bag "
                                f"({shape} → {arr.shape})"
                            ),
                        }
                    )
                    ok = False
                    break
                stack.append(arr)
            if ok and stack:
                arrays[f"{topic}/{key}"] = np.stack(stack)

    if not arrays:
        return None

    np.savez(output_path, **arrays)
    return EncodedArtifact(
        slot=slot,
        path=output_path,
        bytes_size=output_path.stat().st_size,
        topics=list(per_topic.keys()),
    )


# ── message flatteners ────────────────────────────────────────────────
#
# Each flattener takes a deserialised rosbags message and returns a
# flat `dict[str, scalar | ndarray]`. Fixed-length fields (e.g.
# `Twist.linear` is always (x, y, z)) become arrays; truly free-form
# fields (variable-length lists, strings) get stringified for metadata
# or skipped — we don't try to NPZ-pack them.
#
# Specific flatteners are registered by msgtype; everything else falls
# through to `_flatten_generic`, which walks the dataclass and pulls
# out numeric leaves.


FlattenFn = Any  # callable(msg, *, np) -> dict[str, Any]; loosely typed for the registry


def _flattener_for(msgtype: str) -> FlattenFn:
    return _FLATTENERS.get(msgtype, _flatten_generic)


def _flatten_joint_state(msg: Any, *, np: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field_name in ("position", "velocity", "effort"):
        values = getattr(msg, field_name, None)
        if values is None:
            continue
        arr = np.asarray(values, dtype=np.float32)
        if arr.size > 0:
            out[field_name] = arr
    return out


def _flatten_imu(msg: Any, *, np: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if hasattr(msg, "orientation"):
        q = msg.orientation
        out["orientation"] = np.asarray(
            [q.x, q.y, q.z, q.w], dtype=np.float32
        )
    if hasattr(msg, "angular_velocity"):
        v = msg.angular_velocity
        out["angular_velocity"] = np.asarray([v.x, v.y, v.z], dtype=np.float32)
    if hasattr(msg, "linear_acceleration"):
        a = msg.linear_acceleration
        out["linear_acceleration"] = np.asarray([a.x, a.y, a.z], dtype=np.float32)
    return out


def _flatten_twist(msg: Any, *, np: Any) -> dict[str, Any]:
    return {
        "linear": np.asarray(
            [msg.linear.x, msg.linear.y, msg.linear.z], dtype=np.float32
        ),
        "angular": np.asarray(
            [msg.angular.x, msg.angular.y, msg.angular.z], dtype=np.float32
        ),
    }


def _flatten_twist_stamped(msg: Any, *, np: Any) -> dict[str, Any]:
    return _flatten_twist(msg.twist, np=np)


def _flatten_wrench(msg: Any, *, np: Any) -> dict[str, Any]:
    return {
        "force": np.asarray(
            [msg.force.x, msg.force.y, msg.force.z], dtype=np.float32
        ),
        "torque": np.asarray(
            [msg.torque.x, msg.torque.y, msg.torque.z], dtype=np.float32
        ),
    }


def _flatten_wrench_stamped(msg: Any, *, np: Any) -> dict[str, Any]:
    return _flatten_wrench(msg.wrench, np=np)


def _flatten_pose_stamped(msg: Any, *, np: Any) -> dict[str, Any]:
    p = msg.pose.position
    o = msg.pose.orientation
    return {
        "position": np.asarray([p.x, p.y, p.z], dtype=np.float32),
        "orientation": np.asarray([o.x, o.y, o.z, o.w], dtype=np.float32),
    }


def _flatten_odometry(msg: Any, *, np: Any) -> dict[str, Any]:
    p = msg.pose.pose.position
    o = msg.pose.pose.orientation
    v = msg.twist.twist.linear
    a = msg.twist.twist.angular
    return {
        "position": np.asarray([p.x, p.y, p.z], dtype=np.float32),
        "orientation": np.asarray([o.x, o.y, o.z, o.w], dtype=np.float32),
        "linear_velocity": np.asarray([v.x, v.y, v.z], dtype=np.float32),
        "angular_velocity": np.asarray([a.x, a.y, a.z], dtype=np.float32),
    }


def _flatten_generic(msg: Any, *, np: Any) -> dict[str, Any]:
    """Last-resort flattener for unknown message types.

    Walks the dataclass-like structure and pulls out numeric scalars
    and fixed-shape numeric arrays. Any field that's a string,
    variable-length list of complex objects, or nested message we
    don't recognise gets dropped silently — the caller can always
    register a specific flattener if those fields matter.
    """
    out: dict[str, Any] = {}
    _walk_into(msg, "", out, np)
    return out


def _walk_into(node: Any, prefix: str, out: dict[str, Any], np: Any) -> None:
    if node is None:
        return
    if hasattr(node, "__slots__") or hasattr(node, "__dataclass_fields__"):
        # rosbags messages expose `__slots__` (per ros2 message gen);
        # core builtin_interfaces use dataclasses. Both expose attrs.
        for attr in _attr_names(node):
            value = getattr(node, attr, None)
            _walk_into(value, _join(prefix, attr), out, np)
        return
    if isinstance(node, (int, float, bool)):
        out[prefix or "value"] = float(node)
        return
    # numpy arrays from rosbags `.data` for variable-length numeric
    # fields — only keep them if they're 1-D and fixed-length looking.
    if hasattr(node, "shape") and getattr(node, "ndim", None) == 1:
        out[prefix or "value"] = np.asarray(node, dtype=np.float32)
        return
    # Skip strings, bytes, opaque nested objects.


def _attr_names(node: Any) -> list[str]:
    slots = getattr(node, "__slots__", None)
    if slots:
        return [s for s in slots if not s.startswith("_")]
    df = getattr(node, "__dataclass_fields__", None)
    if df:
        return [k for k in df if not k.startswith("_")]
    return []


def _join(prefix: str, attr: str) -> str:
    return f"{prefix}.{attr}" if prefix else attr


_FLATTENERS: dict[str, FlattenFn] = {
    "sensor_msgs/msg/JointState": _flatten_joint_state,
    "sensor_msgs/msg/Imu": _flatten_imu,
    "geometry_msgs/msg/Twist": _flatten_twist,
    "geometry_msgs/msg/TwistStamped": _flatten_twist_stamped,
    "geometry_msgs/msg/Wrench": _flatten_wrench,
    "geometry_msgs/msg/WrenchStamped": _flatten_wrench_stamped,
    "geometry_msgs/msg/PoseStamped": _flatten_pose_stamped,
    "nav_msgs/msg/Odometry": _flatten_odometry,
}


# ── lazy imports ──────────────────────────────────────────────────────


def _import_anyreader() -> type:
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as exc:
        raise ConfigurationError(
            "the ROS 2 adapter needs the `rosbags` library. "
            "Install with `pip install 'robotrace-dev[ros2]==0.1.0a2'`."
        ) from exc
    return AnyReader


def _import_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise ConfigurationError(
            "the ROS 2 adapter needs `numpy`. Install with "
            "`pip install 'robotrace-dev[ros2]==0.1.0a2'` (which pulls it in)."
        ) from exc
    return np


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise ConfigurationError(
            "encoding ROS 2 Image topics into MP4 needs OpenCV. "
            "Install with `pip install 'robotrace-dev[ros2,video]==0.1.0a2'` "
            "(both extras together — the [video] extra carries "
            "opencv-python so a sensor-only bag doesn't pay the "
            "install cost)."
        ) from exc
    return cv2
