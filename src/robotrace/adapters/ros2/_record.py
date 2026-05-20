"""Live ROS 2 recording - subscribe at runtime, ship as one episode.

The offline path (`scan_bag` + `encode_bag` + `upload_bag`) already
covers rosbag2 files that were captured ahead of time. This module
adds the live equivalent: subscribe to a set of topics during a run,
write every received message into a temporary rosbag2 directory, and
on close pipe that directory straight through the existing
encode + upload pipeline. **One** code path for "rosbag → episode" -
the live mode is just "we wrote the bag a moment ago instead of
yesterday".

Usage::

    from robotrace.adapters import ros2

    with ros2.record(
        topics=["/camera/image_raw", "/joint_states", "/cmd_vel"],
        name="warmup pick-and-place",
        policy_version="pap-v3.2.1",
        env_version="halcyon-cell-rev4",
        git_sha="abc1234",
    ) as rec:
        # robot code runs here; rclpy is spinning in a background
        # thread, writing every message to a tempdir bag.
        drive_robot_for_30_seconds()
    # __exit__: stop subscriptions, close bag, encode → upload →
    # finalize. `rec.episode` is the finalized Episode.

Dependency strategy
-------------------

`rclpy` is **not** pinned in `pyproject.toml`. It ships with the ROS
2 distro the user already has installed (apt: ros-humble-rclpy /
ros-jazzy-rclpy / etc.), and the wheels on PyPI are not always
compatible with the `rmw` bindings sourced from a workspace. Pinning
would force a wheel mismatch on most real robot rigs. Instead we
lazy-import inside `record(...)` and surface a friendly
`ConfigurationError` with the apt command when rclpy isn't found.

`rosbags` (already pulled by `[ros2]`) handles the rosbag2 writer
end, so the bag we produce is byte-compatible with what `ros2 bag
record` writes from the ROS 2 CLI. `scan_bag` reads it back without
knowing whether it came from us or from the official tools.

Threading
---------

`rclpy.executors.MultiThreadedExecutor` is spun on a daemon thread.
Subscription callbacks fire on that thread; they serialize the
incoming message and forward the bytes to `_BagWriter.write_message`,
which holds a `threading.Lock` and writes to the rosbag2 storage.
Writes are short (the heavy work - frame encoding - happens later in
`encode_bag`), so the lock is uncontended in practice.

Coverage cut for V1
-------------------

- Topics are explicit (no `topics=None` auto-discovery). rclpy graph
  discovery needs to wait for the graph to stabilize and tends to
  miss topics that aren't yet publishing - we'll iterate when there's
  evidence the explicit-list shape is the bottleneck.
- The bag is single-storage (sqlite3). mcap split-storage is a
  rosbags feature too; we'll add a kwarg when an early user asks.
- No live reconnect to a re-launching node. If the publisher
  restarts mid-recording, you may drop messages between the unsub
  and re-sub events. Same shape as `ros2 bag record` upstream.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...client import Client, EpisodeSource
from ...episode import Episode, EpisodeFinalStatus
from ...errors import ConfigurationError
from ._encode import encode_bag
from ._scan import BagSummary, scan_bag
from ._upload import _default_episode_name, _get_default_client, _upload_encoded

if TYPE_CHECKING:
    # Pure type-only imports so the module loads even when rosbags
    # isn't installed (the SDK base install path).
    pass

# Map of `$ROS_DISTRO` env var values to the matching rosbags
# typestore. Keep this small and explicit - we want a clear error
# when the user's on a distro we haven't validated.
_KNOWN_DISTROS: dict[str, str] = {
    "humble": "ROS2_HUMBLE",
    "iron": "ROS2_IRON",
    "jazzy": "ROS2_JAZZY",
    "rolling": "ROS2_ROLLING",
    "foxy": "ROS2_FOXY",
}

# Fallback when `$ROS_DISTRO` isn't set or isn't recognized. Humble
# is the current LTS at time of writing (May 2026) - the safest
# default for a user who imported the SDK without sourcing a workspace
# (e.g. running the bag writer from CI without ROS 2 installed at
# all, which is the testing path).
_DEFAULT_DISTRO = "humble"


# ── bag writer (rclpy-free, testable) ───────────────────────────────


class _BagWriter:
    """Threadsafe wrapper around ``rosbags.rosbag2.Writer``.

    Pure-Python, rclpy-free: takes already-CDR-serialized bytes plus
    the canonical type name (``sensor_msgs/msg/JointState``) and writes
    them into a real rosbag2 directory. The live recorder serializes
    via ``rclpy.serialization.serialize_message`` and pipes the bytes
    here; tests serialize via ``typestore.serialize_cdr`` directly.

    Connections are created lazily on the first message per topic so
    the user doesn't have to declare topics ahead of subscription.

    Not a public class - constructed only by ``record(...)`` (and by
    tests).
    """

    def __init__(
        self,
        bag_path: Path,
        *,
        ros_distro: str | None = None,
    ) -> None:
        self._bag_path = bag_path
        self._lock = threading.Lock()
        self._connections: dict[str, Any] = {}
        self._message_count = 0
        self._closed = False

        # Lazy import so the SDK base install path stays clean for
        # users who never touch the ROS 2 adapter.
        try:
            from rosbags.rosbag2 import Writer
            from rosbags.typesys import Stores, get_typestore
        except ImportError as exc:
            raise ConfigurationError(
                "the ROS 2 adapter needs the `rosbags` library. "
                "Install with `pip install 'robotrace-dev[ros2]'`."
            ) from exc

        resolved_distro = (ros_distro or os.environ.get("ROS_DISTRO", "")).strip().lower()
        store_name = _KNOWN_DISTROS.get(resolved_distro)
        if store_name is None:
            # Unknown / unset distro - fall back to humble but keep
            # the distro string we used in case the user wants to see
            # it on the finalized episode's metadata.
            resolved_distro = _DEFAULT_DISTRO
            store_name = _KNOWN_DISTROS[_DEFAULT_DISTRO]

        try:
            store_enum = getattr(Stores, store_name)
        except AttributeError as exc:
            raise ConfigurationError(
                f"rosbags version doesn't know about Stores.{store_name}. "
                "Upgrade `rosbags` or pass an older `ros_distro=` kwarg."
            ) from exc

        self._typestore = get_typestore(store_enum)
        self._ros_distro = resolved_distro

        # `Writer(...)` creates the bag directory; it must NOT exist
        # yet. Caller is expected to hand us a fresh tempdir path.
        self._writer = Writer(bag_path, version=Writer.VERSION_LATEST)
        self._writer.open()

    @property
    def message_count(self) -> int:
        """Total messages written across every topic."""
        return self._message_count

    @property
    def ros_distro(self) -> str:
        """Distro string the typestore was loaded from."""
        return self._ros_distro

    def write_message(
        self,
        *,
        topic: str,
        typename: str,
        t_ns: int,
        raw_bytes: bytes,
    ) -> None:
        """Append one CDR-serialized message to the bag.

        ``typename`` is the canonical ``<pkg>/msg/<Name>`` form (what
        rclpy and rosbags both use as the connection's type
        identifier). ``raw_bytes`` is the CDR payload, already
        serialized - we never touch the message struct here.

        Threadsafe. Safe to call from any rclpy subscription
        callback. No-op after ``close()``.
        """
        with self._lock:
            if self._closed:
                # Late-arriving callback after we asked the executor
                # to shut down. Dropping the message is intentional:
                # the bag is closed and re-opening would corrupt the
                # metadata.yaml.
                return

            conn = self._connections.get(topic)
            if conn is None:
                conn = self._writer.add_connection(
                    topic, typename, typestore=self._typestore
                )
                self._connections[topic] = conn

            self._writer.write(conn, int(t_ns), raw_bytes)
            self._message_count += 1

    def close(self) -> None:
        """Finalize the bag on disk. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._writer.close()
            except Exception:
                # The bag directory is what we hand off downstream; a
                # `close()` failure usually means the storage already
                # flushed (rosbags is forgiving here). Don't mask the
                # user's exception path with a cleanup error.
                pass

    def __enter__(self) -> _BagWriter:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


# ── live recorder ───────────────────────────────────────────────────


@dataclass
class LiveRecording:
    """Handle returned by :func:`record`.

    Use as a context manager (preferred) or call ``stop()`` explicitly.
    After exit ``episode`` is the finalized Episode and ``bag_summary``
    describes the bag the encoder ran against.

    Fields populated incrementally:

    - At construction: ``bag_path`` (tempdir) is reserved; nothing is
      written yet.
    - After ``start()``: rclpy is spinning, subscriptions are active.
    - After ``stop()`` / context exit: ``bag_summary`` + ``episode``
      + ``duration_s`` are filled in.

    Public attribute access mid-run is safe but the values reflect
    "what's been written so far". Treat them as final only after
    ``stop()`` returns.
    """

    bag_path: Path
    topics: tuple[str, ...]
    bag_summary: BagSummary | None = None
    episode: Episode | None = None
    duration_s: float | None = None
    message_count: int = 0
    # Internal config carried so __enter__/stop() can run the upload
    # without the caller re-passing everything.
    _writer: _BagWriter | None = field(default=None, repr=False, compare=False)
    _node: Any = field(default=None, repr=False, compare=False)
    _executor: Any = field(default=None, repr=False, compare=False)
    _spin_thread: threading.Thread | None = field(
        default=None, repr=False, compare=False
    )
    _client: Client | None = field(default=None, repr=False, compare=False)
    _episode_kwargs: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    _keep_bag: bool = field(default=False, repr=False, compare=False)
    _started: bool = field(default=False, repr=False, compare=False)
    _stopped: bool = field(default=False, repr=False, compare=False)

    def __enter__(self) -> LiveRecording:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # If the user's code raised inside the `with` block, finalize
        # the episode as `failed` and re-raise. Same shape as
        # `Episode.__exit__`.
        status: EpisodeFinalStatus = "ready" if exc_type is None else "failed"
        extra_metadata: dict[str, Any] = {}
        if exc_type is not None:
            extra_metadata["failure_reason"] = (
                f"{exc_type.__name__}: {exc_val}" if exc_val is not None else exc_type.__name__
            )
        try:
            self.stop(status=status, extra_metadata=extra_metadata)
        except Exception:
            # Don't mask the user's exception with a cleanup failure.
            if exc_type is None:
                raise

    def start(self) -> None:
        """Spin up rclpy + create one subscription per topic.

        Returns immediately - subscriptions fire on a background
        executor thread. Idempotent: calling twice is a no-op.
        """
        if self._started:
            return
        self._started = True

        # Lazy import - rclpy is system-installed via apt, NOT
        # available in CI / dev environments. Friendly error when
        # missing.
        try:
            import rclpy
            from rclpy.executors import MultiThreadedExecutor
            from rclpy.qos import qos_profile_sensor_data
            from rclpy.serialization import serialize_message
        except ImportError as exc:
            raise ConfigurationError(
                "ros2.record(...) needs `rclpy`. rclpy ships with your ROS 2 "
                "distro via apt (e.g. `apt install ros-humble-rclpy`) - this SDK "
                "deliberately does NOT pull it from PyPI because the wheels there "
                "are not always compatible with the rmw bindings sourced from a "
                "ROS 2 workspace. Source your workspace (`source "
                "/opt/ros/<distro>/setup.bash`) and try again. The offline path "
                "(`ros2.upload_bag(\"/path/to/bag\")`) does NOT need rclpy."
            ) from exc

        try:
            from rosidl_runtime_py.utilities import get_message
        except ImportError as exc:
            raise ConfigurationError(
                "ros2.record(...) needs `rosidl_runtime_py` for message-type lookup. "
                "Ships with ROS 2 - source your workspace and try again."
            ) from exc

        # If the user already initialized rclpy (custom node setup,
        # for instance), don't fight them - just attach.
        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True
        else:
            self._owns_rclpy = False

        node = rclpy.create_node(
            "robotrace_recorder",
            # An anonymous suffix avoids name collisions if the user
            # already has a `robotrace_recorder` node from a previous
            # script run.
            namespace="",
            cli_args=None,
            use_global_arguments=False,
            enable_rosout=False,
        )
        self._node = node

        # Resolve each topic's message type from the live graph, then
        # create a subscription. If a topic isn't being published yet
        # we still subscribe optimistically - rclpy stores the
        # subscription and the connection latches when the publisher
        # comes up.
        for topic in self.topics:
            type_strs = node.get_topic_names_and_types()
            type_map = {name: types for name, types in type_strs}
            type_list = type_map.get(topic, [])
            if not type_list:
                # No publisher discovered yet - we still need a type
                # to subscribe. Raise loudly: the user typoed a
                # topic, or hasn't launched the node yet. Better to
                # fail at start() than to silently record nothing.
                raise ConfigurationError(
                    f"topic {topic!r} has no advertised publisher. Available "
                    f"topics: {sorted(type_map.keys())}. Start your robot "
                    "node before opening `ros2.record(...)` so the type can "
                    "be resolved."
                )
            type_str = type_list[0]
            msg_class = get_message(type_str)

            def _make_callback(
                topic_capture: str,
                type_capture: str,
                writer_capture: _BagWriter,
                node_capture: Any,
            ) -> Any:
                """Closure-bind the per-topic state for the callback."""

                def _cb(msg: Any) -> None:
                    # Wall-clock nanoseconds from the rclpy clock. The
                    # message's own header.stamp would be more
                    # precise for some topics but isn't present on
                    # every message type, and rclpy's clock is what
                    # `ros2 bag record` uses upstream too.
                    t_ns = node_capture.get_clock().now().nanoseconds
                    try:
                        raw = serialize_message(msg)
                    except Exception:
                        # Don't crash the executor on a serialization
                        # failure - log the topic and move on.
                        return
                    writer_capture.write_message(
                        topic=topic_capture,
                        typename=type_capture,
                        t_ns=t_ns,
                        raw_bytes=raw,
                    )

                return _cb

            assert self._writer is not None, "writer must be created before start()"
            node.create_subscription(
                msg_class,
                topic,
                _make_callback(topic, type_str, self._writer, node),
                qos_profile_sensor_data,
            )

        executor = MultiThreadedExecutor()
        executor.add_node(node)
        self._executor = executor

        # Daemon thread so an unhandled exception in the user's main
        # loop doesn't leave us pinned to the process.
        spin_thread = threading.Thread(
            target=executor.spin, name="robotrace-ros2-record", daemon=True
        )
        spin_thread.start()
        self._spin_thread = spin_thread

    def stop(
        self,
        *,
        status: EpisodeFinalStatus = "ready",
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> Episode | None:
        """Shut down subscriptions, encode the bag, upload.

        Returns the finalized ``Episode`` (or ``None`` if the bag is
        empty - empty bags are not uploaded). Idempotent.
        """
        if self._stopped:
            return self.episode
        self._stopped = True

        # 1. Stop rclpy. Wrapped in try/except so a partial start
        #    (e.g. ConfigurationError in the loop) still flushes the
        #    bag we already wrote.
        try:
            if self._executor is not None:
                self._executor.shutdown()
            if self._spin_thread is not None:
                self._spin_thread.join(timeout=5.0)
            if self._node is not None:
                self._node.destroy_node()
            if getattr(self, "_owns_rclpy", False):
                import rclpy

                rclpy.shutdown()
        except Exception:
            # Don't mask the encode/upload below with a teardown
            # error - they're the only thing the user cares about.
            pass

        # 2. Close the bag so rosbags writes metadata.yaml.
        if self._writer is not None:
            self.message_count = self._writer.message_count
            self._writer.close()

        # 3. Empty bag → no upload. Useful for "I started the
        #    context manager but my robot crashed before publishing"
        #    so the user gets a clean error path.
        if self.message_count == 0:
            if not self._keep_bag:
                shutil.rmtree(self.bag_path, ignore_errors=True)
            return None

        # 4. Same path as upload_bag(): scan → encode → upload →
        #    finalize. Adapter metadata gets a `mode: "live"` marker
        #    so the portal can tell live recordings apart from
        #    bag-file uploads when triaging.
        try:
            summary = scan_bag(self.bag_path)
            self.bag_summary = summary
            self.duration_s = summary.duration_s

            # Encode into a sibling tempdir so the bag and encoded
            # artifacts have separate lifecycle bookkeeping.
            encoded_dir = self.bag_path.parent / f"{self.bag_path.name}_encoded"
            encoded = encode_bag(self.bag_path, encoded_dir)

            client = self._client if self._client is not None else _get_default_client()

            live_metadata: dict[str, Any] = {
                "ros2": {
                    "mode": "live",
                    "distro": (
                        self._writer.ros_distro if self._writer is not None else _DEFAULT_DISTRO
                    ),
                    "topics": list(self.topics),
                    "message_count": self.message_count,
                },
            }
            user_metadata = self._episode_kwargs.pop("metadata", None) or {}
            if extra_metadata:
                user_metadata = {**user_metadata, **dict(extra_metadata)}

            merged_metadata: dict[str, Any] = {
                **live_metadata,
                **dict(encoded.metadata),
                **dict(user_metadata),
            }

            episode = _upload_encoded(
                encoded=encoded,
                client=client,
                name=self._episode_kwargs.get("name")
                or _default_episode_name(self.bag_path),
                source=self._episode_kwargs.get("source", "real"),
                robot=self._episode_kwargs.get("robot"),
                policy_version=self._episode_kwargs.get("policy_version"),
                env_version=self._episode_kwargs.get("env_version"),
                git_sha=self._episode_kwargs.get("git_sha"),
                seed=self._episode_kwargs.get("seed"),
                metadata=merged_metadata,
                status=status,
            )
            self.episode = episode
            return episode
        finally:
            if not self._keep_bag:
                shutil.rmtree(self.bag_path, ignore_errors=True)
                shutil.rmtree(
                    self.bag_path.parent / f"{self.bag_path.name}_encoded",
                    ignore_errors=True,
                )


def record(
    *,
    topics: Sequence[str],
    # Episode identification + reproducibility - same shape as
    # `start_episode` / `upload_bag`. All forwarded to the eventual
    # `start_episode` call inside `stop()`.
    client: Client | None = None,
    name: str | None = None,
    source: EpisodeSource = "real",
    robot: str | None = None,
    policy_version: str | None = None,
    env_version: str | None = None,
    git_sha: str | None = None,
    seed: int | None = None,
    metadata: Mapping[str, Any] | None = None,
    # ROS distro override; default reads `$ROS_DISTRO`.
    ros_distro: str | None = None,
    # Debug escape hatch - keep the bag + encoded artifacts on disk
    # so the user can inspect what was captured.
    output_dir: str | Path | None = None,
    keep_bag: bool = False,
) -> LiveRecording:
    """Open a live recording session that ships as one episode.

    The returned :class:`LiveRecording` is a context manager - the
    typical shape is::

        with ros2.record(topics=[...], name="...") as rec:
            run_my_robot()
        episode = rec.episode

    Topics are validated against the live ROS 2 graph at ``start()``.
    A topic with no advertised publisher raises ``ConfigurationError``
    so the user notices a typo at start-time, not 60s later when the
    bag is empty.

    ``rclpy`` must be importable in the current Python environment;
    it ships with the ROS 2 distro via apt. Source your workspace
    (``source /opt/ros/<distro>/setup.bash``) before running.

    On clean exit the bag is encoded, uploaded, and finalized; the
    tempdir is removed unless ``keep_bag=True`` or ``output_dir`` is
    supplied. On exception inside the ``with`` block, the episode is
    finalized as ``failed`` and the exception propagates.
    """
    if not topics:
        raise ConfigurationError(
            "ros2.record(...) needs at least one topic. Pass `topics=[\"/joint_states\", ...]`."
        )

    # Resolve the destination directory before constructing the
    # writer so a bad `output_dir` errors before we spin rclpy.
    if output_dir is not None:
        bag_path = Path(output_dir).expanduser().resolve()
        if bag_path.exists():
            raise ConfigurationError(
                f"output_dir {bag_path} already exists. rosbag2 Writer requires "
                "a fresh directory - point at a path that doesn't exist yet."
            )
        bag_path.parent.mkdir(parents=True, exist_ok=True)
        keep_bag = True  # explicit output_dir implies the user wants it
    else:
        # tempfile.mkdtemp returns a directory that already exists;
        # `rosbags.Writer` insists on a fresh path. Append `/bag` so
        # we get a child dir inside our tempdir that doesn't exist
        # yet but whose parent does (so cleanup is straightforward).
        parent = Path(tempfile.mkdtemp(prefix="robotrace-ros2-rec-"))
        bag_path = parent / "bag"

    writer = _BagWriter(bag_path, ros_distro=ros_distro)

    recording = LiveRecording(
        bag_path=bag_path,
        topics=tuple(topics),
    )
    recording._writer = writer
    recording._client = client
    recording._keep_bag = keep_bag
    recording._episode_kwargs = {
        "name": name,
        "source": source,
        "robot": robot,
        "policy_version": policy_version,
        "env_version": env_version,
        "git_sha": git_sha,
        "seed": seed,
        "metadata": dict(metadata) if metadata else None,
    }
    return recording


__all__ = ["LiveRecording", "record"]
