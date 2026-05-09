"""ROS 2 adapter tests.

Synthesises a tiny rosbag2 on disk with three topics — one camera,
one JointState, one Twist — using `rosbags.rosbag2.Writer` so the
test suite never needs to ship a binary fixture or talk to a real
ROS 2 install.

Coverage:

    test_scan_bag_classifies_topics       — auto-classifier picks
                                            video / sensors / actions
                                            from message types.
    test_encode_bag_writes_artifacts      — encode produces video.mp4,
                                            sensors.npz, actions.npz
                                            with the expected keys
                                            and shapes.
    test_encode_bag_skips_unknown         — empty topic-list overrides
                                            actually exclude a slot.
    test_upload_bag_uses_start_episode    — one-shot upload hits the
                                            three ingest endpoints in
                                            the right order with the
                                            right per-slot URLs.

The fixture skips the whole module if `rosbags` or `cv2` aren't
installed (CI without the ros2/video extras).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

import robotrace as rt

# Skip the module gracefully when the optional deps aren't installed —
# the SDK's base tests must still pass on an environment that only
# has `pip install -e ".[dev]"`.
pytest.importorskip("rosbags")
pytest.importorskip("cv2")

from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from robotrace.adapters import ros2

NS_PER_S = 1_000_000_000


# ── synthetic bag fixture ─────────────────────────────────────────────


@pytest.fixture
def synthetic_bag(tmp_path: Path) -> Path:
    """Build a tiny rosbag2 with one of every classified topic.

    Layout:

        /camera/image_raw   sensor_msgs/Image          5 frames @ 10 fps
        /joint_states       sensor_msgs/JointState     5 messages
        /cmd_vel            geometry_msgs/Twist        3 messages

    Camera frames are deterministic 32x24 BGR images so opencv can
    encode them; the actual pixels don't matter for the test.
    """
    bag_path = tmp_path / "fixture_bag"
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    Image = typestore.types["sensor_msgs/msg/Image"]
    JointState = typestore.types["sensor_msgs/msg/JointState"]
    Twist = typestore.types["geometry_msgs/msg/Twist"]
    Header = typestore.types["std_msgs/msg/Header"]
    Time = typestore.types["builtin_interfaces/msg/Time"]
    Vector3 = typestore.types["geometry_msgs/msg/Vector3"]

    def header(t_ns: int, frame_id: str = "world") -> Any:
        return Header(stamp=Time(sec=t_ns // NS_PER_S, nanosec=t_ns % NS_PER_S), frame_id=frame_id)

    with Writer(bag_path, version=Writer.VERSION_LATEST) as writer:
        cam_conn = writer.add_connection(
            "/camera/image_raw", Image.__msgtype__, typestore=typestore
        )
        js_conn = writer.add_connection(
            "/joint_states", JointState.__msgtype__, typestore=typestore
        )
        cmd_conn = writer.add_connection(
            "/cmd_vel", Twist.__msgtype__, typestore=typestore
        )

        # 5 image frames, 100ms apart → infers 10 fps.
        for i in range(5):
            t_ns = (i + 1) * NS_PER_S // 10
            pixels = (np.full((24, 32, 3), 16 * (i + 1), dtype=np.uint8)).tobytes()
            img = Image(
                header=header(t_ns),
                height=24,
                width=32,
                encoding="bgr8",
                is_bigendian=0,
                step=32 * 3,
                data=np.frombuffer(pixels, dtype=np.uint8),
            )
            writer.write(cam_conn, t_ns, typestore.serialize_cdr(img, Image.__msgtype__))

        # 5 JointState messages, 6 joints each, 200ms apart.
        joint_names = ["j1", "j2", "j3", "j4", "j5", "j6"]
        for i in range(5):
            t_ns = (i + 1) * NS_PER_S * 2 // 10
            position = np.linspace(0.0, 1.0, len(joint_names)) + i * 0.01
            velocity = np.full(len(joint_names), 0.05, dtype=np.float64)
            effort = np.zeros(len(joint_names), dtype=np.float64)
            js = JointState(
                header=header(t_ns),
                name=joint_names,
                position=position.astype(np.float64),
                velocity=velocity,
                effort=effort,
            )
            writer.write(
                js_conn, t_ns, typestore.serialize_cdr(js, JointState.__msgtype__)
            )

        # 3 cmd_vel messages.
        for i in range(3):
            t_ns = (i + 1) * NS_PER_S // 3
            twist = Twist(
                linear=Vector3(x=float(i + 1) * 0.1, y=0.0, z=0.0),
                angular=Vector3(x=0.0, y=0.0, z=float(i + 1) * 0.05),
            )
            writer.write(cmd_conn, t_ns, typestore.serialize_cdr(twist, Twist.__msgtype__))

    return bag_path


# ── scan_bag ──────────────────────────────────────────────────────────


def test_scan_bag_classifies_topics(synthetic_bag: Path) -> None:
    summary = ros2.scan_bag(synthetic_bag)

    assert summary.message_count == 5 + 5 + 3
    assert summary.duration_s is not None
    assert summary.duration_s > 0

    by_topic = {t.topic: t for t in summary.topics}
    assert by_topic["/camera/image_raw"].slot == "video"
    assert by_topic["/camera/image_raw"].classified_by == "msgtype"
    assert by_topic["/joint_states"].slot == "sensors"
    assert by_topic["/joint_states"].classified_by == "default"
    assert by_topic["/cmd_vel"].slot == "actions"
    # /cmd_vel is geometry_msgs/Twist — caught by msgtype rule, not
    # the topic-name fallback. The fallback exists for project-specific
    # custom message types.
    assert by_topic["/cmd_vel"].classified_by == "msgtype"

    # Report renders without crashing and includes every topic.
    report = summary.report()
    assert "/camera/image_raw" in report
    assert "/joint_states" in report
    assert "/cmd_vel" in report


def test_scan_bag_rejects_non_directory(tmp_path: Path) -> None:
    not_a_bag = tmp_path / "missing"
    with pytest.raises(rt.ConfigurationError, match="does not exist"):
        ros2.scan_bag(not_a_bag)


def test_scan_bag_rejects_directory_without_metadata(tmp_path: Path) -> None:
    empty = tmp_path / "empty_dir"
    empty.mkdir()
    with pytest.raises(rt.ConfigurationError, match=r"metadata\.yaml"):
        ros2.scan_bag(empty)


# ── encode_bag ────────────────────────────────────────────────────────


def test_encode_bag_writes_all_three_artifacts(synthetic_bag: Path, tmp_path: Path) -> None:
    out = tmp_path / "encoded"
    encoded = ros2.encode_bag(synthetic_bag, out)

    assert encoded.video is not None
    assert encoded.sensors is not None
    assert encoded.actions is not None
    assert encoded.video.path.is_file()
    assert encoded.sensors.path.is_file()
    assert encoded.actions.path.is_file()

    # 5 frames at 100ms intervals → 10 fps inferred.
    assert encoded.fps == 10.0

    # Sensors NPZ: JointState packed under /joint_states/<field>.
    sensors = np.load(encoded.sensors.path)
    assert "/joint_states/_t_ns" in sensors.files
    assert "/joint_states/position" in sensors.files
    assert "/joint_states/velocity" in sensors.files
    assert sensors["/joint_states/position"].shape == (5, 6)
    assert sensors["/joint_states/_t_ns"].dtype == np.int64

    # Actions NPZ: Twist packed under /cmd_vel/{linear,angular}.
    actions = np.load(encoded.actions.path)
    assert "/cmd_vel/_t_ns" in actions.files
    assert "/cmd_vel/linear" in actions.files
    assert "/cmd_vel/angular" in actions.files
    assert actions["/cmd_vel/linear"].shape == (3, 3)


def test_encode_bag_excludes_slot_with_empty_list(
    synthetic_bag: Path, tmp_path: Path
) -> None:
    """Passing `video_topics=[]` deliberately drops that slot."""
    out = tmp_path / "encoded_no_video"
    encoded = ros2.encode_bag(synthetic_bag, out, video_topics=[])
    assert encoded.video is None
    assert encoded.sensors is not None
    assert encoded.actions is not None
    # opencv was never imported because no image topic ran — no
    # video.mp4 on disk.
    assert not (out / "video.mp4").exists()


# ── upload_bag ────────────────────────────────────────────────────────


def test_upload_bag_runs_full_ingest_flow(
    synthetic_bag: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """upload_bag opens an episode, uploads three artifacts, finalizes.

    The signed PUT URLs go to a different host with no auth header, so
    the SDK uses a *fresh* `httpx.Client` per upload (see `_http.py`).
    That means MockTransport on the main client can't intercept the
    PUTs — instead we patch `HTTPClient.upload_file` to record without
    hitting the network. The test still proves that:

      * the create call lands with the right slots and metadata
      * one upload per (existing) artifact happens, with the correct
        signed URL routed to the matching slot
      * finalize lands with `duration_s` and `fps` populated by the
        adapter
    """
    captured_json: list[httpx.Request] = []
    uploads: list[tuple[str, str, str]] = []  # (url, content_type, file_basename)

    def handler(request: httpx.Request) -> httpx.Response:
        captured_json.append(request)
        if request.url.path == "/api/ingest/episode" and request.method == "POST":
            return httpx.Response(
                201,
                json={
                    "episode_id": "ep_ros2_test",
                    "status": "recording",
                    "storage": "r2",
                    "upload_urls": [
                        {
                            "kind": "video",
                            "url": "https://r2.test/video?sig=v",
                            "expires_at": "2026-05-08T13:00:00Z",
                            "public_url": "https://cdn.test/video.mp4",
                        },
                        {
                            "kind": "sensors",
                            "url": "https://r2.test/sensors?sig=s",
                            "expires_at": "2026-05-08T13:00:00Z",
                            "public_url": "https://cdn.test/sensors.npz",
                        },
                        {
                            "kind": "actions",
                            "url": "https://r2.test/actions?sig=a",
                            "expires_at": "2026-05-08T13:00:00Z",
                            "public_url": "https://cdn.test/actions.npz",
                        },
                    ],
                },
            )
        if request.url.path.endswith("/finalize"):
            return httpx.Response(
                200,
                json={
                    "episode_id": "ep_ros2_test",
                    "status": "ready",
                    "updated_at": "2026-05-08T13:01:00Z",
                },
            )
        return httpx.Response(500, json={"error": f"unexpected {request.method} {request.url}"})

    transport = httpx.MockTransport(handler)
    client = rt.Client(
        api_key="rt_test", base_url="https://example.test", transport=transport
    )

    # Patch upload_file on this Client's HTTPClient instance so the
    # signed-URL PUTs don't actually fly to r2.test.
    def fake_upload_file(self: Any, url: str, path: Any, *, content_type: str) -> int:
        p = Path(path)
        size = p.stat().st_size
        uploads.append((url, content_type, p.name))
        return size

    monkeypatch.setattr(
        client._http, "upload_file", fake_upload_file.__get__(client._http)
    )

    episode = ros2.upload_bag(
        synthetic_bag,
        client=client,
        name="ros2 fixture run",
        policy_version="pap-v3.2.1",
        env_version="halcyon-cell-rev4",
        git_sha="abc1234",
        seed=8124,
    )

    assert episode.id == "ep_ros2_test"
    assert episode.status == "ready"

    # The two JSON requests we control: create + finalize.
    methods_paths = [(r.method, r.url.path) for r in captured_json]
    assert methods_paths[0] == ("POST", "/api/ingest/episode")
    assert methods_paths[-1] == ("POST", "/api/ingest/episode/ep_ros2_test/finalize")

    create_payload = json.loads(captured_json[0].content)
    assert create_payload["policy_version"] == "pap-v3.2.1"
    assert create_payload["env_version"] == "halcyon-cell-rev4"
    assert create_payload["git_sha"] == "abc1234"
    assert create_payload["seed"] == 8124
    assert create_payload["request_uploads"] == ["video", "sensors", "actions"]
    assert create_payload["metadata"]["adapter"] == "ros2"
    assert "fps" in create_payload

    # One upload per slot, each routed to the matching signed URL with
    # the right Content-Type (video/mp4 vs application/octet-stream).
    upload_by_url = {url: (ct, name) for url, ct, name in uploads}
    assert upload_by_url["https://r2.test/video?sig=v"] == ("video/mp4", "video.mp4")
    assert upload_by_url["https://r2.test/sensors?sig=s"] == (
        "application/octet-stream",
        "sensors.npz",
    )
    assert upload_by_url["https://r2.test/actions?sig=a"] == (
        "application/octet-stream",
        "actions.npz",
    )

    finalize_payload = json.loads(captured_json[-1].content)
    assert finalize_payload["status"] == "ready"
    assert finalize_payload["duration_s"] > 0
    assert finalize_payload["fps"] == 10.0


# ── classifier unit tests ─────────────────────────────────────────────


def test_classify_topic_falls_through_to_topic_name() -> None:
    """An unknown msgtype on a `cmd_*` topic still routes to actions."""
    decision = ros2.classify_topic(
        "/my_robot/cmd_arm", "my_pkg/msg/CustomCommand"
    )
    assert decision.slot == "actions"
    assert decision.reason == "topic-name"


def test_classify_topic_defaults_to_sensors() -> None:
    decision = ros2.classify_topic("/some/other/topic", "my_pkg/msg/Reading")
    assert decision.slot == "sensors"
    assert decision.reason == "default"
