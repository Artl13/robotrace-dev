"""ROS 2 adapter — rosbag2 → RoboTrace episode.

Three entry points, ordered by how much you want the SDK to do:

    from robotrace.adapters import ros2

    # 1. Inspect a bag without writing anything.
    summary = ros2.scan_bag("./run_2026-05-08/")
    print(summary.report())

    # 2. Encode the bag's classified topics to MP4 + NPZ files. No
    #    network. Useful when you want to inspect / post-process the
    #    artifacts before uploading.
    encoded = ros2.encode_bag("./run_2026-05-08/", "/tmp/encoded/")

    # 3. One-shot: encode to a tempdir, upload, finalize. The shape
    #    99% of users will reach for.
    ros2.upload_bag(
        "./run_2026-05-08/",
        name="warmup pick-and-place",
        policy_version="pap-v3.2.1",
        env_version="halcyon-cell-rev4",
        git_sha="abc1234",
    )

Topic auto-classification (`sensor_msgs/Image` → video, JointState /
Imu / etc → sensors, Twist / cmd_* → actions) covers the common case
without any kwargs. Override per-slot with `video_topics=`,
`sensor_topics=`, `action_topics=` (an empty list deliberately
*excludes* a slot). Pick a single canonical camera from a multi-cam
bag with `canonical_video_topic="/camera/image_raw"`.

Storage: `rosbags` reads both sqlite3 and mcap rosbag2 backends out
of the box. ROS 1 (`rosbag1`) is **out of scope** per AGENTS.md.

Coming next in `robotrace 0.2`: a live `ros2.record(topics=[...])`
context manager that subscribes to topics during a run and ships the
result as an episode on exit. The encoder/uploader plumbing here
will back it.
"""

from __future__ import annotations

from ._classify import TopicClass, classify_topic
from ._encode import EncodedArtifact, EncodedBag, encode_bag
from ._scan import BagSummary, TopicInfo, scan_bag
from ._upload import upload_bag

__all__ = [
    # Primary surface — the three verbs.
    "scan_bag",
    "encode_bag",
    "upload_bag",
    # Result types.
    "BagSummary",
    "TopicInfo",
    "EncodedBag",
    "EncodedArtifact",
    # Classifier (exposed so users can sanity-check or extend it).
    "TopicClass",
    "classify_topic",
]
