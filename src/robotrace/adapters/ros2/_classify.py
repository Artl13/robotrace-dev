"""Heuristics for sorting ROS 2 topics into RoboTrace artifact slots.

A rosbag2 typically contains a mix of sensor streams, camera streams,
and command streams. The RoboTrace episode model has three artifact
slots — `video`, `sensors`, `actions` — and the adapter has to decide
which slot every topic belongs in.

Rules (ordered, first match wins):

    1. Image-shaped messages (`sensor_msgs/(Compressed)?Image`,
       `theora_image_transport/Packet`, …) → ``video``.
    2. Action-shaped messages (`geometry_msgs/Twist*`,
       `trajectory_msgs/JointTrajectory*`,
       `control_msgs/JointJog`, anything where the topic name ends
       in `/cmd_*` or `/command`) → ``actions``.
    3. Everything else → ``sensors``.

Users always win: an explicit override (``video_topics=…``,
``sensor_topics=…``, ``action_topics=…`` on the public API) skips
the heuristic entirely. The classifier just provides a sane default
so the common case — a one-camera, one-IMU, one-cmd_vel bag — works
without any kwargs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Slot labels — match the SDK's ArtifactKind in episode.py exactly.
# Re-imported here as a Literal to avoid a circular import path
# (adapters → core → adapters) when this module is loaded eagerly
# from `adapters.ros2.__init__`.
Slot = Literal["video", "sensors", "actions"]


# Message types that carry one frame of pixel data each. We keep the
# list explicit (rather than substring matching `/Image`) so a
# user-defined message named `MyImageDescriptor` doesn't accidentally
# get encoded as a frame stream.
_IMAGE_MSGTYPES: frozenset[str] = frozenset(
    {
        "sensor_msgs/msg/Image",
        "sensor_msgs/msg/CompressedImage",
    }
)

# Message types that almost always represent a controller / planner
# command (i.e. an action). Matching is exact for the same reason as
# above. Topic-name fallback (`/cmd_*`, `/command`) catches the long
# tail of project-specific Twist wrappers.
_ACTION_MSGTYPES: frozenset[str] = frozenset(
    {
        "geometry_msgs/msg/Twist",
        "geometry_msgs/msg/TwistStamped",
        "geometry_msgs/msg/Wrench",
        "geometry_msgs/msg/WrenchStamped",
        "trajectory_msgs/msg/JointTrajectory",
        "trajectory_msgs/msg/MultiDOFJointTrajectory",
        "control_msgs/msg/JointJog",
    }
)


@dataclass(frozen=True)
class TopicClass:
    """One classification decision for one topic."""

    topic: str
    msgtype: str
    slot: Slot
    # Reason the slot was picked — surfaced in `BagSummary.report()`
    # so the user can sanity-check before uploading. "msgtype" means
    # we matched the message type list; "topic-name" means we fell
    # through to the cmd_*/command suffix rule; "default" means
    # neither matched and we routed to sensors.
    reason: Literal["msgtype", "topic-name", "default"]


def classify_topic(topic: str, msgtype: str) -> TopicClass:
    """Return the slot decision for one (topic, msgtype) pair.

    Pure function — no side effects, no IO. Tests can call this with
    arbitrary (topic, msgtype) pairs to pin behavior.
    """
    if msgtype in _IMAGE_MSGTYPES:
        return TopicClass(topic=topic, msgtype=msgtype, slot="video", reason="msgtype")
    if msgtype in _ACTION_MSGTYPES:
        return TopicClass(topic=topic, msgtype=msgtype, slot="actions", reason="msgtype")
    if _looks_like_command_topic(topic):
        return TopicClass(topic=topic, msgtype=msgtype, slot="actions", reason="topic-name")
    return TopicClass(topic=topic, msgtype=msgtype, slot="sensors", reason="default")


def _looks_like_command_topic(topic: str) -> bool:
    """True when the topic name suggests a control output.

    Matches the conventional ROS 2 pattern of putting commands under
    `…/cmd_*` (e.g. `/cmd_vel`, `/turtle1/cmd_vel`) or `…/command`
    (`/joint_trajectory_controller/command`). Lowercased before
    matching so projects that use `/CMD_VEL` still classify.
    """
    name = topic.rsplit("/", 1)[-1].lower()
    return name.startswith("cmd_") or name == "command"
