"""Tests for `robotrace.types` - typed metadata classes.

Two layers of behavior to cover:

  1. Per-class construction + validation (length / sign / coercion).
  2. End-to-end wire format: typed values inside `metadata={...}`
     get encoded to `__type`-tagged dicts on the way out, and plain
     dicts pass through unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest

import robotrace as rt
from robotrace import types as rt_types


# ── per-class construction ──────────────────────────────────────────


def test_joint_state_round_trips_basic() -> None:
    js = rt.JointState(positions=[0.1, 0.2, 0.3])
    out = js.to_dict()
    assert out == {
        "__type": "robotrace.JointState",
        "positions": [0.1, 0.2, 0.3],
        "velocities": None,
        "efforts": None,
        "names": None,
    }


def test_joint_state_accepts_full_parallel_arrays() -> None:
    js = rt.JointState(
        positions=[0.0, 1.5, -0.3],
        velocities=[0.01, 0.0, -0.02],
        efforts=[1.1, 2.2, 3.3],
        names=["joint1", "joint2", "joint3"],
    )
    out = js.to_dict()
    assert out["velocities"] == [0.01, 0.0, -0.02]
    assert out["efforts"] == [1.1, 2.2, 3.3]
    assert out["names"] == ["joint1", "joint2", "joint3"]


def test_joint_state_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="velocities"):
        rt.JointState(positions=[1.0, 2.0], velocities=[1.0])
    with pytest.raises(ValueError, match="efforts"):
        rt.JointState(positions=[1.0, 2.0], efforts=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="names"):
        rt.JointState(positions=[1.0, 2.0], names=["a"])


def test_joint_state_rejects_empty_positions() -> None:
    with pytest.raises(ValueError, match="at least one"):
        rt.JointState(positions=[])


def test_pose3d_rejects_wrong_dimensions() -> None:
    with pytest.raises(ValueError, match="translation"):
        rt.Pose3D(translation=[1.0, 2.0], rotation=[0, 0, 0, 1])
    with pytest.raises(ValueError, match="rotation"):
        rt.Pose3D(translation=[1.0, 2.0, 3.0], rotation=[0, 0, 1])


def test_pose3d_quaternion_order_documented() -> None:
    # The canonical [x, y, z, w] quaternion - we don't normalize or
    # reorder, the caller's responsibility. This test pins the field
    # order so a future "let's flip to [w, x, y, z]" PR breaks
    # something visible.
    p = rt.Pose3D(translation=[1.0, 2.0, 3.0], rotation=[0.0, 0.0, 0.0, 1.0])
    out = p.to_dict()
    assert out["rotation"] == [0.0, 0.0, 0.0, 1.0]
    assert out["__type"] == "robotrace.Pose3D"


def test_twist_round_trips() -> None:
    t = rt.Twist(linear=[0.5, 0.0, 0.0], angular=[0.0, 0.0, 0.1])
    out = t.to_dict()
    assert out == {
        "__type": "robotrace.Twist",
        "linear": [0.5, 0.0, 0.0],
        "angular": [0.0, 0.0, 0.1],
    }


def test_imu_orientation_optional() -> None:
    no_orient = rt.Imu(
        linear_acceleration=[0, 0, 9.81],
        angular_velocity=[0, 0, 0],
    )
    assert no_orient.to_dict()["orientation"] is None

    with_orient = rt.Imu(
        linear_acceleration=[0, 0, 9.81],
        angular_velocity=[0, 0, 0],
        orientation=[0, 0, 0, 1],
    )
    assert with_orient.to_dict()["orientation"] == [0.0, 0.0, 0.0, 1.0]


def test_battery_percent_range() -> None:
    ok = rt.Battery(percent=42.0, voltage_v=12.4, charging=False)
    assert ok.to_dict()["percent"] == 42.0
    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        rt.Battery(percent=120.0)
    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        rt.Battery(percent=-0.5)


def test_episode_outcome_all_optional() -> None:
    empty = rt.EpisodeOutcome()
    out = empty.to_dict()
    assert out["__type"] == "robotrace.EpisodeOutcome"
    assert out["success"] is None
    assert out["reward_total"] is None
    assert out["collision_count"] is None
    assert out["time_to_goal_s"] is None


def test_episode_outcome_rejects_negative_collision_count() -> None:
    with pytest.raises(ValueError, match="collision_count"):
        rt.EpisodeOutcome(collision_count=-1)


def test_episode_outcome_rejects_negative_time() -> None:
    with pytest.raises(ValueError, match="time_to_goal_s"):
        rt.EpisodeOutcome(time_to_goal_s=-0.1)


def test_typed_classes_are_frozen() -> None:
    js = rt.JointState(positions=[1.0])
    with pytest.raises(Exception):  # FrozenInstanceError subclass varies
        js.positions = [2.0]  # type: ignore[misc]


# ── encoder ─────────────────────────────────────────────────────────


def test_encode_passthrough_for_scalars() -> None:
    assert rt_types.encode(42) == 42
    assert rt_types.encode("hello") == "hello"
    assert rt_types.encode(None) is None
    assert rt_types.encode(True) is True


def test_encode_descends_into_mapping() -> None:
    raw = {"a": rt.JointState(positions=[1.0]), "b": 7}
    out = rt_types.encode(raw)
    assert isinstance(out, dict)
    assert out["a"]["__type"] == "robotrace.JointState"
    assert out["b"] == 7
    # Input untouched - encode returns a new container.
    assert isinstance(raw["a"], rt.JointState)


def test_encode_descends_into_list_and_tuple() -> None:
    raw = [rt.Twist(linear=[0, 0, 0], angular=[0, 0, 0]), (rt.Battery(percent=12.0),)]
    out = rt_types.encode(raw)
    assert out[0]["__type"] == "robotrace.Twist"
    assert out[1][0]["__type"] == "robotrace.Battery"
    # Tuples normalize to lists - the wire is JSON and arrays don't
    # distinguish list vs tuple anyway.
    assert isinstance(out[1], list)


def test_encode_passes_unknown_objects_through() -> None:
    class Custom:
        pass

    c = Custom()
    assert rt_types.encode(c) is c
    # Inside a dict, the foreign object survives - we encode only our
    # own typed values, never random user objects.
    out = rt_types.encode({"thing": c})
    assert out["thing"] is c


# ── end-to-end: metadata gets encoded before hitting the wire ──────


@dataclass
class StubFixture:
    client: rt.Client
    requests: list[httpx.Request] = field(default_factory=list)


def _make_fixture(responses: dict[tuple[str, str], dict[str, Any]]) -> StubFixture:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        body = responses.get((request.method, request.url.path))
        if body is None:
            return httpx.Response(500, json={"error": "no stub"})
        return httpx.Response(201 if request.method == "POST" else 200, json=body)

    transport = httpx.MockTransport(handler)
    client = rt.Client(
        api_key="rt_id123_secret456",
        base_url="https://example.test",
        transport=transport,
        verbose=False,
    )
    return StubFixture(client=client, requests=captured)


@pytest.fixture
def fix() -> StubFixture:
    return _make_fixture(
        {
            ("POST", "/api/ingest/episode"): {
                "episode_id": "ep_test_typed",
                "status": "recording",
                "storage": "unconfigured",
                "upload_urls": [],
            },
            ("POST", "/api/ingest/episode/ep_test_typed/finalize"): {
                "status": "ready",
            },
        }
    )


def _body(req: httpx.Request) -> dict[str, Any]:
    return json.loads(req.content.decode())


def test_start_episode_encodes_typed_metadata(fix: StubFixture) -> None:
    fix.client.start_episode(
        metadata={
            "outcome": rt.EpisodeOutcome(success=True, reward_total=12.3),
            "task": "pick_and_place",
            "joints": rt.JointState(positions=[0.1, 0.2]),
        },
    )
    payload = _body(fix.requests[0])["metadata"]
    assert payload["task"] == "pick_and_place"
    assert payload["outcome"] == {
        "__type": "robotrace.EpisodeOutcome",
        "success": True,
        "reward_total": 12.3,
        "collision_count": None,
        "time_to_goal_s": None,
    }
    assert payload["joints"]["__type"] == "robotrace.JointState"
    assert payload["joints"]["positions"] == [0.1, 0.2]


def test_plain_dict_metadata_still_works(fix: StubFixture) -> None:
    # The encoder is a pure superset - existing customers passing
    # plain JSON-friendly dicts must see no behavior change.
    fix.client.start_episode(metadata={"task": "pick", "scene": "kitchen"})
    payload = _body(fix.requests[0])["metadata"]
    assert payload == {"task": "pick", "scene": "kitchen"}


def test_log_episode_finalize_encodes_typed_metadata(fix: StubFixture) -> None:
    fix.client.log_episode(
        metadata={"outcome": rt.EpisodeOutcome(success=False, collision_count=2)},
    )
    # First request is create, second is finalize - both carry the
    # encoded metadata.
    create_meta = _body(fix.requests[0])["metadata"]
    finalize_meta = _body(fix.requests[1])["metadata"]
    for meta in (create_meta, finalize_meta):
        assert meta["outcome"]["__type"] == "robotrace.EpisodeOutcome"
        assert meta["outcome"]["success"] is False
        assert meta["outcome"]["collision_count"] == 2


def test_nested_typed_values_are_descended_into(fix: StubFixture) -> None:
    fix.client.start_episode(
        metadata={
            "history": [
                rt.Pose3D(translation=[0, 0, 0], rotation=[0, 0, 0, 1]),
                rt.Pose3D(translation=[1, 0, 0], rotation=[0, 0, 0, 1]),
            ],
        },
    )
    poses = _body(fix.requests[0])["metadata"]["history"]
    assert len(poses) == 2
    for p in poses:
        assert p["__type"] == "robotrace.Pose3D"
