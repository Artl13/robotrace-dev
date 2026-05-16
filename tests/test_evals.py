"""Tests for the replay regression harness — `robotrace.evals`.

Two scopes covered here:

  • Metric math (`action_l2_distance`, `ood_action_share`,
    `materialize_observations`, `extract_outcome`) tested with
    in-memory dict-of-arrays — no network, no MockTransport.
  • End-to-end loop (`create_run` → `run_against` → `complete_run`)
    against an httpx MockTransport, exercising the wire shape the
    Web API expects and the failure paths (policy raises, baseline
    404, dry-run skips upload).

The MockTransport responses mirror the real ingest routes; if the
server-side schema in `apps/web/lib/evals/ingest-schemas.ts` changes,
these tests will fail loudly.
"""

from __future__ import annotations

import io
from typing import Any

import httpx
import numpy as np
import pytest

import robotrace as rt
from robotrace import evals as evals_mod

# ── metric math ──────────────────────────────────────────────────────


def test_action_l2_distance_matches_baseline_returns_zero() -> None:
    """Two identical action streams have L2 distance of zero."""
    baseline = {
        "/cmd_vel/linear": np.tile(np.array([0.5, 0.0, 0.0]), (10, 1)),
        "/cmd_vel/_t_ns": np.arange(10) * 1_000_000,
    }
    candidate = [
        {"/cmd_vel/linear": np.array([0.5, 0.0, 0.0])} for _ in range(10)
    ]
    distance = evals_mod.metric_action_l2_distance(baseline, candidate, np=np)
    assert distance is not None
    assert distance == pytest.approx(0.0, abs=1e-9)


def test_action_l2_distance_known_offset() -> None:
    """Constant per-step offset yields the offset's L2 norm."""
    baseline = {
        "/cmd_vel/linear": np.zeros((5, 3)),
        "/cmd_vel/_t_ns": np.arange(5) * 1_000_000,
    }
    # Each candidate step differs by (1, 0, 0). L2 norm per step is 1.
    candidate = [{"/cmd_vel/linear": np.array([1.0, 0.0, 0.0])} for _ in range(5)]
    distance = evals_mod.metric_action_l2_distance(baseline, candidate, np=np)
    assert distance == pytest.approx(1.0)


def test_ood_action_share_in_distribution_is_zero() -> None:
    """Candidate actions inside the baseline's z<3 cone are not OOD."""
    rng = np.random.default_rng(seed=42)
    baseline = {
        "/cmd_vel/linear": rng.normal(loc=0.0, scale=1.0, size=(200, 3)),
        "/cmd_vel/_t_ns": np.arange(200),
    }
    candidate = [
        {"/cmd_vel/linear": np.array([0.1, -0.05, 0.0])} for _ in range(20)
    ]
    share = evals_mod.metric_ood_action_share(baseline, candidate, np=np)
    assert share == pytest.approx(0.0)


def test_ood_action_share_out_of_distribution_flags_steps() -> None:
    """Candidate values 10 sigmas away are flagged as OOD."""
    baseline = {
        "/cmd_vel/linear": np.zeros((100, 3)),  # std becomes 1.0 fallback
        "/cmd_vel/_t_ns": np.arange(100),
    }
    candidate = [
        {"/cmd_vel/linear": np.array([20.0, 0.0, 0.0])} for _ in range(5)
    ]
    share = evals_mod.metric_ood_action_share(baseline, candidate, np=np)
    assert share == pytest.approx(1.0)


def test_action_l2_distance_returns_none_for_missing_baseline() -> None:
    """No baseline → no metric (rollup ignores it)."""
    assert evals_mod.metric_action_l2_distance(None, [], np=np) is None
    assert evals_mod.metric_action_l2_distance({}, [{"x": np.zeros(3)}], np=np) is None


def test_materialize_observations_strips_timestamps_and_aligns() -> None:
    """Walks namespaced sensor arrays into per-step dicts."""
    sensors = {
        "/joint_states/position": np.arange(6).reshape(3, 2).astype(np.float32),
        "/joint_states/_t_ns": np.arange(3).astype(np.int64) * 1_000_000,
        "/imu/orientation": np.ones((3, 4), dtype=np.float32),
        "/imu/_t_ns": np.arange(3).astype(np.int64) * 1_000_000,
    }
    obs = evals_mod.materialize_observations(sensors, np=np)
    assert len(obs) == 3
    assert set(obs[0].keys()) == {
        "/joint_states/position",
        "/joint_states/_t_ns",
        "/imu/orientation",
        "/imu/_t_ns",
    }
    np.testing.assert_array_equal(
        obs[1]["/joint_states/position"], np.array([2.0, 3.0])
    )


def test_extract_outcome_pulls_from_lerobot_block() -> None:
    """Reads `next.success` / `next.reward_sum` out of LeRobot metadata."""
    meta = {
        "lerobot_episode_outcome": {
            "next.success": True,
            "next.reward_sum": 12.4,
        },
    }
    out = evals_mod.extract_outcome(meta)
    assert out["success"] is True
    assert out["reward_total"] == pytest.approx(12.4)


def test_extract_outcome_prefers_explicit_outcome_block() -> None:
    """SDK-stamped `outcome.success` wins over `lerobot_episode_outcome`."""
    meta = {
        "outcome": {"success": False, "reward_total": 3.2},
        "lerobot_episode_outcome": {"next.success": True, "next.reward_sum": 99},
    }
    out = evals_mod.extract_outcome(meta)
    assert out["success"] is False
    # Reward picks up from the same source it found `success` in.
    assert out["reward_total"] == pytest.approx(3.2)


# ── end-to-end loop with MockTransport ───────────────────────────────


def _build_npz_bytes(arrays: dict[str, Any]) -> bytes:
    """Serialize a dict-of-arrays into NPZ bytes the runner can `np.load`."""
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def _make_runner_fixture(
    baseline_ids: list[str],
    *,
    sensors_arrays: dict[str, Any] | None = None,
    actions_arrays: dict[str, Any] | None = None,
    artifact_404_for: set[str] | None = None,
    baseline_metadata: dict[str, Any] | None = None,
) -> tuple[rt.Client, list[httpx.Request]]:
    """Wire a MockTransport that emulates the eval-run ingest routes,
    the artifact resolver, and the episode-create / finalize routes.

    Returns (client, captured_requests). The captured list is the
    test's window into "what did the SDK actually send".
    """
    captured: list[httpx.Request] = []
    sensors_npz = _build_npz_bytes(
        sensors_arrays
        or {
            "/joint_states/position": np.zeros((3, 2), dtype=np.float32),
            "/joint_states/_t_ns": np.arange(3, dtype=np.int64) * 1_000_000,
        }
    )
    actions_npz = _build_npz_bytes(
        actions_arrays
        or {
            "/cmd_vel/linear": np.zeros((3, 3), dtype=np.float32),
            "/cmd_vel/_t_ns": np.arange(3, dtype=np.int64) * 1_000_000,
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        path = request.url.path
        # Eval-run create.
        if request.method == "POST" and path == "/api/ingest/eval-run":
            return httpx.Response(
                201,
                json={
                    "eval_run_id": "00000000-0000-0000-0000-000000000eee",
                    "status": "pending",
                    "episode_count": len(baseline_ids),
                },
            )
        # Per-result push.
        if request.method == "POST" and path.endswith("/result"):
            return httpx.Response(
                200,
                json={
                    "eval_result_id": "00000000-0000-0000-0000-000000000aaa",
                    "run_status": "running",
                    "episodes_completed": 1,
                    "episodes_failed": 0,
                },
            )
        # Finalize.
        if request.method == "POST" and path.endswith("/finalize"):
            return httpx.Response(
                200,
                json={
                    "eval_run_id": "00000000-0000-0000-0000-000000000eee",
                    "status": "completed",
                    "summary": {
                        "success_rate": {
                            "baseline": 1.0,
                            "candidate": 1.0,
                            "delta": 0.0,
                            "delta_is_better": False,
                        },
                        "recommend": "hold",
                        "better_count": 0,
                        "metric_total": 1,
                    },
                },
            )
        # Episode create (called when the runner mints the candidate
        # episode for each baseline).
        if request.method == "POST" and path == "/api/ingest/episode":
            return httpx.Response(
                201,
                json={
                    "episode_id": "00000000-0000-0000-0000-000000000111",
                    "status": "recording",
                    "storage": "unconfigured",
                    "upload_urls": [],
                },
            )
        if request.method == "POST" and path.startswith("/api/ingest/episode/"):
            # Per-episode finalize.
            return httpx.Response(
                200,
                json={
                    "episode_id": "00000000-0000-0000-0000-000000000111",
                    "status": "ready",
                    "updated_at": "2026-05-17T00:00:00Z",
                },
            )
        # Artifact resolver — returns the NPZ bytes directly.
        if request.method == "GET" and path.startswith("/api/episodes/"):
            parts = path.split("/")
            if len(parts) >= 6 and parts[4] == "artifact":
                kind = parts[5]
                episode_id = parts[3]
                if (
                    artifact_404_for is not None
                    and episode_id in artifact_404_for
                ):
                    return httpx.Response(404, json={"error": "not found"})
                if kind == "actions":
                    return httpx.Response(
                        200,
                        content=actions_npz,
                        headers={"Content-Type": "application/octet-stream"},
                    )
                if kind == "sensors":
                    return httpx.Response(
                        200,
                        content=sensors_npz,
                        headers={"Content-Type": "application/octet-stream"},
                    )
            # Episode metadata fallback (the SDK soft-imports this).
            if len(parts) == 4:
                if baseline_metadata is not None:
                    return httpx.Response(
                        200,
                        json={
                            "id": parts[3],
                            "metadata": baseline_metadata,
                        },
                    )
                return httpx.Response(404, json={"error": "not found"})

        return httpx.Response(500, json={"error": f"no stub for {path}"})

    client = rt.Client(
        api_key="rt_id_secret",
        base_url="https://example.test",
        transport=httpx.MockTransport(handler),
    )
    return client, captured


def test_create_run_round_trips_payload() -> None:
    """create_run posts the candidate version + ids the server expects."""
    client, captured = _make_runner_fixture(
        baseline_ids=["00000000-0000-0000-0000-000000000abc"],
    )
    try:
        run = evals_mod.create_run(
            candidate_policy_version="pap-v13",
            baseline_episode_ids=["00000000-0000-0000-0000-000000000abc"],
            baseline_policy_version="pap-v12",
            name="nightly",
            client=client,
        )
    finally:
        client.close()

    assert run.id == "00000000-0000-0000-0000-000000000eee"
    assert run.candidate_policy_version == "pap-v13"
    assert run.status == "pending"
    posted = [r for r in captured if r.url.path == "/api/ingest/eval-run"]
    assert len(posted) == 1
    import json

    body = json.loads(posted[0].content.decode())
    assert body["candidate_policy_version"] == "pap-v13"
    assert body["baseline_policy_version"] == "pap-v12"
    assert body["baseline_episode_ids"] == [
        "00000000-0000-0000-0000-000000000abc"
    ]
    assert body["name"] == "nightly"


def test_create_run_rejects_empty_baseline_list() -> None:
    """Empty baseline list raises ConfigurationError, no network."""
    with pytest.raises(rt.ConfigurationError):
        evals_mod.create_run(
            candidate_policy_version="v1",
            baseline_episode_ids=[],
        )


def test_create_run_rejects_empty_candidate_version() -> None:
    with pytest.raises(rt.ConfigurationError):
        evals_mod.create_run(
            candidate_policy_version="   ",
            baseline_episode_ids=["00000000-0000-0000-0000-000000000abc"],
        )


def test_run_against_pushes_result_per_episode() -> None:
    """Happy path: one baseline → one /result POST."""
    client, captured = _make_runner_fixture(
        baseline_ids=["00000000-0000-0000-0000-0000000000bb"],
    )

    def policy(_obs: evals_mod.Observation) -> evals_mod.Action:
        return {"/cmd_vel/linear": np.array([0.0, 0.0, 0.0])}

    try:
        run = evals_mod.create_run(
            candidate_policy_version="v2",
            baseline_episode_ids=["00000000-0000-0000-0000-0000000000bb"],
            client=client,
        )
        results = evals_mod.run_against(run, policy_callable=policy)
    finally:
        client.close()

    assert len(results) == 1
    assert results[0].status == "complete"
    assert run.episodes_completed == 1
    posted_results = [r for r in captured if r.url.path.endswith("/result")]
    assert len(posted_results) == 1


def test_run_against_records_failure_when_policy_raises() -> None:
    """A throwing policy yields a failed result row + continues."""
    client, captured = _make_runner_fixture(
        baseline_ids=[
            "00000000-0000-0000-0000-0000000000aa",
            "00000000-0000-0000-0000-0000000000bb",
        ],
    )

    calls = {"n": 0}

    # _replay_one aborts on the first raise, so we only need to throw
    # on the very first invocation to fail baseline 1; baseline 2's
    # three steps then all succeed.
    def policy(_obs: evals_mod.Observation) -> evals_mod.Action:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("policy exploded")
        return {"/cmd_vel/linear": np.array([0.0, 0.0, 0.0])}

    try:
        run = evals_mod.create_run(
            candidate_policy_version="v3",
            baseline_episode_ids=[
                "00000000-0000-0000-0000-0000000000aa",
                "00000000-0000-0000-0000-0000000000bb",
            ],
            client=client,
        )
        results = evals_mod.run_against(run, policy_callable=policy)
    finally:
        client.close()

    assert len(results) == 2
    assert results[0].status == "failed"
    assert "policy exploded" in (results[0].error or "")
    assert results[1].status == "complete"
    # Both /result rows posted; failure didn't bail the loop.
    posted_results = [r for r in captured if r.url.path.endswith("/result")]
    assert len(posted_results) == 2


def test_run_against_handles_missing_baseline_artifacts() -> None:
    """Baseline with no NPZ artifacts still produces a metric row.

    The runner short-circuits the per-step loop (no observations to
    iterate), records a result with empty metrics, and moves on.
    """
    bid = "00000000-0000-0000-0000-0000000000dd"
    client, _captured = _make_runner_fixture(
        baseline_ids=[bid],
        artifact_404_for={bid},
    )

    def policy(_obs: evals_mod.Observation) -> evals_mod.Action:
        raise AssertionError("policy must not be called with no observations")

    try:
        run = evals_mod.create_run(
            candidate_policy_version="v4",
            baseline_episode_ids=[bid],
            client=client,
        )
        results = evals_mod.run_against(run, policy_callable=policy)
    finally:
        client.close()

    assert len(results) == 1
    assert results[0].status == "complete"
    # No L2 distance computable without baseline actions.
    assert results[0].metrics.get("action_l2_distance") is None


def test_run_against_dry_run_skips_uploads() -> None:
    """`dry_run=True` runs the policy locally but doesn't POST /result."""
    client, captured = _make_runner_fixture(
        baseline_ids=["00000000-0000-0000-0000-0000000000bb"],
    )

    def policy(_obs: evals_mod.Observation) -> evals_mod.Action:
        return {"/cmd_vel/linear": np.array([0.0, 0.0, 0.0])}

    try:
        run = evals_mod.create_run(
            candidate_policy_version="v5",
            baseline_episode_ids=["00000000-0000-0000-0000-0000000000bb"],
            client=client,
        )
        results = evals_mod.run_against(
            run, policy_callable=policy, dry_run=True
        )
    finally:
        client.close()

    assert len(results) == 1
    assert results[0].status == "complete"
    posted_results = [r for r in captured if r.url.path.endswith("/result")]
    assert posted_results == []
    # No candidate episode minted either.
    posted_episodes = [
        r for r in captured if r.url.path == "/api/ingest/episode"
    ]
    assert posted_episodes == []


def test_run_against_handles_pickled_object_arrays_in_npz() -> None:
    """Regression: an NPZ that ``np.savez`` had to pickle (because one
    of the arrays is dtype=object — e.g. variable-length strings or a
    ragged column the encoder couldn't flatten) must load cleanly.

    numpy 2.x defaults to ``allow_pickle=False`` on load, so the
    runner has to opt in explicitly. The trust-boundary argument is
    in the comment inside ``_fetch_npz``; this test pins the
    behaviour so a future "let's default-deny" refactor doesn't
    silently break every replay that touches a real ROS 2 / LeRobot
    encode output.
    """
    bid = "00000000-0000-0000-0000-0000000000cc"
    # One numeric column + one object column. `np.savez` will pickle
    # the object array on disk, which is what trips `allow_pickle`.
    actions_with_object_field = {
        "/cmd_vel/linear": np.zeros((3, 3), dtype=np.float32),
        "/cmd_vel/_t_ns": np.arange(3, dtype=np.int64) * 1_000_000,
        # Object array: variable-length topic-name strings, the kind
        # of column the LeRobot adapter emits for free-text metadata
        # columns it can't flatten into a homogeneous tensor.
        "/meta/labels": np.asarray(["pick", "lift", "place"], dtype=object),
    }
    client, _captured = _make_runner_fixture(
        baseline_ids=[bid],
        actions_arrays=actions_with_object_field,
    )

    def policy(_obs: evals_mod.Observation) -> evals_mod.Action:
        return {"/cmd_vel/linear": np.array([0.0, 0.0, 0.0])}

    try:
        run = evals_mod.create_run(
            candidate_policy_version="v-pickle",
            baseline_episode_ids=[bid],
            client=client,
        )
        results = evals_mod.run_against(run, policy_callable=policy)
    finally:
        client.close()

    # The whole point of the regression: completes, doesn't blow up
    # with `ValueError: This file contains pickled (object) data`.
    assert len(results) == 1
    assert results[0].status == "complete"


def test_complete_run_posts_finalize() -> None:
    """complete_run hits the finalize endpoint and returns the summary."""
    client, captured = _make_runner_fixture(
        baseline_ids=["00000000-0000-0000-0000-0000000000bb"],
    )
    try:
        run = evals_mod.create_run(
            candidate_policy_version="v6",
            baseline_episode_ids=["00000000-0000-0000-0000-0000000000bb"],
            client=client,
        )
        body = evals_mod.complete_run(run)
    finally:
        client.close()

    assert body["status"] == "completed"
    assert "summary" in body
    finalize_calls = [
        r for r in captured if r.url.path.endswith("/finalize")
    ]
    assert len(finalize_calls) == 1
