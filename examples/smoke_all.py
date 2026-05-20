"""End-to-end smoke test for a RoboTrace deployment.

What this exercises - in order - against your live deployment:

  [1/3] Episodes
        • Log a baseline ready episode (sensors + actions uploaded).
        • Log a deliberately-failed episode with `metadata.failure_reason`
          so the **Failure Intelligence** analyzer should attach findings
          on the portal detail page.

  [2/3] Evals
        • Create an eval run with the baseline above.
        • Replay it through a tiny noisy candidate policy.
        • Finalize the run and print the rollup summary.

  [3/3] Verify
        • Promote the baseline episode to a `critical` verification
          scenario.
        • Call `rt.verify.check_gate(...)` (expect blocked the first
          time - no result for the candidate yet).
        • Run `rt.verify.run_check(...)` with an identity policy and
          confirm the gate flips to `passed`.

Usage:

    cd packages/sdk-python
    pip install -e .

    export ROBOTRACE_API_KEY=rt_<id>_<secret>
    export ROBOTRACE_BASE_URL=https://app.robotrace.dev   # or localhost

    python examples/smoke_all.py

Optional flags:

    --skip-evals      stop after Episodes
    --skip-verify     stop after Evals
    --tag <label>     custom run tag (default: timestamp)

Re-runs are safe - every episode / eval / scenario name carries the
run tag so nothing collides with a previous pass.

Exit codes:
    0  all three pillars green
    1  any pillar failed (the failing section's error is printed)
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import os
import sys
import time
from pathlib import Path
from typing import Any

import robotrace as rt
from robotrace import APIError, ConfigurationError


# ── small helpers for pretty section output ──────────────────────────


GREEN = "\033[32m"
RED = "\033[31m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _color(s: str, code: str) -> str:
    if not sys.stdout.isatty():
        return s
    return f"{code}{s}{RESET}"


def _section(label: str) -> None:
    print()
    print(_color(f"━━━ {label} ━━━", BOLD))


def _ok(msg: str) -> None:
    print(f"  {_color('ok', GREEN)}  {msg}")


def _fail(msg: str) -> None:
    print(f"  {_color('x ', RED)} {msg}")


def _info(msg: str) -> None:
    print(f"  {_color('··', DIM)} {msg}")


# ── synthetic artifacts (tiny - keep the smoke run snappy) ───────────


def _write_synthetic_sensors(path: Path, n: int = 64) -> None:
    """Write a small NPZ in the namespaced shape the eval harness
    expects: `<topic>/<field>` + `<topic>/_t_ns`."""
    import numpy as np

    t_ns = (np.arange(n, dtype=np.int64) * 33_333_333)  # ~30 fps
    pos = np.linspace(-0.2, 0.35, n, dtype=np.float32)
    vel = np.cos(np.linspace(0, 6.28, n, dtype=np.float32))
    np.savez(
        path,
        **{
            "/joint_states/_t_ns": t_ns,
            "/joint_states/position": pos,
            "/joint_states/velocity": vel,
        },
    )


def _write_synthetic_actions(path: Path, n: int = 64) -> None:
    """Match the sensors shape one-for-one so eval metrics can pair up."""
    import numpy as np

    t_ns = (np.arange(n, dtype=np.int64) * 33_333_333)
    cmd = np.linspace(0.0, 1.0, n, dtype=np.float32)
    np.savez(
        path,
        **{
            "/cmd_vel/_t_ns": t_ns,
            "/cmd_vel/linear": cmd,
        },
    )


# ── 1/3 Episodes ─────────────────────────────────────────────────────


@dataclasses.dataclass
class EpisodesResult:
    baseline_id: str
    failed_id: str
    baseline_url: str
    failed_url: str


def _section_episodes(tag: str, base_url: str) -> EpisodesResult:
    _section("[1/3] Episodes")

    sensors = Path(f"/tmp/robotrace_smoke_sensors_{tag}.npz")
    actions = Path(f"/tmp/robotrace_smoke_actions_{tag}.npz")
    _write_synthetic_sensors(sensors)
    _write_synthetic_actions(actions)
    _info(f"wrote synthetic sensors → {sensors.name} ({sensors.stat().st_size:,} B)")
    _info(f"wrote synthetic actions → {actions.name} ({actions.stat().st_size:,} B)")

    baseline = rt.log_episode(
        name=f"smoke baseline · {tag}",
        source="sim",
        robot="smoke-rig",
        policy_version=f"smoke-baseline-{tag}",
        env_version="smoke-env-v1",
        git_sha="smoke000",
        seed=42,
        sensors=str(sensors),
        actions=str(actions),
        duration_s=2.1,
        fps=30,
        metadata={
            "task": "smoke_test",
            # Stamp a clean outcome so the eval harness has something
            # to compare candidates against and verify's pass criterion
            # (`success_candidate is True`) succeeds on identity replay.
            "outcome": {"success": True, "reward_total": 1.0},
            "_smoke_tag": tag,
        },
        status="ready",
    )
    _ok(f"baseline episode: {baseline.id}  ({baseline.status}, storage={baseline.storage})")

    failed = rt.log_episode(
        name=f"smoke failure · {tag}",
        source="sim",
        robot="smoke-rig",
        policy_version=f"smoke-baseline-{tag}",
        env_version="smoke-env-v1",
        git_sha="smoke000",
        seed=43,
        duration_s=0.4,
        metadata={
            # Failure Intelligence should pick this up via the
            # `failure_reason_in_metadata` rule and surface it on the
            # episode detail page.
            "failure_reason": "RuntimeError: gripper stalled mid-grasp",
            "outcome": {"success": False},
            "_smoke_tag": tag,
        },
        status="failed",
    )
    _ok(f"failed episode:   {failed.id}  ({failed.status})")
    _info("Failure Intelligence runs on the server when status flips to")
    _info("`failed`. Open the failed episode in the portal - the")
    _info("\"Failure insights\" card should list ranked findings.")

    return EpisodesResult(
        baseline_id=baseline.id,
        failed_id=failed.id,
        baseline_url=f"{base_url.rstrip('/')}/portal/episodes/{baseline.id}",
        failed_url=f"{base_url.rstrip('/')}/portal/episodes/{failed.id}",
    )


# ── 2/3 Evals ────────────────────────────────────────────────────────


def _noisy_policy(obs: dict[str, Any]) -> dict[str, Any]:
    """Per-step policy that returns the baseline action with a small
    additive offset - enough to produce non-zero L2 distance without
    blowing up the OOD share."""
    import numpy as np

    pos = obs.get("/joint_states/position")
    if pos is None:
        return {"/cmd_vel/linear": np.float32(0.0)}
    arr = np.asarray(pos, dtype=np.float32).reshape(-1)
    return {"/cmd_vel/linear": (arr.mean() + 0.05).astype(np.float32)}


def _section_evals(tag: str, baseline_id: str, base_url: str) -> str:
    _section("[2/3] Evals")

    run = rt.evals.create_run(
        candidate_policy_version=f"smoke-candidate-{tag}",
        baseline_episode_ids=[baseline_id],
        baseline_policy_version=f"smoke-baseline-{tag}",
        name=f"smoke eval · {tag}",
    )
    _ok(f"created eval run {run.id} (candidate={run.candidate_policy_version})")

    results = rt.evals.run_against(run, policy_callable=_noisy_policy)
    if not results:
        raise RuntimeError("eval run produced zero per-episode results")
    completed = sum(1 for r in results if r.status == "complete")
    failed = sum(1 for r in results if r.status != "complete")
    _ok(f"replayed {len(results)} episode(s): {completed} complete, {failed} failed")

    body = rt.evals.complete_run(run, status="completed")
    summary = body.get("summary") or {}
    success = summary.get("success_rate") or {}
    _ok(
        "rollup ready: "
        f"baseline_success={success.get('baseline')} candidate_success={success.get('candidate')}"
    )

    return f"{base_url.rstrip('/')}/portal/evals/{run.id}"


# ── 3/3 Verify ───────────────────────────────────────────────────────


def _identity_policy(obs: dict[str, Any]) -> dict[str, Any]:
    """Policy that mirrors the baseline observation - produces a clean
    `success_candidate=True` for the verify pass case."""
    import numpy as np

    pos = obs.get("/joint_states/position")
    if pos is None:
        return {"/cmd_vel/linear": np.float32(0.0)}
    arr = np.asarray(pos, dtype=np.float32).reshape(-1)
    return {"/cmd_vel/linear": arr.mean().astype(np.float32)}


def _section_verify(tag: str, baseline_id: str) -> None:
    _section("[3/3] Verify")

    scenario_name = f"smoke critical · {tag}"
    candidate = f"smoke-verify-{tag}"

    # `verify.promote` is idempotent server-side - a second call with
    # the same baseline returns 200 + `promoted: false` and the existing
    # `scenario_id`, so re-runs of this smoke test reuse the scenario
    # instead of failing.
    promoted = rt.verify.promote(
        baseline_episode_id=baseline_id,
        name=scenario_name,
        description="Created by examples/smoke_all.py",
        severity="critical",
    )
    scenario_id = promoted.get("scenario_id")
    fresh = promoted.get("promoted", True)
    _ok(
        f"baseline promoted to critical scenario {scenario_id}"
        + ("" if fresh else " (already existed, reused)")
    )

    gate_before = rt.verify.check_gate(candidate_policy_version=candidate)
    passed_before = bool(gate_before.get("passed"))
    blocking = sum(
        1
        for s in (gate_before.get("scenarios") or [])
        if s.get("severity") == "critical" and s.get("latest_status") != "pass"
    )
    _ok(
        f"check_gate (pre-replay): passed={passed_before}, "
        f"critical scenarios still blocking={blocking}"
    )

    exit_code, gate_after = rt.verify.run_check(
        candidate_policy_version=candidate,
        policy_callable=_identity_policy,
    )
    passed_after = bool(gate_after.get("passed"))
    if exit_code != 0 or not passed_after:
        raise RuntimeError(
            f"verify.run_check failed: exit={exit_code}, "
            f"passed={passed_after}, body={gate_after}"
        )
    _ok(f"run_check passed (exit_code=0, gate.passed={passed_after})")


# ── main ─────────────────────────────────────────────────────────────


def _resolve_base_url() -> str:
    """Pull base_url from env or the credentials file - matches the
    Client's own resolution order."""
    explicit = os.environ.get("ROBOTRACE_BASE_URL")
    if explicit:
        return explicit
    try:
        from robotrace._credentials import read_credentials

        creds = read_credentials()
        if creds and creds.base_url:
            return creds.base_url
    except Exception:
        pass
    return "https://app.robotrace.dev"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--skip-evals",
        action="store_true",
        help="stop after the Episodes section",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="stop after the Evals section",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="custom run tag (default: UTC timestamp)",
    )
    args = parser.parse_args()

    tag = args.tag or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    started = time.monotonic()

    base_url = _resolve_base_url()
    print(_color(f"RoboTrace smoke · tag={tag}", BOLD))
    print(f"  base_url:  {base_url}")
    print(f"  sdk:       robotrace=={rt.__version__}")

    try:
        rt.init()
    except ConfigurationError as exc:
        print()
        print(_color("Configuration error.", RED))
        print(f"  {exc}")
        print("  Set ROBOTRACE_API_KEY and ROBOTRACE_BASE_URL or run `robotrace login`.")
        return 1

    episodes: EpisodesResult | None = None
    eval_url: str | None = None

    try:
        episodes = _section_episodes(tag=tag, base_url=base_url)
        if not args.skip_evals:
            eval_url = _section_evals(
                tag=tag,
                baseline_id=episodes.baseline_id,
                base_url=base_url,
            )
        if not args.skip_evals and not args.skip_verify:
            _section_verify(tag=tag, baseline_id=episodes.baseline_id)
    except APIError as exc:
        _fail(f"APIError (status={exc.status_code}): {exc}")
        if exc.response_body:
            body = (
                exc.response_body
                if isinstance(exc.response_body, str)
                else repr(exc.response_body)
            )
            for line in body.splitlines()[:8]:
                print(f"     {line}")
        return 1
    except Exception as exc:
        _fail(f"{type(exc).__name__}: {exc}")
        return 1

    elapsed = time.monotonic() - started
    print()
    print(_color("All sections green.", GREEN) + f"  ({elapsed:.1f}s)")
    print()
    print("  Open these in the portal to spot-check the UI:")
    if episodes is not None:
        print(f"    baseline episode      → {episodes.baseline_url}")
        print(f"    failed episode        → {episodes.failed_url}")
        print("                              (look for the 'Failure insights' card)")
    if eval_url:
        print(f"    eval run rollup       → {eval_url}")
    if episodes is not None and not args.skip_verify:
        print(f"    verification scenarios → {base_url.rstrip('/')}/portal/verify")
    return 0


if __name__ == "__main__":
    sys.exit(main())
