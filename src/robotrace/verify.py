"""Regression verification scenarios - mandatory replay gates.

Three verbs mirror the eval harness pattern:

    rt.verify.check_gate(candidate_policy_version="pap-v13")
    rt.verify.record_result(scenario_id=..., candidate_policy_version=..., ...)
    rt.verify.run_check(candidate_policy_version="pap-v13", policy_callable=fn)

`run_check` is the CI entry point: it replays every active critical
scenario whose baseline hasn't passed for this candidate yet, records
verification results, then re-evaluates the gate. Exits non-zero when
any critical scenario still blocks deploy.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .errors import APIError, ConfigurationError, RobotraceError

if TYPE_CHECKING:
    from .client import Client
    from .evals import PolicyCallable


def check_gate(
    *,
    candidate_policy_version: str,
    client: Client | None = None,
) -> dict[str, Any]:
    """Evaluate the deploy gate for a candidate policy version.

    Returns the parsed JSON body from ``POST /api/verify/check``.
    HTTP 422 still parses - the caller reads ``passed`` from the body.
    """
    version = candidate_policy_version.strip()
    if not version:
        raise ConfigurationError(
            "`candidate_policy_version` is required and cannot be empty."
        )
    c = _resolve_client(client)
    try:
        return c._http.request(
            "POST",
            "/api/verify/check",
            json={"candidate_policy_version": version},
        )
    except APIError as exc:
        # The gate endpoint returns 422 with the same JSON shape when
        # critical scenarios block - surface the body to the caller.
        if exc.status_code == 422 and isinstance(exc.response_body, dict):
            return exc.response_body
        raise


def record_result(
    *,
    scenario_id: str,
    candidate_policy_version: str,
    status: str | None = None,
    candidate_episode_id: str | None = None,
    eval_run_id: str | None = None,
    metrics: Mapping[str, Any] | None = None,
    error: str | None = None,
    client: Client | None = None,
) -> dict[str, Any]:
    """Record one verification replay result."""
    c = _resolve_client(client)
    payload: dict[str, Any] = {
        "scenario_id": scenario_id,
        "candidate_policy_version": candidate_policy_version.strip(),
        "metrics": dict(metrics or {}),
    }
    if status is not None:
        payload["status"] = status
    if candidate_episode_id is not None:
        payload["candidate_episode_id"] = candidate_episode_id
    if eval_run_id is not None:
        payload["eval_run_id"] = eval_run_id
    if error is not None:
        payload["error"] = error
    return c._http.request(
        "POST",
        "/api/verify/result",
        json=payload,
        retry_safe=True,
    )


def promote(
    *,
    baseline_episode_id: str,
    name: str | None = None,
    description: str | None = None,
    severity: str = "warning",
    client: Client | None = None,
) -> dict[str, Any]:
    """Promote a baseline episode to a verification scenario."""
    c = _resolve_client(client)
    payload: dict[str, Any] = {
        "baseline_episode_id": baseline_episode_id,
        "severity": severity,
    }
    if name is not None:
        payload["name"] = name
    if description is not None:
        payload["description"] = description
    return c._http.request(
        "POST",
        "/api/verify/promote",
        json=payload,
        retry_safe=True,
    )


def run_check(
    *,
    candidate_policy_version: str,
    policy_callable: PolicyCallable | None = None,
    dry_run: bool = False,
    client: Client | None = None,
) -> tuple[int, dict[str, Any]]:
    """Run the full verification check loop.

    Returns ``(exit_code, gate_body)``. Exit code is 0 when every
    active critical scenario passes for this candidate.
    """
    from . import evals

    version = candidate_policy_version.strip()
    gate = check_gate(candidate_policy_version=version, client=client)
    if gate.get("passed"):
        return 0, gate

    scenarios = gate.get("scenarios") or []
    to_run = [
        s
        for s in scenarios
        if s.get("severity") == "critical"
        and s.get("latest_status") != "pass"
    ]
    if not to_run:
        return (0 if gate.get("passed") else 1), gate

    if policy_callable is None:
        raise ConfigurationError(
            "Critical verification scenarios still need replay for "
            f"{version!r}. Pass `policy_callable=` or run "
            "`robotrace verify check --policy module:fn --candidate ...`."
        )

    baseline_ids = [str(s["baseline_episode_id"]) for s in to_run]
    scenario_by_baseline = {
        str(s["baseline_episode_id"]): str(s["scenario_id"]) for s in to_run
    }

    c = _resolve_client(client)
    run = evals.create_run(
        candidate_policy_version=version,
        baseline_episode_ids=baseline_ids,
        name=f"verify {version}",
        metadata={"verification_check": True},
        client=c,
    )
    results = evals.run_against(
        run,
        policy_callable=policy_callable,
        dry_run=dry_run,
    )

    if not dry_run:
        for result in results:
            scenario_id = scenario_by_baseline.get(result.baseline_episode_id)
            if not scenario_id:
                continue
            v_status = "error" if result.status == "failed" else None
            if result.status == "complete":
                success = result.metrics.get("success_candidate")
                v_status = "pass" if success is True else "fail"
            record_result(
                scenario_id=scenario_id,
                candidate_policy_version=version,
                status=v_status,
                candidate_episode_id=result.candidate_episode_id,
                eval_run_id=run.id,
                metrics=result.metrics,
                error=result.error,
                client=c,
            )
        evals.complete_run(run, client=c)

    final_gate = check_gate(candidate_policy_version=version, client=client)
    return (0 if final_gate.get("passed") else 1), final_gate


def _resolve_client(client: Client | None = None) -> Client:
    if client is not None:
        return client
    from . import _ensure_default_client

    return _ensure_default_client()


__all__ = ["check_gate", "promote", "record_result", "run_check"]
