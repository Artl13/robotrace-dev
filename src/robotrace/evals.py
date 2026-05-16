"""Replay regression harness — customer-side runner.

Lets a robotics team re-roll a candidate policy against historical
RoboTrace episodes without booking real-robot time. Three verbs:

    rt.evals.create_run(
        candidate_policy_version="pap-v13",
        baseline_episode_ids=[...],
        name="nightly v13 sweep",
    )
    rt.evals.run_against(run, policy_callable=my_policy)
    rt.evals.complete_run(run)

What `run_against` actually does, per baseline episode:

  1. Fetches the baseline episode's `actions.npz` + `sensors.npz`
     via the artifact resolver route (signed R2 GET URL, same auth
     guard as the portal).
  2. Iterates per-step observations through the customer's
     ``policy_callable``, producing candidate actions.
  3. Computes per-episode metrics (success/reward/L2/OOD) and uploads
     them via ``POST /api/ingest/eval-run/<id>/result`` after
     `log_episode(source="replay", policy_version=run.candidate_policy_version)`
     mints a new candidate episode that the runner *links* to the
     baseline via ``metadata={"eval_run_id", "baseline_episode_id"}``.

The runner is intentionally **synchronous** in V0. Robotics teams
already run multi-hour sweeps; making the loop async would only buy
us complexity without changing the wall clock. A future hosted
runner (V1) can implement the same contract on the server side.

Per AGENTS.md the customer's weights never touch our infrastructure
— ``policy_callable`` is invoked locally and only the per-step
metric blob is uploaded.
"""

from __future__ import annotations

import io
import math
import sys
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .errors import APIError, ConfigurationError, NotFoundError

if TYPE_CHECKING:

    from .client import Client

# ── public types ─────────────────────────────────────────────────────


# Per-step observation handed to the policy callable. Mirrors the
# namespaced NPZ layout the ROS 2 and LeRobot adapters emit:
# `{ "/joint_states/position": ndarray, "/joint_states/_t_ns": int, ... }`.
# We pass it through as a dict so the customer's code can index by
# whatever topic / column name they recorded with — no implicit
# reshape, no opaque tensor that hides which sensor is which.
Observation = dict[str, Any]

# Per-step action returned by the policy callable. Same dict shape
# as `Observation` — keyed by the original action namespace (e.g.
# `/cmd_vel/linear`, `action/value`) so we can L2-diff against the
# baseline action at the same key.
Action = dict[str, Any]

# Signature the customer's policy must implement. Sync because robotics
# inference is sync at the call site; the runner is itself sync.
PolicyCallable = Callable[[Observation], Action]


@dataclass
class EvalRun:
    """Handle to an open eval-run campaign.

    Returned by :func:`create_run`. Hold onto it across
    :func:`run_against` (per-episode loop) and :func:`complete_run`
    (rollup). The runner state we need to track between calls fits
    in this dataclass — no global registry.
    """

    id: str
    candidate_policy_version: str
    baseline_policy_version: str | None
    baseline_episode_ids: list[str]
    name: str | None
    runner_kind: str = "client"
    status: str = "pending"
    episodes_completed: int = 0
    episodes_failed: int = 0

    # Internal — populated at create time; not part of the user surface.
    _client: Client | None = field(default=None, repr=False, compare=False)


@dataclass
class EvalResult:
    """Per-episode result returned by :func:`run_against` (one per
    baseline episode walked). Useful for callers that want to react
    in-line (e.g. fail-fast a CI run when success drops below a
    threshold) without re-fetching from the API.
    """

    baseline_episode_id: str
    candidate_episode_id: str | None
    status: str  # "complete" | "failed"
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


# ── public verbs ─────────────────────────────────────────────────────


def create_run(
    *,
    candidate_policy_version: str,
    baseline_episode_ids: Sequence[str],
    baseline_policy_version: str | None = None,
    name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    client: Client | None = None,
) -> EvalRun:
    """Open a new eval-run campaign on the server.

    Mirrors :meth:`Client.start_episode` ergonomically — all kwargs,
    explicit `client=` for tests, falls back to the module-level
    default client (constructed lazily from env / credentials).
    """
    if not candidate_policy_version or not candidate_policy_version.strip():
        raise ConfigurationError(
            "`candidate_policy_version` is required and cannot be empty."
        )
    if not baseline_episode_ids:
        raise ConfigurationError(
            "Pass at least one baseline episode id to evaluate against."
        )
    ids = [str(x).strip() for x in baseline_episode_ids]
    for bid in ids:
        if not bid:
            raise ConfigurationError(
                "`baseline_episode_ids` contains an empty string."
            )

    c = _resolve_client(client)
    payload: dict[str, Any] = {
        "candidate_policy_version": candidate_policy_version,
        "baseline_episode_ids": ids,
    }
    if baseline_policy_version is not None:
        payload["baseline_policy_version"] = baseline_policy_version
    if name is not None:
        payload["name"] = name
    if metadata is not None:
        payload["metadata"] = dict(metadata)

    # `retry_safe=True`: a 429 from the create route means the run
    # row was never inserted, so the auto-retry path is free of
    # duplicate-row risk.
    body = c._http.request(
        "POST",
        "/api/ingest/eval-run",
        json=payload,
        retry_safe=True,
    )

    run = EvalRun(
        id=str(body["eval_run_id"]),
        candidate_policy_version=candidate_policy_version,
        baseline_policy_version=baseline_policy_version,
        baseline_episode_ids=ids,
        name=name,
        status=str(body.get("status", "pending")),
    )
    run._client = c
    return run


def run_against(
    run: EvalRun,
    *,
    policy_callable: PolicyCallable,
    on_episode: Callable[[EvalResult], None] | None = None,
    dry_run: bool = False,
) -> list[EvalResult]:
    """Walk every baseline episode and replay it through the candidate.

    For each baseline episode:

      • download `actions.npz` + `sensors.npz` (best-effort — a
        baseline with no NPZ artifacts still produces a metric-only
        row),
      • iterate per-step observations into `policy_callable`,
      • compute the diff metrics,
      • upload a new candidate episode via :meth:`Client.log_episode`
        (``source="replay"``, ``metadata={"eval_run_id": ...,
        "baseline_episode_id": ...}``) so the portal can drill from
        the eval result back to the replay run,
      • POST `/api/ingest/eval-run/<id>/result` with the metric blob.

    Pass ``dry_run=True`` to skip the per-episode `log_episode` and
    `/result` upload — useful for offline development of the policy
    callable when you just want the printed metrics. The dry-run
    path still talks to the artifact resolver, so it requires a
    valid API key.

    Returns the list of :class:`EvalResult` objects in baseline-id
    order. Errors inside ``policy_callable`` are caught per-episode
    and recorded as ``status="failed"`` rows — the loop continues
    so one bad observation doesn't sink the whole campaign.
    """
    c = run._client or _resolve_client()
    np = _import_numpy()

    results: list[EvalResult] = []
    for baseline_id in run.baseline_episode_ids:
        try:
            metrics, candidate_episode_id = _replay_one(
                client=c,
                run=run,
                baseline_episode_id=baseline_id,
                policy_callable=policy_callable,
                dry_run=dry_run,
                np=np,
            )
            result = EvalResult(
                baseline_episode_id=baseline_id,
                candidate_episode_id=candidate_episode_id,
                status="complete",
                metrics=metrics,
            )
        except Exception as exc:
            # Print the traceback to the customer's terminal so they
            # can debug locally — we only upload a truncated string
            # form. Mirrors the runner-side guidance in the plan's
            # "Risks worth naming" → "we don't see failures either".
            traceback.print_exc(file=sys.stderr)
            result = EvalResult(
                baseline_episode_id=baseline_id,
                candidate_episode_id=None,
                status="failed",
                metrics={},
                error=f"{type(exc).__name__}: {exc}",
            )

        if not dry_run:
            _push_result(c, run, result)

        if result.status == "complete":
            run.episodes_completed += 1
        else:
            run.episodes_failed += 1

        if on_episode is not None:
            try:
                on_episode(result)
            except Exception:
                # Never let a customer callback take down the loop.
                traceback.print_exc(file=sys.stderr)
        results.append(result)

    return results


def complete_run(
    run: EvalRun,
    *,
    status: str = "completed",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Close out the campaign and trigger the server-side rollup.

    Returns the parsed JSON response which carries the same
    ``summary`` shape the portal `EvalDiffCard` renders. CI scripts
    can read the rollup directly:

        out = rt.evals.complete_run(run)
        success_delta = out["summary"]["success_rate"]["delta"]
    """
    c = run._client or _resolve_client()
    payload: dict[str, Any] = {"status": status}
    if metadata is not None:
        payload["metadata"] = dict(metadata)
    body = c._http.request(
        "POST",
        f"/api/ingest/eval-run/{run.id}/finalize",
        json=payload,
    )
    run.status = str(body.get("status", status))
    return body


# ── internal: per-episode replay ─────────────────────────────────────


def _replay_one(
    *,
    client: Client,
    run: EvalRun,
    baseline_episode_id: str,
    policy_callable: PolicyCallable,
    dry_run: bool,
    np: Any,
) -> tuple[dict[str, Any], str | None]:
    """Replay one baseline episode against the candidate policy.

    Returns (metrics_dict, candidate_episode_id). `candidate_episode_id`
    is None for ``dry_run=True``.

    Failure modes inside this function propagate to the caller so
    :func:`run_against` can mark the result row as failed.
    """
    baseline_actions = _fetch_npz(
        client, baseline_episode_id, "actions", np=np, optional=True
    )
    baseline_sensors = _fetch_npz(
        client, baseline_episode_id, "sensors", np=np, optional=True
    )

    # Read baseline reward / success / collisions from the episode
    # metadata. Robotics teams usually stamp `metadata.outcome` at
    # finalize time; LeRobot also stuffs `next.reward_sum` into
    # `metadata.lerobot_episode_outcome`. Be lenient about where we
    # look — the customer's recording habit is the source of truth.
    baseline_meta = _fetch_episode_metadata(client, baseline_episode_id)
    baseline_outcome = _extract_outcome(baseline_meta)

    # Walk per-step observations. We zip the namespaced arrays back
    # together by index — assumes the runner has the same step count
    # across topics, which the encoders enforce (`shape changed mid-bag`
    # is a hard skip on the ingest side, see adapters/_encode.py).
    observations = _materialize_observations(baseline_sensors, np=np)
    n_steps = len(observations)

    candidate_actions: list[Action] = []
    for obs in observations:
        action = policy_callable(obs)
        if not isinstance(action, Mapping):
            raise TypeError(
                "policy_callable must return a Mapping[str, ndarray]; "
                f"got {type(action).__name__}"
            )
        candidate_actions.append(dict(action))

    # Per-step diff metrics. L2 distance averaged over steps;
    # OOD share computed against the baseline action distribution.
    action_l2 = _action_l2_distance(baseline_actions, candidate_actions, np=np)
    ood_share = _ood_action_share(baseline_actions, candidate_actions, np=np)

    # Candidate reward / success / collisions: V0 mirrors the baseline
    # values unless the customer's `policy_callable` returns them
    # explicitly via a sentinel key. We document the sentinels in the
    # docs page; ergonomic for the 80% case where success is computed
    # from the trajectory not from per-step action.
    cand_outcome = _extract_outcome_from_actions(candidate_actions)

    metrics: dict[str, Any] = {
        "success_baseline": baseline_outcome.get("success"),
        "success_candidate": cand_outcome.get("success", baseline_outcome.get("success")),
        "reward_total_baseline": baseline_outcome.get("reward_total"),
        "reward_total_candidate": cand_outcome.get(
            "reward_total", baseline_outcome.get("reward_total")
        ),
        "collision_count_baseline": baseline_outcome.get("collision_count"),
        "collision_count_candidate": cand_outcome.get(
            "collision_count", baseline_outcome.get("collision_count")
        ),
        "time_to_goal_s_baseline": baseline_outcome.get("time_to_goal_s"),
        "time_to_goal_s_candidate": cand_outcome.get(
            "time_to_goal_s", baseline_outcome.get("time_to_goal_s")
        ),
        "action_l2_distance": action_l2,
        "ood_action_share": ood_share,
        "step_count": n_steps,
    }

    if dry_run:
        return metrics, None

    # Mint the candidate episode so the portal can drill from the
    # eval result back to its replay. Metadata-only — we don't try to
    # re-upload a video for the candidate (no rendering on the
    # customer side in V0). The portal episode detail page renders a
    # "Part of eval run" pill via the `metadata.eval_run_id` link.
    cand_episode = client.log_episode(
        name=(run.name or f"replay {run.candidate_policy_version}")
        + f" · {baseline_episode_id[:8]}",
        source="replay",
        policy_version=run.candidate_policy_version,
        metadata={
            "eval_run_id": run.id,
            "baseline_episode_id": baseline_episode_id,
            "metrics": metrics,
        },
    )
    return metrics, cand_episode.id


# ── internal: artifact + metadata fetch ──────────────────────────────


def _fetch_npz(
    client: Client,
    episode_id: str,
    kind: str,
    *,
    np: Any,
    optional: bool,
) -> dict[str, Any] | None:
    """GET a baseline NPZ artifact via the artifact resolver route.

    The route 302s to a short-lived signed R2 URL — `httpx` follows
    redirects by default, so we get the bytes in one round-trip.
    Returns the parsed dict-of-arrays, or None when the episode has
    no artifact in this slot and ``optional=True``.
    """
    import httpx

    url = f"/api/episodes/{episode_id}/artifact/{kind}"
    # Reuse the client's authenticated httpx — same Bearer header.
    # We can't use `client._http.request()` because that expects
    # JSON; this endpoint returns binary.
    try:
        # `httpx.Client` from HTTPClient already has the Authorization
        # header configured. We use it directly with stream=False so
        # the redirect to R2 + the byte download happens server-side.
        response = client._http._client.get(url, follow_redirects=True)
    except httpx.HTTPError as exc:
        raise APIError(
            f"transport error fetching baseline {kind} for {episode_id}: {exc}",
            status_code=0,
        ) from exc

    if response.status_code == 404:
        if optional:
            return None
        raise NotFoundError(
            f"baseline episode {episode_id} has no {kind} artifact",
            status_code=404,
        )
    if response.status_code >= 400:
        raise APIError(
            f"failed to fetch baseline {kind} for {episode_id}",
            status_code=response.status_code,
            response_body=response.text[:500],
        )

    # `allow_pickle=True` on purpose: this NPZ was uploaded by the
    # user's own SDK to their own R2 bucket, and we just downloaded
    # it via a signed URL that's gated on their API key. The trust
    # boundary is "bytes I uploaded vs bytes I'm loading" — same as
    # `import my_module`, not "arbitrary network bytes". Both the
    # ROS 2 and LeRobot encoders use `np.savez(...)` which
    # transparently falls back to pickle for any column numpy can't
    # store as a homogeneous tensor (variable-length strings, ragged
    # arrays, structured records, …). Loading with the default
    # `allow_pickle=False` rejects those files with the cryptic
    # error the user hit in 0.1.0a4, so we opt in here.
    with np.load(io.BytesIO(response.content), allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _fetch_episode_metadata(
    client: Client, episode_id: str
) -> dict[str, Any]:
    """Best-effort metadata pull for the baseline.

    The /api/episodes route doesn't exist for SDK callers yet; we
    fall back to an empty dict so the runner stays usable today.
    When the read-side endpoint lands the implementation flips here
    without changing the rest of the runner.
    """
    try:
        body = client._http.request("GET", f"/api/episodes/{episode_id}")
        meta = body.get("metadata")
        if isinstance(meta, dict):
            return meta
    except (NotFoundError, APIError):
        # 404 = the endpoint isn't shipped yet (Phase 1 has only
        # /artifact/<kind>). Carry on with empty metadata.
        return {}
    return {}


def _extract_outcome(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Pull the canonical outcome signals out of episode metadata.

    Handles the three conventions we've seen in the wild so far:
      • SDK-stamped: `{"outcome": {"success": True, "reward_total": ...}}`
      • LeRobot adapter: `{"lerobot_episode_outcome": {...}}`
      • Flat top-level: `{"success": True}`
    """
    out: dict[str, Any] = {}
    sources: list[Mapping[str, Any]] = [metadata]
    nested = metadata.get("outcome")
    if isinstance(nested, Mapping):
        sources.append(nested)
    lerobot = metadata.get("lerobot_episode_outcome")
    if isinstance(lerobot, Mapping):
        sources.append(lerobot)

    for src in sources:
        if "success" not in out:
            for key in ("success", "next.success"):
                if key in src and isinstance(src[key], bool):
                    out["success"] = src[key]
                    break
        if "reward_total" not in out:
            for key in ("reward_total", "next.reward_sum", "reward"):
                v = src.get(key)
                if isinstance(v, (int, float)):
                    out["reward_total"] = float(v)
                    break
        if "collision_count" not in out:
            v = src.get("collision_count")
            if isinstance(v, (int, float)):
                out["collision_count"] = float(v)
        if "time_to_goal_s" not in out:
            v = src.get("time_to_goal_s")
            if isinstance(v, (int, float)):
                out["time_to_goal_s"] = float(v)
    return out


def _extract_outcome_from_actions(actions: list[Action]) -> dict[str, Any]:
    """Look for the runner-side outcome sentinel keys.

    Customers who can compute success at policy time return a
    sentinel dict in the *last* action's value. V0 supports:

        return {"_outcome": {"success": True, "reward_total": 12.4}}

    Keeps `policy_callable` strictly per-step at the type level while
    giving an opt-in escape hatch for "I already know the answer".
    """
    if not actions:
        return {}
    last = actions[-1]
    sentinel = last.get("_outcome")
    if isinstance(sentinel, Mapping):
        out: dict[str, Any] = {}
        for key in ("success", "reward_total", "collision_count", "time_to_goal_s"):
            if key in sentinel:
                out[key] = sentinel[key]
        return out
    return {}


# ── internal: metric math ────────────────────────────────────────────


def _materialize_observations(
    sensors: dict[str, Any] | None, *, np: Any
) -> list[Observation]:
    """Turn a baseline `sensors.npz` dict-of-arrays into a list of
    per-step :class:`Observation` dicts.

    The NPZ shape is `{ "<topic>/<field>": ndarray[N, ...], "<topic>/_t_ns": int64[N] }`.
    We walk by the first topic's `_t_ns` length and pick row `i` from
    every field — same convention as the ROS 2 + LeRobot encoders.
    Topics with different lengths are truncated to the min.
    """
    if not sensors:
        return []
    # Group by namespace (everything before the last `/`) and find
    # the minimum step count.
    namespaces: dict[str, dict[str, Any]] = {}
    for key, arr in sensors.items():
        if "/" not in key:
            namespaces.setdefault("", {})[key] = arr
            continue
        ns, field_name = key.rsplit("/", 1)
        namespaces.setdefault(ns, {})[field_name] = arr

    lengths: list[int] = []
    for ns_fields in namespaces.values():
        for arr in ns_fields.values():
            if hasattr(arr, "__len__"):
                lengths.append(len(arr))
    if not lengths:
        return []
    n_steps = min(lengths)

    obs_list: list[Observation] = []
    for i in range(n_steps):
        obs: Observation = {}
        for ns, fields in namespaces.items():
            for field_name, arr in fields.items():
                key = f"{ns}/{field_name}" if ns else field_name
                obs[key] = np.asarray(arr[i])
        obs_list.append(obs)
    return obs_list


def _is_numeric_array(arr: Any, *, np: Any) -> bool:
    """Return True iff `arr` has (or coerces to) a numeric numpy dtype.

    The metric helpers below assume they can ``np.asarray(arr,
    dtype=float64)`` every column they iterate. NPZ artifacts in
    the wild ship plenty of non-numeric columns — variable-length
    strings the LeRobot adapter writes for free-text metadata,
    ragged label arrays, pickled object blobs — and forcing a
    float coercion on those raises ``ValueError`` and crashes the
    whole replay. We skip them here so the metric falls back to
    "the numeric subset of the column set," which is exactly what a
    user reading the DiffCard wants.
    """
    try:
        a = np.asarray(arr)
    except Exception:
        return False
    return bool(np.issubdtype(a.dtype, np.number))


def _action_l2_distance(
    baseline_actions: dict[str, Any] | None,
    candidate_actions: list[Action],
    *,
    np: Any,
) -> float | None:
    """Mean per-step L2 distance between baseline and candidate.

    Sums the squared distance across every action key the two have
    in common (e.g. `/cmd_vel/linear`, `/cmd_vel/angular`), takes a
    sqrt, then averages over steps. Returns None when we can't pair
    up any keys (different recording conventions, etc.) — the metric
    column on the result row becomes null and the rollup ignores it.
    """
    if not baseline_actions or not candidate_actions:
        return None
    namespaces: dict[str, dict[str, Any]] = {}
    for key, arr in baseline_actions.items():
        if key.endswith("/_t_ns") or "/" not in key:
            continue
        if not _is_numeric_array(arr, np=np):
            continue
        ns, field_name = key.rsplit("/", 1)
        namespaces.setdefault(ns, {})[field_name] = arr
    if not namespaces:
        return None

    # Build per-step baseline vector by concatenating every field
    # under every action namespace, then compare to the candidate at
    # the matching key.
    n_steps = min(
        min(len(arr) for arr in fields.values())
        for fields in namespaces.values()
    )
    n_steps = min(n_steps, len(candidate_actions))
    if n_steps == 0:
        return None

    per_step: list[float] = []
    for i in range(n_steps):
        cand = candidate_actions[i]
        sq_sum = 0.0
        seen_keys = 0
        for ns, fields in namespaces.items():
            for field_name, baseline_arr in fields.items():
                full_key = f"{ns}/{field_name}" if ns else field_name
                cand_v = cand.get(full_key, cand.get(ns))
                if cand_v is None:
                    continue
                base_v = np.asarray(baseline_arr[i], dtype=np.float64).ravel()
                cand_arr = np.asarray(cand_v, dtype=np.float64).ravel()
                if base_v.shape != cand_arr.shape:
                    continue
                diff = base_v - cand_arr
                sq_sum += float(np.dot(diff, diff))
                seen_keys += 1
        if seen_keys > 0:
            per_step.append(math.sqrt(sq_sum))
    if not per_step:
        return None
    return float(sum(per_step) / len(per_step))


def _ood_action_share(
    baseline_actions: dict[str, Any] | None,
    candidate_actions: list[Action],
    *,
    np: Any,
) -> float | None:
    """Share of candidate steps that lie outside the baseline action
    distribution (per-step z-score > 3 against the per-key baseline
    mean+std). Returns None when the baseline is too short to
    compute meaningful stats.
    """
    if not baseline_actions or not candidate_actions:
        return None
    namespaces: dict[str, dict[str, Any]] = {}
    for key, arr in baseline_actions.items():
        if key.endswith("/_t_ns") or "/" not in key:
            continue
        if not _is_numeric_array(arr, np=np):
            continue
        ns, field_name = key.rsplit("/", 1)
        namespaces.setdefault(ns, {})[field_name] = arr

    stats: dict[str, tuple[Any, Any]] = {}
    for ns, fields in namespaces.items():
        for field_name, arr in fields.items():
            full_key = f"{ns}/{field_name}" if ns else field_name
            a = np.asarray(arr, dtype=np.float64)
            if a.ndim < 1 or a.shape[0] < 5:
                continue
            mean = a.mean(axis=0)
            std = a.std(axis=0)
            # Replace zero stds with 1 to avoid div-by-zero; a
            # constant baseline channel is "in-distribution" by
            # definition, so any candidate value matching the mean
            # is z=0 and any deviation is huge — that's the right
            # behavior.
            std = np.where(std < 1e-9, 1.0, std)
            stats[full_key] = (mean, std)

    if not stats:
        return None

    ood = 0
    counted = 0
    for cand in candidate_actions:
        max_z = 0.0
        has_keyed_value = False
        for full_key, (mean, std) in stats.items():
            v = cand.get(full_key)
            if v is None:
                continue
            has_keyed_value = True
            arr_v = np.asarray(v, dtype=np.float64).ravel()
            mean_v = np.asarray(mean, dtype=np.float64).ravel()
            std_v = np.asarray(std, dtype=np.float64).ravel()
            if arr_v.shape != mean_v.shape:
                continue
            z = np.abs((arr_v - mean_v) / std_v)
            max_z = max(max_z, float(z.max()))
        if not has_keyed_value:
            continue
        counted += 1
        if max_z > 3.0:
            ood += 1
    if counted == 0:
        return None
    return ood / counted


# ── internal: result push ────────────────────────────────────────────


def _push_result(client: Client, run: EvalRun, result: EvalResult) -> None:
    """POST one per-episode result to the ingest route."""
    payload: dict[str, Any] = {
        "baseline_episode_id": result.baseline_episode_id,
        "status": result.status,
        "metrics": result.metrics,
    }
    if result.candidate_episode_id is not None:
        payload["candidate_episode_id"] = result.candidate_episode_id
    if result.error is not None:
        payload["error"] = result.error
    # `retry_safe=True`: the server upserts on (eval_run_id,
    # baseline_episode_id), so a 429-driven retry just lands the
    # same row again.
    client._http.request(
        "POST",
        f"/api/ingest/eval-run/{run.id}/result",
        json=payload,
        retry_safe=True,
    )


# ── internal: client + numpy resolution ──────────────────────────────


def _resolve_client(client: Client | None = None) -> Client:
    if client is not None:
        return client
    # Late import dodges circular ref between client.py and evals.py.
    from . import _ensure_default_client

    return _ensure_default_client()


def _import_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise ConfigurationError(
            "the replay regression harness needs `numpy`. "
            "Install with `pip install 'robotrace-dev[numpy]==0.1.0a4'`."
        ) from exc
    return np


__all__ = [
    "Action",
    "EvalResult",
    "EvalRun",
    "Observation",
    "PolicyCallable",
    "complete_run",
    "create_run",
    "run_against",
]


# Re-exports of helpers used by the test suite — leading-underscore
# names are still importable, but the tests `from .evals import _foo`
# spelling is brittle to ruff re-orderings. Expose stable handles.
metric_action_l2_distance: Callable[..., Any] = _action_l2_distance
metric_ood_action_share: Callable[..., Any] = _ood_action_share
materialize_observations: Callable[..., Any] = _materialize_observations
extract_outcome: Callable[..., Any] = _extract_outcome


def iter_baseline_steps(
    baseline_sensors: dict[str, Any] | None,
) -> Iterable[Observation]:
    """Convenience iterator — public alias for `_materialize_observations`."""
    return iter(_materialize_observations(baseline_sensors, np=_import_numpy()))
