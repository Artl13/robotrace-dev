# Changelog

All notable changes to the `robotrace-dev` Python SDK
(import name: `robotrace`).

The SDK follows [Semantic Versioning](https://semver.org/). Pre-1.0
releases (`0.x`) may make breaking changes between minor versions.
Once we cut `1.0.0`, the [`log_episode`](./README.md#log_episode-the-sacred-call)
signature is **locked** per AGENTS.md - breakages require a major
bump and at least one minor of `DeprecationWarning` first.

## [Unreleased]

## [0.3.0] - 2026-06-13

### Added

- **HDF5 adapter** (`robotrace.adapters.hdf5`) behind the new
  `[hdf5]` extra (`pip install 'robotrace-dev[hdf5]'`). Imports
  imitation-learning demonstration files - **robomimic**
  (`data/demo_*` multi-demo files) and **ALOHA / ACT**
  (one-file-per-episode, `/action` + `/observations/{qpos,images/*}`) -
  turning each trajectory into a RoboTrace episode. Four verbs mirror
  the LeRobot adapter: `scan_file`, `encode_episode`, `upload_episode`,
  and `upload_dataset` (bulk).
  - Depends on `h5py` only (~few MB) - **not** robomimic, lerobot, or
    torch. Image stacks (`(T, H, W, C)` uint8) encode to mp4 via the
    existing `[video]` extra; a sensor-only file never pays the opencv
    cost.
  - `classify_dataset(...)` is a pure function that routes dataset
    names into slots (`action*` → actions; `images/*` / `*_image` →
    video; `rewards` / `dones` / `success` → episode metadata;
    everything else → sensors).
  - fps is read from robomimic's `data.attrs["env_args"]`
    (`control_freq`) when present; otherwise pass `fps=` (ALOHA ≈ 50,
    robomimic ≈ 20). When neither declares a clock the adapter assumes
    30 and marks `fps_assumed` in the episode metadata.
  - Additive only - no change to the frozen `0.2.x` core surface.
- **LeRobot v3.0 support** in `robotrace.adapters.lerobot`. The
  adapter now reads the multi-episode-shard layout (`lerobot >= 0.3.x`)
  in addition to the v2.0 / v2.1 one-file-per-episode layout - same
  four verbs, same `video.mp4 / sensors.npz / actions.npz` output, no
  version flag (it's detected from `info.json`).
  - Reads `meta/episodes/*.parquet` into per-episode locators
    (`data/chunk_index`, `dataset_from_index` / `dataset_to_index`, and
    per-camera `videos/<key>/{chunk_index, file_index, from_timestamp,
    to_timestamp}`), surfaced on `EpisodeMeta` alongside a new
    `VideoLocator` type. `scan_dataset` stays metadata-only.
  - `encode_episode` slices the shared data parquet down to the
    episode's rows (filtering on `episode_index`) and trims each camera
    clip out of the shared mp4 by its `[from, to)` window. v3.0 video
    requires the `[video]` extra (opencv) even for a single camera -
    there's no per-episode mp4 to copy.
  - v4+ / unrecognized `codebase_version` values now raise a clear
    `ConfigurationError` instead of being mistaken for a known layout.
  - Additive only - no change to the frozen `0.2.x` core surface.

### Compatibility

- **No breaking changes.** The `0.2.x` public surface
  (`robotrace`, `.client`, `.episode`, `.errors`, `.evals`,
  `.types`, `.verify`) is byte-for-byte unchanged - the freeze
  guard (`tests/test_api_surface_freeze.py`) passes against the
  committed baseline. Upgrading from any `0.2.x` pin needs no code
  changes.
- The deprecated `Episode.upload_video` / `upload_sensors` /
  `upload_actions` shortcuts (flagged in `0.1.0a13` with `0.3.0` as
  their *earliest* possible removal) are **kept** in `0.3.0`. They
  still emit a `DeprecationWarning`; use `Episode.upload(kind, path)`.

## [0.2.1] - 2026-06-11

### Changed

- **README demo SVG** (`assets/robotrace-demo.svg`) now shows
  `Successfully installed robotrace-dev-0.2.1` instead of the stale
  `0.1.0a13` line. The PyPI/GitHub hero animation matches the stable
  release.
- **README status badge** flips from alpha to stable.

### Compatibility

- **No API or behaviour changes.** Surface is identical to `0.2.0`.
  Safe to upgrade from any `0.2.0` pin with no code changes.

## [0.2.0] - 2026-06-09

### Public commitment

`0.2.0` is the **first stable contract** for `robotrace-dev`.
From this release onward:

- The public symbol set of `robotrace`, `robotrace.client`,
  `robotrace.episode`, `robotrace.errors`, `robotrace.evals`,
  `robotrace.types`, and `robotrace.verify` follows strict semver.
  Breaking changes (removals, signature narrowings, required-param
  flips, kind changes, positional reorderings) require a **major**
  bump (`1.0.0`) and a full minor of `DeprecationWarning` first.
  The mechanical guard for this rule is
  `tests/test_api_surface_freeze.py` diffing the live surface
  against `packages/sdk-python/api-surface.json` on every CI run.
- The `0.1.0aN` alpha line is **closed**. Existing pins
  (`robotrace-dev==0.1.0a13`) keep working forever - PyPI is
  append-only - but new installs should drop the pin and run
  bare `pip install robotrace-dev`, which now picks `0.2.0` by
  default (pip stops auto-relaxing the pre-release filter the
  moment a stable release exists).
- The `Episode.upload(kind, path)` form is now the **canonical**
  upload API. The legacy `upload_video` / `upload_sensors` /
  `upload_actions` shortcuts (deprecated in `0.1.0a13`) stay
  available with a `DeprecationWarning` through `0.2.x`; their
  removal lands in `0.3.0` at the earliest.

### Removed (alpha-era warts)

- _Nothing yet._ Every alpha-era deprecation rolls forward intact:
  the `0.2.0` cut is deliberately **bit-for-bit compatible** with
  `0.1.0a13` at the wire and import level, so an existing pinned
  caller upgrading to `0.2.0` sees no behavioural change beyond
  `__version__` flipping from `"0.1.0a13"` to `"0.2.0"`.

  Scheduled for removal **in `0.3.0`** (the next minor after
  `0.2.0`), at least one minor of `DeprecationWarning` after they
  were first marked:

  - `Episode.upload_video(path)` - migrated to
    `Episode.upload("video", path)` in `0.1.0a13`.
  - `Episode.upload_sensors(path)` - migrated to
    `Episode.upload("sensors", path)` in `0.1.0a13`.
  - `Episode.upload_actions(path)` - migrated to
    `Episode.upload("actions", path)` in `0.1.0a13`.

  Their warning messages already point at the canonical
  replacement and the planned removal version, so callers running
  with `-W error::DeprecationWarning` (a sensible CI default) find
  every site automatically.

### Added

- **First stable contract.** The `0.1.0aN` alpha line is closed.
  All public symbols across `robotrace`, `robotrace.client`,
  `robotrace.episode`, `robotrace.errors`, `robotrace.evals`,
  `robotrace.types`, and `robotrace.verify` are now semver-locked.
  The mechanical guard is `tests/test_api_surface_freeze.py`
  diffing the live surface against `api-surface.json` on every CI
  run. Additive alpha work through `0.1.0a15` (including
  `failure_time_s`, typed metadata, OTel correlation, framework
  adapters, and the deprecation helper) rolls into this stable
  line unchanged at the wire and import level.

### Compatibility

- **Wire format:** unchanged from `0.1.0a13`. The dual-shape
  `/api/ingest/episode/[id]/finalize` route added in `0.1.0a13`
  stays in place - the server transparently accepts both the
  pre-`0.1.0a13` failure-fields shape (`failure_time_s` +
  `metadata.failure_reason`) and the structured
  `failure: { time_s, reason }` shape that newer SDKs may emit.
- **Imports:** unchanged. `from robotrace import log_episode,
  start_episode, Client, Episode, ...` keeps working.
- **Pip behaviour:** users running bare `pip install robotrace-dev`
  pick up `0.2.0` automatically. Users with explicit pins
  (`==0.1.0aN`) stay on their pinned alpha forever - we never
  yank an alpha from PyPI.
- **Marketing:** the portal's `SdkInstallCard` derives its
  maturity pill from `SDK_MATURITY` in
  `apps/web/lib/sdk/version.ts`, which is itself derived from the
  version regex - bumping `SDK_VERSION` to `"0.2.0"` automatically
  flips the pill from amber `ALPHA` to emerald `STABLE`. No
  manual marketing edit required for that specific pill. The
  broader "Phase 1 / Early access / invite-only" framing on the
  marketing site is a **product-policy** decision (per
  `AGENTS.md`) and survives the SDK going stable - those badges
  stay until we open self-serve signup.

### Migration

- **Nothing required.** Code that runs against `0.1.0a13` without
  emitting `DeprecationWarning`s runs unchanged against `0.2.0`.
- If you do see warnings, each one names its canonical
  replacement and removal version inline (the
  `_deprecation.warn_deprecated` message format). The mechanical
  rename for `0.1.0a13`'s `upload_*` deprecations:

  ```python
  # before
  episode.upload_video("/tmp/run.mp4")
  episode.upload_sensors("/tmp/sensors.bin")
  episode.upload_actions("/tmp/actions.parquet")

  # after
  episode.upload("video", "/tmp/run.mp4")
  episode.upload("sensors", "/tmp/sensors.bin")
  episode.upload("actions", "/tmp/actions.parquet")
  ```

## [0.1.0a13] - 2026-05-26

### Added

- **`robotrace._deprecation.warn_deprecated()` helper.** Internal
  helper (underscore-prefixed module, not exported from
  `robotrace.__init__`) that wraps `warnings.warn(DeprecationWarning,
  ...)` with the canonical message format

      <Name> is deprecated since <since> and will be removed in
      <removed_in>. Use <replacement> instead. <hint> (RoboTrace SDK)

  Stacklevel is bumped internally so the warning location points at
  the user's call site, not at the SDK wrapper. Used as part of
  the SDK 0.2.0 readiness work (gate 3 - "a real DeprecationWarning
  helper exists and has been exercised end-to-end"); see the
  `Deprecated` section below for its first real exercise.

- Server admin UI for the SDK surface freeze clock landed in the
  same window (web-side, no SDK code change): `/admin/clients`
  hosts a card showing "Day X / 14" plus the three gate conditions
  with super-admin Start / Reset. The mechanical proof on the SDK
  side is `tests/test_api_surface_freeze.py`, which diffs the live
  surface against `packages/sdk-python/api-surface.json` on every
  CI run. Additions pass silently; removals / signature narrowings
  / required-param flips / kind changes / positional reorderings
  fail loudly with a per-symbol report.

### Deprecated

- **`Episode.upload_video(path)` / `upload_sensors(path)` /
  `upload_actions(path)`** in favour of the canonical
  `Episode.upload(kind, path)`. The three shortcuts continue to
  work (and continue to delegate to `upload(kind, path)`
  internally), but each emits a `DeprecationWarning` pointing at
  the user's call site. Scheduled for removal in `0.3.0`.

  Why: the shortcuts fragment the public surface (every new
  artifact kind - point clouds, depth maps, lidar - would need
  its own wrapper, growing the API combinatorially). The
  canonical form `episode.upload("video", path)` reads
  identically (one extra character), works for any artifact kind,
  and matches the `ArtifactKind` Literal that already drives
  signed-URL routing.

  Migration: mechanical rename.

  ```python
  # before
  episode.upload_video("/tmp/run.mp4")
  episode.upload_sensors("/tmp/sensors.bin")
  episode.upload_actions("/tmp/actions.parquet")

  # after
  episode.upload("video", "/tmp/run.mp4")
  episode.upload("sensors", "/tmp/sensors.bin")
  episode.upload("actions", "/tmp/actions.parquet")
  ```

  All internal callers (gymnasium / lerobot / ros2 adapters,
  `Client.log_episode`, examples, README, top-level
  `__init__.py` docstring) migrated in this same release so the
  SDK never fires its own deprecation warnings on the user.

### Compatibility

- Pure additive change. Every existing call site keeps working
  unchanged through at least one minor version (the
  `DeprecationWarning` is a warning, not an error). Suite is now
  **115 passed, 1 skipped** (7 new tests cover the helper format,
  the stacklevel contract, the three shortcut behaviours, and
  the stdlib per-call-site dedup).

## [0.1.0a12] - 2026-05-25

### Added

- **`failure_time_s` on `Episode.finalize(...)` and `Client.log_episode(...)`**
  - a canonical frame-accurate failure-instant column on the episode
  row. When the SDK knows exactly when a run went wrong (collision
  watchdog tripped, goal-deadline missed, e-stop pressed), it can
  pin that timestamp into the database directly instead of leaving
  the portal to guess from a fuzzy end-of-run heuristic.

  ```python
  ep.finalize(
      status="failed",
      duration_s=18.4,
      failure_time_s=12.34,            # collision watchdog at 12.34 s
      metadata={"failure_reason": "wrist collision"},
  )
  ```

  - Validated **client-side** before the request leaves the
    process: `failure_time_s` must be non-negative, and the SDK
    raises `ValueError` if you combine it with `status="ready"`
    (so mis-wired error handlers fail loudly instead of silently
    pinning a `success` run).
  - Validated **server-side** too: the ingest finalize route
    clamps the value to `[0, duration_s + 0.001]` before persist,
    defending against floating-point edge cases that would
    otherwise trip the new `episodes_failure_time_window` CHECK
    constraint (migration `0021_episodes_failure_time`).
  - The portal's replay scrubber promotes `failure_time_s` to a
    thicker amber pin so SDK truth visually outranks the V1
    end-of-run heuristic marker, and the "Failed at X.Ys" stat
    card on the episode detail page reads from the same column.
  - `failure_time_s` is also the canonical input for Failure
    Intelligence V2's seek-hint chain - rules that don't know the
    timestamp themselves (joint-limit breach, sensor flatline)
    now share the same scrubber-marker plumbing.

### Compatibility

- Pure additive change. Omitting `failure_time_s` keeps the V1
  behaviour bit-for-bit identical. SDK suite green (3 new tests
  in `tests/test_request_shape.py` cover the happy path plus the
  two ValueErrors).

## [0.1.0a11] - 2026-05-20

### Changed

- **PyPI README + docs pin sweep** — the project description on PyPI
  no longer references `0.1.0a6`. Status line, adapter install pins,
  and web docs quickstart/SDK pages now point at `0.1.0a11`.
- **ConfigurationError install hints** — adapter and eval extras
  messages now call `_version.install_command(...)` so the suggested
  pin tracks `__version__` automatically.

No code or API changes vs `0.1.0a10`.

## [0.1.0a10] - 2026-05-20

### Added

- **Live `ros2.record(topics=[...])`** - subscribes to a set of ROS
  2 topics via `rclpy` during a run, writes every message into a
  temporary rosbag2 directory, and on close pipes that bag straight
  through the existing `encode_bag` + `upload_bag` plumbing.
  Same artifact contract as the offline path - the only difference
  is which side wrote the bag.

  ```python
  from robotrace.adapters import ros2

  with ros2.record(
      topics=["/camera/image_raw", "/joint_states", "/cmd_vel"],
      name="warmup pick-and-place",
      policy_version="pap-v3.2.1",
  ) as rec:
      drive_robot_for_30_seconds()
  print(rec.episode.id)
  ```

  - Context manager + explicit `start()` / `stop(status=...)` API.
    Failure inside the `with` block finalizes the episode as
    `failed` with the traceback in `metadata.failure_reason` before
    re-raising.
  - Topics validated against the live ROS 2 graph at `start()` so
    typos fail loudly instead of producing an empty bag.
  - Empty bags (no messages captured) are silently dropped - no
    upload, no orphaned tempdir.
  - Live runs stamp `metadata.ros2.mode = "live"` so the portal can
    tell live captures apart from bag-file uploads.

### Compatibility

- `rclpy` is **not** pinned in `pyproject.toml`. It ships with the
  ROS 2 distro via apt (`apt install ros-<distro>-rclpy`); the
  wheels on PyPI are not always compatible with the `rmw` bindings
  sourced from a workspace. Calling `ros2.record(...)` without a
  sourced workspace raises `ConfigurationError` pointing at the apt
  command. The offline `upload_bag(...)` path stays zero-`rclpy`.
- New `_BagWriter` is rclpy-free and testable: feed it CDR-serialized
  bytes via `rosbags.typesys` and it writes a real rosbag2 directory.
  All 7 new tests exercise the round-trip
  (`write → scan_bag → encode_bag`) without rclpy.
- SDK suite is now 112 passing.

## [0.1.0a9] - 2026-05-20

### Added

- **`robotrace.types`** - typed metadata classes for the half-dozen
  payloads that show up in every ROS 2 / LeRobot / Gymnasium
  integration. Pass them straight into `log_episode(metadata={...})`
  / `start_episode(metadata={...})` and the SDK serializes each one
  to a `__type`-tagged dict the portal renders with a per-shape
  widget (joint sparklines, pose grids, battery pills, outcome
  stats). New classes:
  - `JointState(positions, velocities=None, efforts=None, names=None)`
    - mirrors `sensor_msgs/JointState`. Parallel-array contract
      enforced (matching lengths) so a typo doesn't reach the wire.
  - `Pose3D(translation, rotation)` - Cartesian pose. Translation
    in meters, rotation as a `[x, y, z, w]` quaternion (ROS 2 /
    Eigen order - **not** `[w, x, y, z]`).
  - `Twist(linear, angular)` - linear m/s + angular rad/s.
  - `Imu(linear_acceleration, angular_velocity, orientation=None)`
    - mirrors `sensor_msgs/Imu`. Orientation optional - many IMUs
      publish only accel + gyro.
  - `Battery(percent=None, voltage_v=None, current_a=None, charging=None)`
    - all fields optional; `percent` is in `[0, 100]`.
  - `EpisodeOutcome(success=None, reward_total=None, collision_count=None, time_to_goal_s=None)`
    - episode-level outcome; mirrors the eval harness `_outcome`
      sentinel so non-replay episodes can report the same numbers.
- **`robotrace.types.encode(value)`** - the recursive encoder used
  internally by `log_episode` / `start_episode` / `Episode.finalize`.
  Exported for callers who want to round-trip a typed value through a
  non-SDK path. Plain dicts / lists / scalars pass through unchanged.

### Compatibility

- Pure superset: existing customers passing free-form `metadata=`
  dicts see zero behaviour change. The new typed classes are an
  *additive* convenience on top of the same `metadata jsonb` column.
- Forward-compat: the server validates known `__type` values but
  passes unknown ones through. A future SDK release that adds, say,
  `robotrace.Wrench` will work against today's server without a
  coordinated release.
- 21 new tests; SDK suite is now 88 passing.

## [0.1.0a8] - 2026-05-20

### Added

- **`robotrace.verify`** - promote a failed episode to a verification
  scenario and the next candidate has to replay it without re-failing
  before it can ship. Four public verbs:
  - `verify.promote(baseline_episode_id, *, name=None, severity="warning", description=None) → dict`
    - turn an episode into a scenario. Idempotent: re-promoting the
      same episode returns the existing `scenario_id` with
      `promoted=False`.
  - `verify.check_gate(*, candidate_policy_version) → dict` - read
    the current deploy gate state for a candidate. Returns
    `{"passed": bool, "critical_passed": int, "critical_failed": int,
    "critical_pending": int, "scenarios": [...], "blockers": [...]}`.
    `HTTP 422` is the blocked response; the body shape is identical
    to `200`.
  - `verify.record_result(*, scenario_id, candidate_policy_version,
    status=None, metrics=None, candidate_episode_id=None,
    eval_run_id=None, error=None) → dict` - upload one
    pass/fail/error result. When `status` is omitted, the server
    derives it from `metrics` (matches the replay-harness rollup
    conventions).
  - `verify.run_check(*, candidate_policy_version, policy_callable=None,
    dry_run=False) → tuple[int, dict]` - one call that reads the
    gate, opens a small replay run for every pending critical
    scenario (when `policy_callable` is provided), records the
    results, re-checks the gate, and returns `(exit_code, gate_body)`.
    Same customer-side runner as `robotrace.evals.run_against` -
    weights never touch RoboTrace infra.
- **CLI verb `robotrace verify check`** - the CI entry point. Pass
  `--candidate <version>` and (when needed) `--policy module:fn`;
  the command runs the replays + records the results + re-checks the
  gate. Exits `0` (pass) or `1` (blocked). Prints a compact
  pass/fail/pending summary with an OSC 8 hyperlink to the portal.
  `--dry-run` skips the upload step (still fetches baselines, still
  runs the policy). `--profile` honors `~/.robotrace/credentials`.
- Verification-aware error mapping: `ConfigurationError` when a
  critical scenario is pending and no `policy_callable` is passed;
  `NotFoundError` for missing scenarios / baselines; `AuthError` for
  cross-tenant or revoked-key responses; `ValidationError` for
  rejected result payloads.

### Notes

- The replay-regression harness (`robotrace.evals`) was already
  customer-side and weight-local; `robotrace.verify` reuses that
  runner directly. Customers running `robotrace replay run` whose
  baselines overlap their verification set don't need a second CLI
  call - the server mirrors matching results onto scenarios on
  finalize.

## [0.1.0a7] - 2026-05-19

### Added

- **`robotrace.adapters.gymnasium`** - first runtime adapter. Three
  verbs mirror ROS 2 / LeRobot: `scan_env`, `encode_rollout`,
  `upload_rollout`. Runs a Gymnasium env loop, packs observations into
  `sensors.npz`, actions into `actions.npz`, optional `video.mp4` from
  `env.render()` when `render_mode="rgb_array"`. Install via
  `pip install 'robotrace-dev[gymnasium]'`; add `[video]` for mp4
  encoding. MuJoCo and other sims work through Gymnasium env ids once
  users install their optional deps.

- **`examples/gymnasium_smoke.py`** - CartPole rollout against a local
  dev server.

- **6 new unit tests** (`tests/test_gymnasium_adapter.py`) with a
  fake rgb_array env for video tests (no pygame in CI).

## [0.1.0a6] - 2026-05-16

### Changed

- **`robotrace login`** - friendlier opener (“Welcome to RoboTrace!”),
  clearer verification URL / user-code section, spinner uses
  clear-to-EOL so wide terminals don’t keep stale padding, and optional
  ANSI styling when stdout is a TTY (`NO_COLOR` still disables).

## [0.1.0a5.post1] - 2026-05-22

### Fixed

- **README-only republish** (PEP 440 post-release; same code as
  `0.1.0a5`). PyPI project description now documents **CLI login**
  (`robotrace login`), **`robotrace whoami`** / **`logout`**, a
  **command-line interface** table, credential resolution vs env
  vars, and refreshed **`[project]`** `description` / `keywords`
  (`cli`). Pin installs with `pip install robotrace-dev==0.1.0a5.post1`
  to match this doc refresh, or keep `==0.1.0a5` for the original wheel
  - behavior is unchanged.

## [0.1.0a5] - 2026-05-21

### Fixed

- **OpenTelemetry `traceparent`** - `capture_trace_context()` now emits the
  upstream `trace_flags` byte from `SpanContext` (masked to 8 bits) instead of
  forcing the sampled bit to `01`. Aligns with W3C Trace Context so APMs that
  honor the flag stay consistent with the customer's sampler.

## [0.1.0a3.post2] - 2026-05-10

### Fixed

- **README-only republish** (PEP 440 post-release; identical wheel
  contents to `0.1.0a3.post1` functionally - only the README on
  PyPI's project page changes). The user-facing pin stays at
  `0.1.0a3`; post-releases are install-equivalent.
- README "Layout (current)" tree was missing three real modules
  that ship in the wheel: `_credentials.py` (netrc / keyring / env
  resolution), `_otel.py` (optional OpenTelemetry hook), and
  `cli.py` (the `robotrace` CLI entrypoint registered under
  `[project.scripts]`). The tree now matches `find src/robotrace
  -name '*.py'` exactly so users grepping the README before
  vendoring can trust it.

## [0.1.0a3.post1] - 2026-05-10

### Fixed

- **README-only republish** (PEP 440 post-release; same wheel
  contents as `0.1.0a3` functionally - the only delta is the
  README rendered on PyPI's project page). The user-facing pin
  in `apps/web/lib/sdk/version.ts` and across docs / portal stays
  at `0.1.0a3`; post-releases are install-equivalent.
- README quickstart pointed prospective users at the internal CMS
  flow (`/admin/episodes`, "Admin → Clients → API access"), which
  only RoboTrace staff can reach. Now points at the customer-
  facing portal: sign-in deep-link to
  `app.robotrace.dev/login?next=/portal/api-keys` (the proxy at
  `apps/web/proxy.ts` resolves `?next=` after a successful
  password / magic-link login), and run-list URL at
  `app.robotrace.dev/portal/episodes`.
- README "Layout" section claimed "ROS 2 / LeRobot adapters land
  later under `src/robotrace/adapters/`" - false since `0.1.0a1`
  (ROS 2) and `0.1.0a3` (LeRobot). Replaced with a real Adapters
  section (install matrix, one-shot examples for both adapters,
  rationale for the lean `[lerobot]` extra) and a corrected
  Layout tree showing the actual `adapters/{ros2,lerobot}/`
  packages with version stamps.

## [0.1.0a3] - 2026-05-10

### Added

- **`robotrace.adapters.lerobot`** - Hugging Face LeRobot datasets
  (format **v2.1**) → RoboTrace episodes, one trajectory per episode.
  Four public verbs:
  - `lerobot.scan_dataset(repo_or_path) → DatasetSummary` - read-only
    introspection. Pulls only `meta/*` from the Hub (info.json,
    episodes.jsonl, tasks.jsonl); never downloads a parquet shard or
    an mp4. Returns fps, episode count, frame count, camera list,
    and per-episode lengths and tasks.
  - `lerobot.encode_episode(repo, idx, output_dir) → EncodedEpisode` -
    fetches one episode's parquet + per-camera mp4s and writes
    `video.mp4` / `sensors.npz` / `actions.npz` with provenance
    metadata. No upload.
  - `lerobot.upload_episode(repo, idx, **episode_kwargs) → Episode` -
    one-shot for a single trajectory. `source="replay"` default
    matches the LeRobot context (logged trajectories replayed
    against new policies).
  - `lerobot.upload_dataset(repo, **episode_kwargs) → list[Episode]`
    - bulk: walks every trajectory (or `episode_indices=range(...)`)
    and uploads each as its own RoboTrace episode. Sequential, with
    optional `on_progress=` callback for tqdm-style reporting in
    user code.
- Column auto-classification routes `observation.images.<cam>` to
  `video`, `action[.x]` to `actions`, `next.{reward,done,success,*}` to
  episode-level metadata, and everything `observation.*` plus unknown
  columns to `sensors`. Internal LeRobot bookkeeping
  (`timestamp` / `frame_index` / `episode_index` / `index` /
  `task_index`) is filtered out.
- Multi-camera datasets get tiled horizontally into one `video.mp4`;
  pass `canonical_camera="observation.images.<key>"` to pick one
  camera instead and skip the opencv code path entirely (single-cam
  copies the source mp4 byte-for-byte).
- Episode outcome (`next.reward_sum`, `next.done`, `next.success`)
  rolls into `metadata.lerobot_episode_outcome` so the portal can
  surface "did the trajectory succeed?" without unpacking actions.npz.
- 9 new unit tests (`tests/test_lerobot_adapter.py`) with a
  programmatically-built v2.1 fixture (parquet + per-camera mp4s on
  disk) - never touches the HF Hub during CI.

### Changed

- **`[project.optional-dependencies].lerobot`** changed from the
  unbounded `["lerobot"]` to
  `["huggingface_hub>=0.20,<1", "pyarrow>=14", "numpy>=1.26"]`.
  Reason: the `lerobot` PyPI package pulls torch, torchvision,
  torchaudio, datasets, pyav, and several CUDA wheels (multi-GB
  install). Reading the v2.1 on-disk format directly with pyarrow
  + huggingface_hub keeps the install footprint loyal to the SDK's
  "lean install" rule (~20 MB, comparable to `[ros2]`).
- Marketing-side: the LeRobot integration card on the landing page
  flipped from "Soon" to "Shipped".

### Notes

- LeRobot dataset format **v3.0** (multi-episode parquet shards,
  introduced late 2025) raises a clear `ConfigurationError`
  pointing at the v2.1 revision fallback. v3.0 support is tracked
  for a follow-up release once we see real-user demand - most
  public `lerobot/*` Hub datasets are still v2.1 as of this release.
- The `[lerobot]` extra deliberately does NOT depend on the
  `lerobot` PyPI package. Set `HF_TOKEN` in your environment for
  private / gated datasets - `huggingface_hub` reads it
  automatically.

## [0.1.0a2] - 2026-05-09

### Changed

- **PyPI distribution name is `robotrace-dev`**, matching our
  `robotrace.dev` domain. The un-hyphenated `robotrace` namespace on
  PyPI was claimed in March 2026 by an unrelated robotics
  observability project, and PyPI's typo-squat protector blocks any
  single-edit-distance variant - including `robo-trace`, which we
  tried first and PyPI rejected with `400 Bad Request`.
  `robotrace-dev` clears the similarity threshold (Damerau-Levenshtein
  ≥ 3 from both `robotrace` and `robotrace-sdk`) and reads as the
  obvious match for our domain. The *import* name is still
  `robotrace` (no hyphen, no `-dev`) - same convention as
  `pip install python-dateutil` → `import dateutil`. **No code
  changes required** in your application; the install command is
  `pip install robotrace-dev`.
- Earlier `0.1.0a*` releases were never published to PyPI, so
  there's no in-the-wild upgrade path to worry about - `0.1.0a2` is
  the first published release.

### Added

- **`robotrace-dev[otel]`** extra - opt-in OpenTelemetry trace
  correlation. Pulls only `opentelemetry-api>=1.20` (~30 KB), not
  the heavy `opentelemetry-sdk`. When `start_episode` is called
  inside an active OTel span, the SDK reads the ambient context via
  `opentelemetry.trace.get_current_span()` and attaches:
  - `trace_id` (32-char lowercase hex)
  - `span_id` (16-char lowercase hex)
  - `traceparent` (W3C `00-<trace>-<span>-01` format)
  to the create-episode payload. The server stores them under
  `episodes.metadata.otel`; the portal renders a Tracing card on
  the episode detail page with copy buttons and an optional
  one-click "Open trace" deep-link via the
  `NEXT_PUBLIC_TRACE_URL_TEMPLATE` env var (Datadog, Honeycomb,
  Grafana Tempo, Jaeger).
- New `robotrace._otel` module with `capture_trace_context()`. Soft
  imports - never raises if `opentelemetry` is missing or the active
  span is invalid / unsampled.
- 7 new unit tests (`tests/test_otel.py`) covering: not-installed,
  no-active-span, active-span happy path, unsampled flag, OTel
  module misbehavior, public API exposure, and the `log_episode`
  round-trip.
- The "sacred" `log_episode` / `start_episode` signature is
  **unchanged** - OTel context is read implicitly. No new kwargs to
  learn, no opt-in flag, no deprecation warnings on existing callers.

## [0.1.0a1] - 2026-05-08

### Added

- **`robotrace.adapters.ros2`** - read rosbag2 directories (sqlite3 +
  mcap backends) and turn them into RoboTrace episodes without
  needing an `rclpy` install. Three public verbs:
  - `ros2.scan_bag(path) → BagSummary` - read-only introspection
    with topic catalog, auto-classifier decisions, and bag duration.
  - `ros2.encode_bag(path, output_dir) → EncodedBag` - writes
    `video.mp4`, `sensors.npz`, `actions.npz` and returns the file
    paths plus inferred `duration_s` / `fps`. No network.
  - `ros2.upload_bag(path, **episode_kwargs) → Episode` - one-shot
    scan + encode-to-tempdir + `start_episode` + `upload_*` +
    `finalize`. The headline call.
- Auto-classification routes `sensor_msgs/Image` /
  `CompressedImage` to `video`, `geometry_msgs/Twist*`,
  `Wrench*`, `trajectory_msgs/JointTrajectory*`,
  `control_msgs/JointJog`, and topics ending in `/cmd_*` /
  `/command` to `actions`. Everything else lands in `sensors`.
  Override per-slot via `video_topics=` / `sensor_topics=` /
  `action_topics=` (empty list = exclude the slot).
- Multi-camera bags get tiled horizontally into a single
  `video.mp4`; pass `canonical_video_topic=...` to pick one camera
  instead. Frame rate inferred from the camera with the most
  frames; falls back to 10 fps when timestamps are unusable.
- Built-in flatteners for `sensor_msgs/JointState`, `Imu`,
  `geometry_msgs/Twist[Stamped]`, `Wrench[Stamped]`, `PoseStamped`,
  `nav_msgs/Odometry`. Unknown message types fall through to a
  generic dataclass walker that pulls every numeric leaf.

### Changed

- Bumped `[project.optional-dependencies].ros2` from `[]` to
  `["rosbags>=0.11,<0.12", "numpy>=1.26"]`. Pure Python - no real
  ROS 2 install required to ingest bags.
- Image-topic encoding lives behind the existing `[video]` extra
  (opencv-python). Install combo for camera bags is
  `pip install 'robotrace-dev[ros2,video]'`; sensor-only bags can stick
  with `[ros2]` and skip opencv entirely.

## [0.1.0a0] - 2026-05-02

First public alpha. Contract under iteration.

### Added

- `robotrace.init(...)`, `robotrace.start_episode(...)`,
  `robotrace.log_episode(...)` top-level convenience.
- `robotrace.Client(...)` for explicit, multi-deployment use.
- `robotrace.Episode` handle with `upload_video` / `upload_sensors` /
  `upload_actions` / `finalize` and context-manager auto-finalize
  (clean exit → `ready`, exception → `failed`).
- Streaming PUT to Cloudflare R2 via signed URLs (no body buffering).
- Typed exception hierarchy (`RobotraceError` → `AuthError`,
  `NotFoundError`, `ConflictError`, `ValidationError`, `ServerError`,
  `TransportError`, `ConfigurationError`).
- `pytest`-based smoke tests pinning the wire format to
  `/api/ingest/episode` and `/api/ingest/episode/{id}/finalize`.
