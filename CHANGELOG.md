# Changelog

All notable changes to the `robotrace-dev` Python SDK
(import name: `robotrace`).

The SDK follows [Semantic Versioning](https://semver.org/). Pre-1.0
releases (`0.x`) may make breaking changes between minor versions.
Once we cut `1.0.0`, the [`log_episode`](./README.md#log_episode-the-sacred-call)
signature is **locked** per AGENTS.md — breakages require a major
bump and at least one minor of `DeprecationWarning` first.

## [Unreleased]

## [0.1.0a3] — 2026-05-10

### Added

- **`robotrace.adapters.lerobot`** — Hugging Face LeRobot datasets
  (format **v2.1**) → RoboTrace episodes, one trajectory per episode.
  Four public verbs:
  - `lerobot.scan_dataset(repo_or_path) → DatasetSummary` — read-only
    introspection. Pulls only `meta/*` from the Hub (info.json,
    episodes.jsonl, tasks.jsonl); never downloads a parquet shard or
    an mp4. Returns fps, episode count, frame count, camera list,
    and per-episode lengths and tasks.
  - `lerobot.encode_episode(repo, idx, output_dir) → EncodedEpisode` —
    fetches one episode's parquet + per-camera mp4s and writes
    `video.mp4` / `sensors.npz` / `actions.npz` with provenance
    metadata. No upload.
  - `lerobot.upload_episode(repo, idx, **episode_kwargs) → Episode` —
    one-shot for a single trajectory. `source="replay"` default
    matches the LeRobot context (logged trajectories replayed
    against new policies).
  - `lerobot.upload_dataset(repo, **episode_kwargs) → list[Episode]`
    — bulk: walks every trajectory (or `episode_indices=range(...)`)
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
  disk) — never touches the HF Hub during CI.

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
  for a follow-up release once we see real-user demand — most
  public `lerobot/*` Hub datasets are still v2.1 as of this release.
- The `[lerobot]` extra deliberately does NOT depend on the
  `lerobot` PyPI package. Set `HF_TOKEN` in your environment for
  private / gated datasets — `huggingface_hub` reads it
  automatically.

## [0.1.0a2] — 2026-05-09

### Changed

- **PyPI distribution name is `robotrace-dev`**, matching our
  `robotrace.dev` domain. The un-hyphenated `robotrace` namespace on
  PyPI was claimed in March 2026 by an unrelated robotics
  observability project, and PyPI's typo-squat protector blocks any
  single-edit-distance variant — including `robo-trace`, which we
  tried first and PyPI rejected with `400 Bad Request`.
  `robotrace-dev` clears the similarity threshold (Damerau-Levenshtein
  ≥ 3 from both `robotrace` and `robotrace-sdk`) and reads as the
  obvious match for our domain. The *import* name is still
  `robotrace` (no hyphen, no `-dev`) — same convention as
  `pip install python-dateutil` → `import dateutil`. **No code
  changes required** in your application; the install command is
  `pip install robotrace-dev`.
- Earlier `0.1.0a*` releases were never published to PyPI, so
  there's no in-the-wild upgrade path to worry about — `0.1.0a2` is
  the first published release.

### Added

- **`robotrace-dev[otel]`** extra — opt-in OpenTelemetry trace
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
  imports — never raises if `opentelemetry` is missing or the active
  span is invalid / unsampled.
- 7 new unit tests (`tests/test_otel.py`) covering: not-installed,
  no-active-span, active-span happy path, unsampled flag, OTel
  module misbehavior, public API exposure, and the `log_episode`
  round-trip.
- The "sacred" `log_episode` / `start_episode` signature is
  **unchanged** — OTel context is read implicitly. No new kwargs to
  learn, no opt-in flag, no deprecation warnings on existing callers.

## [0.1.0a1] — 2026-05-08

### Added

- **`robotrace.adapters.ros2`** — read rosbag2 directories (sqlite3 +
  mcap backends) and turn them into RoboTrace episodes without
  needing an `rclpy` install. Three public verbs:
  - `ros2.scan_bag(path) → BagSummary` — read-only introspection
    with topic catalog, auto-classifier decisions, and bag duration.
  - `ros2.encode_bag(path, output_dir) → EncodedBag` — writes
    `video.mp4`, `sensors.npz`, `actions.npz` and returns the file
    paths plus inferred `duration_s` / `fps`. No network.
  - `ros2.upload_bag(path, **episode_kwargs) → Episode` — one-shot
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
  `["rosbags>=0.11,<0.12", "numpy>=1.26"]`. Pure Python — no real
  ROS 2 install required to ingest bags.
- Image-topic encoding lives behind the existing `[video]` extra
  (opencv-python). Install combo for camera bags is
  `pip install 'robotrace-dev[ros2,video]'`; sensor-only bags can stick
  with `[ros2]` and skip opencv entirely.

## [0.1.0a0] — 2026-05-02

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
