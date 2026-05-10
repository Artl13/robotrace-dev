# robotrace-dev (Python SDK)

> The official Python SDK for [RoboTrace](https://robotrace.dev) —
> observability and evals for AI-powered robots.

```bash
pip install robotrace-dev==0.1.0a3
```

> **Why the pin?** Pinning is the most reliable install during alpha.
> The pin goes away once we cut `1.0` — `pip install robotrace-dev`
> will be enough then.

> **Distribution name vs. import name.** PyPI distributes us as
> `robotrace-dev` (matching our `robotrace.dev` domain). The
> un-hyphenated `robotrace` PyPI namespace is held by an unrelated
> robotics project, and PyPI's typo-squat protector blocks any
> single-edit-distance variant (so `robo-trace` was rejected too).
> The *import* name stays `import robotrace` — same pattern as
> `pip install python-dateutil` → `import dateutil`.

> **Status:** alpha (`0.1.0a3`). The public API in this README is the
> shape we're iterating against; once we cut `1.0.0`, the
> [`log_episode`](#log_episode-the-sacred-call) signature is locked
> and breakages require a major bump (per `AGENTS.md` in the
> RoboTrace monorepo).

## Quickstart

Mint an API key in your RoboTrace admin console
(**Admin → Clients → \<client\> → API access**), then:

```python
import robotrace as rt

rt.init(
    api_key="rt_…",
    base_url="https://app.robotrace.dev",   # or http://localhost:3000 in dev
)

rt.log_episode(
    name="pick_and_place v3 morning warmup",
    source="real",
    robot="halcyon-bimanual-01",
    policy_version="pap-v3.2.1",
    env_version="halcyon-cell-rev4",
    git_sha="abc1234",
    seed=8124,
    video="/tmp/run.mp4",
    sensors="/tmp/sensors.bin",
    actions="/tmp/actions.parquet",
    duration_s=47.2,
    fps=30,
    metadata={"task": "pick_and_place", "scene": "tabletop"},
)
```

The episode appears in `/admin/episodes` immediately, with the four
reproducibility fields (policy / env / git / seed) front-and-center
on the detail page.

### From environment variables

Same call without hardcoding the key:

```bash
export ROBOTRACE_API_KEY=rt_…
export ROBOTRACE_BASE_URL=https://app.robotrace.dev
```

```python
import robotrace as rt

# init() is optional when both env vars are set — the default
# client is constructed lazily on first use.
rt.log_episode(
    name="…",
    policy_version="…",
    video="/tmp/run.mp4",
)
```

## API

### `log_episode` — the sacred call

The one-shot entrypoint. Equivalent to `start_episode` → upload all
artifacts → `finalize`. Use this for the 95% case of "I have files
on disk, log them and move on."

```python
rt.log_episode(
    *,
    # Identification
    name: str | None = None,
    source: Literal["real", "sim", "replay"] = "real",
    robot: str | None = None,

    # Reproducibility — load-bearing per AGENTS.md
    policy_version: str | None = None,
    env_version: str | None = None,
    git_sha: str | None = None,
    seed: int | None = None,

    # Artifact paths (uploaded to object storage via signed PUT URLs)
    video: str | Path | None = None,
    sensors: str | Path | None = None,
    actions: str | Path | None = None,

    # Run details
    duration_s: float | None = None,
    fps: float | None = None,
    metadata: Mapping[str, Any] | None = None,

    # Final state
    status: Literal["ready", "failed"] = "ready",
) -> Episode
```

Returns the finalized `Episode`. On failure during upload the SDK
flips the run to `status="failed"` and re-raises so your program
sees what went wrong.

### `start_episode` — explicit lifecycle

When you want fine-grained control (stream uploads, defer finalize,
react to upload errors per-artifact), use `start_episode` and the
returned `Episode` handle:

```python
with rt.start_episode(
    name="pick_and_place v3 morning warmup",
    policy_version="pap-v3.2.1",
    artifacts=["video", "sensors"],     # only request the slots you'll fill
) as ep:
    ep.upload_video("/tmp/run.mp4")
    ep.upload_sensors("/tmp/sensors.bin")
    # No explicit finalize — context manager handles it:
    #   • clean exit → status="ready"
    #   • exception  → status="failed", with metadata.failure_reason set
```

Or explicit:

```python
ep = rt.start_episode(name="…", policy_version="…", artifacts=["video"])
ep.upload_video("/tmp/run.mp4")
ep.finalize(status="ready", duration_s=47.2, fps=30)
```

### `Client` — explicit instance

Skip the module-level default when you need multiple deployments at
once (e.g. shipping the same run to staging + production), or for
clean dependency injection in tests:

```python
with rt.Client(api_key="rt_…", base_url="https://…") as client:
    client.log_episode(name="…", policy_version="…", video="…")
```

`Client` holds a connection pool — construct it once at process
startup, reuse across many episodes, and `close()` (or use as a
context manager) on shutdown.

## Errors

Every SDK error inherits from `robotrace.RobotraceError`. Catch by
type rather than parsing message strings:

| Exception            | When                                                   |
| -------------------- | ------------------------------------------------------ |
| `ConfigurationError` | Missing `api_key` / `base_url`, file path doesn't exist |
| `TransportError`     | Network / DNS / TLS / timeout                          |
| `AuthError`          | 401 — bad / missing / revoked key                      |
| `NotFoundError`      | 404 — episode id doesn't exist (or cross-tenant)       |
| `ConflictError`      | 409 — episode is archived, etc.                        |
| `ValidationError`    | 400 — payload didn't pass server-side validation       |
| `ServerError`        | 5xx — flag for retries                                 |

```python
from robotrace import RobotraceError, AuthError

try:
    rt.log_episode(...)
except AuthError:
    # mint a fresh key and reload
    raise
except RobotraceError:
    # generic recovery / alert
    raise
```

## Storage

Artifact uploads go to Cloudflare R2 via short-lived signed PUT URLs
the server mints for each call. The SDK streams from disk so memory
stays flat regardless of file size.

When the deployment hasn't wired R2 yet (`R2_ACCOUNT_ID` etc. are
blank), the create response has `storage="unconfigured"` and any
`upload_*` call raises `ConfigurationError` with a pointer to the
production setup checklist. Metadata-only runs still work — useful
for testing the SDK contract end-to-end before R2 is provisioned.

## Layout (current)

```
src/robotrace/
├── __init__.py          # public API + module-level default client
├── _version.py
├── client.py            # Client class
├── episode.py           # Episode handle + UploadUrl + ArtifactKind
├── errors.py            # RobotraceError + typed subclasses
└── _http.py             # internal httpx wrapper
```

ROS 2 / LeRobot adapters land later under `src/robotrace/adapters/`.

## Contributing

The public source lives at
[github.com/Artl13/robotrace-dev](https://github.com/Artl13/robotrace-dev) —
a read-only mirror auto-synced from our internal monorepo. File
issues and PRs against the mirror; we'll cherry-pick approved
changes back into the private repo and they'll flow out on the
next sync.

The web app at `apps/web` (private) exposes the ingest API the SDK
talks to — coordinate breaking changes by emailing the
[`/api/ingest/episode`](https://robotrace.dev/docs/api/ingest)
contract owner before opening a SDK PR that depends on a server
change.

## License

MIT.
