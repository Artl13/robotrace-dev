"""Read-only metadata loader for LeRobot datasets (v2.0 / v2.1 / v3.0).

Loads from either a local directory (``./my_dataset/``) or a HF Hub
repo id (``lerobot/aloha_static_cups_open``). Hub access fetches only
the small ``meta/`` files - never a data parquet shard or an mp4 - so a
``scan_dataset(...)`` call is fast and cheap regardless of the dataset's
total size.

Two on-disk layouts are supported:

v2.0 / v2.1 (one file per episode)
    * ``meta/info.json`` - fps, total_episodes, features schema,
      codebase_version.
    * ``meta/episodes.jsonl`` - per-episode length and task index.
    * ``meta/tasks.jsonl`` - task index → human-readable description.

v3.0 (many episodes per shard)
    * ``meta/info.json`` - same fields plus ``data_path`` / ``video_path``
      templates and ``chunks_size``.
    * ``meta/episodes/chunk-XXX/file-YYY.parquet`` - per-episode records
      (length, tasks, and the data/video shard locators that say which
      parquet/mp4 holds the episode and at what row/timestamp range).
    * ``meta/tasks.parquet`` - task index → description.

We deliberately avoid reading ``meta/stats.json`` (or
``meta/episodes_stats.jsonl`` in v2.1) - normalization stats matter
for training, not for replay/observability, and they can be 100s of
KB on big datasets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..._version import install_command
from ...errors import ConfigurationError

# Datasets we know how to read end-to-end. v2.0 / v2.1 are the
# one-file-per-episode layouts used by virtually every public
# `lerobot/*` Hub dataset through 2025; v3.0 is the multi-episode-shard
# layout (`lerobot >= 0.3.x`). Anything else hard-fails with a hint.
_SUPPORTED_CODEBASE_VERSIONS: frozenset[str] = frozenset(
    {"v2.0", "v2.1", "v3.0", "v3.1"}
)

# v3.0 default path templates (from `lerobot.datasets.utils`). Used when
# `info.json` omits them - older v3 pre-releases sometimes did.
_V3_DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
_V3_DEFAULT_VIDEO_PATH = (
    "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
)
_V3_EPISODES_DIR = "meta/episodes"
_V3_TASKS_PATH = "meta/tasks.parquet"


@dataclass(frozen=True)
class VideoLocator:
    """Where one camera's frames for one episode live in a v3.0 shard.

    v3.0 concatenates many episodes into one mp4 per camera. To pull a
    single episode back out you need the shard file (chunk + file index)
    and the ``[from_timestamp, to_timestamp)`` window inside it.
    """

    video_key: str
    chunk_index: int
    file_index: int
    from_timestamp: float
    to_timestamp: float


@dataclass(frozen=True)
class EpisodeMeta:
    """Per-episode facts.

    For v2.x these come from ``meta/episodes.jsonl``; the v3.0 locator
    fields stay ``None`` / empty. For v3.0 they come from one row of a
    ``meta/episodes/*.parquet`` shard, and the locator fields point at
    the data parquet shard + per-camera video windows that hold this
    episode.
    """

    episode_index: int
    length: int  # number of frames in the episode
    tasks: list[str]  # human-readable task description(s); rarely empty
    # v3.0-only locators. None for v2.x (each episode is its own file).
    data_chunk_index: int | None = None
    data_file_index: int | None = None
    dataset_from_index: int | None = None  # global row range start (inclusive)
    dataset_to_index: int | None = None  # global row range end (exclusive)
    # Per-camera video window, keyed by the full feature name
    # (e.g. ``observation.images.laptop``).
    video_locators: dict[str, VideoLocator] = field(default_factory=dict)


@dataclass
class DatasetSummary:
    """What a LeRobot dataset contains and how the adapter would treat it.

    Returned by `scan_dataset`. Computed before any frame data is
    downloaded - print this first, decide what to upload, then call
    `upload_episode` / `upload_dataset`.

    Attributes
    ----------
    repo_id_or_path
        The string the user passed in. Either a local path or an HF
        repo id (``namespace/dataset-name``).
    is_local
        True if the input was a local directory; False if it was a
        Hub repo id.
    codebase_version
        The format version string from ``info.json`` (e.g. ``"v2.1"``).
    fps
        Frames per second from ``info.json``. Carried through to the
        RoboTrace episode so the portal renders frame-accurate timing.
    total_episodes
        Number of trajectories in the dataset. ``upload_dataset(...)``
        defaults to walking all of them.
    total_frames
        Total frame count across the whole dataset. Useful for
        estimating how much data ``upload_dataset`` will move.
    feature_columns
        Every column name declared in ``info.json["features"]``. The
        encoder uses this to pre-validate that the requested episode
        will yield a non-empty NPZ.
    camera_keys
        The full ``observation.images.<key>`` column names. Used to
        find video files under ``videos/.../<camera_key>/...mp4``.
    episodes
        Per-episode lengths + task descriptions, indexed by
        episode_index.
    """

    repo_id_or_path: str
    is_local: bool
    codebase_version: str
    fps: float
    total_episodes: int
    total_frames: int
    feature_columns: list[str] = field(default_factory=list)
    camera_keys: list[str] = field(default_factory=list)
    episodes: list[EpisodeMeta] = field(default_factory=list)
    # v3.0 path templates from info.json (``data_path`` / ``video_path``).
    # None for v2.x, where the encoder uses hardcoded per-episode paths.
    data_path: str | None = None
    video_path: str | None = None

    @property
    def is_v3(self) -> bool:
        """True for the multi-episode-shard layout (v3.x)."""
        return self.codebase_version.startswith("v3")

    def report(self) -> str:
        """Human-readable summary, one line per section.

        Used by docs and by users who do
        ``print(scan_dataset(repo_id).report())`` before invoking
        ``upload_dataset(...)``. Stable enough to assert in tests but
        not promised as a parser-friendly format.
        """
        loc = "local" if self.is_local else "hub"
        lines = [
            f"{self.repo_id_or_path}  ({loc}, {self.codebase_version}, "
            f"{self.fps:g} fps)",
            f"  episodes: {self.total_episodes}, frames: {self.total_frames}",
        ]
        if self.camera_keys:
            lines.append(f"  cameras: {', '.join(self.camera_keys)}")
        non_camera = [
            c
            for c in self.feature_columns
            if not c.startswith("observation.images.")
            and c not in {"timestamp", "frame_index", "episode_index", "index", "task_index"}
        ]
        if non_camera:
            lines.append(f"  features: {', '.join(non_camera[:8])}")
            if len(non_camera) > 8:
                lines.append(f"            (+{len(non_camera) - 8} more)")
        if self.episodes and self.total_episodes <= 5:
            for ep in self.episodes:
                task = ep.tasks[0] if ep.tasks else "(no task)"
                lines.append(
                    f"  ep {ep.episode_index}: {ep.length} frames, task={task!r}"
                )
        return "\n".join(lines)

    def episode(self, episode_index: int) -> EpisodeMeta:
        """Look up one episode's meta - raises if it doesn't exist."""
        for ep in self.episodes:
            if ep.episode_index == episode_index:
                return ep
        raise ConfigurationError(
            f"episode_index={episode_index} is out of range for "
            f"{self.repo_id_or_path} (total_episodes={self.total_episodes})."
        )


def scan_dataset(
    repo_id_or_path: str,
    *,
    revision: str | None = None,
) -> DatasetSummary:
    """Open a LeRobot dataset (local or Hub) and describe its contents.

    Parameters
    ----------
    repo_id_or_path
        Either a local directory (containing ``meta/info.json``) or
        an HF Hub repo id of the form ``namespace/dataset-name``.
        The HF API uses the same string shape; we discriminate by
        checking whether the path exists locally.
    revision
        Optional Hub revision (branch / tag / commit sha). Forwarded
        to ``huggingface_hub.hf_hub_download``. Ignored for local
        paths. Defaults to ``main``.

    Raises ``ConfigurationError`` if:
      * the format version isn't v2.0 / v2.1 / v3.0
      * the meta files are missing or malformed
      * (Hub only) ``huggingface_hub`` isn't installed
    """
    is_local = _looks_local(repo_id_or_path)
    if is_local:
        meta_dir = Path(repo_id_or_path).expanduser().resolve() / "meta"
        if not meta_dir.is_dir():
            raise ConfigurationError(
                f"no meta/ directory in {repo_id_or_path} - is this a "
                "LeRobot dataset root? The expected layout is "
                "<root>/{meta,data,videos}/."
            )
        info_path = meta_dir / "info.json"
    else:
        info_path = _hub_download(repo_id_or_path, "meta/info.json", revision=revision)

    info = _load_info(info_path)
    _check_format_version(repo_id_or_path, info)

    feature_columns = sorted((info.get("features") or {}).keys())
    camera_keys = _camera_keys(info, feature_columns)
    version = str(info.get("codebase_version") or "unknown")

    if version.startswith("v3"):
        episodes = _load_episodes_v3(
            repo_id_or_path, is_local, camera_keys, revision=revision
        )
        data_path = str(info.get("data_path") or _V3_DEFAULT_DATA_PATH)
        video_path = str(info.get("video_path") or _V3_DEFAULT_VIDEO_PATH)
    else:
        episodes = _load_episodes_v2(repo_id_or_path, is_local, revision=revision)
        data_path = None
        video_path = None

    return DatasetSummary(
        repo_id_or_path=repo_id_or_path,
        is_local=is_local,
        codebase_version=version,
        fps=float(info.get("fps") or 0),
        total_episodes=int(info.get("total_episodes") or len(episodes)),
        total_frames=int(info.get("total_frames") or sum(e.length for e in episodes)),
        feature_columns=feature_columns,
        camera_keys=camera_keys,
        episodes=episodes,
        data_path=data_path,
        video_path=video_path,
    )


def _camera_keys(info: dict[str, Any], feature_columns: list[str]) -> list[str]:
    """Camera feature names. v3 marks them ``dtype: "video"``; older
    datasets just use the ``observation.images.`` prefix."""
    features = info.get("features") or {}
    by_dtype = [
        k for k, v in features.items() if isinstance(v, dict) and v.get("dtype") == "video"
    ]
    if by_dtype:
        return sorted(by_dtype)
    return [c for c in feature_columns if c.startswith("observation.images.")]


def _load_episodes_v2(
    repo_id_or_path: str, is_local: bool, *, revision: str | None
) -> list[EpisodeMeta]:
    """v2.x: read ``meta/episodes.jsonl`` (+ optional ``meta/tasks.jsonl``)."""
    if is_local:
        meta_dir = Path(repo_id_or_path).expanduser().resolve() / "meta"
        episodes_path: Path = meta_dir / "episodes.jsonl"
        tasks_path: Path | None = meta_dir / "tasks.jsonl"
    else:
        episodes_path = _hub_download(
            repo_id_or_path, "meta/episodes.jsonl", revision=revision
        )
        # tasks.jsonl is optional - older v2.0 datasets sometimes ship
        # the task field inline on each episode row instead.
        try:
            tasks_path = _hub_download(
                repo_id_or_path, "meta/tasks.jsonl", revision=revision
            )
        except ConfigurationError:
            tasks_path = None

    tasks_by_index: dict[int, str] = {}
    if tasks_path is not None and Path(tasks_path).is_file():
        tasks_by_index = _load_tasks(Path(tasks_path))

    return _load_episodes(Path(episodes_path), tasks_by_index)


# ── internals ─────────────────────────────────────────────────────────


def _looks_local(s: str) -> bool:
    """Heuristic: is this a local path or a Hub repo id?

    HF repo ids are always ``namespace/dataset-name`` - exactly one
    slash, no other path separators, no leading dot or tilde. Anything
    that looks like a real filesystem path (absolute, relative, with
    leading ``./`` / ``~``, or with a path that exists on disk) is
    treated as local.
    """
    p = Path(s).expanduser()
    if p.exists():
        return True
    if s.startswith(("/", "./", "../", "~")):
        return True
    # `name/repo` with no other path-shaped chars → Hub
    if s.count("/") == 1 and not s.startswith("/") and "\\" not in s:
        return False
    # Fall through: probably a typo'd path; Hub call would 404 anyway
    return False


def _load_info(info_path: Path) -> dict[str, Any]:
    try:
        parsed = json.loads(info_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(
            f"failed to read {info_path} - corrupted info.json? ({exc})"
        ) from exc
    if not isinstance(parsed, dict):
        raise ConfigurationError(
            f"{info_path} did not parse to a JSON object (got {type(parsed).__name__})"
        )
    return parsed


def _check_format_version(repo_id_or_path: str, info: dict[str, Any]) -> None:
    version = str(info.get("codebase_version") or "")
    if version in _SUPPORTED_CODEBASE_VERSIONS:
        return
    if version.startswith("v3"):
        # We read v3.0 / v3.1; warn-but-proceed for an unseen v3.x point
        # release rather than hard-failing on a layout we likely handle.
        return
    if version.startswith(("v4", "v5", "v6", "v7", "v8", "v9")):
        raise ConfigurationError(
            f"{repo_id_or_path} is a LeRobot dataset format {version!r}, which is "
            "newer than anything this robotrace release knows how to read "
            "(supported: v2.0, v2.1, v3.0). Upgrade robotrace, or open an issue "
            "at https://github.com/Artl13/robotrace-dev/issues with your dataset."
        )
    raise ConfigurationError(
        f"{repo_id_or_path} declares codebase_version={version!r}, which the "
        "robotrace adapter doesn't recognise. Supported: v2.0, v2.1, v3.0."
    )


def _load_episodes(
    episodes_path: Path,
    tasks_by_index: dict[int, str],
) -> list[EpisodeMeta]:
    """Read ``meta/episodes.jsonl`` line by line.

    Format on each line (v2.1):
        {"episode_index": 0, "tasks": ["pick up the cup"], "length": 250}
    Older v2.0 datasets sometimes use ``"task"`` (singular) or
    ``"task_index"`` referencing tasks.jsonl. We accept all three
    shapes - robotics teams in the wild really do mix versions.
    """
    out: list[EpisodeMeta] = []
    try:
        with episodes_path.open() as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                ep_idx = int(row.get("episode_index", len(out)))
                length = int(row.get("length") or row.get("episode_length") or 0)
                tasks = _normalize_tasks(row, tasks_by_index)
                out.append(EpisodeMeta(episode_index=ep_idx, length=length, tasks=tasks))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(
            f"failed to read {episodes_path}: {exc}"
        ) from exc
    return out


def _normalize_tasks(
    row: dict[str, Any],
    tasks_by_index: dict[int, str],
) -> list[str]:
    """Coerce per-episode task fields into a uniform list[str].

    ``tasks`` (plural list) - v2.1 modern shape.
    ``task`` (singular string) - older v2.0.
    ``task_index`` (int) - references tasks.jsonl.
    """
    if "tasks" in row and isinstance(row["tasks"], list):
        return [str(t) for t in row["tasks"]]
    task_value = row.get("task")
    if task_value:
        return [str(task_value)]
    if "task_index" in row:
        try:
            idx = int(row["task_index"])
        except (TypeError, ValueError):
            return []
        if idx in tasks_by_index:
            return [tasks_by_index[idx]]
    return []


def _load_tasks(tasks_path: Path) -> dict[int, str]:
    """Read ``meta/tasks.jsonl`` into ``{task_index: description}``."""
    out: dict[int, str] = {}
    try:
        with tasks_path.open() as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                idx_raw = row.get("task_index")
                desc = row.get("task") or row.get("description") or ""
                if idx_raw is None:
                    continue
                try:
                    idx = int(idx_raw)
                except (TypeError, ValueError):
                    continue
                out[idx] = str(desc)
    except (OSError, json.JSONDecodeError):
        return {}
    return out


# ── v3.0 episode metadata (parquet shards) ────────────────────────────


def _load_episodes_v3(
    repo_id_or_path: str,
    is_local: bool,
    camera_keys: list[str],
    *,
    revision: str | None,
) -> list[EpisodeMeta]:
    """v3.0: read every ``meta/episodes/*.parquet`` shard into EpisodeMeta.

    Each row is one episode and carries the locators (which data parquet
    shard + which row range, and per-camera the video shard + timestamp
    window) needed to pull that episode back out of the concatenated
    shards later, in ``encode_episode``.
    """
    shard_paths = _v3_episode_shard_paths(repo_id_or_path, is_local, revision=revision)
    if not shard_paths:
        raise ConfigurationError(
            f"{repo_id_or_path} declares a v3.0 layout but has no "
            "meta/episodes/*.parquet shards. Is the dataset fully uploaded? "
            "(v3 writers must call `dataset.finalize()` before pushing.)"
        )

    tasks_by_index = _load_tasks_v3(repo_id_or_path, is_local, revision=revision)

    out: list[EpisodeMeta] = []
    for path in shard_paths:
        for row in _read_parquet_rows(path):
            out.append(_episode_from_v3_row(row, camera_keys, tasks_by_index))
    out.sort(key=lambda e: e.episode_index)
    return out


def _episode_from_v3_row(
    row: dict[str, Any],
    camera_keys: list[str],
    tasks_by_index: dict[int, str],
) -> EpisodeMeta:
    ep_idx = int(row.get("episode_index", 0))
    length = int(row.get("length") or 0)
    tasks = _normalize_tasks(row, tasks_by_index)

    locators: dict[str, VideoLocator] = {}
    for key in camera_keys:
        chunk = row.get(f"videos/{key}/chunk_index")
        file_ = row.get(f"videos/{key}/file_index")
        from_ts = row.get(f"videos/{key}/from_timestamp")
        to_ts = row.get(f"videos/{key}/to_timestamp")
        if chunk is None or file_ is None or from_ts is None or to_ts is None:
            continue
        locators[key] = VideoLocator(
            video_key=key,
            chunk_index=int(chunk),
            file_index=int(file_),
            from_timestamp=float(from_ts),
            to_timestamp=float(to_ts),
        )

    return EpisodeMeta(
        episode_index=ep_idx,
        length=length,
        tasks=tasks,
        data_chunk_index=_opt_int(row.get("data/chunk_index")),
        data_file_index=_opt_int(row.get("data/file_index")),
        dataset_from_index=_opt_int(row.get("dataset_from_index")),
        dataset_to_index=_opt_int(row.get("dataset_to_index")),
        video_locators=locators,
    )


def _opt_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _v3_episode_shard_paths(
    repo_id_or_path: str,
    is_local: bool,
    *,
    revision: str | None,
) -> list[Path]:
    if is_local:
        root = Path(repo_id_or_path).expanduser().resolve() / _V3_EPISODES_DIR
        return sorted(root.rglob("*.parquet"))
    rels = [
        f
        for f in _list_hub_files(repo_id_or_path, revision=revision)
        if f.startswith(_V3_EPISODES_DIR + "/") and f.endswith(".parquet")
    ]
    return [_hub_download(repo_id_or_path, rel, revision=revision) for rel in sorted(rels)]


def _load_tasks_v3(
    repo_id_or_path: str,
    is_local: bool,
    *,
    revision: str | None,
) -> dict[int, str]:
    """Best-effort ``meta/tasks.parquet`` → ``{task_index: description}``.

    v3 also embeds a ``tasks`` list directly on each episode row, so this
    map is only a fallback for datasets that store ``task_index`` instead.
    A missing or oddly-shaped tasks file is non-fatal.
    """
    if is_local:
        path = Path(repo_id_or_path).expanduser().resolve() / _V3_TASKS_PATH
        if not path.is_file():
            return {}
    else:
        try:
            path = _hub_download(repo_id_or_path, _V3_TASKS_PATH, revision=revision)
        except ConfigurationError:
            return {}

    out: dict[int, str] = {}
    try:
        rows = _read_parquet_rows(path)
    except ConfigurationError:
        return {}
    for i, row in enumerate(rows):
        idx_raw = row.get("task_index", i)
        desc = (
            row.get("task")
            or row.get("description")
            or row.get("__index_level_0__")
            or ""
        )
        idx = _opt_int(idx_raw)
        if idx is not None:
            out[idx] = str(desc)
    return out


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    """Read a parquet file into a list of per-row dicts (column-keyed)."""
    pq = _import_pyarrow_parquet()
    try:
        table = pq.read_table(path)
    except Exception as exc:  # pyarrow raises a variety of error types
        raise ConfigurationError(
            f"failed to read parquet {path}: {exc}. Corrupted shard, or a "
            "v3 dataset that wasn't finalized before upload?"
        ) from exc
    rows: list[dict[str, Any]] = table.to_pylist()
    return rows


def _import_pyarrow_parquet() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ConfigurationError(
            "reading LeRobot v3.0 metadata needs `pyarrow`. Install with "
            f"`{install_command('lerobot')}`."
        ) from exc
    return pq


# ── HF Hub plumbing ───────────────────────────────────────────────────


def _list_hub_files(repo_id: str, *, revision: str | None) -> list[str]:
    """List every file path in an HF Hub dataset repo.

    Used to discover the variable number of ``meta/episodes/*.parquet``
    shards in a v3.0 dataset without guessing chunk/file indices.
    """
    hub = _import_huggingface_hub()
    try:
        return list(
            hub.HfApi().list_repo_files(
                repo_id=repo_id, repo_type="dataset", revision=revision
            )
        )
    except Exception as exc:
        raise ConfigurationError(
            f"huggingface_hub failed to list files in {repo_id}: {exc}. "
            "Common causes: typo'd repo id, dataset is private (set HF_TOKEN), "
            "or no internet."
        ) from exc


def _hub_download(
    repo_id: str,
    filename: str,
    *,
    revision: str | None,
) -> Path:
    """Pull one file from an HF Hub dataset repo and return its path.

    Caches in ``~/.cache/huggingface/hub`` per `huggingface_hub`'s
    default - repeat calls don't re-download. Raises
    ConfigurationError with a clear install hint if the lib is missing.
    """
    hub = _import_huggingface_hub()
    try:
        path = hub.hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            repo_type="dataset",
        )
    except Exception as exc:  # huggingface_hub raises a forest of types
        # Most useful failure modes: missing repo, missing file, no network.
        raise ConfigurationError(
            f"huggingface_hub failed to fetch {filename} from {repo_id}: {exc}. "
            "Common causes: typo'd repo id, dataset is private (set HF_TOKEN), "
            "no internet, or this dataset doesn't ship the expected meta/* files."
        ) from exc
    return Path(path)


def _import_huggingface_hub() -> Any:
    try:
        import huggingface_hub
    except ImportError as exc:
        raise ConfigurationError(
            "the LeRobot adapter needs `huggingface_hub` to talk to the HF Hub. "
            f"Install with `{install_command('lerobot')}`. "
            "If you only ever pass local dataset paths you can install "
            "`pip install huggingface_hub` separately and skip the rest of "
            "the extra."
        ) from exc
    return huggingface_hub
