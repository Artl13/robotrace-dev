"""Read-only metadata loader for LeRobot v2.1 datasets.

Loads from either a local directory (``./my_dataset/``) or a HF Hub
repo id (``lerobot/aloha_static_cups_open``). Hub access fetches only
the small ``meta/`` files — never a parquet shard or an mp4 — so a
``scan_dataset(...)`` call is fast and cheap regardless of the dataset's
total size.

Files we read:
  * ``meta/info.json`` — fps, total_episodes, features schema,
    codebase_version (used to gate v3.0 with a friendly error).
  * ``meta/episodes.jsonl`` — per-episode length and task index.
  * ``meta/tasks.jsonl`` — task index → human-readable description.

We deliberately avoid reading ``meta/stats.json`` (or
``meta/episodes_stats.jsonl`` in v2.1) — normalization stats matter
for training, not for replay/observability, and they can be 100s of
KB on big datasets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...errors import ConfigurationError

# Datasets we know how to read end-to-end. v2.1 is the on-disk format
# used by virtually every public `lerobot/*` Hub dataset as of
# May 2026; v3.0 is the new multi-episode-shard format and lands as a
# separate ship in 0.1.0a4. Anything else hard-fails with a hint to
# pin to a v2.1 revision.
_SUPPORTED_CODEBASE_VERSIONS: frozenset[str] = frozenset({"v2.0", "v2.1"})


@dataclass(frozen=True)
class EpisodeMeta:
    """Per-episode facts read from ``meta/episodes.jsonl``."""

    episode_index: int
    length: int  # number of frames in the episode
    tasks: list[str]  # human-readable task description(s); rarely empty


@dataclass
class DatasetSummary:
    """What a LeRobot dataset contains and how the adapter would treat it.

    Returned by `scan_dataset`. Computed before any frame data is
    downloaded — print this first, decide what to upload, then call
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
        """Look up one episode's meta — raises if it doesn't exist."""
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
      * the format version isn't v2.0 / v2.1 (v3.0 fails clearly)
      * the meta files are missing or malformed
      * (Hub only) ``huggingface_hub`` isn't installed
    """
    is_local = _looks_local(repo_id_or_path)
    if is_local:
        meta_dir = Path(repo_id_or_path).expanduser().resolve() / "meta"
        if not meta_dir.is_dir():
            raise ConfigurationError(
                f"no meta/ directory in {repo_id_or_path} — is this a "
                "LeRobot dataset root? The expected layout is "
                "<root>/{meta,data,videos}/."
            )
        info_path = meta_dir / "info.json"
        episodes_path = meta_dir / "episodes.jsonl"
        tasks_path = meta_dir / "tasks.jsonl"
    else:
        info_path = _hub_download(repo_id_or_path, "meta/info.json", revision=revision)
        episodes_path = _hub_download(
            repo_id_or_path, "meta/episodes.jsonl", revision=revision
        )
        # tasks.jsonl is optional — older v2.0 datasets sometimes ship
        # the task field inline on each episode row instead.
        try:
            tasks_path = _hub_download(
                repo_id_or_path, "meta/tasks.jsonl", revision=revision
            )
        except ConfigurationError:
            tasks_path = None

    info = _load_info(info_path)
    _check_format_version(repo_id_or_path, info)

    feature_columns = sorted((info.get("features") or {}).keys())
    camera_keys = [c for c in feature_columns if c.startswith("observation.images.")]

    tasks_by_index: dict[int, str] = {}
    if tasks_path is not None and Path(tasks_path).is_file():
        tasks_by_index = _load_tasks(Path(tasks_path))

    episodes = _load_episodes(Path(episodes_path), tasks_by_index)

    return DatasetSummary(
        repo_id_or_path=repo_id_or_path,
        is_local=is_local,
        codebase_version=str(info.get("codebase_version") or "unknown"),
        fps=float(info.get("fps") or 0),
        total_episodes=int(info.get("total_episodes") or len(episodes)),
        total_frames=int(info.get("total_frames") or sum(e.length for e in episodes)),
        feature_columns=feature_columns,
        camera_keys=camera_keys,
        episodes=episodes,
    )


# ── internals ─────────────────────────────────────────────────────────


def _looks_local(s: str) -> bool:
    """Heuristic: is this a local path or a Hub repo id?

    HF repo ids are always ``namespace/dataset-name`` — exactly one
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
            f"failed to read {info_path} — corrupted info.json? ({exc})"
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
        raise ConfigurationError(
            f"{repo_id_or_path} is a LeRobot dataset format {version!r}, which "
            "uses multi-episode parquet shards instead of one-file-per-episode. "
            "The robotrace adapter currently supports v2.0 / v2.1 only — "
            "v3.0 is on the roadmap for robotrace 0.1.0a4. Workarounds: "
            "(a) pin to a v2.1 revision of the dataset (`revision='v2.1'`), "
            "(b) convert the dataset locally with `lerobot`'s "
            "`convert_dataset_v21_to_v30.py` script in reverse, or "
            "(c) open an issue at "
            "https://github.com/Artl13/robotrace-dev/issues with your dataset."
        )
    raise ConfigurationError(
        f"{repo_id_or_path} declares codebase_version={version!r}, which the "
        "robotrace adapter doesn't recognise. Supported: v2.0, v2.1."
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
    shapes — robotics teams in the wild really do mix versions.
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

    ``tasks`` (plural list) — v2.1 modern shape.
    ``task`` (singular string) — older v2.0.
    ``task_index`` (int) — references tasks.jsonl.
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


# ── HF Hub plumbing ───────────────────────────────────────────────────


def _hub_download(
    repo_id: str,
    filename: str,
    *,
    revision: str | None,
) -> Path:
    """Pull one file from an HF Hub dataset repo and return its path.

    Caches in ``~/.cache/huggingface/hub`` per `huggingface_hub`'s
    default — repeat calls don't re-download. Raises
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
            "Install with `pip install 'robotrace-dev[lerobot]==0.1.0a3'`. "
            "If you only ever pass local dataset paths you can install "
            "`pip install huggingface_hub` separately and skip the rest of "
            "the extra."
        ) from exc
    return huggingface_hub
