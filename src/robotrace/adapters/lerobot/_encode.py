"""`encode_episode(...)` — write artifacts for one LeRobot episode.

Walks one episode's parquet file (``data/chunk-XXX/episode_NNNNNN.parquet``)
and the matching per-camera mp4s (``videos/observation.images.<key>/chunk-XXX/
episode_NNNNNN.mp4``), and produces the standard RoboTrace artifact shape:

    video.mp4    one camera passthrough, OR multi-camera horizontally tiled
    sensors.npz  every observation.* column packed under
                 ``<column>/<value-shape>`` keys + ``<column>/_t_ns``
    actions.npz  every action[.X] column packed the same way
    metadata     episode_meta (next.reward, next.done, …) + dataset
                 provenance (repo_id, codebase_version, episode_index,
                 task description) for the SDK to merge into
                 `start_episode(metadata=...)`

The encoder never opens the network for the *upload* — it only fetches
the parquet + mp4s from the Hub (or reads them from the local path)
into a tempdir. Use `upload_episode(...)` for the full pipeline.

Memory model
------------

A LeRobot episode is small by RoboTrace standards (single trajectory,
typically <1 minute, <10 MB of parquet for the modal `lerobot/*`
dataset). We materialize the full parquet table into pyarrow before
extracting columns. If we hit a real-world dataset where this is a
problem we can swap to row-group iteration without changing the public
surface.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...errors import ConfigurationError
from ._classify import Slot, classify_column
from ._meta import DatasetSummary, EpisodeMeta, scan_dataset


@dataclass
class EncodedArtifact:
    """One encoded file ready to upload."""

    slot: Slot
    path: Path
    bytes_size: int
    columns: list[str] = field(default_factory=list)


@dataclass
class EncodedEpisode:
    """The product of `encode_episode(...)`.

    Mirrors the ROS 2 adapter's `EncodedBag` shape so callers that
    composed against ROS 2 can reuse the same upload pattern with no
    branching.
    """

    output_dir: Path
    summary: DatasetSummary
    episode: EpisodeMeta
    video: EncodedArtifact | None = None
    sensors: EncodedArtifact | None = None
    actions: EncodedArtifact | None = None
    duration_s: float | None = None
    fps: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def encode_episode(
    repo_id_or_path: str,
    episode_index: int,
    output_dir: str | Path,
    *,
    revision: str | None = None,
    canonical_camera: str | None = None,
    summary: DatasetSummary | None = None,
) -> EncodedEpisode:
    """Encode one LeRobot trajectory to ``video.mp4 / sensors.npz / actions.npz``.

    Parameters
    ----------
    repo_id_or_path
        Hub repo id (``namespace/dataset-name``) or local dataset
        directory.
    episode_index
        Which trajectory to encode (0-indexed).
    output_dir
        Directory to write artifacts into. Created if missing. Three
        filenames are reserved: ``video.mp4``, ``sensors.npz``,
        ``actions.npz``.
    revision
        Optional Hub revision (branch / tag / commit). Forwarded to
        ``huggingface_hub.hf_hub_download``. Defaults to ``main``.
    canonical_camera
        Pick one camera as the only video output (skipping the
        multi-cam horizontal tile) when more than one camera exists.
        Pass the *full* column name (e.g. ``"observation.images.laptop"``)
        — same format the parquet schema uses.
    summary
        Pre-computed `DatasetSummary` from `scan_dataset(...)`. Skips
        the meta-fetch round-trip when the caller already scanned.
    """
    if summary is None:
        summary = scan_dataset(repo_id_or_path, revision=revision)

    ep_meta = summary.episode(episode_index)
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = _resolve_episode_parquet(
        repo_id_or_path, episode_index, summary, revision=revision
    )

    pa = _import_pyarrow()
    table = pa.parquet.read_table(parquet_path)
    columns = list(table.column_names)

    # LeRobot v2.1 stores cameras as MP4 files referenced by feature
    # name in info.json — NOT as parquet columns. So the source of
    # truth for "which cameras does this dataset have?" is
    # `summary.camera_keys`, populated from info.json["features"].
    # The parquet classifier only handles non-image features
    # (observation.state, action, next.*, etc.).
    video_columns = list(summary.camera_keys)

    column_decisions = {c: classify_column(c) for c in columns}
    sensors_columns = [c for c, d in column_decisions.items() if d.slot == "sensors"]
    actions_columns = [c for c, d in column_decisions.items() if d.slot == "actions"]
    episode_meta_columns = [
        c for c, d in column_decisions.items() if d.slot == "episode_meta"
    ]

    if canonical_camera is not None:
        if canonical_camera not in video_columns:
            raise ConfigurationError(
                f"canonical_camera={canonical_camera!r} is not a camera "
                f"in this dataset. Available cameras (from info.json): "
                f"{video_columns}."
            )
        video_columns = [canonical_camera]

    metadata: dict[str, Any] = {
        "adapter": "lerobot",
        "lerobot_repo_id": repo_id_or_path,
        "lerobot_codebase_version": summary.codebase_version,
        "lerobot_episode_index": episode_index,
        "lerobot_episode_length": ep_meta.length,
    }
    if ep_meta.tasks:
        metadata["lerobot_tasks"] = ep_meta.tasks

    skipped: list[dict[str, Any]] = []

    encoded_video = _encode_video(
        repo_id_or_path=repo_id_or_path,
        episode_index=episode_index,
        camera_columns=video_columns,
        output_path=out_dir / "video.mp4",
        revision=revision,
        skipped=skipped,
    )

    encoded_sensors = _encode_sensors_or_actions(
        table=table,
        columns=sensors_columns,
        slot="sensors",
        output_path=out_dir / "sensors.npz",
        skipped=skipped,
    )
    encoded_actions = _encode_sensors_or_actions(
        table=table,
        columns=actions_columns,
        slot="actions",
        output_path=out_dir / "actions.npz",
        skipped=skipped,
    )

    if episode_meta_columns:
        outcome = _extract_episode_outcome(table, episode_meta_columns)
        if outcome:
            metadata["lerobot_episode_outcome"] = outcome

    if skipped:
        metadata["skipped_columns"] = skipped

    duration_s = ep_meta.length / summary.fps if summary.fps > 0 else None
    fps = summary.fps if summary.fps > 0 else None

    return EncodedEpisode(
        output_dir=out_dir,
        summary=summary,
        episode=ep_meta,
        video=encoded_video,
        sensors=encoded_sensors,
        actions=encoded_actions,
        duration_s=duration_s,
        fps=fps,
        metadata=metadata,
    )


# ── parquet path resolution ───────────────────────────────────────────


def _resolve_episode_parquet(
    repo_id_or_path: str,
    episode_index: int,
    summary: DatasetSummary,
    *,
    revision: str | None,
) -> Path:
    """Locate the parquet file for one episode.

    LeRobot v2.1 always stores them at:
        ``data/chunk-{NNN}/episode_{NNNNNN}.parquet``
    where ``NNN`` is the chunk number (episode_index // chunks_size,
    chunks_size defaults to 1000). We compute both candidates rather
    than hardcoding the chunk size — older datasets sometimes used 100,
    and v2.0 used a flatter layout.
    """
    relative_candidates = _candidate_parquet_paths(episode_index)
    if summary.is_local:
        root = Path(summary.repo_id_or_path).expanduser().resolve()
        for rel in relative_candidates:
            candidate = root / rel
            if candidate.is_file():
                return candidate
        raise ConfigurationError(
            f"could not find parquet file for episode {episode_index} in "
            f"{root}. Tried: {relative_candidates}."
        )
    from ._meta import _hub_download

    last_error: Exception | None = None
    for rel in relative_candidates:
        try:
            return _hub_download(repo_id_or_path, rel, revision=revision)
        except ConfigurationError as exc:
            last_error = exc
            continue
    raise ConfigurationError(
        f"could not find parquet file for episode {episode_index} in "
        f"{repo_id_or_path}. Tried: {relative_candidates}. "
        f"Last error: {last_error}"
    )


def _candidate_parquet_paths(episode_index: int) -> list[str]:
    """Plausible relative paths for one episode's parquet file."""
    # Most common: chunk size 1000 → chunk-NNN.
    chunk_1000 = episode_index // 1000
    # Older datasets used chunk size 100 (e.g. early aloha).
    chunk_100 = episode_index // 100
    suffix = f"episode_{episode_index:06d}.parquet"
    return [
        f"data/chunk-{chunk_1000:03d}/{suffix}",
        f"data/chunk-{chunk_100:03d}/{suffix}",
    ]


def _candidate_video_paths(camera_column: str, episode_index: int) -> list[str]:
    """Plausible relative paths for one camera's mp4 for one episode."""
    chunk_1000 = episode_index // 1000
    chunk_100 = episode_index // 100
    suffix = f"episode_{episode_index:06d}.mp4"
    return [
        f"videos/{camera_column}/chunk-{chunk_1000:03d}/{suffix}",
        f"videos/{camera_column}/chunk-{chunk_100:03d}/{suffix}",
    ]


# ── video encoder ─────────────────────────────────────────────────────


def _encode_video(
    *,
    repo_id_or_path: str,
    episode_index: int,
    camera_columns: Sequence[str],
    output_path: Path,
    revision: str | None,
    skipped: list[dict[str, Any]],
) -> EncodedArtifact | None:
    """Resolve mp4(s) for the requested cameras and produce one output mp4.

    Single camera → the source mp4 is copied (no transcode — the
    server transcodes for browser playback if needed).
    Multi camera → horizontally tile per-frame using opencv. Falls back
    to "first camera only" if opencv isn't installed, with a `skipped`
    note pointing the user at the `[lerobot,video]` extras combination.
    """
    if not camera_columns:
        return None

    resolved: list[tuple[str, Path]] = []
    for cam in camera_columns:
        try:
            mp4_path = _resolve_video_path(
                repo_id_or_path, cam, episode_index, revision=revision
            )
        except ConfigurationError as exc:
            skipped.append({"column": cam, "reason": f"video file unavailable: {exc}"})
            continue
        resolved.append((cam, mp4_path))

    if not resolved:
        return None

    if len(resolved) == 1:
        cam, mp4_path = resolved[0]
        shutil.copyfile(mp4_path, output_path)
        return EncodedArtifact(
            slot="video",
            path=output_path,
            bytes_size=output_path.stat().st_size,
            columns=[cam],
        )

    # Multi-camera tile. Needs opencv.
    try:
        cv2 = _import_cv2()
        np = _import_numpy()
    except ConfigurationError as exc:
        skipped.append(
            {
                "reason": (
                    "multi-camera tile needs opencv — install "
                    "`pip install 'robotrace-dev[lerobot,video]==0.1.0a5'`. "
                    f"Falling back to canonical_camera={resolved[0][0]!r}."
                ),
                "exc": str(exc),
            }
        )
        cam, mp4_path = resolved[0]
        shutil.copyfile(mp4_path, output_path)
        return EncodedArtifact(
            slot="video",
            path=output_path,
            bytes_size=output_path.stat().st_size,
            columns=[cam],
        )

    columns_in_order = [cam for cam, _ in resolved]
    paths = [p for _, p in resolved]
    _tile_videos_horizontal(paths, output_path, cv2=cv2, np=np)
    return EncodedArtifact(
        slot="video",
        path=output_path,
        bytes_size=output_path.stat().st_size,
        columns=columns_in_order,
    )


def _resolve_video_path(
    repo_id_or_path: str,
    camera_column: str,
    episode_index: int,
    *,
    revision: str | None,
) -> Path:
    """Find the mp4 for one camera + one episode, local or Hub."""
    relative_candidates = _candidate_video_paths(camera_column, episode_index)
    is_local = Path(repo_id_or_path).expanduser().exists()
    if is_local:
        root = Path(repo_id_or_path).expanduser().resolve()
        for rel in relative_candidates:
            candidate = root / rel
            if candidate.is_file():
                return candidate
        raise ConfigurationError(
            f"no video file under {root} for camera {camera_column!r} "
            f"episode {episode_index}. Tried: {relative_candidates}."
        )
    from ._meta import _hub_download

    last_error: Exception | None = None
    for rel in relative_candidates:
        try:
            return _hub_download(repo_id_or_path, rel, revision=revision)
        except ConfigurationError as exc:
            last_error = exc
            continue
    raise ConfigurationError(
        f"no video file in {repo_id_or_path} for camera {camera_column!r} "
        f"episode {episode_index}. Tried: {relative_candidates}. "
        f"Last error: {last_error}"
    )


def _tile_videos_horizontal(
    paths: list[Path], output_path: Path, *, cv2: Any, np: Any
) -> None:
    """Horizontally tile N mp4s into one. Frames aligned by index.

    Pads heights with black if cameras mismatch. Output fps is the
    first input's fps. Output codec is mp4v for the same reason as
    the ROS 2 adapter (most portable across opencv wheels).
    """
    caps = [cv2.VideoCapture(str(p)) for p in paths]
    try:
        if not all(c.isOpened() for c in caps):
            raise ConfigurationError(
                f"opencv could not open one of {paths}. Most likely the "
                "mp4 codec isn't compiled into your opencv wheel."
            )
        fps = caps[0].get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            fps = 30.0
        widths: list[int] = []
        heights: list[int] = []
        for c in caps:
            widths.append(int(c.get(cv2.CAP_PROP_FRAME_WIDTH)))
            heights.append(int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out_w = sum(widths)
        out_h = max(heights)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))
        if not writer.isOpened():
            raise ConfigurationError(
                f"opencv VideoWriter failed to open {output_path}. "
                "Try `pip install --upgrade opencv-python`."
            )
        try:
            while True:
                cells: list[Any] = []
                for c, w, h in zip(caps, widths, heights, strict=True):
                    ok, frame = c.read()
                    if not ok:
                        cells = []
                        break
                    if h < out_h:
                        pad = np.zeros((out_h - h, w, 3), dtype=np.uint8)
                        frame = np.vstack([frame, pad])
                    cells.append(frame)
                if not cells:
                    break
                writer.write(np.hstack(cells))
        finally:
            writer.release()
    finally:
        for c in caps:
            c.release()


# ── sensors / actions encoder ─────────────────────────────────────────


def _encode_sensors_or_actions(
    *,
    table: Any,  # pyarrow.Table — left untyped to avoid hard dep
    columns: Sequence[str],
    slot: Slot,
    output_path: Path,
    skipped: list[dict[str, Any]],
) -> EncodedArtifact | None:
    """Pack the requested columns into one NPZ.

    Layout (mirrors the ROS 2 adapter for consistent server-side
    parsing):

        {
          "<column>/_t_ns":   int64[N]   — timestamp nanoseconds
          "<column>/value":   float32[N, K]  — column values
        }

    For columns whose value type is a list-of-floats (the usual case
    for ``observation.state``, ``action``, ``observation.imu.angular_velocity``,
    etc.) we stack into 2-D. For scalar columns (rare in LeRobot v2.1
    but possible — e.g. ``next.reward`` would land here if it weren't
    filtered into episode_meta) we stack into 1-D.
    """
    if not columns:
        return None

    np = _import_numpy()

    # `timestamp` is the canonical per-frame clock in LeRobot v2.1 —
    # we reuse it as the nanosecond timestamp for every column in the
    # NPZ. Convert seconds (float32) → nanoseconds (int64) once.
    if "timestamp" not in table.column_names:
        skipped.append(
            {
                "slot": slot,
                "reason": "no `timestamp` column in parquet — cannot align rows",
            }
        )
        return None
    timestamps_s = table.column("timestamp").to_numpy(zero_copy_only=False).astype(
        "float64"
    )
    timestamps_ns = (timestamps_s * 1_000_000_000).astype(np.int64)

    arrays: dict[str, Any] = {}
    used_columns: list[str] = []

    for col in columns:
        if col not in table.column_names:
            skipped.append(
                {"column": col, "reason": "declared in features but absent from data"}
            )
            continue
        try:
            stacked = _column_to_ndarray(table.column(col), np)
        except ValueError as exc:
            skipped.append({"column": col, "reason": f"unsupported value shape: {exc}"})
            continue
        if stacked is None:
            continue
        arrays[f"{col}/_t_ns"] = timestamps_ns
        arrays[f"{col}/value"] = stacked
        used_columns.append(col)

    if not arrays:
        return None

    np.savez(output_path, **arrays)
    return EncodedArtifact(
        slot=slot,
        path=output_path,
        bytes_size=output_path.stat().st_size,
        columns=used_columns,
    )


def _column_to_ndarray(column: Any, np: Any) -> Any:
    """Coerce a pyarrow column into a numpy 1-D or 2-D float32 array.

    Handles the three shapes LeRobot uses:
      * `float32` / `float64` scalars              → 1-D float32
      * `list<float>` per row (canonical state)    → 2-D float32 (N, K)
      * `fixed_size_list<float>` per row           → 2-D float32 (N, K)
    Anything else (struct, nested list, string)    → ValueError, which
    the caller turns into a `skipped` entry.
    """
    pa = _import_pyarrow()
    pa_lib = pa  # alias to keep type-checker quiet

    arrow_type = column.type

    if pa_lib.types.is_floating(arrow_type) or pa_lib.types.is_integer(arrow_type):
        return column.to_numpy(zero_copy_only=False).astype(np.float32)

    if pa_lib.types.is_list(arrow_type) or pa_lib.types.is_fixed_size_list(arrow_type):
        py_rows = column.to_pylist()
        if not py_rows:
            return None
        # Reject ragged lists — LeRobot state vectors are always fixed
        # length. Variable-length would only happen if the user shoved
        # `observation.tokens` or similar into a non-image observation.
        first_len = len(py_rows[0]) if py_rows[0] is not None else 0
        for i, row in enumerate(py_rows):
            if row is None or len(row) != first_len:
                raise ValueError(
                    f"row {i} has length {0 if row is None else len(row)}, "
                    f"expected {first_len}"
                )
        return np.asarray(py_rows, dtype=np.float32)

    raise ValueError(f"arrow type {arrow_type} not supported")


def _extract_episode_outcome(table: Any, columns: Sequence[str]) -> dict[str, Any]:
    """Roll up `next.reward` / `next.done` / `next.success` for the row.

    Returns the *last frame's* value for each — the typical "did the
    episode succeed?" signal. Reward is summed across the trajectory
    in addition (LeRobot stores per-step reward).
    """
    np = _import_numpy()
    out: dict[str, Any] = {}
    for col in columns:
        if col not in table.column_names:
            continue
        try:
            arr = table.column(col).to_numpy(zero_copy_only=False)
        except (ValueError, TypeError, ArithmeticError):
            continue
        if len(arr) == 0:
            continue
        last = arr[-1]
        # `bool_` and `np.bool_` aren't JSON-serialisable on every numpy
        # build; coerce to Python primitives.
        if hasattr(last, "item"):
            last = last.item()
        out[col] = last
        if col == "next.reward":
            try:
                out["next.reward_sum"] = float(np.asarray(arr, dtype=np.float64).sum())
            except (TypeError, ValueError):
                pass
    return out


# ── lazy imports ──────────────────────────────────────────────────────


def _import_pyarrow() -> Any:
    try:
        # `pyarrow.parquet` and `pyarrow.types` are submodules that must
        # be explicitly imported before they're available as
        # `pyarrow.parquet.read_table` / `pyarrow.types.is_floating`.
        # We use both later — these aren't unused imports.
        import pyarrow
        import pyarrow.parquet
        import pyarrow.types

        return pyarrow
    except ImportError as exc:
        raise ConfigurationError(
            "the LeRobot adapter needs `pyarrow` to read parquet files. "
            "Install with `pip install 'robotrace-dev[lerobot]==0.1.0a5'`."
        ) from exc


def _import_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise ConfigurationError(
            "the LeRobot adapter needs `numpy`. Install with "
            "`pip install 'robotrace-dev[lerobot]==0.1.0a5'` (which pulls it in)."
        ) from exc
    return np


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise ConfigurationError(
            "tiling multi-camera LeRobot videos needs OpenCV. "
            "Install with `pip install 'robotrace-dev[lerobot,video]==0.1.0a5'`. "
            "Or pass `canonical_camera=...` to encode/upload one camera only "
            "(no opencv needed in that path)."
        ) from exc
    return cv2
