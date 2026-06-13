"""LeRobot adapter - Hugging Face LeRobot datasets → RoboTrace episodes.

Three entry points, ordered by how much you want the SDK to do:

    from robotrace.adapters import lerobot

    # 1. Inspect a dataset without downloading any frames.
    summary = lerobot.scan_dataset("lerobot/aloha_static_cups_open")
    print(summary.report())

    # 2. Encode one episode's parquet + per-camera mp4s into the
    #    standard RoboTrace artifact shape (video.mp4, sensors.npz,
    #    actions.npz). No upload.
    encoded = lerobot.encode_episode(
        "lerobot/aloha_static_cups_open",
        episode_index=0,
        output_dir="/tmp/encoded/",
    )

    # 3. One-shot: encode + upload one trajectory as a RoboTrace
    #    episode. The shape 99% of users will reach for.
    lerobot.upload_episode(
        "lerobot/aloha_static_cups_open",
        episode_index=0,
        policy_version="aloha-v1",
        env_version="aloha-cell-1",
    )

    # 4. Bulk: walk every episode (or a slice) and upload each as
    #    its own RoboTrace episode.
    episodes = lerobot.upload_dataset(
        "lerobot/aloha_static_cups_open",
        policy_version="aloha-v1",
        env_version="aloha-cell-1",
        episode_indices=range(0, 50),  # default = all
    )

Each LeRobot trajectory becomes one RoboTrace episode - the natural
mapping. Reproducibility fields (`policy_version`, `env_version`,
`git_sha`, `seed`) come from the caller; the adapter populates the
LeRobot-side identifiers (repo id, dataset codebase_version, episode
index, task description) into `metadata` so the portal shows the
provenance.

Format support
--------------

LeRobot dataset formats **v2.0, v2.1, and v3.0** are supported.

* **v2.0 / v2.1** - one parquet per episode, one mp4 per episode per
  camera. The layout used by the vast majority of public `lerobot/*`
  Hub datasets through 2025. A single camera's mp4 is passed through
  untouched; multiple cameras are tiled with opencv.
* **v3.0** - the multi-episode-shard layout (`lerobot >= 0.3.x`): many
  episodes are concatenated into shared parquet/mp4 files and addressed
  through relational metadata under `meta/episodes/*.parquet`. The
  adapter reads each episode's locator (data shard + row range, and the
  per-camera video shard + `[from, to)` timestamp window), slices the
  data parquet down to the episode's rows, and trims each camera clip
  out of its shared mp4. v3.0 video always needs the `[video]` extra
  (opencv) because there's no per-episode file to copy.

Newer (v4+) or unrecognized `codebase_version` values raise a clear
`ConfigurationError`.

Dependency strategy
-------------------

The adapter does **not** depend on the `lerobot` PyPI package - that
package pulls in torch, torchvision, torchaudio, datasets, pyav, and
several CUDA wheels (multi-GB install footprint). We read the
on-disk format directly with `pyarrow` (parquet) +
`huggingface-hub` (download) + `numpy`, plus our own `[video]`
extra (opencv) for tiling multi-camera mp4s. Total install adds
~20 MB on top of the base SDK - comparable to the ROS 2 adapter.

Storage / download behavior
---------------------------

`huggingface_hub.hf_hub_download` caches per-file in
`~/.cache/huggingface/hub` by default - repeated runs against the
same dataset don't re-download. Per-episode upload is sequential
(no parallel HF downloads) so a flaky network only loses one
episode's worth of progress, never a full dataset.
"""

from __future__ import annotations

from ._classify import ColumnClass, Slot, classify_column
from ._encode import EncodedArtifact, EncodedEpisode, encode_episode
from ._meta import DatasetSummary, EpisodeMeta, VideoLocator, scan_dataset
from ._upload import upload_dataset, upload_episode

__all__ = [
    "scan_dataset",
    "encode_episode",
    "upload_episode",
    "upload_dataset",
    "DatasetSummary",
    "EpisodeMeta",
    "VideoLocator",
    "EncodedEpisode",
    "EncodedArtifact",
    "ColumnClass",
    "Slot",
    "classify_column",
]
