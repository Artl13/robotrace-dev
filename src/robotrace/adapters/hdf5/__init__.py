"""HDF5 adapter - imitation-learning demos → RoboTrace episodes.

Reads the two dominant on-disk layouts for imitation / behavior-cloning
datasets and turns each trajectory into a RoboTrace episode:

* **robomimic** - one HDF5 file with many trajectories under
  ``data/demo_0``, ``data/demo_1``, … Each demo holds ``actions``, an
  ``obs`` group (proprioception + camera image stacks), ``rewards``,
  ``dones``, ``states``. One demo → one episode.
* **ALOHA / ACT** - one file per episode: ``/action`` plus an
  ``/observations`` group (``qpos``, ``qvel``, ``effort``, and
  ``images/<camera>`` stacks). The whole file → one episode.

Four entry points, ordered by how much you want the SDK to do::

    from robotrace.adapters import hdf5

    # 1. Inspect a file without decoding frames.
    summary = hdf5.scan_file("demo.hdf5", fps=20)
    print(summary.report())

    # 2. Encode one trajectory to video.mp4 / sensors.npz / actions.npz.
    encoded = hdf5.encode_episode("demo.hdf5", "/tmp/out/", episode_index=0)

    # 3. One-shot: encode + upload + finalize one trajectory.
    hdf5.upload_episode(
        "episode_0.hdf5",
        policy_version="act-v1",
        env_version="aloha-cell-1",
        fps=50,
    )

    # 4. Bulk: upload every demo in a multi-demo robomimic file.
    episodes = hdf5.upload_dataset("low_dim.hdf5", policy_version="bc-v3")

Slot mapping
------------

Dataset names within a trajectory are routed by `classify_dataset`:
``action*`` → actions; image stacks (``observations/images/*``,
``*_image``) → video; ``rewards`` / ``dones`` / ``success`` →
episode metadata; everything else (``qpos``, ``robot0_eef_pos``,
``states``, custom keys) → sensors. The classifier is a pure function
- call it with arbitrary names to pin behavior.

Reproducibility fields (`policy_version`, `env_version`, `git_sha`,
`seed`) come from the caller; the adapter populates HDF5-side
provenance (source filename, layout, demo key, trajectory length,
episode outcome) into `metadata`.

Dependency strategy
--------------------

The adapter depends on `h5py` only (a thin libhdf5 wrapper, ~few MB) -
**not** robomimic, lerobot, or torch. Image stacks encode to mp4 via
the separate ``[video]`` extra (opencv); a sensor-only file never pays
that cost. Install with ``pip install 'robotrace-dev[hdf5]'`` (add
``,video`` for camera streams).

Timestamps
----------

HDF5 imitation files rarely store a per-step clock, so timestamps are
synthesised from ``fps`` (uniform spacing by construction). Pass the
real capture rate - ALOHA is typically 50, robomimic 20 - via ``fps=``;
otherwise the adapter assumes 30 and marks ``fps_assumed`` in the
episode metadata.
"""

from __future__ import annotations

from ._classify import DatasetClass, Slot, classify_dataset
from ._encode import EncodedArtifact, EncodedEpisode, ImageColor, encode_episode
from ._scan import EpisodeRef, FileSummary, scan_file
from ._upload import upload_dataset, upload_episode

__all__ = [
    "DatasetClass",
    "EncodedArtifact",
    "EncodedEpisode",
    "EpisodeRef",
    "FileSummary",
    "ImageColor",
    "Slot",
    "classify_dataset",
    "encode_episode",
    "scan_file",
    "upload_dataset",
    "upload_episode",
]
