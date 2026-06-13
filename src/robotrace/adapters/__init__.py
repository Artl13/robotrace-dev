"""Framework adapters.

Adapters slurp third-party dataset / recording formats into the
canonical RoboTrace `log_episode` contract. None of them are loaded
by default - each lives behind an extras pin (`pip install
'robotrace-dev[ros2]'`, `'robotrace-dev[lerobot]'`, …) so the base
install stays slim.

Today:
  * `robotrace.adapters.ros2` - rosbag2 (sqlite3 + mcap) → episode.
  * `robotrace.adapters.lerobot` - Hugging Face LeRobot v2.1
    datasets → one episode per trajectory.
  * `robotrace.adapters.gymnasium` - Gymnasium env rollout → episode.
  * `robotrace.adapters.hdf5` - imitation-learning HDF5 files
    (robomimic `data/demo_*`, ALOHA/ACT `/action` + `/observations`)
    → one episode per trajectory.

Soon: Genesis, Isaac Sim, LeRobot v3.0, RLDS / Open X-Embodiment.
MuJoCo isn't a separate adapter - it logs through the Gymnasium
adapter via `gymnasium[mujoco]` env ids.

Adapters never reach inside the SDK's private modules - they go
through the same public `Client.start_episode` / `upload_*` /
`finalize` surface that user code does. This keeps the contract
between adapter and core honest and means a third-party adapter
written by a user gets the same affordances as a first-party one.
"""

from __future__ import annotations
