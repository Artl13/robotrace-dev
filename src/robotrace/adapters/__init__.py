"""Framework adapters.

Adapters slurp third-party dataset / recording formats into the
canonical RoboTrace `log_episode` contract. None of them are loaded
by default — each lives behind an extras pin (`pip install
'robotrace[ros2]'`, `'robotrace[lerobot]'`, …) so the base install
stays slim.

Today: `robotrace.adapters.ros2` (rosbag2 → episode).
Soon: `robotrace.adapters.lerobot`, MuJoCo, Genesis, Isaac Sim.

Adapters never reach inside the SDK's private modules — they go
through the same public `Client.start_episode` / `upload_*` /
`finalize` surface that user code does. This keeps the contract
between adapter and core honest and means a third-party adapter
written by a user gets the same affordances as a first-party one.
"""

from __future__ import annotations
