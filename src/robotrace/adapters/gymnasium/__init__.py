"""Gymnasium adapter - env rollout → RoboTrace episode.

Three entry points, ordered by how much you want the SDK to do:

    from robotrace.adapters import gymnasium as rt_gym
    import gymnasium as gym

    env = gym.make("CartPole-v1", render_mode="rgb_array")

    # 1. Inspect an env without running a rollout.
    summary = rt_gym.scan_env(env)
    print(summary.report())

    # 2. Run one episode and write artifacts to disk. No upload.
    encoded = rt_gym.encode_rollout(
        env,
        "/tmp/rollout/",
        policy=lambda obs, info: 1,
        seed=42,
    )

    # 3. One-shot: rollout + encode + upload + finalize.
    rt_gym.upload_rollout(
        env,
        policy=my_policy,
        policy_version="cartpole-v1",
        env_version="cartpole-v1",
        seed=42,
    )

Video comes from ``env.render()`` only - create the env with
``render_mode="rgb_array"`` and install the ``[video]`` extra for mp4
encoding. MuJoCo and other sims work through Gymnasium env ids once
you install their optional deps (``gymnasium[mujoco]``, etc.) - no
separate MuJoCo adapter is required for most teams.

Roadmap (not yet shipped):
  * ``record()`` context manager for mid-training logging
  * ``upload_rollouts()`` batch helper for RL training loops
"""

from __future__ import annotations

from ._encode import EncodedArtifact, EncodedRollout, Policy, encode_rollout
from ._scan import EnvSummary, scan_env
from ._upload import upload_rollout

__all__ = [
    "scan_env",
    "encode_rollout",
    "upload_rollout",
    "EnvSummary",
    "EncodedRollout",
    "EncodedArtifact",
    "Policy",
]
