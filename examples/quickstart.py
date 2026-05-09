"""Smallest possible RoboTrace SDK demo.

Run from inside `packages/sdk-python/`:

    pip install -e .
    export ROBOTRACE_API_KEY=rt_<id>_<secret>      # mint in /admin/clients/<id>
    export ROBOTRACE_BASE_URL=http://localhost:3000

    python examples/quickstart.py

The episode appears in /admin/episodes immediately. Without R2
configured on the deployment, this still works — `storage` will be
"unconfigured" in the response and no artifacts get uploaded
(the example doesn't pass any).
"""

from __future__ import annotations

import robotrace as rt


def main() -> None:
    # init() reads ROBOTRACE_API_KEY / ROBOTRACE_BASE_URL from env
    # if you don't pass them. Calling it explicitly is the same.
    rt.init()

    episode = rt.log_episode(
        name="quickstart smoke run",
        source="sim",
        robot="example-rig",
        # Reproducibility — fill these in real code, even when seeded
        # from a notebook. Future you re-rolling this run depends on it.
        policy_version="quickstart-v0.0.1",
        env_version="example-env",
        git_sha="0000000",
        seed=42,
        # Run details
        duration_s=1.5,
        fps=30,
        metadata={"task": "quickstart", "purpose": "smoke"},
    )

    print(f"Logged episode {episode.id}")
    print(f"  status: {episode.status}")
    print(f"  storage: {episode.storage}")


if __name__ == "__main__":
    main()
