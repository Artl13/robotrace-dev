"""Streaming-style episode with the context-manager API.

Demonstrates the failure path: an exception inside the `with` block
auto-flips the run to status="failed" and records the failure reason
in metadata, so it shows up correctly on the episode detail page
without any explicit error handling.

Run from inside `packages/sdk-python/`:

    pip install -e .
    export ROBOTRACE_API_KEY=rt_<id>_<secret>
    export ROBOTRACE_BASE_URL=http://localhost:3000

    python examples/with_context.py
"""

from __future__ import annotations

import random

import robotrace as rt
from robotrace import RobotraceError


def main() -> None:
    rt.init()

    # Simulate two runs — one that succeeds, one that crashes.
    for i in range(2):
        try:
            with rt.start_episode(
                name=f"context-demo {i}",
                source="sim",
                policy_version="ctx-demo-v0.0.1",
                env_version="example-env",
                git_sha="deadbee",
                seed=1000 + i,
                fps=60,
                metadata={"trial": i},
                # Don't request signed URLs — we're not uploading
                # any files in this example.
                artifacts=[],
            ) as ep:
                print(f"[{i}] opened episode {ep.id} ({ep.storage})")

                # Pretend to do work…
                if random.random() < 0.5:
                    raise RuntimeError("policy diverged at step 12")

                print(f"[{i}] clean exit — context manager will mark ready")

        except RobotraceError:
            # Anything from the SDK itself — auth, transport, server.
            raise
        except RuntimeError as exc:
            # Our own simulated failure. The context manager already
            # marked the episode failed; we just log and continue.
            print(f"[{i}] crashed: {exc} — episode auto-flipped to 'failed'")


if __name__ == "__main__":
    random.seed()
    main()
