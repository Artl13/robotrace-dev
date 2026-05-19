"""Seed a Gymnasium sim episode through the adapter (CLI auth).

Runs CartPole-v1 with a trivial policy, packs observations into
`sensors.npz` and actions into `actions.npz`, encodes `env.render()`
frames to `video.mp4` by default, and uploads via
`robotrace.adapters.gymnasium.upload_rollout`.

Run (after `robotrace login`):

    cd packages/sdk-python

    python3 -m venv .venv
    source .venv/bin/activate

    pip install -e ".[gymnasium,video]"
    pip install 'gymnasium[classic-control]'   # pygame - required for CartPole render

    export ROBOTRACE_BASE_URL=http://localhost:3000

    robotrace login
    python examples/gymnasium_smoke.py

Credentials come from `~/.robotrace/credentials` after
`robotrace login` - no `ROBOTRACE_API_KEY` export needed.

Sensor-only (no mp4): `python examples/gymnasium_smoke.py --no-video`

Then open http://localhost:3000/portal/episodes and click the new
row (source=sim, adapter=gymnasium in metadata).
"""

from __future__ import annotations

import argparse
import sys

import gymnasium as gym

import robotrace as rt
from robotrace import APIError
from robotrace.adapters import gymnasium as rt_gym


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload one Gymnasium rollout as a RoboTrace episode.")
    p.add_argument(
        "--no-video",
        action="store_true",
        help="Skip env.render() / video.mp4 (sensors.npz + actions.npz only).",
    )
    p.add_argument(
        "--env-id",
        default="CartPole-v1",
        help="Gymnasium env id (default: CartPole-v1).",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Step cap for the rollout (default: 200).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="env.reset(seed=...) (default: 42).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    record_video = not args.no_video
    render_mode = "rgb_array" if record_video else None

    print(f"env: {args.env_id!r}  video={record_video}  max_steps={args.max_steps}")

    env = gym.make(args.env_id, render_mode=render_mode)
    try:
        summary = rt_gym.scan_env(env)
        print(summary.report())
        print()

        rt.init()  # ~/.robotrace/credentials after `robotrace login`

        episode = rt_gym.upload_rollout(
            env,
            policy=lambda obs, info: 1,
            name=f"Gymnasium smoke · {args.env_id}",
            policy_version="gymnasium-smoke-v1",
            env_version=args.env_id,
            git_sha="smoke0001",
            seed=args.seed,
            max_steps=args.max_steps,
            record_video=record_video,
            metadata={"task": "gymnasium_smoke", "_smoke": True},
        )
    except gym.error.DependencyNotInstalled as exc:
        print()
        print(f"x {exc}")
        print()
        print("CartPole video needs pygame. Install it, then re-run:")
        print("  pip install 'gymnasium[classic-control]'")
        print("  python examples/gymnasium_smoke.py")
        print()
        print("Or upload sensors/actions only:")
        print("  python examples/gymnasium_smoke.py --no-video")
        sys.exit(1)
    except APIError as exc:
        print()
        print(f"x APIError: {exc}")
        print(f"  status_code: {exc.status_code}")
        print("  response body:")
        print("  " + "-" * 60)
        body_text = (
            exc.response_body
            if isinstance(exc.response_body, str)
            else repr(exc.response_body)
        )
        for line in body_text.splitlines():
            print(f"  {line}")
        print("  " + "-" * 60)
        sys.exit(1)
    finally:
        env.close()

    slots = sorted(episode.upload_urls.keys()) if episode.upload_urls else []

    print(f"episode {episode.id}")
    print(f"  status:   {episode.status}")
    print(f"  storage:  {episode.storage}")
    print(f"  artifacts: {', '.join(slots) or '(none - check R2 config)'}")

    if record_video and "video" not in slots:
        print()
        print("! no video on this episode - R2 may be unconfigured or upload failed.")
    elif not record_video:
        print()
        print("! ran with --no-video - only NPZ artifacts uploaded.")

    if episode.storage != "r2":
        print()
        print("! storage is not 'r2' - episode row exists but bytes may not be in object storage.")
        print("  Check R2 env vars on the Next.js dev server (apps/web/.env.local).")
    else:
        print()
        print("ok - open http://localhost:3000/portal/episodes")


if __name__ == "__main__":
    main()
