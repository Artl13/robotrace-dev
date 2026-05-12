"""Seed the canonical demo episode for the empty-state portal.

Generates a synthetic pick-and-place rollout (top-down 480×270 sim
view, ~6 s @ 30 fps) plus tiny **sensors** (`.npz`) and **actions**
(`.csv`) blobs, uploads all three slots through the SDK to R2, and
prints a one-liner you paste into `apps/web/.env.local` as
`DEMO_EPISODE_VIDEO_KEY`.

What you'll see in the portal:

    - A "tabletop" with a target zone and a cube to grasp
    - A "gripper" rectangle that drives to the cube, closes,
      carries it to the target zone, releases, then returns home
    - A HUD overlay with policy_version / env_version / seed and
      a frame counter — same fields the reproducibility section
      surfaces, so the video matches the metadata 1:1
    - Episode detail also lists **sensors** (`.npz`) and **actions**
      (`.csv`) blobs — synthetic tabular traces aligned to the timeline
    - A subtle "RoboTrace · SAMPLE" watermark so nobody mistakes
      it for production data

It's deliberately not a real-robot clip. We don't ship one because
(a) shipping a binary mp4 in the repo bloats clones for everyone,
and (b) public-domain robot teleop footage with a license clear
enough for marketing-adjacent use is rare. Synthetic + visually
honest beats "looks too real to be a sample".

Запуск:

    cd packages/sdk-python
    .venv/bin/pip install -e ".[video]"            # if opencv missing
    export ROBOTRACE_API_KEY=rt_<id>_<secret>
    export ROBOTRACE_BASE_URL=http://localhost:3000
    .venv/bin/python examples/seed_demo_episode.py

After it finishes:

    1. Copy the printed `DEMO_EPISODE_VIDEO_KEY=...` line into
       `apps/web/.env.local`.
    2. Ctrl+C the dev server and re-run `npm run dev` so the new
       env is picked up.
    3. Open `/portal/episodes` with no real episodes in the table
       — the "Sample run" row is now clickable, and the detail
       page plays this clip via the auth-gated signed-GET-URL
       route.

Re-run any time you want a fresher-looking demo; each run uploads
to a new R2 key, so update `DEMO_EPISODE_VIDEO_KEY` after each
re-seed (we don't reach into the user's R2 to delete the old one
— that's a manual sweep through the Cloudflare dashboard if
needed).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from urllib.parse import urlparse

import robotrace as rt
from robotrace import APIError

# ── canvas / animation constants ─────────────────────────────────────
#
# Kept small on purpose — 480×270 keeps the upload under ~150 KB even
# with the longer duration, which means the smoke flow over the demo
# stays fast on residential connections. The portal player scales it
# up cleanly because the chrome is composed of solid geometry, not
# detailed textures.
WIDTH = 480
HEIGHT = 270
FPS = 30
DURATION_S = 6.0
TOTAL_FRAMES = int(FPS * DURATION_S)

# Phase boundaries in frames. A real RL rollout segments cleanly:
# approach → grasp → transport → release → retreat.
APPROACH_END = int(TOTAL_FRAMES * 0.30)   # 0.0 – 1.8 s
GRASP_END = int(TOTAL_FRAMES * 0.40)      # 1.8 – 2.4 s
TRANSPORT_END = int(TOTAL_FRAMES * 0.70)  # 2.4 – 4.2 s
RELEASE_END = int(TOTAL_FRAMES * 0.80)    # 4.2 – 4.8 s
# Frames after RELEASE_END are the retreat back to home.

# ── colors (BGR — opencv uses BGR not RGB) ───────────────────────────
BG = (28, 30, 34)            # slate background
TABLE = (45, 50, 58)
GRID = (62, 70, 80)
TARGET_ZONE = (60, 100, 60)  # muted green
CUBE = (55, 165, 230)        # warm orange — high contrast vs target
GRIPPER = (235, 235, 240)
GRIPPER_DARK = (180, 180, 190)
TEXT = (220, 220, 225)
TEXT_DIM = (140, 145, 155)
ACCENT = (210, 160, 80)      # cyan-ish, used for the watermark


def _ease_in_out(t: float) -> float:
    """Smooth-step easing — keeps the gripper from teleporting between
    waypoints. Standard 3t² − 2t³ formula on [0, 1]."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _make_demo_mp4(out_path: Path) -> None:
    """Render the pick-and-place rollout to disk."""
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np
    except ImportError as exc:
        sys.stderr.write(
            "this script needs opencv-python + numpy. install with\n"
            "    pip install -e '.[video]'\n"
            f"original error: {exc}\n"
        )
        sys.exit(2)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(FPS), (WIDTH, HEIGHT))
    if not writer.isOpened():
        sys.stderr.write(
            f"opencv VideoWriter failed to open {out_path}. most often "
            "the mp4v codec isn't compiled into your opencv wheel — try "
            "`pip install --upgrade opencv-python`.\n"
        )
        sys.exit(2)

    # ── waypoints (in pixel space, top-down view) ───────────────────
    home = (60, 50)
    cube_pickup = (160, 200)
    target_drop = (380, 80)

    # Dimensions
    cube_size = 22
    target_w, target_h = 60, 60
    gripper_open_w = 36
    gripper_closed_w = 26
    gripper_h = 18

    try:
        for f in range(TOTAL_FRAMES):
            frame = np.full((HEIGHT, WIDTH, 3), BG, dtype=np.uint8)

            # ── tabletop with subtle grid ─────────────────────────
            cv2.rectangle(frame, (20, 20), (WIDTH - 20, HEIGHT - 60), TABLE, -1)
            for gx in range(40, WIDTH - 20, 40):
                cv2.line(frame, (gx, 20), (gx, HEIGHT - 60), GRID, 1)
            for gy in range(40, HEIGHT - 60, 40):
                cv2.line(frame, (20, gy), (WIDTH - 20, gy), GRID, 1)

            # ── target zone (drop area), drawn first so cube sits on it
            tx0 = target_drop[0] - target_w // 2
            ty0 = target_drop[1] - target_h // 2
            tx1 = tx0 + target_w
            ty1 = ty0 + target_h
            # Filled translucent fill via overlay blend
            overlay = frame.copy()
            cv2.rectangle(overlay, (tx0, ty0), (tx1, ty1), TARGET_ZONE, -1)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, dst=frame)
            cv2.rectangle(frame, (tx0, ty0), (tx1, ty1), TARGET_ZONE, 1)
            cv2.putText(
                frame, "TARGET",
                (tx0 + 6, ty1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, TARGET_ZONE, 1, cv2.LINE_AA,
            )

            # ── compute gripper + cube positions for this frame ──
            if f < APPROACH_END:
                # Phase 1: home → cube_pickup
                t = _ease_in_out(f / max(1, APPROACH_END - 1))
                gx = _lerp(home[0], cube_pickup[0], t)
                gy = _lerp(home[1], cube_pickup[1], t)
                cube_x, cube_y = cube_pickup
                gripper_w = gripper_open_w
                phase_label = "APPROACH"
            elif f < GRASP_END:
                # Phase 2: hover at cube and close gripper
                t = (f - APPROACH_END) / max(1, GRASP_END - APPROACH_END - 1)
                gx, gy = cube_pickup
                cube_x, cube_y = cube_pickup
                gripper_w = _lerp(gripper_open_w, gripper_closed_w, _ease_in_out(t))
                phase_label = "GRASP"
            elif f < TRANSPORT_END:
                # Phase 3: cube_pickup → target_drop, cube follows gripper
                t = _ease_in_out((f - GRASP_END) / max(1, TRANSPORT_END - GRASP_END - 1))
                gx = _lerp(cube_pickup[0], target_drop[0], t)
                gy = _lerp(cube_pickup[1], target_drop[1], t)
                cube_x, cube_y = int(gx), int(gy)
                gripper_w = gripper_closed_w
                phase_label = "TRANSPORT"
            elif f < RELEASE_END:
                # Phase 4: hover at target and open gripper
                t = (f - TRANSPORT_END) / max(1, RELEASE_END - TRANSPORT_END - 1)
                gx, gy = target_drop
                cube_x, cube_y = target_drop
                gripper_w = _lerp(gripper_closed_w, gripper_open_w, _ease_in_out(t))
                phase_label = "RELEASE"
            else:
                # Phase 5: retreat to home, cube stays in target zone
                t = _ease_in_out((f - RELEASE_END) / max(1, TOTAL_FRAMES - RELEASE_END - 1))
                gx = _lerp(target_drop[0], home[0], t)
                gy = _lerp(target_drop[1], home[1], t)
                cube_x, cube_y = target_drop
                gripper_w = gripper_open_w
                phase_label = "RETREAT"

            # ── draw cube (skip during transport: cube is *inside* gripper) ──
            cv2.rectangle(
                frame,
                (cube_x - cube_size // 2, cube_y - cube_size // 2),
                (cube_x + cube_size // 2, cube_y + cube_size // 2),
                CUBE, -1,
            )
            cv2.rectangle(
                frame,
                (cube_x - cube_size // 2, cube_y - cube_size // 2),
                (cube_x + cube_size // 2, cube_y + cube_size // 2),
                (40, 110, 160), 1,
            )

            # ── draw gripper (two pads + a connector) ─────────────
            gw = int(gripper_w)
            pad_w = 6
            pad_h = gripper_h
            # Connector / wrist
            cv2.rectangle(
                frame,
                (int(gx) - 2, int(gy) - gripper_h),
                (int(gx) + 2, int(gy) - gripper_h // 2),
                GRIPPER_DARK, -1,
            )
            # Left pad
            cv2.rectangle(
                frame,
                (int(gx) - gw // 2 - pad_w, int(gy) - pad_h // 2),
                (int(gx) - gw // 2, int(gy) + pad_h // 2),
                GRIPPER, -1,
            )
            # Right pad
            cv2.rectangle(
                frame,
                (int(gx) + gw // 2, int(gy) - pad_h // 2),
                (int(gx) + gw // 2 + pad_w, int(gy) + pad_h // 2),
                GRIPPER, -1,
            )

            # ── HUD bar at the bottom ─────────────────────────────
            cv2.rectangle(frame, (0, HEIGHT - 40), (WIDTH, HEIGHT), (18, 20, 24), -1)
            cv2.line(frame, (0, HEIGHT - 40), (WIDTH, HEIGHT - 40), (60, 65, 75), 1)

            t_s = f / FPS
            hud_left = f"policy demo/pap-v3.2.1   env halcyon-cell-rev4   seed 8124"
            hud_right = f"t {t_s:5.2f}s   f {f:>3}/{TOTAL_FRAMES}"
            cv2.putText(
                frame, hud_left,
                (12, HEIGHT - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT, 1, cv2.LINE_AA,
            )
            cv2.putText(
                frame, hud_right,
                (12, HEIGHT - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT_DIM, 1, cv2.LINE_AA,
            )

            # Phase chip — top-right
            chip_pad = 6
            (tw, th), _ = cv2.getTextSize(
                phase_label, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1,
            )
            cv2.rectangle(
                frame,
                (WIDTH - tw - chip_pad * 2 - 12, 8),
                (WIDTH - 12, 8 + th + chip_pad * 2),
                (40, 45, 55), -1,
            )
            cv2.putText(
                frame, phase_label,
                (WIDTH - tw - chip_pad - 12, 8 + th + chip_pad - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT, 1, cv2.LINE_AA,
            )

            # Watermark — top-left, intentionally subtle
            cv2.putText(
                frame, "ROBOTRACE  •  SAMPLE",
                (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, ACCENT, 1, cv2.LINE_AA,
            )

            # Recording dot — pulses while in non-RETREAT phases, like
            # a sim viewer's record indicator. Tiny touch but reads as
            # "this is live data" instead of a static composition.
            if phase_label != "RETREAT":
                pulse = 0.5 + 0.5 * math.sin(f * 0.4)
                radius = 3 + int(pulse * 1.5)
                cv2.circle(frame, (WIDTH - 18, HEIGHT - 22), radius, (60, 70, 230), -1)

            writer.write(frame)
    finally:
        writer.release()


def _write_demo_sensors_npz(path: Path, *, fps: float, n_frames: int) -> None:
    """One row per video frame — enough to exercise the sensors slot."""
    import numpy as np

    t = np.arange(n_frames, dtype=np.float64) / float(fps)
    phase_ix = np.arange(n_frames, dtype=np.float32)
    ee_x = np.linspace(-0.2, 0.35, n_frames, dtype=np.float32)
    ee_y = (0.12 * np.sin(phase_ix * 0.08)).astype(np.float32)
    gripper = np.clip(
        np.where(
            phase_ix < APPROACH_END,
            1.0,
            np.where(phase_ix < GRASP_END, 0.0, 1.0),
        ),
        0.0,
        1.0,
    ).astype(np.float32)
    np.savez(
        path,
        time_s=t,
        ee_xy=np.stack([ee_x, ee_y], axis=-1),
        gripper_open=gripper,
        schema=np.array("robotrace.demo.v1"),
    )


def _write_demo_actions_csv(path: Path, *, fps: float, n_frames: int) -> None:
    """Tiny CSV — `.csv` is not heuristically `sensors`, so `actions=` is valid."""
    lines = ["t_s,cmd_vx,cmd_vy,gripper_target\n"]
    for i in range(n_frames):
        ts = i / fps
        phase = i / max(n_frames - 1, 1)
        cmd_vx = 0.8 * math.sin(phase * math.pi)
        cmd_vy = 0.15 * math.cos(phase * 2 * math.pi)
        g_target = (
            1.0 if i < APPROACH_END else (0.0 if i < TRANSPORT_END else 1.0)
        )
        lines.append(f"{ts:.6f},{cmd_vx:.6f},{cmd_vy:.6f},{g_target:.2f}\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    out_path = Path("/tmp/robotrace_demo_episode.mp4")
    sensors_path = Path("/tmp/robotrace_demo_sensors.npz")
    actions_path = Path("/tmp/robotrace_demo_actions.csv")

    _make_demo_mp4(out_path)
    print(f"wrote {out_path} ({out_path.stat().st_size:,} bytes)")

    _write_demo_sensors_npz(sensors_path, fps=FPS, n_frames=TOTAL_FRAMES)
    print(f"wrote {sensors_path} ({sensors_path.stat().st_size:,} bytes)")

    _write_demo_actions_csv(actions_path, fps=FPS, n_frames=TOTAL_FRAMES)
    print(f"wrote {actions_path} ({actions_path.stat().st_size:,} bytes)")

    rt.init()  # reads ROBOTRACE_API_KEY / ROBOTRACE_BASE_URL from env
    try:
        episode = rt.log_episode(
            name="Sample run · pick & place",
            source="real",
            robot="halcyon-bimanual-01",
            policy_version="demo/pap-v3.2.1",
            env_version="halcyon-cell-rev4",
            git_sha="demo0001",
            seed=8124,
            video=str(out_path),
            sensors=str(sensors_path),
            actions=str(actions_path),
            duration_s=DURATION_S,
            fps=FPS,
            metadata={
                "task": "pick_and_place",
                "scene": "kitchen-tabletop",
                "operator": "demo",
                # Marker so anyone inspecting the metadata knows this
                # is the canonical sample, not customer data.
                "_demo": True,
            },
        )
    except APIError as exc:
        # Surface the R2 XML body if this 4xxs — without it,
        # diagnosing a NoSuchBucket / SignatureDoesNotMatch is awful.
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

    print(f"episode {episode.id}")
    print(f"  status:  {episode.status}")
    print(f"  storage: {episode.storage}")

    if episode.storage != "r2":
        print()
        print("x storage is not 'r2' — R2 env vars likely not picked up.")
        print("  check apps/web/.env.local has R2_ACCOUNT_ID, R2_ACCESS_KEY_ID,")
        print("  R2_SECRET_ACCESS_KEY, R2_BUCKET_EPISODES — then restart")
        print("  `npm run dev` so the new env is loaded.")
        sys.exit(1)

    object_key = _extract_object_key(episode)
    print()
    if object_key:
        print("Add this line to apps/web/.env.local, then restart `npm run dev`:")
        print()
        print(f"    DEMO_EPISODE_VIDEO_KEY={object_key}")
        print()
        print("Once the dev server reboots, an empty `/portal/episodes`")
        print("shows a clickable 'Sample run · pick & place' row that")
        print("opens this clip in the demo episode page.")
    else:
        print("Demo episode uploaded, but we couldn't recover the R2")
        print("object key from the signed PUT URL. Open the new episode")
        print("in the portal, copy the video artifact key shown under")
        print("Artifacts, and paste it into apps/web/.env.local as")
        print("DEMO_EPISODE_VIDEO_KEY=<key>. Restart `npm run dev` after.")
        print()
        print(f"episode page: /portal/episodes/{episode.id}")


def _extract_object_key(episode: rt.Episode) -> str | None:
    """Recover the R2 object key from the signed PUT URL.

    The Episode dataclass doesn't carry the canonical key directly
    (the SDK only tracks the signed URL it was handed). The key is
    deterministic — `episodes/<client>/<episode>/video.mp4` — and
    appears verbatim in the URL path after the bucket segment, so
    we just split on `/episodes/` and reattach the prefix.

    Returns `None` if the URL doesn't match the expected shape, in
    which case the caller falls back to the manual-copy instruction.
    """
    upload = episode.upload_urls.get("video")
    if upload is None:
        return None
    path = urlparse(upload.url).path  # /<bucket>/episodes/<client>/<episode>/video.mp4
    marker = "/episodes/"
    idx = path.find(marker)
    if idx < 0:
        return None
    # Drop the leading slash; we want a key, not a path.
    return path[idx + 1 :]


if __name__ == "__main__":
    main()
