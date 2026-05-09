"""Smoke test: проверяет что R2 действительно сконфигурирован и работает.

Скрипт делает три вещи:
  1. Создаёт мелкий тестовый .mp4 (24 кадра @ 24 fps, 320×240) через
     opencv. Файл занимает ~5 KB — Cloudflare R2 free tier даёт
     1M Class A операций в месяц, так что прогонять можно сколько угодно.
  2. Зовёт `rt.log_episode(video=...)` против локального dev сервера.
  3. Проверяет что `episode.storage == "r2"` — это и есть подтверждение
     что ingest endpoint увидел R2 env переменные и сминтил подписанный
     PUT URL, а SDK успешно залил файл в bucket.

Запуск:

    cd packages/sdk-python
    export ROBOTRACE_API_KEY=rt_<id>_<secret>
    export ROBOTRACE_BASE_URL=http://localhost:3000
    .venv/bin/python examples/r2_smoke.py

Если `storage: r2` в выводе — всё работает, файл лежит в R2 bucket.
Если `storage: unconfigured` — env переменные R2_* не подхватились,
скорее всего dev сервер запущен до того как ты их добавил в .env.local.
Перезапусти Next.js (`Ctrl+C` + `npm run dev`) и попробуй снова.

`opencv-python` нужен для записи .mp4 — поставь через `[video]` extra:

    .venv/bin/pip install -e ".[video]"

Этот скрипт не использует ROS 2 — для тестов ROS 2 адаптера смотри
`tests/test_ros2_adapter.py` (8 тестов с синтетическим bag-ом).
"""

from __future__ import annotations

import sys
from pathlib import Path

import robotrace as rt
from robotrace import APIError


def _make_test_mp4(out_path: Path) -> None:
    """Записать односекундное видео-«радуга» через opencv VideoWriter.

    Каждый кадр — равномерно серый, яркость растёт от 0 до 192. Получается
    короткий fade-in. Достаточно чтобы R2 PUT прошёл, и чтобы потом на
    странице эпизода в портале было видно «настоящее» видео а не чёрный
    квадрат.
    """
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np
    except ImportError as exc:
        sys.stderr.write(
            "this smoke test needs opencv-python + numpy. install with\n"
            "    pip install -e '.[video]'\n"
            f"original error: {exc}\n"
        )
        sys.exit(2)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, 24.0, (320, 240))
    if not writer.isOpened():
        sys.stderr.write(
            f"opencv VideoWriter failed to open {out_path}. most often "
            "the mp4v codec isn't compiled into your opencv wheel — try "
            "`pip install --upgrade opencv-python`.\n"
        )
        sys.exit(2)
    try:
        for i in range(24):
            frame = np.full((240, 320, 3), 8 * i, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    out_path = Path("/tmp/robotrace_r2_smoke.mp4")
    _make_test_mp4(out_path)
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")

    rt.init()  # читает ROBOTRACE_API_KEY / ROBOTRACE_BASE_URL из env
    try:
        episode = rt.log_episode(
            name="r2 smoke test",
            source="sim",
            robot="smoke-rig",
            policy_version="r2-smoke-v1",
            env_version="local-dev",
            git_sha="0000000",
            seed=1,
            video=str(out_path),
            duration_s=1.0,
            fps=24,
            metadata={"task": "r2_smoke", "purpose": "verify R2 wiring"},
        )
    except APIError as exc:
        # R2 возвращает XML body на 403 — `<Code>NoSuchBucket</Code>`,
        # `<Code>SignatureDoesNotMatch</Code>`, и т.п. SDK его захватил
        # в `response_body`, но не печатает по умолчанию (валит на
        # сообщение). Тут мы достаём явно — без этого диагностировать
        # 403 от R2 практически невозможно.
        print()
        print(f"✗ APIError: {exc}")
        print(f"  status_code: {exc.status_code}")
        print("  response body:")
        print("  " + "-" * 60)
        body_text = (
            exc.response_body if isinstance(exc.response_body, str) else repr(exc.response_body)
        )
        for line in body_text.splitlines():
            print(f"  {line}")
        print("  " + "-" * 60)
        sys.exit(1)

    print(f"episode {episode.id}")
    print(f"  status:  {episode.status}")
    print(f"  storage: {episode.storage}")

    if episode.storage == "r2":
        print()
        print("✓ R2 wired correctly — the .mp4 is in your bucket.")
        print(f"  open the episode in the portal: {episode.id}")
    else:
        print()
        print("✗ storage is not 'r2' — R2 env vars likely not picked up.")
        print("  check apps/web/.env.local has R2_ACCOUNT_ID, R2_ACCESS_KEY_ID,")
        print("  R2_SECRET_ACCESS_KEY, R2_BUCKET_EPISODES — then restart")
        print("  `npm run dev` so the new env is loaded.")
        sys.exit(1)


if __name__ == "__main__":
    main()
