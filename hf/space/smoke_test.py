"""End-to-end smoke test of the demo through its public API (gradio_client).

Usage::

    python hf/space/smoke_test.py http://127.0.0.1:7860
    python hf/space/smoke_test.py https://cristianlazoquispe-wholebodyposedemo.hf.space --backends mediapipe rtmpose

Calls the ``/video`` endpoint once per backend on an example video and the ``/image`` endpoint
once, and checks that an MP4, an ``.npz`` and a ``.json`` come back with ``frames > 0``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from gradio_client import Client, handle_file

EXAMPLES = Path(__file__).parent / "examples"


def _path(value) -> str:
    """Gradio returns files as paths or as ``{"video": path}`` / ``{"path": path}`` dicts."""
    if isinstance(value, dict):
        value = value.get("video") or value.get("path") or value.get("name")
    return str(value)


def _check_files(files, prefix: str) -> None:
    paths = [Path(_path(f)) for f in files]
    suffixes = sorted(p.suffix for p in paths)
    assert suffixes == [".json", ".npz"], f"{prefix}: expected .npz + .json, got {paths}"
    with np.load(next(p for p in paths if p.suffix == ".npz")) as data:
        assert data["keypoints"].shape[-2:] == (133, 2), f"{prefix}: bad keypoints {data['keypoints'].shape}"


def check_video(client: Client, backend: str, video: Path, fps: float) -> dict:
    out, files, summary = client.predict(
        video=handle_file(str(video)),
        backend=backend,
        processing_fps=fps,
        api_name="/video",
    )
    mp4 = Path(_path(out))
    assert mp4.suffix == ".mp4" and mp4.stat().st_size > 0, f"{backend}: no mp4 ({mp4})"
    _check_files(files, backend)
    assert summary["frames"] > 0, f"{backend}: no frames processed ({summary})"
    assert summary["frames_with_person"] > 0, f"{backend}: nobody detected ({summary})"
    return summary


def check_image(client: Client, backend: str, image: Path) -> dict:
    out, files, summary = client.predict(image=handle_file(str(image)), backend=backend, api_name="/image")
    assert Path(_path(out)).stat().st_size > 0, "image: no output"
    _check_files(files, f"image/{backend}")
    assert summary["frames"] == 1 and summary["people"] > 0, f"image: nobody detected ({summary})"
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url", help="Demo URL, e.g. http://127.0.0.1:7860 or the *.hf.space URL")
    parser.add_argument("--backends", nargs="+", default=["rtmpose", "mediapipe", "vitpose"])
    parser.add_argument("--video", type=Path, default=EXAMPLES / "sign_goodbye.mp4")
    parser.add_argument("--image", type=Path, default=EXAMPLES / "people.jpg")
    parser.add_argument("--fps", type=float, default=10, help="processing FPS sent to the video endpoint")
    parser.add_argument("--hf-token", default=None, help="token for a private Space")
    args = parser.parse_args(argv)

    client = Client(args.url, token=args.hf_token, verbose=False)
    failures = 0
    jobs = [("video", b) for b in args.backends] + [("image", args.backends[0])]
    for kind, backend in jobs:
        t0 = time.perf_counter()
        try:
            if kind == "video":
                summary = check_video(client, backend, args.video, args.fps)
            else:
                summary = check_image(client, backend, args.image)
        except Exception as e:  # noqa: BLE001 - report every endpoint, fail at the end
            failures += 1
            print(f"FAIL {kind:5s} {backend:9s} {type(e).__name__}: {e}")
            continue
        print(f"OK   {kind:5s} {backend:9s} {time.perf_counter() - t0:6.1f}s  {summary}")
    print("smoke test", "FAILED" if failures else "passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
