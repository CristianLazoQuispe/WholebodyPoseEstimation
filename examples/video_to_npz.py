"""Video -> training-ready pose sequence (``.npz``) plus an optional rendered MP4.

    python examples/video_to_npz.py data/videos/videoTest.mp4 --backend rtmpose --fps 15
    python examples/video_to_npz.py data/videos/videoTest.mp4 -b mediapipe --max-frames 30 --no-render

The ``.npz`` holds ``keypoints (T, 133, 2)``, ``scores (T, 133)`` and JSON metadata; read it back
with ``wholebodypose.load_poses``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from wholebodypose import PoseEstimator, load_poses, save_poses


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--backend", "-b", default="rtmpose", choices=["rtmpose", "vitpose", "mediapipe"])
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--output", "-o", type=Path, default=Path("outputs"))
    ap.add_argument("--fps", type=float, default=None, help="sub-sample the input to about this FPS")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--resize-width", type=int, default=640, help="paper setting: 640; 0 keeps the original size")
    ap.add_argument("--no-render", action="store_true", help="only save keypoints, no MP4")
    args = ap.parse_args()

    est = PoseEstimator(args.backend, device=args.device, verbose=False)

    def progress(done: int, total: int) -> None:
        if done % 10 == 0 or done == total:
            print(f"\r  frame {done}/{total or '?'}", end="", flush=True)

    res = est.process_video(
        str(args.video),
        None if args.no_render else str(args.output),
        video_processing_fps=args.fps,
        max_frames=args.max_frames,
        resize_width=args.resize_width or None,
        progress=progress,
    )
    print()

    out = save_poses(
        args.output / f"{args.video.stem}_{args.backend}.npz",
        res["total_keypoints"],
        res["total_scores"],
        backend=args.backend,
        source=str(args.video),
        fps=res["processing_fps"],
        frame_size=res["frame_size"],
    )
    keypoints, scores, meta = load_poses(out)
    detected = int((scores.sum(axis=1) > 0).sum())
    print(f"{res['n_frames']} frames at {res['processing_fps']:.1f} fps, person found in {detected}")
    print(f"keypoints {keypoints.shape}, scores {scores.shape}, meta {meta}")
    print(f"-> {out}" + (f"\n-> {res['filepath_result']}" if "filepath_result" in res else ""))


if __name__ == "__main__":
    main()
