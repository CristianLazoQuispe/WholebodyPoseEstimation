"""Whole-body pose on one image: keypoints, a drawn overlay and a JSON file.

    python examples/image_inference.py data/images/image.jpg --backend mediapipe
    python examples/image_inference.py data/images/image.jpg --backend rtmpose --mode lightweight

Needs the backend's extra, e.g. ``pip install "wholebodypose[mediapipe]"``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from wholebodypose import PoseEstimator, save_poses

# COCO-WholeBody 133 index ranges (docs/guide/keypoints.md)
PARTS = {"body": (0, 17), "feet": (17, 23), "face": (23, 91), "left_hand": (91, 112), "right_hand": (112, 133)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=Path)
    ap.add_argument("--backend", "-b", default="mediapipe", choices=["rtmpose", "vitpose", "mediapipe"])
    ap.add_argument("--mode", default=None, help="RTMPose only: performance | balanced | lightweight")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--output", "-o", type=Path, default=Path("outputs"))
    args = ap.parse_args()

    kwargs = {"mode": args.mode} if args.mode else {}
    est = PoseEstimator(args.backend, device=args.device, verbose=False, **kwargs)

    bgr = cv2.imread(str(args.image))
    if bgr is None:
        raise SystemExit(f"cannot read {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # every backend takes RGB

    t0 = time.perf_counter()
    keypoints, scores = est.predict(rgb)  # (N, 133, 2) pixels, (N, 133)
    dt = time.perf_counter() - t0
    print(f"{args.backend}: {len(keypoints)} person(s) in {dt * 1000:.0f} ms, keypoints {keypoints.shape}")

    for i, s in enumerate(scores):
        visible = {name: int((s[a:b] > 0).sum()) for name, (a, b) in PARTS.items()}
        print(f"  person {i}: visible keypoints per part {visible}")

    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"{args.image.stem}_{args.backend}"
    cv2.imwrite(str(args.output / f"{stem}.png"), est.draw(bgr, keypoints, scores))
    # one "frame" holding N people, the same layout as `wholebodypose image`
    save_poses(args.output / f"{stem}.json", [keypoints], [scores], backend=args.backend, source=str(args.image))
    print(f"wrote {args.output / stem}.png and .json")


if __name__ == "__main__":
    main()
