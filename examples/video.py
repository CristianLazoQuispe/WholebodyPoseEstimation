"""Whole-body pose on a video: keypoints saved to .npz and a rendered MP4.

python examples/video.py data/videos/videoTest.mp4 --backend rtmpose --max-frames 60
"""

import argparse
from pathlib import Path

from wholebodypose import PoseEstimator, save_poses

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--backend", "-b", default="rtmpose", choices=["rtmpose", "vitpose", "mediapipe"])
ap.add_argument("--output", "-o", default="outputs")
ap.add_argument("--max-frames", type=int, default=None)
args = ap.parse_args()

est = PoseEstimator(args.backend)
res = est.process_video(args.video, args.output, max_frames=args.max_frames)
out = Path(args.output) / f"{Path(args.video).stem}_{args.backend}.npz"
save_poses(out, res["total_keypoints"], res["total_scores"], backend=args.backend, fps=res["processing_fps"])
print(f"{res['n_frames']} frames -> {out}, {res['filepath_result']}")
