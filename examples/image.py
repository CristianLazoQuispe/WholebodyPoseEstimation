"""Whole-body pose on one image.

python examples/image.py data/images/image.jpg --backend rtmpose
"""

import argparse
from pathlib import Path

import cv2

from wholebodypose import PoseEstimator

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("--backend", "-b", default="rtmpose", choices=["rtmpose", "vitpose", "mediapipe"])
ap.add_argument("--output", "-o", default="outputs/pose.png")
args = ap.parse_args()

est = PoseEstimator(args.backend)
bgr = cv2.imread(args.image)
keypoints, scores = est.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))  # (N, 133, 2), (N, 133)
print(f"{len(keypoints)} person(s), keypoints {keypoints.shape}")
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(args.output, est.draw(bgr, keypoints, scores))
print(f"-> {args.output}")
