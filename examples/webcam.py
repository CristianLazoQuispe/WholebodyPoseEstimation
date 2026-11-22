"""Live whole-body pose from a webcam (mirrored view, FPS overlay). Press q or Esc to quit.

python examples/webcam.py --backend rtmpose --mode lightweight
"""

import argparse
import time

import cv2

from wholebodypose import PoseEstimator

ap = argparse.ArgumentParser()
ap.add_argument("--backend", "-b", default="rtmpose", choices=["rtmpose", "vitpose", "mediapipe"])
ap.add_argument("--mode", default="lightweight", choices=["lightweight", "balanced", "performance"], help="rtmpose")
ap.add_argument("--device", default="auto", help="auto | cuda | coreml | cpu")
ap.add_argument("--camera", type=int, default=0)
args = ap.parse_args()

kwargs = {"mode": args.mode} if args.backend == "rtmpose" else {}
est = PoseEstimator(args.backend, device=args.device, verbose=False, **kwargs)
device = getattr(est.model, "device", "cpu")

cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    raise SystemExit(f"Cannot open camera {args.camera} (on macOS, allow camera access for your terminal).")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

fps, last = 0.0, time.perf_counter()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    keypoints, scores = est.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if len(keypoints):
        frame = est.draw(frame, keypoints, scores)
    now = time.perf_counter()
    fps = 0.9 * fps + 0.1 / (now - last) if fps else 1 / (now - last)
    last = now
    model = f"{args.backend} {args.mode}" if args.backend == "rtmpose" else args.backend
    label = f"{fps:5.1f} FPS  {model}  {device}  people={len(keypoints)}"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("WholeBodyPose", frame)
    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
        break

cap.release()
cv2.destroyAllWindows()
