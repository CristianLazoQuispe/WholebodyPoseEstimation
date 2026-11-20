"""Real-time whole-body pose from a webcam (or any video file / stream URL), with optional smoothing.

    python examples/webcam_realtime.py                       # webcam 0, MediaPipe; press q to quit
    python examples/webcam_realtime.py --backend rtmpose --mode lightweight --smooth nlms
    python examples/webcam_realtime.py --source data/videos/videoTest.mp4 --no-show --max-frames 50

``--smooth nlms`` runs the paper's NLMS tracker online, frame by frame (needs
``wholebodypose[filters]``); ``--smooth mean`` uses a moving mean/median window.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from wholebodypose import PoseEstimator


class OnlineNLMS:
    """Frame-by-frame version of :class:`wholebodypose.FilteringNLMS` for one person."""

    def __init__(self, num_keypoints: int = 133, mu: float = 1.05) -> None:
        from wholebodypose.filters.filtering_nlms import KeypointMemory, initialize_adaptive_filter

        self.filters = initialize_adaptive_filter(num_keypoints, mu=mu)
        self.memories = [KeypointMemory(window=5) for _ in range(num_keypoints)]
        self.t = 0

    def __call__(self, keypoints: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from wholebodypose.filters.filtering_nlms import track_keypoints

        out = track_keypoints(self.t, keypoints, scores, self.filters, self.memories)
        self.t += 1
        return out


class OnlineMeanMedian:
    def __init__(self, window_size: int = 5) -> None:
        from wholebodypose.filters.filtering_mean import MeanMedianFilter

        self.f = MeanMedianFilter(window_size=window_size)

    def __call__(self, keypoints: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.f(keypoints), scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="camera index, video file or stream URL")
    ap.add_argument("--backend", "-b", default="mediapipe", choices=["rtmpose", "vitpose", "mediapipe"])
    ap.add_argument("--mode", default=None, help="RTMPose only: performance | balanced | lightweight")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--width", type=int, default=640, help="resize frames to this width before inference")
    ap.add_argument("--smooth", choices=["none", "mean", "nlms"], default="none")
    ap.add_argument("--no-show", action="store_true", help="do not open a window (headless run)")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    kwargs = {"mode": args.mode} if args.mode else {}
    if args.backend == "mediapipe":
        kwargs["image_mode"] = False  # MediaPipe tracking mode for a frame stream
    est = PoseEstimator(args.backend, device=args.device, verbose=False, **kwargs)
    smoother = {"nlms": OnlineNLMS, "mean": OnlineMeanMedian}.get(args.smooth, lambda: None)()

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.source}")

    n, fps, t_start = 0, 0.0, time.perf_counter()
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            h, w = bgr.shape[:2]
            if args.width and w != args.width:
                bgr = cv2.resize(bgr, (args.width, round(h * args.width / w)))

            t0 = time.perf_counter()
            keypoints, scores = est.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if len(keypoints):  # follow the first person
                kp, sc = keypoints[0], scores[0]
            else:
                kp, sc = np.zeros((133, 2), np.float32), np.zeros(133, np.float32)
            if smoother is not None:
                kp, sc = smoother(kp, sc)
            inst = 1.0 / max(time.perf_counter() - t0, 1e-6)
            fps = inst if n == 0 else 0.9 * fps + 0.1 * inst
            n += 1

            if not args.no_show:
                canvas = est.draw(bgr, kp[None], sc[None])
                cv2.putText(
                    canvas, f"{args.backend} {fps:.1f} fps", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                cv2.imshow("WholeBodyPose", canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_frames and n >= args.max_frames:
                break
    finally:
        cap.release()
        if not args.no_show:
            cv2.destroyAllWindows()

    total = time.perf_counter() - t_start
    print(f"{n} frames in {total:.1f} s, model {fps:.1f} fps (smoothed), smoothing: {args.smooth}")


if __name__ == "__main__":
    main()
