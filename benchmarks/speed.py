"""Steady-state speed of every WholeBodyPose configuration on one image (one person).

    python benchmarks/speed.py --device cpu  --threads 8 -o benchmarks/results/cpu.json
    python benchmarks/speed.py --device cuda             -o benchmarks/results/gpu.json
    python benchmarks/speed.py --device coreml           -o benchmarks/results/coreml.json   # macOS

The frame is resized to 640 px wide (the video pipeline default). Each configuration runs 5 warm-up
frames, then 50 timed frames; the JSON stores the median, 10th and 90th percentile in ms.
MediaPipe runs on CPU only and is skipped with ``--device cuda`` and ``--device coreml``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time

import cv2
import numpy as np

import wholebodypose
from wholebodypose import PoseEstimator

MP_HEAVY, MP_FULL = {"model_complexity": 2}, {"model_complexity": 1}
CONFIGS = [
    ("MediaPipe heavy", "mediapipe", {"pose_config": MP_HEAVY}),
    ("MediaPipe heavy + tracking", "mediapipe", {"pose_config": MP_HEAVY, "image_mode": False}),
    ("MediaPipe full", "mediapipe", {"pose_config": MP_FULL}),
    ("MediaPipe full + tracking", "mediapipe", {"pose_config": MP_FULL, "image_mode": False}),
    (
        "MediaPipe body only, full + tracking",
        "mediapipe",
        {"models": ["pose"], "pose_config": MP_FULL, "image_mode": False},
    ),
    *[(f"RTMPose {m}", "rtmpose", {"mode": m}) for m in ("lightweight", "balanced", "performance")],
    *[
        (f"ViTPose+ {s} + YOLOX-m", "vitpose", {"model_name": f"ViTPose+_{s}_coco_wholebody", "detector": "yolox_m"})
        for s in ("small", "base", "large", "huge")
    ],
    *[
        (f"ViTPose+ {s}, full frame", "vitpose", {"model_name": f"ViTPose+_{s}_coco_wholebody"})
        for s in ("small", "base", "large", "huge")
    ],
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=["cpu", "cuda", "coreml"], default="cpu")
    ap.add_argument("--threads", type=int, default=8, help="ONNX Runtime CPU threads (MediaPipe uses its own)")
    ap.add_argument("--image", default="data/images/image_4.jpeg")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    bgr = cv2.imread(args.image)
    bgr = cv2.resize(bgr, (640, round(bgr.shape[0] * 640 / bgr.shape[1])))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rows = []
    for label, backend, kw in CONFIGS:
        if backend == "mediapipe" and args.device != "cpu":
            continue
        extra = {"intra_op_num_threads": args.threads} if args.device == "cpu" and backend != "mediapipe" else {}
        est = PoseEstimator(backend, device=args.device, verbose=False, **kw, **extra)
        for _ in range(args.warmup):
            est.predict(rgb)
        t = []
        for _ in range(args.frames):
            t0 = time.perf_counter()
            k, _ = est.predict(rgb)
            t.append((time.perf_counter() - t0) * 1e3)
        ms = float(np.median(t))
        rows.append(
            {
                "label": label,
                "backend": backend,
                "options": kw,
                "ms_median": ms,
                "fps": 1e3 / ms,
                "ms_p10": float(np.percentile(t, 10)),
                "ms_p90": float(np.percentile(t, 90)),
                "people": len(k),
            }
        )
        print(f"{label:40s} {ms:7.1f} ms {1e3 / ms:7.1f} FPS", flush=True)
        del est

    import onnxruntime as ort

    meta = {
        "device": args.device,
        "threads": args.threads if args.device == "cpu" else None,
        "image": args.image,
        "frame": list(rgb.shape[1::-1]),
        "frames": args.frames,
        "wholebodypose": wholebodypose.__version__,
        "onnxruntime": ort.__version__,
        "git": os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip() or None,
        "python": platform.python_version(),
        "cpu": platform.processor() or os.uname().machine,
    }
    with open(args.output, "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
