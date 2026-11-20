"""Compare the vendored ONNX Runtime RTMPose/YOLOX code with rtmlib, number by number.

    python scripts/validate_onnx_stack.py compare --out compare.json      # outputs (needs rtmlib)
    python scripts/validate_onnx_stack.py latency --threads 8 --out latency.json

``compare`` runs both implementations on the same frames with the same ``.onnx`` files and reports, per
case, whether keypoints, scores and YOLOX boxes are bit-identical (``np.array_equal``) and otherwise the
maximum absolute difference. Frames: ``data/images/*``, the images in ``hf/space/examples`` and the first
``--frames`` frames of every video in ``data/videos`` and ``hf/space/examples``. Every RTMPose mode is run
on the BGR frame (library default) and on the RGB frame (``legacy_rgb_input=True``, ICMV 2025 preset).
The ViTPose multi-person path (YOLOX boxes -> ViTPose crops) is compared with rtmlib's YOLOX as detector.

``latency`` times vendored vs rtmlib ``Wholebody`` per frame on CPU. Pin the cores with ``taskset``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXT = {".jpg", ".jpeg", ".png"}
VIDEO_EXT = {".mp4", ".webm", ".avi", ".mov"}


def load_frames(n_video_frames: int) -> list[tuple[str, np.ndarray]]:
    """``(name, bgr)`` for every test image and the first frames of every test video."""
    frames = []
    folders = [ROOT / "data" / "images", ROOT / "data" / "videos", ROOT / "hf" / "space" / "examples"]
    for folder in folders:
        for path in sorted(folder.iterdir()):
            ext = path.suffix.lower()
            if ext in IMAGE_EXT:
                frames.append((str(path.relative_to(ROOT)), cv2.imread(str(path))))
            elif ext in VIDEO_EXT:
                cap = cv2.VideoCapture(str(path))
                for i in range(n_video_frames):
                    ok, bgr = cap.read()
                    if not ok:
                        break
                    frames.append((f"{path.relative_to(ROOT)}#{i}", bgr))
                cap.release()
    return frames


def diff(a, b) -> dict:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return {"equal": False, "shape": [list(a.shape), list(b.shape)]}
    eq = bool(np.array_equal(a, b))
    mad = 0.0 if eq or a.size == 0 else float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
    return {"equal": eq, "max_abs_diff": mad, "dtype": [str(a.dtype), str(b.dtype)]}


def build_pair(mode: str, threads: int):
    from rtmlib import Wholebody as RtmWholebody

    from wholebodypose.models.rtmpose._onnx import Wholebody
    from wholebodypose.models.rtmpose.model import MODES
    from wholebodypose.weights import get_weight

    det_key, pose_key, det_size, pose_size = MODES[mode]
    det, pose = str(get_weight(det_key, verbose=False)), str(get_weight(pose_key, verbose=False))
    ours = Wholebody(det, pose, det_size, pose_size, device="cpu", intra_op_num_threads=threads)
    ref = RtmWholebody(det=det, det_input_size=det_size, pose=pose, pose_input_size=pose_size, device="cpu")
    if threads:  # same thread count for the reference sessions
        import onnxruntime as ort

        for tool in (ref.det_model, ref.pose_model):
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = threads
            tool.session = ort.InferenceSession(tool.onnx_model, sess_options=opts, providers=["CPUExecutionProvider"])
    return ours, ref


def cmd_compare(args) -> None:
    from wholebodypose.models.rtmpose.model import MODES, RTMPoseModel

    frames = load_frames(args.frames)
    print(f"{len(frames)} frames")
    rows = []
    for mode in args.modes or list(MODES):
        ours, ref = build_pair(mode, args.threads)
        for name, bgr in frames:
            for colour, img in (("bgr", bgr), ("rgb_legacy", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))):
                b_ours, b_ref = ours.det_model(img), ref.det_model(img)
                k_ours, s_ours = ours(img)
                k_ref, s_ref = ref(img)
                row = {
                    "mode": mode,
                    "frame": name,
                    "input": colour,
                    "people": int(len(k_ref)),
                    "boxes": diff(b_ours, b_ref),
                    "keypoints": diff(k_ours, k_ref),
                    "scores": diff(s_ours, s_ref),
                }
                rows.append(row)
                ok = all(row[k]["equal"] for k in ("boxes", "keypoints", "scores"))
                if not ok or args.verbose:
                    print(json.dumps(row))
        # end-to-end through RTMPoseModel (float32 outputs, thresholds, filter_noise), both engines
        for legacy in (False, True):
            kw = {"mode": mode, "device": "cpu", "verbose": False, "legacy_rgb_input": legacy}
            kw.update(use_thresholding=True, filter_noise=True)
            m_onnx = RTMPoseModel(engine="onnx", intra_op_num_threads=args.threads, **kw)
            m_rtm = RTMPoseModel(engine="rtmlib", **kw)
            m_rtm.model = ref  # same thread settings as the vendored sessions
            for name, bgr in frames:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                (ka, sa), (kb, sb) = m_onnx.predict(rgb), m_rtm.predict(rgb)
                row = {"mode": mode, "frame": name, "input": f"RTMPoseModel(legacy_rgb_input={legacy})"}
                row.update(people=int(len(kb)), keypoints=diff(ka, kb), scores=diff(sa, sb), boxes={"equal": True})
                rows.append(row)
                if not (row["keypoints"]["equal"] and row["scores"]["equal"]) or args.verbose:
                    print(json.dumps(row))
        print(f"{mode}: done")

    if not args.skip_vitpose:
        rows += compare_vitpose(frames, args)

    summary = {}
    for r in rows:
        key = f"{r['mode']} / {r['input'] if r['mode'].startswith('vitpose') else r['input'].split('(')[0]}"
        s = summary.setdefault(key, {"cases": 0, "identical": 0, "max_abs_diff_kpts": 0.0, "max_abs_diff_scores": 0.0})
        s["cases"] += 1
        s["identical"] += int(all(r[k]["equal"] for k in ("boxes", "keypoints", "scores")))
        s["max_abs_diff_kpts"] = max(s["max_abs_diff_kpts"], r["keypoints"].get("max_abs_diff", float("inf")))
        s["max_abs_diff_scores"] = max(s["max_abs_diff_scores"], r["scores"].get("max_abs_diff", float("inf")))
    from importlib.metadata import version

    import onnxruntime as ort

    out = {
        "onnxruntime": ort.__version__,
        "rtmlib": version("rtmlib"),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "threads": args.threads,
        "n_frames": len(frames),
        "summary": summary,
        "rows": rows,
    }
    print(json.dumps(summary, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


def compare_vitpose(frames, args) -> list[dict]:
    """ViTPose multi-person: vendored YOLOX vs rtmlib YOLOX as the person detector."""
    from rtmlib import YOLOX as RtmYOLOX

    from wholebodypose.models.vitpose.model import DETECTORS, VITPoseModel
    from wholebodypose.weights import get_weight

    rows = []
    for det_name in ("yolox_m", "yolox_tiny"):
        m = VITPoseModel(
            device="cpu",
            model_name="ViTPose+_small_coco_wholebody",
            engine="onnx",
            detector=det_name,
            verbose=False,
            intra_op_num_threads=args.threads,
        )
        key, size = DETECTORS[det_name]
        ref_det = RtmYOLOX(str(get_weight(key, verbose=False)), model_input_size=size, device="cpu")
        ref_det.score_thr = m.detector.score_thr
        ours_det = m.detector
        for name, bgr in frames:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            b_ours, b_ref = ours_det(bgr), ref_det(bgr)
            m.detector = ours_det
            ka, sa = m.predict(rgb)
            m.detector = ref_det
            kb, sb = m.predict(rgb)
            row = {"mode": f"vitpose_small+{det_name}", "frame": name, "input": "multiperson", "people": int(len(kb))}
            row.update(boxes=diff(b_ours, b_ref), keypoints=diff(ka, kb), scores=diff(sa, sb))
            rows.append(row)
            if not all(row[k]["equal"] for k in ("boxes", "keypoints", "scores")) or args.verbose:
                print(json.dumps(row))
        m.detector = ours_det
        print(f"vitpose + {det_name}: done")
    return rows


def cmd_latency(args) -> None:
    from wholebodypose.models.rtmpose.model import MODES

    frames = [f for f in load_frames(args.frames) if f[0].endswith(("image.jpg", "#0", "#1", "#2"))]
    out = {"threads": args.threads, "frames": [f[0] for f in frames], "modes": {}}
    for mode in args.modes or list(MODES):
        ours, ref = build_pair(mode, args.threads)
        res = {}
        for fn in (ours, ref):
            for _, bgr in frames[:2]:  # warm-up
                fn(bgr)
        times = {"vendored": [], "rtmlib": []}
        # Paired timing: each frame is run by both implementations back to back, and the order flips
        # every round, so machine load affects both alike.
        pairs = (("vendored", ours), ("rtmlib", ref))
        for r in range(args.repeats):
            for _, bgr in frames:
                for label, fn in pairs if r % 2 == 0 else pairs[::-1]:
                    t0 = time.perf_counter()
                    fn(bgr)
                    times[label].append(time.perf_counter() - t0)
        for label, ts in times.items():
            ts = np.asarray(ts) * 1000
            res[label] = {"median_ms": float(np.median(ts)), "mean_ms": float(ts.mean()), "n": int(ts.size)}
        ratio = np.asarray(times["vendored"]) / np.asarray(times["rtmlib"])
        res["vendored_over_rtmlib"] = {"median": float(np.median(ratio)), "p25": float(np.percentile(ratio, 25)),
                                       "p75": float(np.percentile(ratio, 75))}  # fmt: skip
        out["modes"][mode] = res
        print(mode, json.dumps(res))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("compare", "latency"):
        s = sub.add_parser(name)
        s.add_argument("--modes", nargs="*", default=None)
        s.add_argument("--threads", type=int, default=8, help="onnxruntime intra-op threads (0 = default)")
        s.add_argument("--frames", type=int, default=10, help="frames per video")
        s.add_argument("--out", default=None)
    sub.choices["compare"].add_argument("--skip-vitpose", action="store_true")
    sub.choices["compare"].add_argument("--verbose", action="store_true")
    sub.choices["latency"].add_argument("--repeats", type=int, default=5)
    args = p.parse_args()
    {"compare": cmd_compare, "latency": cmd_latency}[args.cmd](args)


if __name__ == "__main__":
    main()
