"""Command line interface: ``wholebodypose {image,video,backends,download}``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CITE_LINE = "Please cite: Lazo-Quispe et al., WholeBodyPose, ICMV 2025, Proc. SPIE 14114, doi:10.1117/12.3096312"


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", "-b", default="rtmpose", help="rtmpose | vitpose | mediapipe")
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    p.add_argument("--style", choices=["coco133", "openpose135"], default="coco133", help="skeleton drawing style")
    p.add_argument("--quiet", "-q", action="store_true")


def cmd_image(args) -> int:
    import cv2

    from wholebodypose import PoseEstimator, save_poses

    est = PoseEstimator(args.backend, device=args.device, mode_coco=args.style == "coco133", verbose=not args.quiet)
    bgr = cv2.imread(str(args.input))
    if bgr is None:
        print(f"cannot read {args.input}", file=sys.stderr)
        return 1
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    kpts, scores = est.predict(rgb)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(args.input).stem}_{args.backend}"
    drawn = est.draw(bgr, kpts, scores)
    cv2.imwrite(str(out_dir / f"{stem}.png"), drawn)
    # one "frame" holding N people: keypoints (1, N, 133, 2)
    save_poses(out_dir / f"{stem}.{args.format}", [kpts], [scores], backend=args.backend, source=str(args.input))
    print(f"{len(kpts)} person(s) -> {out_dir / stem}.png, .{args.format}")
    return 0


def cmd_video(args) -> int:
    from wholebodypose import PoseEstimator, save_poses

    extra = {}
    if (args.tracking or args.mediapipe_pose) and args.backend != "mediapipe":
        print("--tracking and --mediapipe-pose need -b mediapipe", file=sys.stderr)
        return 2
    if args.tracking:
        extra["image_mode"] = False
    if args.mediapipe_pose:
        extra["pose_config"] = {"model_complexity": 2 if args.mediapipe_pose == "heavy" else 1}
    est = PoseEstimator(
        args.backend,
        device=args.device,
        mode_coco=args.style == "coco133",
        verbose=not args.quiet,
        multiperson=args.multiperson,
        max_people=args.max_people,
        **extra,
    )
    res = est.process_video(
        str(args.input),
        None if args.no_render else args.output,
        multiperson=args.multiperson,
        video_processing_fps=args.fps,
        max_frames=args.max_frames,
        background_plot=args.background,
        resize_width=args.resize_width or None,
    )
    out = Path(args.output) / f"{Path(args.input).stem}_{args.backend}.{args.format}"
    meta = {"backend": args.backend, "source": str(args.input), "fps": res["processing_fps"]}
    meta["frame_size"] = res["frame_size"]
    if "track_ids" in res:  # per-frame person IDs, aligned with the people of each frame
        meta["track_ids"] = [ids.tolist() for ids in res["track_ids"]]
    save_poses(out, res["total_keypoints"], res["total_scores"], **meta)
    print(f"{res['n_frames']} frames -> {out}" + (f", {res['filepath_result']}" if "filepath_result" in res else ""))
    return 0


def cmd_backends(args) -> int:
    import importlib.util

    from wholebodypose.api import BACKENDS, install_command

    for b in BACKENDS.values():
        print(f"{b.name:10s} {install_command(b.extra):40s} {b.description}")
    try:
        import onnxruntime as ort

        print(f"onnxruntime {ort.__version__}: {', '.join(ort.get_available_providers())}")
    except ImportError:
        print("onnxruntime: not installed (pip install onnxruntime or onnxruntime-gpu)")
    optional = {"mediapipe": "mediapipe", "imageio_ffmpeg": "video", "padasip": "filters", "h5py": "datasets"}
    found = [extra for mod, extra in optional.items() if importlib.util.find_spec(mod) is not None]
    print("extras: " + (", ".join(found) if found else "none installed"))
    return 0


def cmd_download(args) -> int:
    from wholebodypose.weights import WEIGHTS, get_weight

    keys = args.keys or list(WEIGHTS)
    for k in keys:
        print(f"{k:16s} {get_weight(k, verbose=not args.quiet)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    from wholebodypose import __version__

    parser = argparse.ArgumentParser(
        prog="wholebodypose",
        description="Whole-body (COCO-133) pose estimation.",
        epilog=CITE_LINE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}\n{CITE_LINE}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("image", help="estimate poses on an image")
    p.add_argument("input", type=Path)
    p.add_argument("--output", "-o", default="outputs")
    p.add_argument("--format", choices=["npz", "json"], default="json")
    _add_common(p)
    p.set_defaults(func=cmd_image)

    p = sub.add_parser("video", help="estimate poses on a video")
    p.add_argument("input", type=Path)
    p.add_argument("--output", "-o", default="outputs")
    p.add_argument("--format", choices=["npz", "json"], default="npz")
    p.add_argument("--multiperson", action="store_true", help="keep every person, with stable track IDs")
    p.add_argument("--max-people", type=int, default=None, help="cap on people (vitpose/mediapipe)")
    p.add_argument("--fps", type=float, default=None, help="sub-sample input to this FPS")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--resize-width", type=int, default=640, help="0 keeps original size")
    p.add_argument("--background", default=None, help="draw on a solid color (black, white, ...)")
    p.add_argument("--no-render", action="store_true", help="only save keypoints")
    p.add_argument("--tracking", action="store_true", help="mediapipe: track across frames (VIDEO mode)")
    p.add_argument("--mediapipe-pose", choices=["heavy", "full"], default=None, help="mediapipe pose model")
    _add_common(p)
    p.set_defaults(func=cmd_video)

    p = sub.add_parser("backends", help="list backends and their install extras")
    p.set_defaults(func=cmd_backends)

    p = sub.add_parser("download", help="pre-download model weights")
    p.add_argument("keys", nargs="*", help="weight keys (default: all)")
    p.add_argument("--quiet", "-q", action="store_true")
    p.set_defaults(func=cmd_download)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
