"""WholeBodyPose demo (Hugging Face Space).

Whole-body (COCO-WholeBody, 133 keypoints) pose estimation on videos and images with RTMPose,
MediaPipe and ViTPose+, built on the public ``wholebodypose`` API. Run locally with::

    pip install -e ".[demo,mediapipe,video]" && python hf/space/app.py
"""

from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import ui
from PIL import Image, ImageOps

from wholebodypose import Video2Pose, __version__, create_model, save_poses
from wholebodypose.models.base import BasePoseModel
from wholebodypose.utils.vision import DrawerPose
from wholebodypose.video.converters import ensure_constant_fps
from wholebodypose.video.video2pose import BACKGROUND_COLORS, select_largest_person

EXAMPLES = Path(__file__).parent / "examples"
MAX_SECONDS = float(os.environ.get("WBP_DEMO_MAX_SECONDS", 15))  # longer inputs are truncated
KPT_THR = 0.5  # drawing threshold (scores are binarized by the paper settings)
RESIZE_WIDTH = 640  # paper setting: frames are resized to 640 px wide before inference


def _onnx_cuda() -> bool:
    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()
    except ImportError:
        return False


GPU = _onnx_cuda()
VITPOSE_SIZE = "huge" if GPU else "base"  # huge is the paper model but too slow on a CPU Space


# value -> (label, library backend, overrides of the paper defaults, multi-person capable)
BACKENDS: dict[str, tuple[str, str, dict, bool]] = {
    "rtmpose": ("RTMPose · RTMW-l 384 (paper model)", "rtmpose", {}, True),
    "rtmpose_lite": ("RTMPose-lite · RTMW-m 256 (faster on CPU)", "rtmpose", {"mode": "lightweight"}, True),
    "mediapipe": ("MediaPipe · pose + hands + face (fastest)", "mediapipe", {}, False),
    "vitpose": (
        f"ViTPose+ {VITPOSE_SIZE}",
        "vitpose",
        {"model_name": f"ViTPose+_{VITPOSE_SIZE}_coco_wholebody", "engine": "onnx"},
        False,
    ),
}
# On 2 CPU cores the paper model ran at 0.5-0.9 frames/s, so videos default to the lighter
# RTMW-m (about 2-3x faster on 2 CPU cores). A single image is cheap: use the paper model.
DEFAULT_VIDEO_BACKEND = "rtmpose" if GPU else "rtmpose_lite"
DEFAULT_IMAGE_BACKEND = "rtmpose"
DEFAULT_FPS = 10
STYLES = [("COCO-WholeBody 133", "coco133"), ("OpenPose-style 135 (+ neck, pelvis)", "openpose135")]
BACKGROUNDS = [("Original frame", "original"), ("Black", "black"), ("White", "white")]

# (file, label, backend, multi-person)
VIDEO_EXAMPLES = [
    ("sign_goodbye.mp4", "ASL 'goodbye'", DEFAULT_VIDEO_BACKEND, False),
    ("sign_book.mp4", "ASL 'book'", DEFAULT_VIDEO_BACKEND, False),
    ("dancing.mp4", "dancer (MediaPipe)", "mediapipe", False),
    ("choreography.mp4", "group choreography (multi-person)", DEFAULT_VIDEO_BACKEND, True),
]
IMAGE_EXAMPLES = [
    ("person.jpg", "one person"),
    ("people.jpg", "four people"),
    ("jump.jpg", "jump"),
    ("yoga.jpg", "yoga"),
]

# --------------------------------------------------------------------------- model cache
_MODELS: dict[str, BasePoseModel] = {}
_MODELS_LOCK = threading.Lock()


def get_model(key: str) -> BasePoseModel:
    """Build each backend once per process (weights are downloaded on first use)."""
    with _MODELS_LOCK:
        if key not in _MODELS:
            _, backend, overrides, _ = BACKENDS[key]
            t0 = time.perf_counter()
            try:
                _MODELS[key] = create_model(backend, device="auto", verbose=False, **overrides)
            except Exception as e:  # noqa: BLE001 - shown to the user
                raise gr.Error(f"Could not load {BACKENDS[key][0]}: {type(e).__name__}: {e}") from e
            print(f"[demo] loaded {key} in {time.perf_counter() - t0:.1f}s", flush=True)
        return _MODELS[key]


def _label(key: str) -> str:
    return BACKENDS[key][0]


def _device(model: BasePoseModel) -> str:
    return str(getattr(model, "device", "cpu"))


def _safe_stem(name: str, fallback: str) -> str:
    """File-system and download friendly stem of an uploaded file name."""
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name).stem.removeprefix("muted_")).strip("._")
    return stem[:60] or fallback


def _save_keypoints(folder: Path, stem: str, keypoints, scores, **meta) -> list[str]:
    meta = {"wholebodypose": __version__, **meta}
    return [str(save_poses(folder / f"{stem}.{ext}", keypoints, scores, **meta)) for ext in ("npz", "json")]


def _check_backend(backend: str, multiperson: bool) -> bool:
    if backend not in BACKENDS:
        raise gr.Error(f"Unknown backend {backend!r}; choose one of {list(BACKENDS)}.")
    return bool(multiperson) and BACKENDS[backend][3]


# --------------------------------------------------------------------------- inference
def estimate_video(
    video: str | None,
    backend: str = DEFAULT_VIDEO_BACKEND,
    style: str = "coco133",
    multiperson: bool = False,
    line_width: int = 2,
    radius: int = 2,
    background: str = "original",
    processing_fps: float = DEFAULT_FPS,
    progress: gr.Progress = gr.Progress(),  # noqa: B008 - Gradio injects the tracker here
):
    """Pose estimation on a video: rendered H.264 MP4, keypoint files (.npz, .json) and a summary."""
    if not video:
        raise gr.Error("Upload a video, record one with the webcam or pick an example.")
    multiperson = _check_backend(backend, multiperson)
    processing_fps = float(min(max(processing_fps or DEFAULT_FPS, 1), 30))
    stem = f"{_safe_stem(video, 'video')}_{backend}"

    # webcam recordings are VFR WebM files whose FPS OpenCV misreads; normalize them first
    video, _ = ensure_constant_fps(video, tempfile.mkdtemp(prefix="wbp_in_"))
    cap = cv2.VideoCapture(video)
    opened = cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not opened:
        raise gr.Error("This file could not be opened as a video. Try an MP4 (H.264) or WebM file.")
    duration = n_frames / fps if n_frames > 0 else 0.0
    step = max(1, math.ceil(fps / processing_fps))
    max_frames = math.ceil(MAX_SECONDS * fps / step)
    truncated = duration > MAX_SECONDS
    if truncated:
        gr.Warning(f"The video is {duration:.0f} s long; only the first {MAX_SECONDS:.0f} s are processed.")

    progress(0, desc=f"Loading {_label(backend)}")
    model = get_model(backend)
    runner = Video2Pose(mode_coco=style == "coco133", model=model, verbose=False)
    out_dir = Path(tempfile.mkdtemp(prefix="wbp_"))

    def on_frame(done: int, total: int) -> None:
        progress((done, total or None), desc="Estimating poses", unit="frames")

    t0 = time.perf_counter()
    try:
        res = runner.predict(
            video,
            str(out_dir),
            line_width=int(line_width),
            radius=int(radius),
            background_plot=None if background == "original" else background,
            multiperson=multiperson,
            video_processing_fps=processing_fps,
            resize_width=RESIZE_WIDTH,
            max_frames=max_frames,
            progress=on_frame,
        )
    except Exception as e:  # noqa: BLE001 - shown to the user
        raise gr.Error(f"Pose estimation failed: {type(e).__name__}: {e}") from e
    elapsed = time.perf_counter() - t0
    if res["n_frames"] == 0:
        raise gr.Error("Could not decode any frame from this video.")

    people = [len(k) for k in res["total_keypoints"]] if multiperson else [1] * res["n_frames"]
    detected = [int((np.asarray(s) > 0).any()) for s in res["total_scores"]]
    if not any(detected):
        gr.Warning("No person was detected. Try another backend or a video where the body is fully visible.")
    rendered = out_dir / f"{stem}_pose.mp4"
    shutil.move(res["filepath_result"], rendered)
    files = _save_keypoints(
        out_dir,
        stem,
        res["total_keypoints"],
        res["total_scores"],
        backend=backend,
        fps=res["processing_fps"],
        frame_size=res["frame_size"],
        multiperson=multiperson,
    )
    summary = {
        "backend": backend,
        "device": _device(model),
        "frames": res["n_frames"],
        "frames_with_person": int(sum(detected)),
        "max_people": max(people) if any(detected) else 0,
        "input": {"fps": round(fps, 2), "duration_s": round(duration, 2), "size": [res["width"], res["height"]]},
        "processing_fps": round(res["processing_fps"], 2),
        "truncated_to_s": MAX_SECONDS if truncated else None,
        "time_s": round(elapsed, 2),
        "speed_fps": round(res["n_frames"] / elapsed, 2),
        "keypoints_shape": "(T, N, 133, 2) ragged" if multiperson else [res["n_frames"], 133, 2],
    }
    return str(rendered), files, summary


def _read_image(image) -> tuple[np.ndarray, str]:
    """RGB array and a file stem from a path (UI, API) or an array (Python callers)."""
    if isinstance(image, np.ndarray):
        return np.ascontiguousarray(image[..., :3]), "image"
    try:
        with Image.open(image) as im:
            rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    except Exception as e:  # noqa: BLE001 - shown to the user
        raise gr.Error(f"This file could not be read as an image ({type(e).__name__}).") from e
    return np.ascontiguousarray(rgb), _safe_stem(str(image), "image")


def estimate_image(
    image: str | np.ndarray | None,
    backend: str = DEFAULT_IMAGE_BACKEND,
    style: str = "coco133",
    multiperson: bool = True,
    line_width: int = 2,
    radius: int = 3,
    background: str = "original",
):
    """Pose estimation on an image: annotated PNG, keypoint files (.npz, .json) and a summary."""
    if image is None:
        raise gr.Error("Upload an image or pick an example.")
    multiperson = _check_backend(backend, multiperson)
    rgb, name = _read_image(image)
    model = get_model(backend)

    t0 = time.perf_counter()
    keypoints, scores = model.predict(rgb)
    elapsed = time.perf_counter() - t0
    if len(keypoints) and not multiperson:
        i = select_largest_person(keypoints, scores)
        keypoints, scores = keypoints[i : i + 1], scores[i : i + 1]
    if not len(keypoints):
        gr.Warning("No person was detected. Try another backend or an image where the body is fully visible.")

    # Draw in BGR like the video pipeline so colors match between both tabs.
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if background != "original":
        canvas = np.full_like(canvas, BACKGROUND_COLORS[background])
    if len(keypoints):
        drawer = DrawerPose(mode_coco=style == "coco133")
        canvas = drawer(canvas, keypoints, scores, KPT_THR, int(radius), int(line_width))

    out_dir = Path(tempfile.mkdtemp(prefix="wbp_"))
    stem = f"{name}_{backend}"
    rendered = out_dir / f"{stem}_pose.png"
    cv2.imwrite(str(rendered), canvas)
    size = list(rgb.shape[1::-1])
    files = _save_keypoints(out_dir, stem, keypoints, scores, backend=backend, image_size=size)
    summary = {
        "backend": backend,
        "device": _device(model),
        "frames": 1,
        "people": len(keypoints),
        "visible_keypoints": [int((s > 0).sum()) for s in scores],
        "image_size": size,
        "time_s": round(elapsed, 3),
        "keypoints_shape": list(keypoints.shape),
    }
    return str(rendered), files, summary


# --------------------------------------------------------------------------- results summary
def summary_markdown(summary: dict | None) -> str:
    """Readable one-glance version of the JSON summary returned by the API."""
    if not summary:
        return ""
    key = summary.get("backend", "?")
    label = _label(key).split(" · ")[0] if key in BACKENDS else key
    device = summary.get("device", "cpu").upper()
    if "people" in summary:  # image
        n = summary["people"]
        if n == 0:
            return f"**No person detected** with {label} ({summary['time_s']} s on {device})."
        visible = ", ".join(str(v) for v in summary["visible_keypoints"])
        return (
            f"**{n} {'person' if n == 1 else 'people'}** detected with {label} in **{summary['time_s']} s** "
            f"on {device}  \nVisible keypoints per person (of 133): {visible}  \n"
            f"Image {summary['image_size'][0]}×{summary['image_size'][1]} px · keypoints `{summary['keypoints_shape']}`"
        )
    inp = summary["input"]
    lines = [
        f"**{summary['frames']} frames** processed with {label} in **{summary['time_s']} s** "
        f"({summary['speed_fps']} frames/s on {device})",
        f"Person found in {summary['frames_with_person']}/{summary['frames']} frames"
        + (f" · up to {summary['max_people']} people per frame" if summary["max_people"] > 1 else ""),
        f"Input {inp['duration_s']} s at {inp['fps']} fps · processed at {summary['processing_fps']} fps"
        + (f" · truncated to the first {summary['truncated_to_s']:.0f} s" if summary.get("truncated_to_s") else ""),
    ]
    return "  \n".join(lines)


# --------------------------------------------------------------------------- UI
def _multiperson_toggle(backend: str):
    if BACKENDS.get(backend, ("", "", {}, False))[3]:
        return gr.update(interactive=True, info="Detect and draw every person in the frame.")
    return gr.update(value=False, interactive=False, info="This backend keeps a single (the largest) person.")


def _controls(default_backend: str, multiperson_default: bool, radius_default: int, with_fps: bool):
    """Backend / style / drawing controls shared by both tabs."""
    choices = [(label, key) for key, (label, *_) in BACKENDS.items()]
    backend = gr.Dropdown(choices, value=default_backend, label="Backend")
    multiperson = gr.Checkbox(
        value=multiperson_default, label="Multi-person (RTMPose)", info="Detect and draw every person in the frame."
    )
    with gr.Accordion("Drawing and speed", open=False):
        style = gr.Radio(STYLES, value="coco133", label="Skeleton style (drawing only)")
        fps = (
            gr.Slider(1, 30, DEFAULT_FPS, step=1, label="Processing FPS", info="Lower is faster") if with_fps else None
        )
        with gr.Row():
            line_width = gr.Slider(1, 10, 2, step=1, label="Line width")
            radius = gr.Slider(1, 10, radius_default, step=1, label="Point radius")
        background = gr.Radio(BACKGROUNDS, value="original", label="Background")
    backend.change(_multiperson_toggle, backend, multiperson, api_visibility="private")
    controls = [backend, style, multiperson, line_width, radius, background]
    return controls + [fps] if with_fps else controls


def _results(kind: str):
    out_cls = gr.Video if kind == "video" else gr.Image
    extra = {"autoplay": True} if kind == "video" else {"format": "png"}
    out = out_cls(label="Skeleton overlay", height=380, **extra)
    summary_md = gr.Markdown(elem_classes="summary")
    files = gr.File(label="Keypoints (.npz, .json)", file_count="multiple", height=150)
    with gr.Accordion("Summary (JSON)", open=False):
        summary = gr.JSON(show_label=False)
    return out, files, summary, summary_md


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="WholeBodyPose Demo") as demo:
        gr.HTML(ui.HEADER_HTML)

        with gr.Tabs():
            with gr.Tab("Video"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=320):
                        video_in = gr.Video(
                            sources=["upload", "webcam"],
                            label=f"Input video (first {MAX_SECONDS:.0f} s are used)",
                            height=380,
                            include_audio=False,
                        )
                        video_controls = _controls(DEFAULT_VIDEO_BACKEND, False, 2, with_fps=True)
                        video_btn = gr.Button("Estimate pose", variant="primary", size="lg")
                        backend_in, _, multi_in = video_controls[:3]
                        gr.Examples(
                            [[str(EXAMPLES / f), b, m] for f, _, b, m in VIDEO_EXAMPLES],
                            inputs=[video_in, backend_in, multi_in],
                            visible_columns=[0],
                            label="Examples: " + ", ".join(label for _, label, _, _ in VIDEO_EXAMPLES),
                            cache_examples=False,
                        )
                    with gr.Column(scale=1, min_width=320):
                        video_out, video_files, video_summary, video_md = _results("video")
                video_btn.click(
                    estimate_video,
                    [video_in, *video_controls],
                    [video_out, video_files, video_summary],
                    api_name="video",
                    concurrency_id="inference",
                ).success(summary_markdown, video_summary, video_md, api_visibility="private")

            with gr.Tab("Image"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=320):
                        image_in = gr.Image(type="filepath", label="Input image", height=380)
                        image_controls = _controls(DEFAULT_IMAGE_BACKEND, True, 3, with_fps=False)
                        image_btn = gr.Button("Estimate pose", variant="primary", size="lg")
                        gr.Examples(
                            [[str(EXAMPLES / f)] for f, _ in IMAGE_EXAMPLES],
                            inputs=[image_in],
                            label="Examples: " + ", ".join(label for _, label in IMAGE_EXAMPLES),
                            cache_examples=False,
                        )
                    with gr.Column(scale=1, min_width=320):
                        image_out, image_files, image_summary, image_md = _results("image")
                image_btn.click(
                    estimate_image,
                    [image_in, *image_controls],
                    [image_out, image_files, image_summary],
                    api_name="image",
                    concurrency_id="inference",
                ).success(summary_markdown, image_summary, image_md, api_visibility="private")

            with gr.Tab("About"):
                gr.Markdown(ui.ABOUT_MD, elem_classes="about")

        gr.Markdown(ui.FOOTER_MD)
    return demo


if __name__ == "__main__":
    print(f"[demo] wholebodypose {__version__}, gradio {gr.__version__}, ONNX Runtime CUDA: {GPU}", flush=True)

    # Warm up the default backends so the first request does not pay for the weight download.
    def _warm_up() -> None:
        for key in dict.fromkeys([DEFAULT_VIDEO_BACKEND, DEFAULT_IMAGE_BACKEND]):
            try:
                get_model(key)
            except Exception as e:  # noqa: BLE001 - the request will report it
                print(f"[demo] warm-up of {key} failed: {e}", flush=True)

    threading.Thread(target=_warm_up, daemon=True).start()
    build_demo().queue(default_concurrency_limit=1, max_size=20).launch(
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"),
        css=ui.CSS,
        max_file_size="50mb",
        show_error=True,
    )
