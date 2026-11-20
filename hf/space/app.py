"""WholeBodyPose demo (Hugging Face Space).

Whole-body (COCO-WholeBody, 133 keypoints) pose estimation on videos and images with RTMPose,
MediaPipe and ViTPose+, built on the public ``wholebodypose`` API. Run locally with::

    pip install -e ".[demo,vitpose]" && python hf/space/app.py
"""

from __future__ import annotations

import math
import os
import tempfile
import threading
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import ui

from wholebodypose import Video2Pose, __version__, create_model, save_poses
from wholebodypose.models.base import BasePoseModel
from wholebodypose.utils.vision import DrawerPose
from wholebodypose.video.video2pose import BACKGROUND_COLORS, select_largest_person

EXAMPLES = Path(__file__).parent / "examples"
VIDEO_EXAMPLES = ["sign_goodbye.mp4", "sign_book.mp4", "dancing.mp4", "choreography.mp4"]
IMAGE_EXAMPLES = ["person.jpg", "people.jpg", "jump.jpg", "yoga.jpg"]
MAX_SECONDS = float(os.environ.get("WBP_DEMO_MAX_SECONDS", 15))  # longer inputs are truncated
KPT_THR = 0.5  # drawing threshold (scores are binarized by the paper settings)
RESIZE_WIDTH = 640  # paper setting: frames are resized to 640 px wide before inference


def _torch_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


GPU = _torch_cuda()
VITPOSE_SIZE = "huge" if GPU else "base"  # huge is the paper model but too slow on a CPU Space

# value -> (label, backend, overrides of the paper defaults)
BACKENDS: dict[str, tuple[str, str, dict]] = {
    "rtmpose": ("⚡ RTMPose (multi-person)", "rtmpose", {}),
    "mediapipe": ("🔷 MediaPipe (fast on CPU)", "mediapipe", {}),
    "vitpose": (
        f"🎯 ViTPose+ {VITPOSE_SIZE}",
        "vitpose",
        {"model_name": f"ViTPose+_{VITPOSE_SIZE}_coco_wholebody"},
    ),
}
STYLES = [("COCO-WholeBody 133", "coco133"), ("OpenPose-style 135 (neck + pelvis)", "openpose135")]
BACKGROUNDS = [("Original frame", "original"), ("Black", "black"), ("White", "white")]

# --------------------------------------------------------------------------- model cache
_MODELS: dict[str, BasePoseModel] = {}
_MODELS_LOCK = threading.Lock()


def get_model(key: str) -> BasePoseModel:
    """Build each backend once per process (weights are downloaded on first use)."""
    with _MODELS_LOCK:
        if key not in _MODELS:
            _, backend, overrides = BACKENDS[key]
            t0 = time.perf_counter()
            _MODELS[key] = create_model(backend, device="auto", verbose=False, **overrides)
            print(f"[demo] loaded {key} in {time.perf_counter() - t0:.1f}s")
        return _MODELS[key]


def _device(model: BasePoseModel) -> str:
    return str(getattr(model, "device", "cpu"))


def _save_keypoints(folder: Path, stem: str, keypoints, scores, **meta) -> list[str]:
    meta = {"wholebodypose": __version__, **meta}
    return [str(save_poses(folder / f"{stem}.{ext}", keypoints, scores, **meta)) for ext in ("npz", "json")]


def _check_backend(backend: str, multiperson: bool) -> bool:
    if backend not in BACKENDS:
        raise gr.Error(f"Unknown backend {backend!r}; choose one of {list(BACKENDS)}.")
    return bool(multiperson) and backend == "rtmpose"


# --------------------------------------------------------------------------- inference
def estimate_video(
    video: str | None,
    backend: str = "rtmpose",
    style: str = "coco133",
    multiperson: bool = False,
    line_width: int = 2,
    radius: int = 2,
    background: str = "original",
    processing_fps: float = 10,
    progress: gr.Progress = gr.Progress(),  # noqa: B008 - Gradio injects the tracker here
):
    """Pose estimation on a video: rendered H.264 MP4, keypoint files (.npz, .json) and a summary."""
    if not video:
        raise gr.Error("Upload a video, record one with the webcam or pick an example.")
    multiperson = _check_backend(backend, multiperson)

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = n_frames / fps if n_frames > 0 else 0.0
    step = max(1, math.ceil(fps / processing_fps))
    max_frames = math.ceil(MAX_SECONDS * fps / step)
    truncated = duration > MAX_SECONDS
    if truncated:
        gr.Warning(f"The video is {duration:.0f} s long; only the first {MAX_SECONDS:.0f} s are processed.")

    progress(0, desc=f"Loading {BACKENDS[backend][0]}")
    model = get_model(backend)
    runner = Video2Pose(mode_coco=style == "coco133", model=model, verbose=False)
    out_dir = Path(tempfile.mkdtemp(prefix="wbp_"))

    def on_frame(done: int, total: int) -> None:
        progress((done, total or None), desc="Estimating poses", unit="frames")

    t0 = time.perf_counter()
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
    elapsed = time.perf_counter() - t0
    if res["n_frames"] == 0:
        raise gr.Error("Could not decode any frame from this video.")

    people = [len(k) for k in res["total_keypoints"]] if multiperson else [1] * res["n_frames"]
    detected = [int((np.asarray(s) > 0).any()) for s in res["total_scores"]]
    stem = f"{Path(video).stem}_{backend}"
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
    return res["filepath_result"], files, summary


def estimate_image(
    image: np.ndarray | None,
    backend: str = "rtmpose",
    style: str = "coco133",
    multiperson: bool = True,
    line_width: int = 2,
    radius: int = 3,
    background: str = "original",
):
    """Pose estimation on an RGB image: annotated image, keypoint files (.npz, .json) and a summary."""
    if image is None:
        raise gr.Error("Upload an image or pick an example.")
    multiperson = _check_backend(backend, multiperson)
    model = get_model(backend)

    rgb = np.ascontiguousarray(image[..., :3])
    t0 = time.perf_counter()
    keypoints, scores = model.predict(rgb)
    elapsed = time.perf_counter() - t0
    if len(keypoints) and not multiperson:
        i = select_largest_person(keypoints, scores)
        keypoints, scores = keypoints[i : i + 1], scores[i : i + 1]

    # Draw in BGR like the video pipeline so colors match between both tabs.
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if background != "original":
        canvas = np.full_like(canvas, BACKGROUND_COLORS[background])
    if len(keypoints):
        drawer = DrawerPose(mode_coco=style == "coco133")
        canvas = drawer(canvas, keypoints, scores, KPT_THR, int(radius), int(line_width))

    out_dir = Path(tempfile.mkdtemp(prefix="wbp_"))
    size = list(rgb.shape[1::-1])
    files = _save_keypoints(out_dir, f"image_{backend}", keypoints, scores, backend=backend, image_size=size)
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
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), files, summary


# --------------------------------------------------------------------------- UI
def _multiperson_toggle(backend: str):
    if backend == "rtmpose":
        return gr.update(interactive=True, info="RTMPose detects every person in the frame.")
    return gr.update(value=False, interactive=False, info="This backend tracks a single (the largest) person.")


def _controls(multiperson_default: bool, radius_default: int, with_fps: bool):
    """Backend / style / drawing controls shared by both tabs."""
    choices = [(label, key) for key, (label, _, _) in BACKENDS.items()]
    backend = gr.Dropdown(choices, value="rtmpose", label="🤖 Backend")
    style = gr.Radio(STYLES, value="coco133", label="🎨 Skeleton style")
    multiperson = gr.Checkbox(value=multiperson_default, label="👥 Multi-person (RTMPose only)")
    with gr.Accordion("🎛️ Drawing & speed", open=False):
        line_width = gr.Slider(1, 10, 2, step=1, label="Skeleton thickness")
        radius = gr.Slider(1, 10, radius_default, step=1, label="Keypoint radius")
        background = gr.Radio(BACKGROUNDS, value="original", label="Background")
        fps = gr.Slider(1, 30, 10, step=1, label="Processing FPS (lower = faster)") if with_fps else None
    backend.change(_multiperson_toggle, backend, multiperson)
    controls = [backend, style, multiperson, line_width, radius, background]
    return controls + [fps] if with_fps else controls


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="WholeBodyPose Demo") as demo:
        gr.Markdown(ui.HEADER_MD, elem_id="header")
        gr.HTML(ui.BADGES_HTML)

        with gr.Tabs():
            with gr.Tab("🎥 Video"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_in = gr.Video(
                            sources=["upload", "webcam"],
                            label=f"Input video (first {MAX_SECONDS:.0f} s are processed)",
                            height=360,
                            include_audio=False,
                        )
                        video_controls = _controls(multiperson_default=False, radius_default=2, with_fps=True)
                        video_btn = gr.Button("🔍 Estimate pose", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        video_out = gr.Video(label="Skeleton overlay (H.264)", height=360, autoplay=True)
                        video_files = gr.File(label="Keypoints (.npz / .json)", file_count="multiple")
                        video_summary = gr.JSON(label="Summary")
                gr.Examples(
                    [[str(EXAMPLES / name)] for name in VIDEO_EXAMPLES],
                    inputs=[video_in],
                    label="Examples: two ASL signs, a dancer and a multi-person choreography",
                    cache_examples=False,
                )
                video_btn.click(
                    estimate_video,
                    [video_in, *video_controls],
                    [video_out, video_files, video_summary],
                    api_name="video",
                )

            with gr.Tab("📷 Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_in = gr.Image(type="numpy", label="Input image", height=360)
                        image_controls = _controls(multiperson_default=True, radius_default=3, with_fps=False)
                        image_btn = gr.Button("🔍 Estimate pose", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        image_out = gr.Image(label="Skeleton overlay", height=360, format="png")
                        image_files = gr.File(label="Keypoints (.npz / .json)", file_count="multiple")
                        image_summary = gr.JSON(label="Summary")
                gr.Examples(
                    [[str(EXAMPLES / name)] for name in IMAGE_EXAMPLES],
                    inputs=[image_in],
                    cache_examples=False,
                )
                image_btn.click(
                    estimate_image,
                    [image_in, *image_controls],
                    [image_out, image_files, image_summary],
                    api_name="image",
                )

        with gr.Accordion("ℹ️ Backends and output format", open=False):
            gr.Markdown(ui.MODEL_INFO_MD)
        gr.Markdown(ui.ABOUT_MD)
        with gr.Accordion("📖 Python usage, datasets & citation", open=False):
            gr.Markdown(ui.CITATION_MD)
        gr.Markdown(ui.FOOTER_MD)
    return demo


if __name__ == "__main__":
    print(f"[demo] wholebodypose {__version__}, gradio {gr.__version__}, torch CUDA: {GPU}")
    # Warm up the default backend so the first request does not pay for the weight download.
    threading.Thread(target=get_model, args=("rtmpose",), daemon=True).start()
    build_demo().queue(default_concurrency_limit=1, max_size=20).launch(
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="purple"),
        css=ui.CSS,
        max_file_size="50mb",
        show_error=True,
    )
