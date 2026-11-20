"""Video -> whole-body pose pipeline.

Reads a video, runs a pose backend on every (sub-sampled) frame, optionally draws the skeleton
and writes a browser-playable MP4, and returns the keypoint sequence.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from wholebodypose.models.base import BasePoseModel
from wholebodypose.tracking import PoseTracker, pose_bbox
from wholebodypose.utils.vision import DrawerPose

DEFAULT_LINE_WIDTH = 2
DEFAULT_RADIUS = 1
DEFAULT_KPT_THRESHOLD = 0.5
DEFAULT_RESIZE_WIDTH = 640
MIN_MODE_COUNT = 4
MODE_TOLERANCE = 1.05
LOWER_THRESHOLD_RATIO = 0.75
UPPER_THRESHOLD_RATIO = 1.33

BACKGROUND_COLORS = {  # BGR
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}


# --------------------------------------------------------------------------- helpers
def get_rotation_angle(filepath: str) -> int:
    """Rotation stored in the video metadata (degrees), read with ffprobe; 0 if unknown."""
    exe = shutil.which("ffprobe")
    if exe is None:
        return 0
    try:
        out = subprocess.run(
            [exe, "-v", "quiet", "-print_format", "json", "-show_streams", filepath],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for stream in json.loads(out).get("streams", []):
            if stream.get("codec_type") == "video":
                for item in stream.get("side_data_list", []):
                    if "rotation" in item:
                        return int(item["rotation"])
                if "rotate" in stream.get("tags", {}):
                    return -int(stream["tags"]["rotate"])
    except Exception:  # noqa: BLE001 - metadata is best effort
        pass
    return 0


def rotate_frame(frame: np.ndarray, rotation: int) -> np.ndarray:
    """Undo a metadata rotation (display-matrix convention: positive = counter-clockwise)."""
    rotation = rotation % 360
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame


def clean_outliers(keypoints: np.ndarray) -> np.ndarray:
    """Zero a "phantom cluster" of keypoints collapsed near one corner.

    Some backends place undetected keypoints at a common location; when at least
    ``MIN_MODE_COUNT`` points share the modal x and y and that mode lies far from the pose
    centroid, those points are reset to ``(0, 0)``. Input/Output shape: ``(K, 2)``.
    """
    keypoints = np.array(keypoints, dtype=np.float32, copy=True)
    rounded = np.round(keypoints * 1000) / 1000
    ux, cx = np.unique(rounded[:, 0], return_counts=True)
    uy, cy = np.unique(rounded[:, 1], return_counts=True)
    mode_x, count_x = ux[cx.argmax()] * MODE_TOLERANCE, cx.max()
    mode_y, count_y = uy[cy.argmax()] * MODE_TOLERANCE, cy.max()
    avg_x, avg_y = rounded[:, 0].mean(), rounded[:, 1].mean()

    clustered = count_x >= MIN_MODE_COUNT and count_y >= MIN_MODE_COUNT and mode_y < avg_y * LOWER_THRESHOLD_RATIO
    far_x = mode_x < avg_x * LOWER_THRESHOLD_RATIO or mode_x > avg_x * UPPER_THRESHOLD_RATIO
    if clustered and far_x:
        mask = (rounded[:, 0] <= mode_x) & (rounded[:, 1] <= mode_y)
        keypoints[mask] = 0
    return keypoints


def select_largest_person(keypoints: np.ndarray, scores: np.ndarray) -> int:
    """Index of the person with the largest bounding box over visible keypoints."""
    areas = []
    for kp, sc in zip(keypoints, scores, strict=True):
        vis = sc > 0
        if not vis.any():
            areas.append(0.0)
            continue
        xs, ys = kp[vis, 0], kp[vis, 1]
        areas.append(float((xs.max() - xs.min()) * (ys.max() - ys.min())))
    return int(np.argmax(areas))


def _draw_ids(canvas: np.ndarray, keypoints: np.ndarray, scores: np.ndarray, ids: np.ndarray) -> None:
    """Write ``#id`` above each person's bounding box (in place)."""
    for k, s, tid in zip(keypoints, scores, ids, strict=True):
        box = pose_bbox(k, s, DEFAULT_KPT_THRESHOLD)
        if box is None:
            continue
        x, y = int(box[0]), max(int(box[1]) - 8, 12)
        cv2.putText(canvas, f"#{tid}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, f"#{tid}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def _open_writer(path: str, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    for codec in ("mp4v", "avc1", "MJPG"):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Could not open a video writer for {path}")


# --------------------------------------------------------------------------- pipeline
class Video2Pose:
    """Whole-body pose estimation over a video file.

    Args:
        mode_coco: Draw COCO-WholeBody (133) skeleton; ``False`` adds neck/pelvis points (135).
        model_name: ``'rtmpose'``, ``'vitpose'`` or ``'mediapipe'`` (ignored if ``model`` given).
        verbose: Print progress information.
        model: An already built backend (any :class:`BasePoseModel`), e.g. to share it across calls.
        device: Device for the backend built from ``model_name``.
        **model_kwargs: Overrides of the backend's paper defaults (see ``wholebodypose.api.BACKENDS``).
    """

    def __init__(
        self,
        mode_coco: bool = True,
        model_name: str = "rtmpose",
        verbose: bool = True,
        model: BasePoseModel | None = None,
        device: str = "auto",
        **model_kwargs,
    ) -> None:
        self.mode_coco = mode_coco
        self.verbose = verbose
        self.draw_skeleton = DrawerPose(mode_coco=mode_coco)
        if model is None:
            from wholebodypose.api import create_model

            model = create_model(model_name, device=device, verbose=verbose, **model_kwargs)
        self.model = model
        self.model_name = getattr(model, "name", model_name)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[wholebodypose] {msg}")

    def predict(
        self,
        filepath: str,
        folder_results: str | None = None,
        show: bool = False,
        line_width: int = DEFAULT_LINE_WIDTH,
        radius: int = DEFAULT_RADIUS,
        save_as_mov: bool = False,
        background_plot: str | None = None,
        multiperson: bool = False,
        track: bool = True,
        draw_ids: bool = True,
        video_processing_fps: float | None = None,
        output_video_fps: float | None = None,
        resize_width: int | None = DEFAULT_RESIZE_WIDTH,
        max_frames: int | None = None,
        web_compatible: bool = True,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict:
        """Run pose estimation on a video.

        Args:
            filepath: Input video.
            folder_results: Where to write ``<stem>_<backend>.mp4``; ``None`` skips rendering.
            show: Display frames in an OpenCV window (press ``q`` to stop).
            line_width, radius: Skeleton drawing style.
            save_as_mov: Also convert the rendered video to MOV.
            background_plot: Draw on a solid background (``red|blue|green|white|black``).
            multiperson: Keep every detected person instead of the largest one.
            track: In multi-person mode, assign stable per-person IDs across frames
                (:class:`wholebodypose.tracking.PoseTracker`), returned as ``track_ids``.
            draw_ids: Draw the track ID above each person in the rendered video.
            video_processing_fps: Sub-sample the input to roughly this FPS.
            output_video_fps: FPS of the rendered video (default: the effective processing FPS,
                so playback speed matches the input).
            resize_width: Frames are resized to this width before inference (paper: 640);
                ``None`` keeps the original size. Keypoints are in this resized frame.
            max_frames: Stop after this many processed frames.
            web_compatible: Re-encode the rendered video to H.264 with ffmpeg when available.
            progress: Callback ``progress(processed, expected_total)``.

        Returns:
            Dict with ``filepath, width, height, fps, total_frames, duration, n_frames,
            processing_fps, frame_size`` and ``total_keypoints`` / ``total_scores``:
            arrays ``(T, 133, 2)`` / ``(T, 133)`` in single-person mode, or lists of
            ``(N_t, 133, 2)`` / ``(N_t, 133)`` arrays in multi-person mode. When rendering,
            ``filepath_result`` points to the output video.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)
        vid = cv2.VideoCapture(filepath)
        if not vid.isOpened():
            raise ValueError(f"Failed to open video file: {filepath}")

        # Let OpenCV apply the rotation metadata; otherwise rotate manually.
        rotation = 0
        if not (hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO") and vid.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)):
            rotation = get_rotation_angle(filepath)

        fps = vid.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        results: dict = {
            "filepath": filepath,
            "width": int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": fps,
            "total_frames": total,
            "duration": total / fps if fps else 0.0,
            "backend": self.model_name,
        }
        frame_skip = max(1, math.ceil(fps / video_processing_fps)) if video_processing_fps else 1
        processing_fps = fps / frame_skip
        output_video_fps = output_video_fps or processing_fps
        expected = math.ceil(total / frame_skip) if total > 0 else 0
        if max_frames:
            expected = min(expected, max_frames) if expected else max_frames
        results["processing_fps"] = processing_fps
        self._log(f"{Path(filepath).name}: {fps:.2f} fps, every {frame_skip} frame(s), rotation {rotation}")

        bg = BACKGROUND_COLORS.get(background_plot.lower(), (0, 0, 0)) if background_plot else None
        writer = None
        tmp_path = None
        kp_list: list[np.ndarray] = []
        sc_list: list[np.ndarray] = []
        id_list: list[np.ndarray] = []
        tracker = PoseTracker() if (multiperson and track) else None
        frame_idx = processed = 0
        if show:
            cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL)

        try:
            while True:
                ok, frame = vid.read()
                if not ok or frame is None:
                    break
                if frame_idx % frame_skip:
                    frame_idx += 1
                    continue
                frame_idx += 1

                frame = rotate_frame(frame, rotation)
                if resize_width:
                    h, w = frame.shape[:2]
                    frame = cv2.resize(frame, (resize_width, round(h * resize_width / w)))
                h, w = frame.shape[:2]

                keypoints, scores = self.model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                keypoints = np.array([clean_outliers(k) for k in keypoints], np.float32).reshape(-1, 133, 2)

                ids = None
                if multiperson:
                    kp_list.append(keypoints)
                    sc_list.append(scores)
                    draw_kp, draw_sc = keypoints, scores
                    if tracker is not None:
                        ids = tracker.update(keypoints, scores)
                        id_list.append(ids)
                else:
                    if len(keypoints):
                        i = select_largest_person(keypoints, scores)
                        kp, sc = keypoints[i], scores[i]
                    else:
                        kp, sc = np.zeros((133, 2), np.float32), np.zeros(133, np.float32)
                    kp_list.append(kp)
                    sc_list.append(sc)
                    draw_kp, draw_sc = kp[None], sc[None]

                if folder_results or show:
                    canvas = np.full_like(frame, bg) if bg is not None else frame
                    canvas = self.draw_skeleton(
                        canvas,
                        draw_kp,
                        draw_sc,
                        kpt_thr=DEFAULT_KPT_THRESHOLD,
                        line_width=line_width,
                        radius=radius,
                    )
                    if ids is not None and draw_ids:
                        _draw_ids(canvas, draw_kp, draw_sc, ids)
                    if folder_results:
                        if writer is None:
                            Path(folder_results).mkdir(parents=True, exist_ok=True)
                            fd, tmp_path = tempfile.mkstemp(suffix=".mp4", dir=folder_results)
                            os.close(fd)
                            writer = _open_writer(tmp_path, output_video_fps, (w, h))
                        writer.write(canvas)
                    if show:
                        cv2.imshow("Pose Estimation", canvas)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                processed += 1
                if progress is not None:
                    progress(processed, expected)
                if max_frames and processed >= max_frames:
                    break
        finally:
            vid.release()
            if writer is not None:
                writer.release()
            if show:
                cv2.destroyAllWindows()

        self._log(f"processed {processed} frames")
        results["n_frames"] = processed
        results["frame_size"] = (w, h) if processed else None
        if multiperson:
            results["total_keypoints"], results["total_scores"] = kp_list, sc_list
            if tracker is not None:
                results["track_ids"] = id_list
        else:
            results["total_keypoints"] = np.array(kp_list, np.float32).reshape(-1, 133, 2)
            results["total_scores"] = np.array(sc_list, np.float32).reshape(-1, 133)

        if folder_results and tmp_path:
            out_path = str(Path(folder_results) / f"{Path(filepath).stem}_{self.model_name}.mp4")
            if web_compatible:
                from wholebodypose.video.converters import reencode_for_web

                ok, _ = reencode_for_web(tmp_path, out_path)
                if ok:
                    os.remove(tmp_path)
                else:
                    os.replace(tmp_path, out_path)
            else:
                os.replace(tmp_path, out_path)
            results["filepath_result"] = out_path
            if save_as_mov:
                from wholebodypose.video.converters import convert_mp4_to_mov

                ok, mov = convert_mp4_to_mov(out_path, folder_results)
                if ok:
                    results["filepath_result_mov"] = mov
        return results
