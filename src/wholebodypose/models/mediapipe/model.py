"""MediaPipe whole-body backend (pose + hands + face mesh mapped to COCO-WholeBody).

Two engines are supported:

* ``"solutions"`` - the legacy ``mp.solutions`` API used in the paper (``mediapipe<0.10.30``);
* ``"tasks"`` - the MediaPipe Tasks API (``PoseLandmarker`` heavy, ``HandLandmarker``,
  ``FaceLandmarker``), available in every current MediaPipe release.

``engine="auto"`` picks ``solutions`` when it is installed and ``tasks`` otherwise.

Running mode: this backend defaults to per-frame detection (``image_mode=True``), as in the paper.
MediaPipe's legacy Solutions API defaults to tracking instead (``static_image_mode=False``).
``image_mode=False`` (CLI ``--tracking``) enables tracking (with ``tasks``, the face
step still runs per frame on a head crop).
Pose model: ``pose_config={"model_complexity": 2}`` (default) is ``heavy``, ``1`` is ``full``
(CLI ``--mediapipe-pose``).
"""

from __future__ import annotations

import contextlib
import copy
import os
import weakref

import numpy as np

from wholebodypose.models.base import BasePoseModel
from wholebodypose.models.mediapipe.map_mp2coco import Hand, MPFrame, landmarks_from_list, to_coco133
from wholebodypose.weights import get_weight

DEFAULT_FACE_CONFIG = {
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "max_num_faces": 1,
    "refine_landmarks": False,
}
DEFAULT_HANDS_CONFIG = {
    "min_detection_confidence": 0.3,
    "min_tracking_confidence": 0.3,
    "max_num_hands": 2,
    "model_complexity": 1,
}
DEFAULT_POSE_CONFIG = {
    "min_detection_confidence": 0.3,
    "min_tracking_confidence": 0.3,
    "model_complexity": 2,
    "smooth_landmarks": True,
    "enable_segmentation": False,
    "smooth_segmentation": True,
}
VALID_PARTS = ("hands", "face", "pose")


def _merge(defaults: dict, override: dict | None) -> dict:
    cfg = copy.deepcopy(defaults)
    if override:
        cfg.update({k: v for k, v in override.items() if k in defaults})
    return cfg


def _has_solutions() -> bool:
    try:
        import mediapipe as mp

        return hasattr(mp, "solutions")
    except ImportError:
        return False


class MediapipeModel(BasePoseModel):
    """Single-person whole-body pose with MediaPipe, returned in COCO-WholeBody order.

    Args:
        models: Parts to run, any of ``'hands'``, ``'face'``, ``'pose'``.
        image_mode: ``True`` treats every frame independently; ``False`` enables tracking (video).
        face_config, hands_config, pose_config: Overrides of the ``DEFAULT_*_CONFIG`` dicts.
        use_thresholding, kpt_thr: See :class:`BasePoseModel`. MediaPipe scores are 0/1.
        engine: ``'auto'``, ``'solutions'`` or ``'tasks'``.
        verbose: Print loading information.
        max_people: Maximum number of people (``1`` = paper setting). Values above 1 need the
            ``tasks`` engine; hands and faces are attached to the nearest body.
    """

    name = "mediapipe"

    def __init__(
        self,
        models: list[str] | tuple[str, ...] = VALID_PARTS,
        image_mode: bool = True,
        face_config: dict | None = None,
        hands_config: dict | None = None,
        pose_config: dict | None = None,
        use_thresholding: bool = False,
        kpt_thr: float = 0.5,
        engine: str = "auto",
        verbose: bool = True,
        max_people: int = 1,
    ) -> None:
        super().__init__(use_thresholding=use_thresholding, kpt_thr=kpt_thr)
        try:
            import mediapipe  # noqa: F401
        except ImportError as e:  # pragma: no cover - depends on extras
            raise ImportError("MediaPipe backend needs: pip install 'wholebodypose[mediapipe]'") from e

        if not 1 <= len(models) <= 3 or not all(m in VALID_PARTS for m in models):
            raise ValueError("'models' must contain 1-3 items among 'hands', 'face' and 'pose'.")
        self.models = [m for m in VALID_PARTS if m in models]
        self.image_mode = image_mode
        self.verbose = verbose
        self.face_config = _merge(DEFAULT_FACE_CONFIG, face_config)
        self.hands_config = _merge(DEFAULT_HANDS_CONFIG, hands_config)
        self.pose_config = _merge(DEFAULT_POSE_CONFIG, pose_config)

        if engine == "auto":
            engine = "solutions" if _has_solutions() else "tasks"
        if engine == "solutions" and not _has_solutions():
            raise ImportError("engine='solutions' needs mediapipe<0.10.30; use engine='tasks'.")
        if engine not in ("solutions", "tasks"):
            raise ValueError(f"engine must be 'auto', 'solutions' or 'tasks', got {engine!r}")
        if max_people < 1:
            raise ValueError("max_people must be >= 1")
        if max_people > 1 and engine == "solutions":
            raise ValueError("max_people > 1 needs engine='tasks' (mp.solutions is single-person).")
        self.max_people = max_people
        self.engine = engine
        self._timestamp_ms = 0
        if not verbose:
            os.environ.setdefault("GLOG_minloglevel", "2")
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        if engine == "solutions":
            self._init_solutions()
        else:
            self._init_tasks()
        # Close graphs deterministically (MediaPipe's own __del__ fails at interpreter shutdown).
        self._finalizer = weakref.finalize(self, _close_all, self._graphs())

    # ------------------------------------------------------------------ solutions API
    def _init_solutions(self) -> None:
        import mediapipe as mp

        s = mp.solutions
        if "face" in self.models:
            c = self.face_config
            self.face_model = s.face_mesh.FaceMesh(
                static_image_mode=self.image_mode,
                min_detection_confidence=c["min_detection_confidence"],
                min_tracking_confidence=c["min_tracking_confidence"],
                max_num_faces=c["max_num_faces"],
                refine_landmarks=c["refine_landmarks"],
            )
        if "hands" in self.models:
            c = self.hands_config
            self.hands_model = s.hands.Hands(
                static_image_mode=self.image_mode,
                min_detection_confidence=c["min_detection_confidence"],
                min_tracking_confidence=c["min_tracking_confidence"],
                max_num_hands=c["max_num_hands"],
                model_complexity=c["model_complexity"],
            )
        if "pose" in self.models:
            c = self.pose_config
            self.pose_model = s.pose.Pose(
                static_image_mode=self.image_mode,
                min_detection_confidence=c["min_detection_confidence"],
                min_tracking_confidence=c["min_tracking_confidence"],
                model_complexity=c["model_complexity"],
                smooth_landmarks=c["smooth_landmarks"],
                enable_segmentation=c["enable_segmentation"],
                smooth_segmentation=c["smooth_segmentation"],
            )

    def _run_solutions(self, image_rgb: np.ndarray) -> MPFrame:
        frame = MPFrame()
        if "pose" in self.models:
            r = self.pose_model.process(image_rgb)
            if r.pose_landmarks is not None:
                frame.pose = landmarks_from_list(r.pose_landmarks.landmark)
        if "face" in self.models:
            r = self.face_model.process(image_rgb)
            if r.multi_face_landmarks:
                lm = landmarks_from_list(r.multi_face_landmarks[0].landmark)
                frame.face = lm
        if "hands" in self.models:
            r = self.hands_model.process(image_rgb)
            for i, hand in enumerate(r.multi_hand_landmarks or []):
                lm = landmarks_from_list(hand.landmark)
                label = None
                if r.multi_handedness and i < len(r.multi_handedness):
                    label = r.multi_handedness[i].classification[0].label
                frame.hands.append(Hand(lm.xy, lm.valid, label))
        return frame

    # ------------------------------------------------------------------ tasks API
    def _init_tasks(self) -> None:
        from mediapipe.tasks.python import BaseOptions, vision

        mode = vision.RunningMode.IMAGE if self.image_mode else vision.RunningMode.VIDEO

        def opts(key):
            return BaseOptions(model_asset_path=str(get_weight(key, verbose=self.verbose)))

        if "pose" in self.models:
            c = self.pose_config
            self.pose_model = vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=opts("mp_pose_heavy" if c["model_complexity"] >= 2 else "mp_pose_full"),
                    running_mode=mode,
                    num_poses=self.max_people,
                    min_pose_detection_confidence=c["min_detection_confidence"],
                    min_tracking_confidence=c["min_tracking_confidence"],
                )
            )
        if "hands" in self.models:
            c = self.hands_config
            self.hands_model = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=opts("mp_hand"),
                    running_mode=mode,
                    num_hands=max(c["max_num_hands"], 2 * self.max_people),
                    min_hand_detection_confidence=c["min_detection_confidence"],
                    min_tracking_confidence=c["min_tracking_confidence"],
                )
            )
        if "face" in self.models:
            c = self.face_config
            # IMAGE mode: the face is searched on a head crop that moves every frame.
            self.face_model = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=opts("mp_face"),
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=max(c["max_num_faces"], self.max_people),
                    min_face_detection_confidence=c["min_detection_confidence"],
                    min_tracking_confidence=c["min_tracking_confidence"],
                )
            )

    def _run_tasks(self, image_rgb: np.ndarray) -> MPFrame:
        import mediapipe as mp

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image_rgb))
        if self.image_mode:

            def call(model):
                return model.detect(image)
        else:
            self._timestamp_ms += 33
            ts = self._timestamp_ms

            def call(model):
                return model.detect_for_video(image, ts)

        frame = MPFrame()
        if "pose" in self.models:
            r = call(self.pose_model)
            frame.poses = [landmarks_from_list(p) for p in (r.pose_landmarks or [])][: self.max_people]
            frame.pose = frame.poses[0] if frame.poses else None
        if "face" in self.models:
            # one head-crop search per body (or one full-image search without bodies)
            faces = [self._detect_face_tasks(image_rgb, p) for p in (frame.poses or [None])]
            frame.faces = [f for f in faces if f is not None]
            frame.face = frame.faces[0] if frame.faces else None
        if "hands" in self.models:
            r = call(self.hands_model)
            for i, hand in enumerate(r.hand_landmarks or []):
                lm = landmarks_from_list(hand)
                label = r.handedness[i][0].category_name if r.handedness and i < len(r.handedness) else None
                frame.hands.append(Hand(lm.xy, lm.valid, label))
        return frame

    def _detect_face_tasks(self, image_rgb: np.ndarray, pose):
        """Face landmarks on a head crop (from the body pose), falling back to the full image.

        The face detector of the Tasks API misses small faces in full-body shots; cropping the
        head first makes it work at typical sign-language framing.
        """
        import mediapipe as mp

        h, w = image_rgb.shape[:2]
        crops = []
        head = range(0, 11)  # BlazePose nose, eyes, ears, mouth
        if pose is not None and pose.valid[list(head)].sum() >= 3:
            pts = pose.xy[list(head)][pose.valid[list(head)]] * [w, h]
            cx, cy = pts.mean(axis=0)
            size = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 16) * 3.0
            x0, y0 = int(max(cx - size / 2, 0)), int(max(cy - size / 2, 0))
            x1, y1 = int(min(cx + size / 2, w)), int(min(cy + size / 2, h))
            if x1 - x0 > 8 and y1 - y0 > 8:
                crops.append((x0, y0, x1, y1))
        crops.append((0, 0, w, h))
        for x0, y0, x1, y1 in crops:
            crop = np.ascontiguousarray(image_rgb[y0:y1, x0:x1])
            r = self.face_model.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
            if r.face_landmarks:
                lm = landmarks_from_list(r.face_landmarks[0])
                lm.xy = (lm.xy * [x1 - x0, y1 - y0] + [x0, y0]) / [w, h]
                return lm
        return None

    # ------------------------------------------------------------------ public API
    def process(self, image_rgb: np.ndarray) -> MPFrame:
        """Run the selected MediaPipe parts and return engine-independent landmarks."""
        if self.engine == "solutions":
            return self._run_solutions(image_rgb)
        return self._run_tasks(image_rgb)

    def _predict(self, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = image_rgb.shape[:2]
        keypoints, scores = to_coco133(self.process(image_rgb), w, h)
        keep = scores.any(axis=1)  # drop empty people (nothing detected)
        return keypoints[keep], scores[keep]

    def _graphs(self) -> list:
        return [getattr(self, n) for n in ("pose_model", "hands_model", "face_model") if hasattr(self, n)]

    def close(self) -> None:
        """Release the MediaPipe graphs."""
        self._finalizer()


def _close_all(graphs: list) -> None:
    for g in graphs:
        with contextlib.suppress(Exception):  # already closed / interpreter shutting down
            g.close()
        # MediaPipe's __del__ calls close() again, which crashes during interpreter shutdown.
        with contextlib.suppress(Exception):
            g.close = lambda: None
