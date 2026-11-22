"""High-level API: one entry point for every backend."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from wholebodypose.models.base import BasePoseModel


@dataclass(frozen=True)
class Backend:
    """Registry entry for a pose backend."""

    name: str
    target: str  # "module:Class"
    extra: str  # pip extra that installs its dependencies ("" = core install)
    description: str
    defaults: dict = field(default_factory=dict)  # paper settings
    multiperson: bool = False  # can return several people (some need an option, see description)


BACKENDS: dict[str, Backend] = {
    b.name: b
    for b in [
        Backend(
            "rtmpose",
            "wholebodypose.models.rtmpose.model:RTMPoseModel",
            "",
            "YOLOX + RTMW (OpenMMLab), multi-person, ONNX Runtime",
            {"mode": "performance", "use_thresholding": True, "filter_noise": True},
            multiperson=True,
        ),
        Backend(
            "vitpose",
            "wholebodypose.models.vitpose.model:VITPoseModel",
            "",
            "ViTPose+ top-down, ONNX Runtime; multi-person with detector='yolox_m'",
            {"model_name": "ViTPose+_huge_coco_wholebody", "use_thresholding": True, "kpt_thr": 0.5},
            multiperson=True,
        ),
        Backend(
            "mediapipe",
            "wholebodypose.models.mediapipe.model:MediapipeModel",
            "mediapipe",
            "MediaPipe pose + hands + face, CPU friendly; multi-person with max_people>1",
            {"use_thresholding": True, "kpt_thr": 0.5},
            multiperson=True,
        ),
    ]
}


# Extra settings that turn a backend into multi-person mode.
MULTIPERSON_DEFAULTS: dict[str, dict] = {
    "rtmpose": {},
    "vitpose": {"detector": "yolox_m"},
    "mediapipe": {"engine": "tasks", "max_people": 4},
}


def install_command(extra: str) -> str:
    """``pip install`` command for a backend extra (``""`` = the core package)."""
    return f"pip install 'wholebodypose[{extra}]'" if extra else "pip install wholebodypose"


def available_backends() -> list[str]:
    """Names of all registered backends."""
    return list(BACKENDS)


def create_model(
    backend: str = "rtmpose",
    device: str = "auto",
    verbose: bool = True,
    multiperson: bool = False,
    max_people: int | None = None,
    **kwargs,
) -> BasePoseModel:
    """Build a backend with the paper defaults, overridden by ``kwargs``.

    ``multiperson=True`` applies ``MULTIPERSON_DEFAULTS`` (YOLOX detector for ViTPose,
    ``max_people`` for MediaPipe; RTMPose is always multi-person). ``max_people`` caps the
    number of people for ViTPose and MediaPipe.

    Example:
        >>> model = create_model("mediapipe")            # doctest: +SKIP
        >>> keypoints, scores = model.predict(image_rgb)  # doctest: +SKIP
    """
    key = backend.lower()
    if key not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}. Choose one of {available_backends()}.")
    spec = BACKENDS[key]
    module_name, cls_name = spec.target.split(":")
    try:
        cls = getattr(importlib.import_module(module_name), cls_name)
    except ImportError as e:
        raise ImportError(f"Backend '{key}' needs: {install_command(spec.extra)} ({e})") from e
    params = {**spec.defaults, **(MULTIPERSON_DEFAULTS[key] if multiperson else {}), **kwargs}
    if max_people is not None and key in ("vitpose", "mediapipe"):
        params["max_people"] = max_people
    if key != "mediapipe":
        params.setdefault("device", device)
    params.setdefault("verbose", verbose)
    return cls(**params)


class PoseEstimator:
    """Whole-body pose estimation with a single, backend-agnostic interface.

    Args:
        backend: ``'rtmpose'``, ``'vitpose'`` or ``'mediapipe'``.
        device: ``'auto'``, ``'cuda'``, ``'coreml'`` or ``'cpu'``.
        mode_coco: Visualization style. ``True``: COCO-WholeBody 133 keypoints;
            ``False``: adds neck and pelvis points (135, OpenPose-like).
        verbose: Print loading information.
        multiperson: Detect every person (see :func:`create_model`); videos then return one
            array per frame plus stable ``track_ids``.
        max_people: Upper bound on people for ViTPose / MediaPipe.
        **kwargs: Backend-specific overrides (e.g. ``model_name='ViTPose+_base_coco_wholebody'``).

    Example:
        >>> from wholebodypose import PoseEstimator           # doctest: +SKIP
        >>> est = PoseEstimator("rtmpose")                   # doctest: +SKIP
        >>> kpts, scores = est.predict(image_rgb)            # (N, 133, 2), (N, 133)
        >>> res = est.process_video("in.mp4", "out/")        # res["total_keypoints"]: (T, 133, 2)
    """

    def __init__(
        self,
        backend: str = "rtmpose",
        device: str = "auto",
        mode_coco: bool = True,
        verbose: bool = True,
        multiperson: bool = False,
        max_people: int | None = None,
        **kwargs,
    ) -> None:
        self.backend = backend.lower()
        self.multiperson = multiperson
        self.model = create_model(
            self.backend, device=device, verbose=verbose, multiperson=multiperson, max_people=max_people, **kwargs
        )
        self.mode_coco = mode_coco
        self.verbose = verbose

    def predict(self, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Keypoints ``(N, 133, 2)`` and scores ``(N, 133)`` for an RGB ``uint8`` image."""
        return self.model.predict(image_rgb)

    __call__ = predict

    def predict_image(self, path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        """Read an image from disk and run :meth:`predict`."""
        import cv2

        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(path)
        return self.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def draw(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
        kpt_thr: float = 0.5,
        radius: int = 2,
        line_width: int = 2,
    ) -> np.ndarray:
        """Return a copy of ``image`` with the skeleton(s) drawn.

        Skeleton colors are defined in BGR (OpenCV convention), so pass a BGR image to get the
        documented colors (as the CLI does before ``cv2.imwrite``); RGB works but swaps red/blue.
        """
        from wholebodypose.utils.vision import DrawerPose

        img = np.ascontiguousarray(image).copy()
        if len(keypoints) == 0:
            return img
        return DrawerPose(mode_coco=self.mode_coco)(img, keypoints, scores, kpt_thr, radius, line_width)

    def process_video(self, filepath: str, folder_results: str | None = None, **kwargs) -> dict:
        """Run the video pipeline; see :meth:`wholebodypose.video.Video2Pose.predict`."""
        from wholebodypose.video.video2pose import Video2Pose

        kwargs.setdefault("multiperson", self.multiperson)
        runner = Video2Pose(mode_coco=self.mode_coco, model=self.model, verbose=self.verbose)
        return runner.predict(filepath, folder_results, **kwargs)
