"""Real backends on a real image. Downloads weights: run with ``pytest -m slow``."""

import numpy as np
import pytest

from wholebodypose import PoseEstimator

pytestmark = pytest.mark.slow

# RTMPose and ViTPose only need the core install (onnxruntime).
CASES = [
    ("rtmpose", {"mode": "lightweight"}, "onnxruntime"),
    ("mediapipe", {}, "mediapipe"),
    ("vitpose", {"model_name": "ViTPose+_small_coco_wholebody"}, "onnxruntime"),
]
IDS = ["rtmpose", "mediapipe", "vitpose"]


@pytest.mark.parametrize("backend,kwargs,module", CASES, ids=IDS)
def test_backend_on_image(backend, kwargs, module, sample_image_path):
    pytest.importorskip(module)
    est = PoseEstimator(backend, device="cpu", verbose=False, **kwargs)
    k, s = est.predict_image(sample_image_path)
    assert k.shape[1:] == (133, 2) and s.shape[1:] == (133,) and len(k) >= 1
    body = s[0, :17] > 0
    assert body.sum() >= 12
    # the person sits in the middle of the 960x640 photo
    cx, cy = k[0, :17][body].mean(axis=0)
    assert 250 < cx < 520 and 150 < cy < 400


def test_backends_agree(sample_image_path):
    pytest.importorskip("mediapipe")
    a, sa = PoseEstimator("rtmpose", device="cpu", verbose=False, mode="lightweight").predict_image(sample_image_path)
    b, sb = PoseEstimator("mediapipe", verbose=False).predict_image(sample_image_path)
    vis = (sa[0, :17] > 0) & (sb[0, :17] > 0)
    err = np.median(np.linalg.norm(a[0, :17][vis] - b[0, :17][vis], axis=1))
    assert err < 30, f"median body disagreement {err:.1f}px"


@pytest.mark.parametrize("backend,kwargs,module", CASES, ids=IDS)
def test_multiperson_two_dancers(backend, kwargs, module):
    """Frame 30 of data/videos/videoTest.mp4 shows two people side by side."""
    import cv2

    pytest.importorskip(module)
    from tests.conftest import ROOT

    cap = cv2.VideoCapture(str(ROOT / "data" / "videos" / "videoTest.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    ok, bgr = cap.read()
    cap.release()
    assert ok
    est = PoseEstimator(backend, device="cpu", verbose=False, multiperson=True, **kwargs)
    k, s = est.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    assert len(k) == 2
    centers = sorted(float(k[i, :17][s[i, :17] > 0, 0].mean()) for i in range(2))
    assert centers[1] - centers[0] > bgr.shape[1] * 0.2  # two distinct people
