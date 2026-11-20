"""ViTPose on ONNX Runtime: NumPy pre/post-processing, model wiring, multi-person (slow)."""

import numpy as np
import pytest

from wholebodypose.models.vitpose import model as vp
from wholebodypose.models.vitpose import onnx_runtime as vo


def _gaussian(h, w, x, y, sigma=2.0):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2)).astype(np.float32)


def test_box_to_center_scale_keeps_aspect():
    c, s = vo.box_to_center_scale(0, 0, 960, 640)  # wide box -> height grows to 960 * 256 / 192
    assert np.allclose(c, [480, 320]) and np.allclose(s * 200, [960, 1280])
    c, s = vo.box_to_center_scale(10, 20, 30, 400)  # tall box -> width grows
    assert np.allclose(c, [25, 220]) and np.allclose(s * 200, [300, 400])


def test_preprocess_maps_box_to_input():
    img = np.zeros((256, 192, 3), np.uint8)  # same aspect ratio: the warp is the identity
    img[100, 50] = (255, 0, 0)
    x, c, s = vo.preprocess(img)
    assert x.shape == (3, 256, 192) and x.dtype == np.float32
    assert np.allclose(c, [96, 128]) and np.allclose(s * 200, [192, 256])
    red = x[0] * vo.STD[0] + vo.MEAN[0]
    assert np.unravel_index(np.argmax(red), red.shape) == (100, 50)


def test_decode_heatmaps_peak_shift_and_transform():
    hm = np.zeros((2, 133, 64, 48), np.float32)
    hm[0, 0] = _gaussian(64, 48, 10, 20)
    hm[0, 1] = _gaussian(64, 48, 10.4, 20.4)  # right/down neighbours higher -> +0.25
    hm[1, 0] = _gaussian(64, 48, 0, 0)  # on the border -> no shift
    centers = np.array([[96, 128], [96, 128]], np.float32)
    scales = np.array([[192, 256], [384, 512]], np.float32) / 200
    k, s = vo.decode_heatmaps(hm, centers, scales)
    assert k.shape == (2, 133, 2) and s.shape == (2, 133)
    assert np.allclose(k[0, 0], [10 * 4, 20 * 4])  # stride 4 at scale 1
    assert np.allclose(k[0, 1], [10.25 * 4, 20.25 * 4])
    assert np.allclose(k[1, 0], [96 - 192, 128 - 256])  # heatmap (0, 0) at scale 2
    assert s[0, 0] == pytest.approx(1.0) and s[0, 2] == 0


def test_model_name_validation():
    with pytest.raises(ValueError, match="model_name"):
        vp.VITPoseModel(model_name="ViTPose_tiny", verbose=False)


def test_weight_keys_registered():
    from wholebodypose.weights import WEIGHTS

    for key in vp.MODEL_NAMES.values():
        w = WEIGHTS[key]
        assert w.filename.endswith(".onnx") and w.upstream is None


class _FakeOnnx:
    device = "cpu"

    def __init__(self, *a, **k):
        self.calls = []

    def __call__(self, crops):
        self.calls.append(len(crops))
        n = len(crops)
        k = np.tile(np.array([[5.0, 7.0]], np.float32), (n, 133, 1))
        return k, np.full((n, 133), 0.9, np.float32)


def test_model_full_frame_and_multiperson(monkeypatch, tmp_path):
    monkeypatch.setattr(vp, "get_weight", lambda key, verbose=True: tmp_path / f"{key}.onnx")
    monkeypatch.setattr(vo, "ViTPoseOnnx", _FakeOnnx)
    m = vp.VITPoseModel(verbose=False, model_name="ViTPose+_small_coco_wholebody")
    assert m.device == "cpu"
    k, s = m.predict(np.zeros((50, 40, 3), np.uint8))
    assert k.shape == (1, 133, 2) and np.allclose(k[0, 0], [5, 7]) and s.shape == (1, 133)

    # multi-person: crops are batched in one call and shifted back by their offsets
    m.detector = lambda bgr: np.array([[10, 10, 30, 40], [0, 0, 39, 49]], np.float32)
    m.box_padding = 0.0
    k, s = m.predict(np.zeros((50, 40, 3), np.uint8))
    assert m.model.calls[-1] == 2 and k.shape == (2, 133, 2)
    assert np.allclose(k[:, 0], [[5, 7], [15, 17]])  # largest box first


# ------------------------------------------------------------------------------------------- slow
@pytest.mark.slow
def test_multiperson_two_dancers():
    """Frame 30 of data/videos/videoTest.mp4 shows two people side by side."""
    import cv2

    from tests.conftest import ROOT

    cap = cv2.VideoCapture(str(ROOT / "data" / "videos" / "videoTest.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    ok, bgr = cap.read()
    cap.release()
    assert ok
    m = vp.VITPoseModel(device="cpu", model_name="ViTPose+_small_coco_wholebody", detector="yolox_m", verbose=False)
    k, s = m.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    assert len(k) == 2
    centers = sorted(float(k[i, :17][s[i, :17] > 0.5, 0].mean()) for i in range(2))
    assert centers[1] - centers[0] > bgr.shape[1] * 0.2
