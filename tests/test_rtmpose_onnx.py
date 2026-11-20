"""Vendored YOLOX + RTMW code on ONNX Runtime: unit tests."""

import numpy as np
import pytest

from wholebodypose.models.rtmpose import model as rm
from wholebodypose.models.rtmpose._onnx import rtmw, yolox
from wholebodypose.models.rtmpose._onnx.wholebody import MODE


def test_mode_table_matches_weights():
    from wholebodypose.weights import WEIGHTS

    for name, m in MODE.items():
        assert rm.MODES[name] == (m["det_key"], m["pose_key"], m["det_input_size"], m["pose_input_size"])
        for key, url in ((m["det_key"], m["det"]), (m["pose_key"], m["pose"])):
            assert WEIGHTS[key].upstream == url
    assert MODE["performance"]["pose_key"] == "rtmw_x_384" and MODE["lightweight"]["det_key"] == "yolox_tiny"


def test_nms_keeps_best_of_overlapping_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    assert yolox.nms(boxes, scores, 0.45) == [0, 2]
    dets, keep = yolox.multiclass_nms(boxes, scores[:, None], nms_thr=0.45, score_thr=0.75)
    assert dets.shape == (1, 6) and keep.tolist() == [0]
    assert yolox.multiclass_nms(boxes, scores[:, None], nms_thr=0.45, score_thr=0.95) == (None, None)


def test_yolox_preprocess_letterbox():
    img = np.full((100, 200, 3), 7, np.uint8)
    padded, ratio = yolox.yolox_preprocess(img, (640, 640))
    assert padded.shape == (640, 640, 3) and ratio == pytest.approx(3.2)
    assert (padded[:320] == 7).all() and (padded[320:] == 114).all()
    same, r1 = yolox.yolox_preprocess(np.zeros((640, 640, 3), np.uint8), (640, 640))
    assert r1 == 1.0 and same.shape == (640, 640, 3)


def test_yolox_postprocess_decodes_one_box():
    size = (64, 64)
    n_anchors = sum((64 // s) ** 2 for s in (8, 16, 32))
    out = np.zeros((1, n_anchors, 6), np.float32)
    out[..., 4] = 0.0  # objectness
    # anchor 9 of the stride-8 grid: cell (x=1, y=1); centre offset 0.5 -> (12, 12); size exp(1)*8
    out[0, 9] = [0.5, 0.5, 1.0, 1.0, 1.0, 0.9]
    boxes, cls = yolox.yolox_postprocess(out, 0.5, size, nms_thr=0.45, score_thr=0.5)
    w = np.exp(np.float32(1.0)) * 8
    assert boxes.shape == (1, 4) and cls.tolist() == [0]
    np.testing.assert_allclose(boxes[0], np.array([12 - w / 2, 12 - w / 2, 12 + w / 2, 12 + w / 2]) / 0.5, rtol=1e-6)
    empty, _ = yolox.yolox_postprocess(np.zeros((1, n_anchors, 6), np.float32), 1.0, size, 0.45, 0.5)
    assert len(empty) == 0


def test_bbox_and_affine_round_trip():
    c, s = rtmw.bbox_xyxy2cs(np.array([10.0, 20.0, 110.0, 220.0]), padding=1.25)
    assert np.allclose(c, [60, 120]) and np.allclose(s, [125, 250])
    img = np.zeros((300, 300, 3), np.uint8)
    img[120, 60] = 255  # box centre -> crop centre
    crop, scale = rtmw.top_down_affine((192, 256), s, c, img)
    assert crop.shape == (256, 192, 3) and np.allclose(scale, [187.5, 250])
    y, x = np.unravel_index(np.argmax(crop[..., 0]), crop.shape[:2])
    assert abs(x - 96) <= 1 and abs(y - 128) <= 1
    fwd = rtmw.get_warp_matrix(c, scale, 0, (192, 256))
    inv = rtmw.get_warp_matrix(c, scale, 0, (192, 256), inv=True)
    assert np.allclose(inv @ np.append(fwd @ [60, 120, 1], 1), [60, 120], atol=1e-3)


def test_simcc_maximum():
    x = np.zeros((1, 2, 10), np.float32)
    y = np.zeros((1, 2, 12), np.float32)
    x[0, 0, 3], y[0, 0, 7] = 0.8, 0.6
    locs, vals = rtmw.get_simcc_maximum(x, y)
    assert locs[0, 0].tolist() == [3, 7] and vals[0, 0] == pytest.approx(0.7)
    assert locs[0, 1].tolist() == [-1, -1] and vals[0, 1] == 0


def test_mode_validation():
    with pytest.raises(ValueError, match="mode"):
        rm.RTMPoseModel(mode="fastest", verbose=False)
