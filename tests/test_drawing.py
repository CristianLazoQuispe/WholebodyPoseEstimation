import numpy as np
import pytest

from wholebodypose.utils.coco133 import coco133
from wholebodypose.utils.vision import DrawerPose, draw_bbox, valid_keypoint_mask


def test_coco133_layout():
    assert len(coco133["keypoint_info"]) == 135  # 133 COCO-WholeBody + neck, pelvis
    names = {v["name"] for v in coco133["keypoint_info"].values()}
    for link in coco133["skeleton_info"].values():
        assert set(link["link"]) <= names


@pytest.mark.parametrize("mode_coco", [True, False])
def test_drawer_draws(mode_coco, image_rgb):
    rng = np.random.default_rng(0)
    k = rng.uniform(20, 200, (2, 133, 2))
    s = np.ones((2, 133))
    out = DrawerPose(mode_coco=mode_coco)(image_rgb.copy(), k, s)
    assert out.shape == image_rgb.shape and not np.array_equal(out, image_rgb)


def test_drawer_respects_threshold(image_rgb):
    k = np.full((133, 2), 50.0)
    out = DrawerPose()(image_rgb.copy(), k, np.zeros(133), kpt_thr=0.5)
    assert np.array_equal(out, image_rgb)


def test_valid_mask_and_bbox(image_rgb):
    k = np.array([[0, 0], [1, 0], [0, 2]], float)
    assert valid_keypoint_mask(k).tolist() == [False, True, True]
    out = draw_bbox(image_rgb.copy(), np.array([[10, 10, 50, 50]]))
    assert not np.array_equal(out, image_rgb)
