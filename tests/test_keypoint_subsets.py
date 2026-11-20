import numpy as np
import pytest

from wholebodypose.utils.keypoint_subsets import SIGNERT_96, select_keypoints


def test_signert96_is_a_valid_unique_subset():
    assert len(SIGNERT_96) == 96
    assert len(set(SIGNERT_96)) == 96
    assert all(0 <= i < 135 for i in SIGNERT_96)
    # no legs, feet, jaw or nose bridge; mid-hip excluded
    assert not set(SIGNERT_96) & set(range(11, 40))
    assert 134 not in SIGNERT_96
    assert SIGNERT_96[53] == 133 and SIGNERT_96[-1] == 71


def test_select_keypoints_axes():
    data = np.arange(7 * 2 * 135, dtype=np.float64).reshape(7, 2, 135)
    out = select_keypoints(data)
    assert out.shape == (7, 2, 96)
    np.testing.assert_array_equal(out[:, :, 53], data[:, :, 133])
    out2 = select_keypoints(np.transpose(data, (0, 2, 1)), axis=1)
    np.testing.assert_array_equal(np.transpose(out2, (0, 2, 1)), out)


def test_select_keypoints_rejects_wrong_layout():
    with pytest.raises(ValueError):
        select_keypoints(np.zeros((3, 2, 133)))
