import numpy as np
import pytest

from wholebodypose.io import load_poses, save_poses


@pytest.mark.parametrize("suffix", [".npz", ".json"])
def test_roundtrip_dense(tmp_path, suffix):
    k = np.random.default_rng(0).uniform(0, 640, (5, 133, 2)).astype(np.float32)
    s = np.random.default_rng(1).uniform(0, 1, (5, 133)).astype(np.float32)
    path = save_poses(tmp_path / f"p{suffix}", k, s, fps=30.0, backend="rtmpose")
    k2, s2, meta = load_poses(path)
    assert k2.shape == k.shape and s2.shape == s.shape
    np.testing.assert_allclose(k2, k, atol=0.01)
    np.testing.assert_allclose(s2, s, atol=1e-4)
    assert meta["fps"] == 30.0 and meta["backend"] == "rtmpose" and meta["layout"] == "coco_wholebody_133"


@pytest.mark.parametrize("suffix", [".npz", ".json"])
def test_roundtrip_multiperson(tmp_path, suffix):
    k = [np.zeros((n, 133, 2), np.float32) + n for n in (1, 0, 3)]
    s = [np.ones((n, 133), np.float32) for n in (1, 0, 3)]
    k2, s2, _ = load_poses(save_poses(tmp_path / f"m{suffix}", k, s))
    assert [x.shape[0] for x in k2] == [1, 0, 3]
    assert [x.shape[0] for x in s2] == [1, 0, 3]
    np.testing.assert_array_equal(k2[2], k[2])


def test_bad_suffix(tmp_path):
    with pytest.raises(ValueError):
        save_poses(tmp_path / "x.csv", np.zeros((1, 133, 2)), np.zeros((1, 133)))
