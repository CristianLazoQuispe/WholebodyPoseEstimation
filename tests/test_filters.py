import numpy as np
import pytest

from wholebodypose.filters.filtering_mean import MeanMedianFilter, process_keypoints
from wholebodypose.filters.sequence import PoseSequenceFilter


def test_mean_median_filter_smooths():
    f = MeanMedianFilter(window_size=3)
    outs = [f(np.full((133, 2), v, np.float32)) for v in (0, 0, 9)]
    assert outs[0].mean() == 0 and outs[1].mean() == 0
    assert outs[2].mean() == pytest.approx((3 + 0) / 2)  # (mean + median) / 2
    buf: list = []
    assert process_keypoints([1.0], buf, window_size=2) == [1.0]


def test_sequence_filters():
    T = 10
    k = np.random.default_rng(0).uniform(1, 100, (T, 133, 2))
    k[3] = 0  # invalid frame
    k[6] = k[5]  # no motion
    flt = PoseSequenceFilter(np.arange(91, 112), np.arange(112, 133))
    kept, masks = flt.apply_all_filters(k)
    assert not masks["invalid"][3]
    assert kept.shape[1:] == (133, 2) and len(kept) < T


def test_hand_filter_requires_indices():
    with pytest.raises(ValueError):
        PoseSequenceFilter().filter_by_hand_visibility(np.zeros((3, 133, 2)))


def test_nlms_is_deterministic():
    pytest.importorskip("padasip")
    from wholebodypose.filters.filtering_nlms import FilteringNLMS

    rng = np.random.default_rng(0)
    k = rng.uniform(0, 100, (8, 133, 2))
    s = rng.uniform(0, 1, (8, 133))
    a = FilteringNLMS(num_keypoints=133, seed=1).apply(k, s)
    b = FilteringNLMS(num_keypoints=133, seed=1).apply(k, s)
    np.testing.assert_array_equal(a[0], b[0])
    assert a[0].shape == k.shape and a[1].shape == s.shape
    hi = s >= 0.5
    np.testing.assert_array_equal(a[0][hi], k[hi])  # confident points pass through
