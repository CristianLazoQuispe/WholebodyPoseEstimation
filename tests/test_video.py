import cv2
import numpy as np
import pytest

from tests.conftest import FakePoseModel
from wholebodypose.video.video2pose import Video2Pose, clean_outliers, rotate_frame, select_largest_person


def test_single_person_sequence(synthetic_video, tmp_path):
    model = FakePoseModel(n_people=2)
    res = Video2Pose(model=model, verbose=False).predict(str(synthetic_video), str(tmp_path / "out"))
    assert res["n_frames"] == 20 == model.calls
    assert res["total_keypoints"].shape == (20, 133, 2)
    assert res["total_scores"].shape == (20, 133)
    assert res["frame_size"] == (640, 480)  # resized to the paper's 640 px width
    out = res["filepath_result"]
    cap = cv2.VideoCapture(out)
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 20
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(10, abs=0.5)  # playback speed preserved
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (640, 480)
    cap.release()


def test_subsampling_and_max_frames(synthetic_video):
    v = Video2Pose(model=FakePoseModel(), verbose=False)
    res = v.predict(str(synthetic_video), video_processing_fps=5, resize_width=None)
    assert res["n_frames"] == 10 and res["processing_fps"] == pytest.approx(5)
    assert res["frame_size"] == (160, 120)
    res = v.predict(str(synthetic_video), max_frames=3)
    assert res["n_frames"] == 3


def test_multiperson_and_background(synthetic_video, tmp_path):
    seen = []
    res = Video2Pose(model=FakePoseModel(n_people=2), verbose=False).predict(
        str(synthetic_video),
        str(tmp_path),
        multiperson=True,
        background_plot="white",
        progress=lambda i, n: seen.append((i, n)),
    )
    assert isinstance(res["total_keypoints"], list) and res["total_keypoints"][0].shape == (2, 133, 2)
    assert seen[-1] == (20, 20)
    cap = cv2.VideoCapture(res["filepath_result"])
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.mean() > 200  # mostly white canvas


def test_no_detection_gives_zeros(synthetic_video):
    class Empty(FakePoseModel):
        def _predict(self, image_rgb):
            return np.zeros((0, 133, 2)), np.zeros((0, 133))

    res = Video2Pose(model=Empty(), verbose=False).predict(str(synthetic_video), max_frames=2)
    assert res["total_keypoints"].shape == (2, 133, 2) and not res["total_keypoints"].any()


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        Video2Pose(model=FakePoseModel(), verbose=False).predict("nope.mp4")


def test_select_largest_person():
    small = np.zeros((133, 2)) + [[10, 10]]
    small[0] = [20, 20]
    big = np.zeros((133, 2)) + [[0, 0]]
    big[0] = [100, 100]
    k = np.stack([small, big])
    s = np.ones((2, 133))
    assert select_largest_person(k, s) == 1
    assert select_largest_person(k, np.zeros((2, 133))) == 0


def test_clean_outliers_removes_corner_cluster():
    rng = np.random.default_rng(1)
    k = rng.uniform(300, 400, size=(133, 2)).astype(np.float32)
    k[:10] = [5.0, 5.0]  # phantom cluster near the origin
    out = clean_outliers(k)
    assert np.all(out[:10] == 0)
    assert np.array_equal(out[10:], k[10:])
    assert k[0, 0] == 5.0, "input must not be modified"


@pytest.mark.parametrize("angle,shape", [(0, (2, 3)), (90, (3, 2)), (180, (2, 3)), (-90, (3, 2)), (270, (3, 2))])
def test_rotate_frame(angle, shape):
    assert rotate_frame(np.zeros((2, 3, 3), np.uint8), angle).shape[:2] == shape


def test_multiperson_tracking_ids(synthetic_video):
    res = Video2Pose(model=FakePoseModel(n_people=2), verbose=False).predict(str(synthetic_video), multiperson=True)
    ids = res["track_ids"]
    assert len(ids) == 20 and all(len(i) == 2 for i in ids)
    assert all(list(i) == list(ids[0]) for i in ids)  # same (static) people keep their IDs
    assert len(set(ids[0].tolist())) == 2
