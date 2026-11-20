import numpy as np

from wholebodypose.tracking import PoseTracker, bbox_iou, pose_bbox


def _person(x, y, size=50):
    k = np.zeros((133, 2), np.float32)
    k[:17] = np.linspace([x, y], [x + size, y + size], 17)
    s = np.zeros(133, np.float32)
    s[:17] = 1
    return k, s


def test_bbox_helpers():
    k, s = _person(10, 20)
    assert pose_bbox(k, s).tolist() == [10, 20, 60, 70]
    assert pose_bbox(k, np.zeros(133)) is None
    assert bbox_iou(np.array([0, 0, 10, 10]), np.array([0, 0, 10, 10])) == 1
    assert bbox_iou(np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30])) == 0


def test_ids_follow_people_and_swap_order():
    t = PoseTracker()
    a, b = _person(0, 0), _person(200, 0)
    ids0 = t.update(np.stack([a[0], b[0]]), np.stack([a[1], b[1]]))
    # next frame: people move slightly and come in the opposite order
    a2, b2 = _person(5, 0), _person(205, 0)
    ids1 = t.update(np.stack([b2[0], a2[0]]), np.stack([b2[1], a2[1]]))
    assert ids1.tolist() == [ids0[1], ids0[0]]


def test_new_person_and_expiry():
    t = PoseTracker(max_age=1)
    a = _person(0, 0)
    first = t.update(a[0][None], a[1][None])[0]
    t.update(np.zeros((0, 133, 2)), np.zeros((0, 133)))
    t.update(np.zeros((0, 133, 2)), np.zeros((0, 133)))  # track expires
    assert t.update(a[0][None], a[1][None])[0] != first
