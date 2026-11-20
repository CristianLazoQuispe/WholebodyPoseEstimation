import numpy as np

from wholebodypose.models.mediapipe.map_mp2coco import Hand, Landmarks, MPFrame, assign_hands, to_coco133


def _pose(left_wrist=(0.3, 0.5), right_wrist=(0.7, 0.5)):
    xy = np.full((33, 2), 0.5, np.float32)
    xy[15], xy[16] = left_wrist, right_wrist
    return Landmarks(xy, np.ones(33, bool))


def _hand(at, label=None):
    xy = np.tile(np.array(at, np.float32), (21, 1))
    return Hand(xy, np.ones(21, bool), label)


def test_hands_assigned_by_wrist_proximity():
    frame = MPFrame(pose=_pose(), hands=[_hand((0.7, 0.5), "Right"), _hand((0.3, 0.5), "Left")])
    assert assign_hands(frame) == ["right", "left"]
    k, s = to_coco133(frame, 100, 100)
    assert k.shape == (1, 133, 2) and s.shape == (1, 133)
    assert np.allclose(k[0, 91:112], [30, 50]) and np.allclose(k[0, 112:133], [70, 50])


def test_hands_fallback_to_mirrored_label():
    frame = MPFrame(hands=[_hand((0.1, 0.1), "Left"), _hand((0.9, 0.9), "Left")])
    assert assign_hands(frame) == ["right", "left"]


def test_body_and_feet_mapping():
    frame = MPFrame(pose=_pose())
    xy = frame.pose.xy
    xy[29], xy[31] = (0.2, 0.9), (0.25, 0.95)  # left heel, left foot index
    k, s = to_coco133(frame, 100, 100)
    assert np.allclose(k[0, 19], [20, 90])  # COCO left heel
    assert np.allclose(k[0, 17], [25, 95]) and np.allclose(k[0, 18], [25, 95])  # toes
    assert s[0, :23].all() and not s[0, 23:].any()


def test_invisible_landmarks_are_zero():
    pose = _pose()
    pose.valid[:] = False
    k, s = to_coco133(MPFrame(pose=pose), 100, 100)
    assert not k.any() and not s.any()
