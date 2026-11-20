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


def test_multi_person_hands_and_faces():
    left_person = _pose(left_wrist=(0.1, 0.5), right_wrist=(0.3, 0.5))
    left_person.xy[0] = (0.2, 0.2)
    right_person = _pose(left_wrist=(0.7, 0.5), right_wrist=(0.9, 0.5))
    right_person.xy[0] = (0.8, 0.2)
    face_r = Landmarks(np.tile(np.array([0.8, 0.21], np.float32), (468, 1)), np.ones(468, bool))
    frame = MPFrame(
        poses=[left_person, right_person],
        hands=[_hand((0.9, 0.5)), _hand((0.1, 0.5)), _hand((0.3, 0.5))],
        faces=[face_r],
    )
    k, s = to_coco133(frame, 100, 100)
    assert k.shape == (2, 133, 2)
    assert np.allclose(k[0, 91:112], [10, 50]) and np.allclose(k[0, 112:133], [30, 50])
    assert np.allclose(k[1, 112:133], [90, 50]) and not s[1, 91:112].any()  # its left hand was not found
    assert s[1, 23:91].all() and not s[0, 23:91].any()  # face attached to the right person


def test_far_hand_is_not_attached():
    frame = MPFrame(poses=[_pose(), _pose()], hands=[_hand((0.5, 0.0))])
    frame.poses[1].xy[:] = 0.9
    k, s = to_coco133(frame, 100, 100)
    assert not s[:, 91:].any()
