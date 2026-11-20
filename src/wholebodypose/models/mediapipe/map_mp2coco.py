"""Map MediaPipe pose / hand / face landmarks to the COCO-WholeBody 133-keypoint layout."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_META = json.loads((Path(__file__).parent / "meta_data.json").read_text())["meta_info"]["mp2coco"]

# MediaPipe BlazePose (33) -> COCO-WholeBody body + feet (0-22).
# Feet follow COCO-WholeBody: 17/20 big toe, 18/21 small toe, 19/22 heel. MediaPipe has a single
# "foot index" point per foot, used for both toes.
POSE_TO_COCO: dict[int, list[int]] = {
    0: [0],
    2: [1],
    5: [2],
    7: [3],
    8: [4],
    11: [5],
    12: [6],
    13: [7],
    14: [8],
    15: [9],
    16: [10],
    23: [11],
    24: [12],
    25: [13],
    26: [14],
    27: [15],
    28: [16],
    31: [17, 18],
    29: [19],  # left foot index -> toes, left heel -> heel
    32: [20, 21],
    30: [22],  # right foot index -> toes, right heel -> heel
}
FACE_TO_COCO: dict[int, int] = {int(k): int(v) for k, v in _META["face"].items()}

LEFT_HAND = slice(91, 112)
RIGHT_HAND = slice(112, 133)
LEFT_WRIST, RIGHT_WRIST = 9, 10

VISIBILITY_THRESHOLD = 0.5
PRESENCE_THRESHOLD = 0.5


@dataclass
class Landmarks:
    """Normalized landmarks of one body part: ``xy`` (L, 2) in [0, 1] and ``valid`` (L,) bool."""

    xy: np.ndarray
    valid: np.ndarray


@dataclass
class Hand(Landmarks):
    #: MediaPipe handedness label ("Left"/"Right"); it assumes a mirrored (selfie) image.
    label: str | None = None


@dataclass
class MPFrame:
    """Engine-independent MediaPipe output for one frame."""

    pose: Landmarks | None = None
    hands: list[Hand] = field(default_factory=list)
    face: Landmarks | None = None


def landmarks_from_list(landmarks) -> Landmarks:
    """Build :class:`Landmarks` from MediaPipe landmark objects (solutions or tasks API)."""
    xy = np.zeros((len(landmarks), 2), np.float32)
    valid = np.zeros(len(landmarks), bool)
    for i, lm in enumerate(landmarks):
        vis = getattr(lm, "visibility", None)
        pres = getattr(lm, "presence", None)
        if hasattr(lm, "HasField"):  # protobuf (solutions API): unset fields read as 0
            vis = lm.visibility if lm.HasField("visibility") else None
            pres = lm.presence if lm.HasField("presence") else None
        if (vis is not None and vis < VISIBILITY_THRESHOLD) or (pres is not None and pres < PRESENCE_THRESHOLD):
            continue
        xy[i] = (lm.x, lm.y)
        valid[i] = True
    return Landmarks(xy, valid)


def to_coco133(frame: MPFrame, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert one :class:`MPFrame` to ``(1, 133, 2)`` pixel keypoints and ``(1, 133)`` scores."""
    kpts = np.zeros((133, 2), np.float32)
    scores = np.zeros(133, np.float32)

    if frame.pose is not None:
        for mp_id, coco_ids in POSE_TO_COCO.items():
            if mp_id < len(frame.pose.valid) and frame.pose.valid[mp_id]:
                kpts[coco_ids] = frame.pose.xy[mp_id]
                scores[coco_ids] = 1.0

    if frame.face is not None:
        for mp_id, coco_id in FACE_TO_COCO.items():
            if mp_id < len(frame.face.xy):
                kpts[coco_id] = frame.face.xy[mp_id]
                scores[coco_id] = 1.0

    for hand, side in zip(frame.hands[:2], assign_hands(frame), strict=True):
        part = LEFT_HAND if side == "left" else RIGHT_HAND
        kpts[part] = hand.xy
        scores[part] = hand.valid.astype(np.float32)

    kpts = np.round(kpts * [width, height]).astype(np.float32)
    kpts[scores == 0] = 0
    return kpts[None], scores[None]


def assign_hands(frame: MPFrame) -> list[str]:
    """Decide which detected hand is the person's left/right hand.

    Uses the distance from each hand's wrist to the body wrists when the body is detected,
    otherwise MediaPipe's handedness label (which is mirrored: "Left" == person's right hand).
    """
    hands = frame.hands[:2]
    sides: list[str] = []
    pose = frame.pose
    body_ok = pose is not None and len(pose.valid) > 16 and pose.valid[15] and pose.valid[16]
    if body_ok and hands:
        lw, rw = pose.xy[15], pose.xy[16]  # BlazePose left/right wrist
        cost = [(np.linalg.norm(h.xy[0] - lw), np.linalg.norm(h.xy[0] - rw)) for h in hands]
        if len(hands) == 1:
            sides = ["left" if cost[0][0] <= cost[0][1] else "right"]
        else:
            straight = cost[0][0] + cost[1][1]
            swapped = cost[0][1] + cost[1][0]
            sides = ["left", "right"] if straight <= swapped else ["right", "left"]
    else:
        for h in hands:
            sides.append("right" if (h.label or "").lower() == "left" else "left")
        if len(sides) == 2 and sides[0] == sides[1]:
            sides[1] = "right" if sides[0] == "left" else "left"
    return sides
