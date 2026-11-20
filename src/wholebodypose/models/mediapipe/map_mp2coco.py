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
    """Engine-independent MediaPipe output for one frame.

    ``poses`` / ``faces`` hold every detected body / face; ``pose`` / ``face`` are shortcuts for a
    single person (both forms are accepted by the constructor).
    """

    pose: Landmarks | None = None
    hands: list[Hand] = field(default_factory=list)
    face: Landmarks | None = None
    poses: list[Landmarks] = field(default_factory=list)
    faces: list[Landmarks] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.pose is not None and not self.poses:
            self.poses = [self.pose]
        if self.face is not None and not self.faces:
            self.faces = [self.face]
        self.pose = self.poses[0] if self.poses else None
        self.face = self.faces[0] if self.faces else None


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


def _person_arrays(pose: Landmarks | None) -> tuple[np.ndarray, np.ndarray]:
    kpts = np.zeros((133, 2), np.float32)
    scores = np.zeros(133, np.float32)
    if pose is not None:
        for mp_id, coco_ids in POSE_TO_COCO.items():
            if mp_id < len(pose.valid) and pose.valid[mp_id]:
                kpts[coco_ids] = pose.xy[mp_id]
                scores[coco_ids] = 1.0
    return kpts, scores


def _put_face(kpts: np.ndarray, scores: np.ndarray, face: Landmarks) -> None:
    for mp_id, coco_id in FACE_TO_COCO.items():
        if mp_id < len(face.xy):
            kpts[coco_id] = face.xy[mp_id]
            scores[coco_id] = 1.0


def _put_hand(kpts: np.ndarray, scores: np.ndarray, hand: Hand, side: str) -> None:
    part = LEFT_HAND if side == "left" else RIGHT_HAND
    kpts[part] = hand.xy
    scores[part] = hand.valid.astype(np.float32)


def _greedy_match(cost: np.ndarray, max_cost: float) -> list[tuple[int, int]]:
    """Greedy minimum-cost one-to-one matching of rows to columns (small problems only)."""
    pairs = []
    cost = cost.astype(np.float64).copy()
    while cost.size and np.isfinite(cost).any():
        r, c = np.unravel_index(np.argmin(cost), cost.shape)
        if cost[r, c] > max_cost:
            break
        pairs.append((int(r), int(c)))
        cost[r, :] = np.inf
        cost[:, c] = np.inf
    return pairs


def _body_scale(pose: Landmarks) -> float:
    pts = pose.xy[pose.valid]
    if len(pts) < 2:
        return 0.1
    return float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 0.05))


def to_coco133(frame: MPFrame, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert one :class:`MPFrame` to ``(N, 133, 2)`` pixel keypoints and ``(N, 133)`` scores.

    With a single (or no) body the paper's single-person logic is used. With several bodies,
    hands are matched to body wrists and faces to noses by greedy minimum distance.
    """
    if len(frame.poses) <= 1:
        kpts, scores = _person_arrays(frame.pose)
        if frame.face is not None:
            _put_face(kpts, scores, frame.face)
        for hand, side in zip(frame.hands[:2], assign_hands(frame), strict=True):
            _put_hand(kpts, scores, hand, side)
        people = [(kpts, scores)]
    else:
        people = [_person_arrays(p) for p in frame.poses]
        # hands -> (person, side) slots, by wrist distance
        slots = [(i, side, wid) for i in range(len(frame.poses)) for side, wid in (("left", 15), ("right", 16))]
        if frame.hands:
            cost = np.full((len(frame.hands), len(slots)), np.inf)
            for h, hand in enumerate(frame.hands):
                for j, (i, _, wid) in enumerate(slots):
                    pose = frame.poses[i]
                    if pose.valid[wid]:
                        cost[h, j] = np.linalg.norm(hand.xy[0] - pose.xy[wid]) / _body_scale(pose)
            for h, j in _greedy_match(cost, max_cost=0.5):
                i, side, _ = slots[j]
                _put_hand(*people[i], frame.hands[h], side)
        # faces -> persons, by face centroid vs nose
        if frame.faces:
            cost = np.full((len(frame.faces), len(frame.poses)), np.inf)
            for f, face in enumerate(frame.faces):
                center = face.xy.mean(axis=0)
                for i, pose in enumerate(frame.poses):
                    if pose.valid[0]:
                        cost[f, i] = np.linalg.norm(center - pose.xy[0]) / _body_scale(pose)
            for f, i in _greedy_match(cost, max_cost=0.5):
                _put_face(*people[i], frame.faces[f])

    kpts = np.stack([k for k, _ in people]) * np.array([width, height], np.float32)
    scores = np.stack([s for _, s in people])
    kpts = np.round(kpts).astype(np.float32)
    kpts[scores == 0] = 0
    return kpts, scores


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
