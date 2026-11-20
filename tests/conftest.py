"""Shared fixtures. Unit tests never download weights; real backends are marked ``slow``."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from wholebodypose.models.base import BasePoseModel

ROOT = Path(__file__).resolve().parents[1]


class FakePoseModel(BasePoseModel):
    """Deterministic backend: one or two people whose pose is a function of the image size."""

    name = "fake"

    def __init__(self, n_people: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_people = n_people
        self.calls = 0

    def _predict(self, image_rgb):
        self.calls += 1
        h, w = image_rgb.shape[:2]
        rng = np.random.default_rng(0)
        people = []
        for p in range(self.n_people):
            scale = 0.5 / (p + 1)
            kp = rng.uniform(0.25, 0.75, size=(133, 2)) * scale * [w, h] + [w * 0.1, h * 0.1]
            people.append(kp)
        kps = np.array(people, np.float32).reshape(-1, 133, 2)
        return kps, np.ones(kps.shape[:2], np.float32)


@pytest.fixture
def fake_model():
    return FakePoseModel()


@pytest.fixture
def image_rgb():
    img = np.full((240, 320, 3), 127, np.uint8)
    cv2.circle(img, (160, 120), 40, (255, 0, 0), -1)
    return img


@pytest.fixture
def synthetic_video(tmp_path) -> Path:
    """A 2-second, 10 fps, 160x120 MP4 with a moving square."""
    path = tmp_path / "synthetic.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    assert writer.isOpened()
    for t in range(20):
        frame = np.zeros((120, 160, 3), np.uint8)
        cv2.rectangle(frame, (10 + 3 * t, 30), (40 + 3 * t, 90), (0, 255, 0), -1)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def sample_image_path() -> Path:
    return ROOT / "data" / "images" / "image.jpg"
