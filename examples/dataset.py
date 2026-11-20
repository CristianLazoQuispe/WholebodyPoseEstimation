"""Build a pose dataset (ISLR HDF5 layout) from a folder of labelled videos.

    videos/<label>/<clip>.mp4  ->  out.hdf5, one group per clip: data (T, 2, 135), label, video_name, width, height

    python examples/dataset.py videos/ outputs/my_dataset.hdf5

Needs ``pip install "wholebodypose[datasets]"``. To rebuild the published ISLR datasets use
``wholebodypose-build-islr`` (docs/datasets).
"""

import argparse
from pathlib import Path

import h5py

from wholebodypose import create_model
from wholebodypose.datasets import to_published_array, write_sample
from wholebodypose.datasets.build import PRESETS, extract_video

ap = argparse.ArgumentParser()
ap.add_argument("root", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--max-frames", type=int, default=None)
args = ap.parse_args()

cfg = PRESETS["v1"]  # RTMPose with the paper thresholds, largest person per frame
model = create_model(cfg.backend, **cfg.model_kwargs)
videos = sorted(p for p in args.root.glob("*/*") if p.suffix.lower() in {".mp4", ".avi", ".mov", ".webm"})
args.out.parent.mkdir(parents=True, exist_ok=True)
with h5py.File(args.out, "w") as h5:
    for i, video in enumerate(videos):
        r = extract_video(model, video, cfg.resize_width, cfg.person, max_frames=args.max_frames)
        data = to_published_array(r["keypoints"], r["width"], r["height"])
        write_sample(h5, i, data, video.parent.name, str(video.relative_to(args.root)), r["width"], r["height"])
        print(f"{video}: {len(data)} frames")
print(f"{len(videos)} videos -> {args.out}")
