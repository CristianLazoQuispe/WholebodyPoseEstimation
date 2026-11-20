"""Clean a saved pose sequence: smooth it, then drop unusable frames.

    python examples/filter_sequence.py outputs/videoTest_rtmpose.npz
    python examples/filter_sequence.py outputs/videoTest_rtmpose.npz --nlms   # needs wholebodypose[filters]

Steps:

1. ``FilteringNLMS`` (``--nlms``): the paper's adaptive NLMS tracker. It keeps confident keypoints,
   blends medium-confidence ones with the filter prediction and fills short gaps of low-confidence
   keypoints from the last good value.
2. ``PoseSequenceFilter``: remove frames with mostly missing keypoints, frames without motion and
   frames where neither hand is visible (sign language preprocessing).
3. ``MeanMedianFilter`` (``--window``): moving (mean + median) / 2 smoothing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wholebodypose import load_poses, save_poses
from wholebodypose.filters.filtering_mean import MeanMedianFilter
from wholebodypose.filters.sequence import PoseSequenceFilter

LEFT_HAND, RIGHT_HAND = np.arange(91, 112), np.arange(112, 133)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("poses", type=Path, help=".npz or .json written by save_poses (single person)")
    ap.add_argument("--nlms", action="store_true", help="run the NLMS tracker first (needs padasip)")
    ap.add_argument("--motion-threshold", type=float, default=1.0, help="min mean pixel motion between frames")
    ap.add_argument("--window", type=int, default=5, help="MeanMedianFilter window (0 or 1 disables)")
    ap.add_argument("--output", "-o", type=Path, default=None)
    args = ap.parse_args()

    keypoints, scores, meta = load_poses(args.poses)
    if not isinstance(keypoints, np.ndarray) or keypoints.ndim != 3:
        raise SystemExit("expected a single-person sequence (T, 133, 2)")
    print(f"input       {keypoints.shape}")

    if args.nlms:
        from wholebodypose import FilteringNLMS

        # num_keypoints defaults to 135 (the dataset layout); match the file instead
        keypoints, scores = FilteringNLMS(num_keypoints=keypoints.shape[1]).apply(keypoints, scores)
        keypoints, scores = keypoints.astype(np.float32), scores.astype(np.float32)
        print(f"NLMS        {keypoints.shape}")

    flt = PoseSequenceFilter(left_hand_indices=LEFT_HAND, right_hand_indices=RIGHT_HAND)
    kept, masks = flt.apply_all_filters(keypoints, motion_threshold=args.motion_threshold)
    # each mask indexes the output of the previous stage
    idx = np.arange(len(keypoints))
    for name, mask in masks.items():
        idx = idx[mask]
        print(f"{name:11s} kept {int(mask.sum())}/{len(mask)} frames")
    keypoints, scores = kept, scores[idx]

    if args.window > 1:
        smoother = MeanMedianFilter(window_size=args.window)
        keypoints = np.stack([smoother(k) for k in keypoints]).astype(np.float32)

    out = args.output or args.poses.with_name(f"{args.poses.stem}_filtered.npz")
    save_poses(out, keypoints, scores, **{**meta, "filtered": True, "kept_frames": idx.tolist()})
    print(f"output      {keypoints.shape} -> {out}")


if __name__ == "__main__":
    main()
