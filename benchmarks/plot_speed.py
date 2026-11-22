"""Speed chart (FPS, CPU vs GPU) from the JSON files written by ``speed.py``.

    python benchmarks/plot_speed.py benchmarks/results -o website/static/images/speed.png

Needs matplotlib. Use a ``.pdf`` or ``.svg`` output for print.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms as mt  # noqa: E402
import numpy as np  # noqa: E402

BARS = [  # (label in the JSON, tick label, small label under the tick)
    ("MediaPipe heavy", "MediaPipe\nheavy", ""),
    ("MediaPipe heavy + tracking", "MediaPipe\nheavy", "+ tracking"),
    ("MediaPipe full", "MediaPipe\nfull", ""),
    ("MediaPipe full + tracking", "MediaPipe\nfull", "+ tracking"),
    ("RTMPose lightweight", "RTMPose\nlight", ""),
    ("RTMPose balanced", "RTMPose\nbalanced", ""),
    ("RTMPose performance", "RTMPose\nperf.", ""),
    *[(f"ViTPose+ {s} + YOLOX-m", f"ViTPose+\n{s[0].upper()}", "") for s in ("small", "base", "large", "huge")],
]
BODY_ONLY = "MediaPipe body only, full + tracking"
DARK, LIGHT, INK, MUTED, GRID = "#0f6f74", "#a9d3d3", "#1f2328", "#6b7279", "#eceef0"


def load(path: Path) -> dict:
    return {r["label"]: r["fps"] for r in json.loads(path.read_text())["rows"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path, help="directory with cpu.json and gpu.json")
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--cpu-name", default="AMD EPYC 7662")
    ap.add_argument("--gpu-name", default="NVIDIA A100")
    args = ap.parse_args()

    cpu, gpu = load(args.results / "cpu.json"), load(args.results / "gpu.json")
    meta = json.loads((args.results / "cpu.json").read_text())["meta"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.edgecolor": "#c9cdd1",
            "xtick.color": INK,
            "ytick.color": MUTED,
        }
    )
    keys = [b[0] for b in BARS]
    x = np.arange(len(keys), dtype=float)
    x[1:] += 0.15
    x[2:] += 0.15
    x[3:] += 0.15
    x[4:] += 0.4
    x[7:] += 0.5
    w = 0.38
    c = np.array([cpu[k] for k in keys])
    g = np.array([gpu.get(k, np.nan) for k in keys])
    cpu_only = np.isnan(g)
    xc = np.where(cpu_only, x, x - w / 2)

    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    ax.bar(xc, c, w, color=LIGHT, zorder=2, label="CPU")
    ax.bar(x + w / 2, np.nan_to_num(g), w, color=DARK, zorder=2, label="GPU")
    for xi, xci, ci, gi in zip(x, xc, c, g, strict=True):
        ax.text(
            xci,
            ci * 1.08,
            f"{ci:.0f}" if ci >= 10 else f"{ci:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=MUTED,
        )
        if not np.isnan(gi):
            ax.text(
                xi + w / 2, gi * 1.08, f"{gi:.0f}", ha="center", va="bottom", fontsize=7.5, color=INK, fontweight="bold"
            )
    ax.text(x[cpu_only].mean(), 25, "CPU only", ha="center", fontsize=6.5, color=MUTED)
    under = mt.blended_transform_factory(ax.transData, ax.transAxes)
    for xi, (_, _, small) in zip(x, BARS, strict=True):
        if small:
            ax.text(xi, -0.155, small, transform=under, ha="center", va="top", fontsize=6.5, color=MUTED)

    ax.set_yscale("log")
    ax.set_ylim(1, 170)
    ax.set_yticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
    ax.axhspan(30, 170, color=DARK, alpha=0.06, zorder=0, lw=0)
    ax.text(x[0] - 0.4, 135, "real time (≥ 30 FPS)", fontsize=7, color=DARK, va="center", fontweight="bold")
    ax.set_ylabel("Frames per second (log scale) ↑", color=MUTED, fontsize=8)
    ax.set_xticks(x, [b[1] for b in BARS], fontsize=7.5)
    ax.tick_params(axis="both", length=0)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=7.5, bbox_to_anchor=(1, 1.08))
    wpx, hpx = meta["frame"]
    note = (
        "All bars: whole-body pose, 133 keypoints (body, feet, face, hands), with person detection. "
        f"MediaPipe body only (33 keypoints, full, tracking): {cpu[BODY_ONLY]:.0f} FPS.\n"
        f"1 person, {wpx}×{hpx} px. CPU: {meta['threads']} threads, {args.cpu_name}. GPU: {args.gpu_name}."
    )
    fig.text(0.01, 0.01, note, fontsize=6.5, color=MUTED)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.26)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)


if __name__ == "__main__":
    main()
