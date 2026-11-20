"""Regenerate the media assets of the project page (website/static/).

Usage (from the repository root)::

    python scripts/make_website_assets.py                # teaser video + poster, OG card, favicons
    python scripts/make_website_assets.py --paper        # also paper thumbnail and Figure 1 crops

Needs Pillow and the ``ffmpeg`` binary; ``--paper`` also needs poppler (``pdftoppm``, ``pdfimages``).
Nothing here runs at deploy time: the generated files are committed.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "website" / "static"
TEASER_GIF = ROOT / "docs" / "assets" / "output_telefono_pucp_slow.gif"
PAPER = SITE / "paper" / "WholeBodyPose_ICMV2025.pdf"
POSTER_TIME = 5.0  # seconds into the GIF used for the poster frame and the OG card

ACCENT = (15, 111, 116)
INK = (17, 20, 23)
INK_2 = (59, 65, 72)
MUTED = (92, 99, 107)
BODY, FACE, HAND = (217, 115, 31), (122, 91, 181), (46, 139, 87)

FONT_DIRS = [
    "/usr/share/fonts",
    "/usr/share/fonts/truetype",
    "/Library/Fonts",
    "/System/Library/Fonts/Supplemental",
    "C:/Windows/Fonts",
]


def _font(names: list[str], size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for base in FONT_DIRS:
        for name in names:
            for path in Path(base).rglob(name) if Path(base).is_dir() else []:
                return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size)


def serif_bold(size: int):
    return _font(
        ["SourceSerif4-SemiBold.ttf", "DejaVuSerif-Bold.ttf", "LiberationSerif-Bold.ttf", "Georgia Bold.ttf"], size
    )


def sans(size: int):
    return _font(["NotoSans-Regular.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf"], size)


def sans_bold(size: int):
    return _font(["NotoSans-SemiBold.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "Arial Bold.ttf"], size)


def run(*cmd: str) -> None:
    subprocess.run(cmd, check=True)


def teaser() -> None:
    """GIF (5.6 MB) -> H.264 MP4 + VP9 WebM (~0.45 MB each) and a JPEG poster."""
    videos = SITE / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    gif = str(TEASER_GIF)
    run("ffmpeg", "-v", "error", "-y", "-i", gif, "-movflags", "+faststart", "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "slow", "-crf", "26",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-an", str(videos / "teaser.mp4"))  # fmt: skip
    run("ffmpeg", "-v", "error", "-y", "-i", gif, "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "36",
        "-row-mt", "1", "-pix_fmt", "yuv420p", "-an", str(videos / "teaser.webm"))  # fmt: skip
    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "poster.png"
        run("ffmpeg", "-v", "error", "-y", "-ss", str(POSTER_TIME), "-i", gif, "-frames:v", "1", str(png))
        Image.open(png).convert("RGB").save(
            SITE / "images" / "teaser_poster.jpg", quality=85, optimize=True, progressive=True
        )


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) <= width:
            line = trial
        else:
            lines.append(line)
            line = word
    return [*lines, line]


def og_card() -> None:
    """1200x630 social card: title block on the left, teaser frame on the right."""
    W, H = 1200, 630
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    frame = Image.open(SITE / "images" / "teaser_poster.jpg").convert("RGB")
    fw = 530
    fh = round(frame.height * fw / frame.width)
    frame = frame.resize((fw, fh), Image.LANCZOS)
    fx, fy = W - fw - 48, (H - fh) // 2
    mask = Image.new("L", (fw, fh), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, fw - 1, fh - 1), radius=18, fill=255)
    d.rounded_rectangle((fx - 2, fy - 2, fx + fw + 1, fy + fh + 1), radius=20, fill=(207, 211, 216))
    img.paste(frame, (fx, fy), mask)

    x, maxw = 60, fx - 60 - 40
    d.rectangle((0, 0, W, 10), fill=ACCENT)
    d.text((x, 70), "ICMV 2025 · PROCEEDINGS OF SPIE", font=sans_bold(20), fill=ACCENT)
    size = 66
    while size > 30 and d.textlength("WholeBodyPose", font=serif_bold(size)) > maxw:
        size -= 2
    d.text((x, 104), "WholeBodyPose", font=serif_bold(size), fill=INK)
    y = 200
    sub = sans(27)
    for line in _wrap(
        d, "A Unified End-to-End Framework for Sign Language Recognition and Pose-Based Training Data", sub, maxw
    ):
        d.text((x, y), line, font=sub, fill=INK_2)
        y += 38
    y += 22
    small = sans(21)
    for line in ["RTMPose · ViTPose+ · MediaPipe", "→ 133 COCO-WholeBody keypoints"]:
        d.text((x, y), line, font=small, fill=ACCENT)
        y += 30
    authors = "Lazo-Quispe, Castro-Cruz, Salazar-Espinosa, Bejarano"
    d.text((x, H - 92), authors, font=sans(18), fill=MUTED)
    d.text((x, H - 64), "cristianlazoquispe.github.io/WholebodyPoseEstimation", font=sans(18), fill=MUTED)
    img.save(SITE / "images" / "og.png", optimize=True)


FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <rect width="64" height="64" rx="14" fill="#0f6f74"/>
  <g fill="none" stroke="#fff" stroke-width="5" stroke-linecap="round" stroke-linejoin="round">
    <path d="M32 24v16M32 40l-8 14M32 40l8 14M18 21l14 7 14-7"/>
  </g>
  <circle cx="32" cy="13" r="6" fill="#fff"/>
  <circle cx="16" cy="20" r="4.5" fill="#f4a259"/>
  <circle cx="48" cy="20" r="4.5" fill="#7fd1a1"/>
</svg>
"""


def _favicon_png(size: int) -> Image.Image:
    """Raster version of FAVICON_SVG (drawn at 8x and downsampled)."""
    s = 8 * size / 64
    big = Image.new("RGBA", (round(64 * s), round(64 * s)), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    d.rounded_rectangle((0, 0, 64 * s - 1, 64 * s - 1), radius=14 * s, fill=ACCENT)
    w = round(5 * s)
    for seg in [
        ((32, 24), (32, 40)),
        ((32, 40), (24, 54)),
        ((32, 40), (40, 54)),
        ((18, 21), (32, 28)),
        ((32, 28), (46, 21)),
    ]:
        (x0, y0), (x1, y1) = seg
        d.line((x0 * s, y0 * s, x1 * s, y1 * s), fill="white", width=w)
        for cx, cy in seg:
            d.ellipse((cx * s - w / 2, cy * s - w / 2, cx * s + w / 2, cy * s + w / 2), fill="white")
    for cx, cy, r, col in [(32, 13, 6, "white"), (16, 20, 4.5, (244, 162, 89)), (48, 20, 4.5, (127, 209, 161))]:
        d.ellipse(((cx - r) * s, (cy - r) * s, (cx + r) * s, (cy + r) * s), fill=col)
    return big.resize((size, size), Image.LANCZOS)


def favicons() -> None:
    images = SITE / "images"
    (images / "favicon.svg").write_text(FAVICON_SVG)
    _favicon_png(32).save(images / "favicon-32.png", optimize=True)
    touch = Image.new("RGB", (180, 180), "white")  # iOS ignores transparency
    touch.paste(_favicon_png(180), (0, 0), _favicon_png(180))
    touch.save(images / "apple-touch-icon.png", optimize=True)


def paper_assets() -> None:
    """First-page thumbnail and the two DrawerPose panels of Figure 1 (page 5)."""
    images = SITE / "images"
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        run("pdftoppm", "-r", "50", "-png", "-f", "1", "-l", "1", "-singlefile", str(PAPER), str(t / "p1"))
        Image.open(t / "p1.png").convert("RGB").save(
            images / "paper_thumb.jpg", quality=85, optimize=True, progressive=True
        )
        run("pdfimages", "-png", "-f", "5", "-l", "5", str(PAPER), str(t / "im"))
        for src, dst in [("im-000.png", "kpts_coco133.jpg"), ("im-002.png", "kpts_openpose135.jpg")]:
            Image.open(t / src).convert("RGB").save(images / dst, quality=88, optimize=True, progressive=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paper", action="store_true", help="also regenerate the paper thumbnail and Figure 1 crops")
    ap.add_argument("--skip-video", action="store_true", help="keep the existing teaser video and poster")
    args = ap.parse_args()
    if not args.skip_video:
        if shutil.which("ffmpeg") is None:
            raise SystemExit("ffmpeg not found (needed for the teaser video); use --skip-video")
        teaser()
    og_card()
    favicons()
    if args.paper:
        paper_assets()
    print(f"assets written to {SITE}")


if __name__ == "__main__":
    main()
