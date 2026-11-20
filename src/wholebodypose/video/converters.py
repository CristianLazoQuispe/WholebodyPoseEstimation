"""Video re-encoding helpers built on the ``ffmpeg`` executable.

The executable is taken from ``imageio-ffmpeg`` (``pip install 'wholebodypose[video]'``,
ships a static binary) or, failing that, from ``ffmpeg`` on ``PATH``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def ffmpeg_exe() -> str | None:
    """Return a usable ffmpeg executable, or ``None``."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - optional dependency
        return shutil.which("ffmpeg")


def _run(cmd: list[str]) -> bool:
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode == 0


def reencode_for_web(input_path: str | Path, output_path: str | Path | None = None, crf: int = 23) -> tuple[bool, str]:
    """Re-encode a video to H.264 / yuv420p / faststart so browsers can play it.

    Returns ``(success, output_path)``. If ffmpeg is unavailable the input path is returned.
    """
    exe = ffmpeg_exe()
    src = Path(input_path)
    if exe is None:
        return False, str(src)
    dst = Path(output_path) if output_path else src.with_name(f"{src.stem}_web.mp4")
    ok = _run(
        [
            exe,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(dst),
        ]
    )
    return ok, str(dst if ok else src)


def convert_mp4_to_mov(input_path: str | Path, output_dir: str | Path, overwrite: bool = True) -> tuple[bool, str]:
    """Convert an MP4 file to MOV (QuickTime) container."""
    exe = ffmpeg_exe()
    src = Path(input_path)
    if exe is None:
        raise RuntimeError("ffmpeg not found. Install with: pip install 'wholebodypose[video]'")
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {src}")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{src.stem}.mov"
    cmd = [exe, "-loglevel", "error", "-i", str(src), "-f", "mov"] + (["-y"] if overwrite else ["-n"]) + [str(dst)]
    return _run(cmd), str(dst)


def ensure_constant_fps(
    input_path: str | Path, output_dir: str | Path | None = None, target_fps: float = 30.0, max_fps: float = 120.0
) -> tuple[str, bool]:
    """Re-encode variable-frame-rate videos (e.g. browser/webcam WebM) to a constant frame rate.

    OpenCV reports a bogus FPS (often 1000) for VFR WebM files, which breaks sub-sampling and
    output timing. If the reported FPS is missing or above ``max_fps`` the video is converted to
    ``target_fps`` with ffmpeg. Returns ``(path, converted)``; the input is returned unchanged
    when it looks fine or ffmpeg is unavailable.
    """
    import cv2

    src = Path(input_path)
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
    cap.release()
    if 0 < fps <= max_fps:
        return str(src), False
    exe = ffmpeg_exe()
    if exe is None:
        return str(src), False
    out_dir = Path(output_dir) if output_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{src.stem}_cfr.mp4"
    ok = _run(
        [
            exe,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-fps_mode",
            "cfr",
            "-r",
            str(target_fps),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(dst),
        ]
    )
    return (str(dst), True) if ok else (str(src), False)
