"""Share the host webcam with a container as an MJPEG stream.

Docker Desktop (Windows, macOS) cannot pass a webcam to a container. Run this on the **host**
(only OpenCV needed), then read the stream from inside the container:

    python docker/stream.py --camera 0 --port 8080                     # on the host
    python examples/webcam_realtime.py --source http://host.docker.internal:8080/ --no-show   # in the container

On Linux you can instead pass the device directly: ``docker run --device /dev/video0 ...``.
"""

from __future__ import annotations

import argparse
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2


def make_handler(cap: cv2.VideoCapture, lock: threading.Lock, quality: int):
    class MJPEGHandler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # silence per-request logs
            return

        def do_GET(self) -> None:
            if self.path != "/":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            self.connection.settimeout(1.0)
            while True:
                with lock:
                    ok, frame = cap.read()
                if not ok:
                    break
                ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if not ok:
                    continue
                try:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    break

    return MJPEGHandler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--quality", type=int, default=85, help="JPEG quality")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {args.camera}")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(cap, threading.Lock(), args.quality))
    server.daemon_threads = True
    print(f"streaming camera {args.camera} at http://localhost:{args.port}/ (Ctrl+C to stop)")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        cap.release()


if __name__ == "__main__":
    main()
