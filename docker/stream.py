import cv2
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import threading
import signal
import socket

PORT = 8080
cap = cv2.VideoCapture(0)

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True  # Threads will terminate automatically when the main thread exits

class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return  # Silence logs

    def do_GET(self):
        if self.path != '/':
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()

        # Force socket timeout to avoid eternal blocking
        self.connection.settimeout(1.0)

        try:
            # Exit loop if server is shutting down
            while not self.server._BaseServer__shutdown_request:
                ret, frame = cap.read()
                if not ret:
                    break
                ret, jpeg = cv2.imencode('.jpg', frame)
                try:
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                except (BrokenPipeError, ConnectionAbortedError, socket.timeout):
                    break
        except Exception as e:
            pass  # Silence other errors

def main():
    server = ThreadingHTTPServer(('', PORT), MJPEGHandler)

    def shutdown_server(sig, frame):
        print("\n Stopping server and releasing camera...")
        cap.release()
        threading.Thread(target=server.shutdown).start()  # Avoid blocking the main thread
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_server)

    print(f"Streaming at http://localhost:{PORT} — press Ctrl+C to stop")
    server.serve_forever(poll_interval=0.5)

if __name__ == '__main__':
    main()
