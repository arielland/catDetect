"""
Live MJPEG camera stream — opens in any browser on your local network.

Usage:
    python stream.py              # stream on port 8080
    python stream.py --port 9000  # custom port

Then open in your browser: http://catdetect.local:8080

NOTE: stops detect.py automatically while streaming.
Ctrl+C to stop — detect.py will restart automatically on next reboot,
or run: nohup venv/bin/python -u detect.py --phase before --interval 30 >> detect.log 2>&1 &
"""

import argparse
import subprocess
import signal
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import cv2

PAGE = b"""
<html>
<head>
  <title>catDetect Live</title>
  <style>
    body { background: #111; display: flex; flex-direction: column;
           align-items: center; justify-content: center; height: 100vh; margin: 0; }
    img  { max-width: 100%; border: 2px solid #444; border-radius: 6px; }
    h2   { color: #aaa; font-family: monospace; margin-bottom: 12px; }
  </style>
</head>
<body>
  <h2>catDetect Live Stream</h2>
  <img src="/stream" />
</body>
</html>
"""

cap = None


def open_camera():
    global cap
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera — is detect.py still running?")


class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # silence access logs

    def do_GET(self):
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    _, jpeg = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(jpeg.tobytes())
                    self.wfile.write(b"\r\n")
                    time.sleep(0.1)   # ~10 fps
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(PAGE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    # Stop detect.py if running
    result = subprocess.run(["pkill", "-f", "detect.py"], capture_output=True)
    if result.returncode == 0:
        print("Stopped detect.py")
        time.sleep(2)

    open_camera()

    server = HTTPServer(("0.0.0.0", args.port), StreamHandler)
    print(f"Stream running → http://catdetect.local:{args.port}")
    print("Press Ctrl+C to stop\n")

    def shutdown(sig, frame):
        print("\nStopping stream...")
        cap.release()
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    server.serve_forever()


if __name__ == "__main__":
    main()
