#!/usr/bin/env python3
"""Local virtual-camera sink for delulucam's web app.

The browser (index.html) does the actual Decart connection and rendering —
this script just receives the transformed frames it captures from its output
<video> element (as JPEG bytes over a local WebSocket) and pushes them into a
real system virtual camera via pyvirtualcam, so any other app (Zoom, Meet,
Discord, OBS...) can select it directly like any other webcam.

Usage:
    python3 delulucam/web/vcam_bridge.py [--port 8421] [--fps 25]

Then in index.html, toggle "Feed to virtual camera" once this is running.
"""

import argparse
import asyncio
import sys

import cv2
import numpy as np

try:
    import websockets
except ImportError:
    print("Missing dependency: pip install websockets", file=sys.stderr)
    sys.exit(1)

try:
    import pyvirtualcam
    from pyvirtualcam import PixelFormat
except ImportError:
    print("Missing dependency: pip install pyvirtualcam", file=sys.stderr)
    sys.exit(1)


class VCamSink:
    """Lazily creates the virtual camera once we know the real frame size —
    the browser's output resolution depends on which Decart model is active
    and isn't known ahead of time."""

    def __init__(self, fps: int):
        self.fps = fps
        self.cam: pyvirtualcam.Camera | None = None
        self.size: tuple[int, int] | None = None

    def send(self, frame_bgr: np.ndarray) -> None:
        h, w = frame_bgr.shape[:2]
        if self.cam is None or self.size != (w, h):
            if self.cam is not None:
                self.cam.close()
            try:
                self.cam = pyvirtualcam.Camera(width=w, height=h, fps=self.fps, fmt=PixelFormat.BGR)
            except RuntimeError as e:
                raise RuntimeError(
                    f"could not start the virtual camera: {e}\n"
                    "  macOS: install OBS Studio once (it ships the virtual cam driver), "
                    "open it, and click 'Start Virtual Camera' one time.\n"
                    "  Linux: sudo modprobe v4l2loopback devices=1 card_label=delulucam"
                ) from e
            self.size = (w, h)
            print(f"[vcam] virtual camera live at {w}x{h}@{self.fps} -> {self.cam.device}")
        self.cam.send(frame_bgr)
        self.cam.sleep_until_next_frame()

    def close(self) -> None:
        if self.cam is not None:
            self.cam.close()
            self.cam = None


async def handle_connection(websocket, sink: VCamSink):
    peer = websocket.remote_address
    print(f"[vcam] browser connected from {peer}")
    frame_count = 0
    try:
        async for message in websocket:
            if not isinstance(message, (bytes, bytearray)):
                continue  # ignore any stray text/control messages
            arr = np.frombuffer(message, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            try:
                sink.send(frame)
            except RuntimeError as e:
                print(f"[vcam] {e}")
                await websocket.close(code=1011, reason="virtual camera unavailable — see console")
                return
            frame_count += 1
            if frame_count % 150 == 0:
                print(f"[vcam] {frame_count} frames relayed")
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        print("[vcam] browser disconnected")


async def main():
    parser = argparse.ArgumentParser(description="Local virtual-camera sink for delulucam web app")
    parser.add_argument("--port", type=int, default=8421)
    parser.add_argument("--fps", type=int, default=25, help="virtual camera fps (default 25)")
    args = parser.parse_args()

    sink = VCamSink(fps=args.fps)

    async def handler(websocket):
        await handle_connection(websocket, sink)

    print(f"[vcam] listening on ws://localhost:{args.port}")
    print("[vcam] waiting for the browser page to connect (toggle 'Feed to virtual camera')")
    try:
        async with websockets.serve(handler, "localhost", args.port, max_size=None):
            await asyncio.Future()
    finally:
        sink.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[vcam] stopped")
