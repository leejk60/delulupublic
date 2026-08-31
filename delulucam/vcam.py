"""Camera input (your OBSBOT or any UVC webcam) and virtual camera output."""

from typing import List, Optional, Tuple

import cv2
import numpy as np


def open_capture(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(
            f"could not open camera index {index} — run with --list-cameras to "
            f"see what is available, and make sure no other app holds the camera"
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def list_cameras(max_index: int = 10) -> List[Tuple[int, str]]:
    """Probe camera indices and report which ones deliver frames."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                found.append((i, f"{w}x{h}"))
        cap.release()
    return found


class VirtualCamera:
    """Thin wrapper over pyvirtualcam; requires OBS Virtual Camera (Windows/
    macOS) or v4l2loopback (Linux) to be installed once, system-wide."""

    def __init__(self, width: int, height: int, fps: int):
        try:
            import pyvirtualcam
            from pyvirtualcam import PixelFormat
        except ImportError as e:
            raise SystemExit("pyvirtualcam is not installed — pip install -r requirements.txt") from e
        try:
            self._cam = pyvirtualcam.Camera(
                width=width, height=height, fps=fps, fmt=PixelFormat.BGR
            )
        except RuntimeError as e:
            raise SystemExit(
                f"could not start virtual camera: {e}\n"
                "  Windows/macOS: install OBS Studio once (it ships the virtual cam driver).\n"
                "  Linux: sudo modprobe v4l2loopback devices=1 card_label=delulucam\n"
                "  Or run with --no-vcam for a preview-only session."
            ) from e
        print(f"[vcam] streaming to virtual camera: {self._cam.device}")

    def send(self, frame_bgr: np.ndarray) -> None:
        self._cam.send(frame_bgr)
        self._cam.sleep_until_next_frame()

    def close(self) -> None:
        self._cam.close()


def open_vcam(width: int, height: int, fps: int, enabled: bool) -> Optional[VirtualCamera]:
    return VirtualCamera(width, height, fps) if enabled else None
