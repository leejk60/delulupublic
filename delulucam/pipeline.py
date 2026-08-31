"""The live loop: capture → detect → swap → (enhance) → preview + virtual cam."""

import time
from typing import List, Optional

import cv2
import numpy as np

from .character import Character

HELP_OVERLAY = "[q]uit  [s]wap  [e]nhance  [m]irror  [ / ] character  [h]ide help"


class Pipeline:
    def __init__(
        self,
        swapper,
        characters: List[Character],
        enhancer=None,
        detect_every: int = 1,
        max_faces: int = 1,
        mirror: bool = False,
        show_fps: bool = True,
    ):
        self.swapper = swapper
        self.characters = characters
        self.enhancer = enhancer
        self.detect_every = max(1, detect_every)
        self.max_faces = max_faces
        self.mirror = mirror
        self.show_fps = show_fps

        self.current = 0
        self.swap_enabled = True
        self.enhance_enabled = enhancer is not None
        self.show_help = True

        self._frame_no = 0
        self._last_faces: list = []
        self._fps = 0.0
        self._fps_t0 = time.perf_counter()
        self._fps_frames = 0

    @property
    def character(self) -> Character:
        return self.characters[self.current]

    def cycle(self, step: int) -> None:
        self.current = (self.current + step) % len(self.characters)
        print(f"[pipeline] character -> {self.character.name}")

    def process(self, frame: np.ndarray) -> np.ndarray:
        if self.mirror:
            frame = cv2.flip(frame, 1)

        if self.swap_enabled:
            if self._frame_no % self.detect_every == 0 or not self._last_faces:
                self._last_faces = self.swapper.analyse(frame)
            if self._last_faces:
                frame = self.swapper.swap(
                    frame, self._last_faces, self.character.ref_face, self.max_faces
                )
                if self.enhance_enabled and self.enhancer is not None:
                    frame = self.enhancer.enhance(frame)
        self._frame_no += 1

        self._fps_frames += 1
        now = time.perf_counter()
        if now - self._fps_t0 >= 1.0:
            self._fps = self._fps_frames / (now - self._fps_t0)
            self._fps_t0, self._fps_frames = now, 0
        return frame

    def draw_hud(self, frame: np.ndarray) -> np.ndarray:
        """Overlay for the local preview window only — never sent to the vcam."""
        out = frame.copy()
        state = f"{self.character.name}" if self.swap_enabled else "swap OFF"
        if self.enhance_enabled and self.swap_enabled:
            state += " +enhance"
        if self.show_fps:
            state += f"  {self._fps:4.1f} fps"
        cv2.putText(out, state, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(out, state, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 120), 2)
        if self.show_help:
            h = out.shape[0]
            cv2.putText(
                out, HELP_OVERLAY, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3
            )
            cv2.putText(
                out, HELP_OVERLAY, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
        return out

    def handle_key(self, key: int) -> bool:
        """Returns False when the loop should stop."""
        if key in (ord("q"), 27):
            return False
        if key == ord("s"):
            self.swap_enabled = not self.swap_enabled
        elif key == ord("e"):
            if self.enhancer is None:
                print("[pipeline] enhancer not available (install requirements-enhance.txt)")
            else:
                self.enhance_enabled = not self.enhance_enabled
        elif key == ord("m"):
            self.mirror = not self.mirror
        elif key == ord("["):
            self.cycle(-1)
        elif key == ord("]"):
            self.cycle(+1)
        elif key == ord("h"):
            self.show_help = not self.show_help
        return True


def run(pipeline: Pipeline, cap, vcam, preview: bool = True) -> None:
    window = "delulucam preview"
    if preview:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print("[pipeline] running — press q in the preview window (or Ctrl-C) to stop")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[pipeline] camera stopped delivering frames, exiting")
                break
            frame = pipeline.process(frame)
            if vcam is not None:
                vcam.send(frame)
            if preview:
                cv2.imshow(window, pipeline.draw_hud(frame))
                if not pipeline.handle_key(cv2.waitKey(1) & 0xFF):
                    break
    except KeyboardInterrupt:
        print("\n[pipeline] interrupted")
    finally:
        cap.release()
        if vcam is not None:
            vcam.close()
        if preview:
            cv2.destroyAllWindows()
