"""Optional GFPGAN face enhancement (sharpens the swapped face; costs FPS).

GFPGAN and torch are heavy, so they are an optional install:
    pip install -r requirements-enhance.txt
"""

from typing import Optional

import numpy as np


class FaceEnhancer:
    def __init__(self, model_path: str, upscale: int = 1):
        try:
            from gfpgan import GFPGANer
        except ImportError as e:
            raise RuntimeError(
                "GFPGAN is not installed — run `pip install -r requirements-enhance.txt` "
                "to enable --enhance"
            ) from e
        self._gfpgan = GFPGANer(
            model_path=model_path, upscale=upscale, arch="clean", channel_multiplier=2
        )

    def enhance(self, frame_bgr: np.ndarray) -> np.ndarray:
        _, _, restored = self._gfpgan.enhance(
            frame_bgr, has_aligned=False, only_center_face=False, paste_back=True
        )
        return restored if restored is not None else frame_bgr


def try_create(model_path: str) -> Optional[FaceEnhancer]:
    try:
        return FaceEnhancer(model_path)
    except RuntimeError as e:
        print(f"[enhancer] disabled: {e}")
        return None
