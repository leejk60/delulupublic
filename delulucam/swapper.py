"""Face analysis + swapping built on InsightFace (buffalo_l + inswapper_128)."""

from typing import List, Optional

import numpy as np


def pick_providers(prefer_gpu: bool = True) -> List[str]:
    """Order ONNX Runtime execution providers: GPU/NPU first, CPU as fallback."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    preferred = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "ROCMExecutionProvider",
        "CoreMLExecutionProvider",
        "DmlExecutionProvider",
    ]
    providers = [p for p in preferred if prefer_gpu and p in available]
    providers.append("CPUExecutionProvider")
    return providers


class FaceSwapper:
    def __init__(self, inswapper_path: str, det_size: int = 640, prefer_gpu: bool = True):
        # Imported here so `--help` and model downloads work without the heavy deps.
        import insightface
        from insightface.app import FaceAnalysis

        providers = pick_providers(prefer_gpu)
        print(f"[swapper] ONNX providers: {providers}")
        self.analyser = FaceAnalysis(
            name="buffalo_l", providers=providers, allowed_modules=["detection", "recognition"]
        )
        self.analyser.prepare(ctx_id=0, det_size=(det_size, det_size))
        self.swapper = insightface.model_zoo.get_model(inswapper_path, providers=providers)

    def analyse(self, frame_bgr: np.ndarray) -> list:
        """Detect faces in a frame, largest first."""
        faces = self.analyser.get(frame_bgr)
        faces.sort(
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True
        )
        return faces

    def swap(
        self, frame_bgr: np.ndarray, faces: list, ref_face, max_faces: Optional[int] = 1
    ) -> np.ndarray:
        """Replace up to `max_faces` detected faces with the character's face."""
        targets = faces if max_faces is None else faces[:max_faces]
        for face in targets:
            frame_bgr = self.swapper.get(frame_bgr, face, ref_face, paste_back=True)
        return frame_bgr
