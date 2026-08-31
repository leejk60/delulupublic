"""Load character sheets and turn them into a swappable identity.

A character sheet often shows the same character several times (front view,
profile, expressions...). We detect every face on the sheet, keep the ones
that match the dominant identity, and average their embeddings — the swap
gets noticeably more stable than from a single crop.
"""

import os
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Cosine similarity above which two faces on one sheet count as the same character.
SAME_CHARACTER_THRESHOLD = 0.35


@dataclass
class Character:
    name: str
    path: str
    ref_face: object  # insightface Face carrying the averaged embedding
    num_views: int = 1
    thumbnail: np.ndarray = field(default=None, repr=False)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def load_character(path: str, swapper) -> Character:
    """Build a Character from one sheet image using the swapper's analyser."""
    from insightface.app.common import Face

    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"could not read image: {path}")

    faces = swapper.analyse(img)
    if not faces:
        raise ValueError(
            f"no face found on character sheet {path!r} — the swap needs a "
            f"clear, mostly-frontal face on the sheet"
        )

    # Anchor on the largest face, then pull in other views of the same character.
    anchor = faces[0]
    views = [anchor]
    for face in faces[1:]:
        if _cosine(anchor.normed_embedding, face.normed_embedding) >= SAME_CHARACTER_THRESHOLD:
            views.append(face)

    avg = np.mean([f.normed_embedding for f in views], axis=0)
    # Face.normed_embedding is derived from .embedding, so storing the averaged
    # vector as .embedding yields a re-normalised average identity.
    ref_face = Face(embedding=avg.astype(np.float32))

    x1, y1, x2, y2 = (int(v) for v in anchor.bbox)
    h, w = img.shape[:2]
    pad = int(0.25 * max(x2 - x1, y2 - y1))
    thumb = img[max(0, y1 - pad) : min(h, y2 + pad), max(0, x1 - pad) : min(w, x2 + pad)]

    name = os.path.splitext(os.path.basename(path))[0]
    return Character(
        name=name, path=path, ref_face=ref_face, num_views=len(views), thumbnail=thumb
    )


def load_characters(paths: List[str], swapper) -> List[Character]:
    """Load characters from image paths and/or directories of images."""
    files: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            for entry in sorted(os.listdir(p)):
                if os.path.splitext(entry)[1].lower() in IMAGE_EXTS:
                    files.append(os.path.join(p, entry))
        else:
            files.append(p)

    characters = []
    for f in files:
        try:
            ch = load_character(f, swapper)
            print(f"[character] loaded {ch.name!r} ({ch.num_views} view(s) on sheet)")
            characters.append(ch)
        except ValueError as e:
            print(f"[character] skipping: {e}")
    if not characters:
        raise SystemExit("no usable character sheets — nothing to swap to")
    return characters
