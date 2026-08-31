"""Load character sheets and turn them into a swappable identity.

Layout-agnostic: sheets can be a single portrait, a grid collage, a
turnaround with full-body views, detail crops, any resolution. The loader
detects every face, clusters them by identity, picks the dominant character,
and builds one identity embedding from the best views:

- tiny faces (full-body tiles) and low-confidence partials are filtered by
  thresholds *relative* to the sheet's own best view, so any resolution works
- views are weighted by quality (size x detection confidence squared), so
  crisp close-ups dominate partial or awkward crops
- stray faces of other characters on the sheet end up in their own identity
  clusters and are ignored
- very large collages whose faces are too small for one detector pass get a
  second, tiled detection pass
"""

import os
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Cosine similarity above which two faces on one sheet count as the same character.
SAME_CHARACTER_THRESHOLD = 0.35

# Absolute floors: below these, identity embeddings are unreliable regardless
# of sheet size (the recognition model works on 112px crops).
MIN_VIEW_DET_SCORE = 0.50
ABS_MIN_VIEW_HEIGHT_PX = 40

# Within the chosen identity cluster, drop views smaller than this fraction of
# the cluster's tallest view — scale-relative, so it adapts to any layout.
REL_MIN_VIEW_HEIGHT = 0.30

# Warn when even the best view is small: the swap will still run, but the
# identity will be mushy. Suggest adding a close-up portrait to the sheet.
GOOD_VIEW_HEIGHT_PX = 110


@dataclass
class Character:
    name: str
    path: str
    ref_face: object  # insightface Face carrying the averaged embedding
    num_views: int = 1
    thumbnail: np.ndarray = field(default=None, repr=False)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _height(face) -> float:
    return float(face.bbox[3] - face.bbox[1])


def _quality(face) -> float:
    # det_score squared: partial or awkward views score visibly lower and
    # should be strongly discounted; size only mildly rewarded.
    return float(np.sqrt(_height(face))) * float(face.det_score) ** 2


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter))


def _read_sheet(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # normalises alpha/grayscale to BGR
    if img is None:
        raise ValueError(f"could not read image: {path}")
    return img


def _detect(img: np.ndarray, swapper) -> list:
    """Detect faces on the sheet; for large collages where one pass over the
    downscaled sheet finds nothing, retry on overlapping tiles at full res."""
    faces = swapper.analyse(img)
    if faces:
        return faces

    h, w = img.shape[:2]
    if max(h, w) < 1000:
        return []

    print("[character] no faces at full scale — retrying with tiled detection")
    th, tw = int(h * 0.6), int(w * 0.6)
    candidates = []
    for oy in (0, h - th):
        for ox in (0, w - tw):
            for face in swapper.analyse(img[oy : oy + th, ox : ox + tw]):
                # map tile-local geometry back onto the full sheet
                face.bbox = face.bbox + np.array([ox, oy, ox, oy], dtype=face.bbox.dtype)
                if face.kps is not None:
                    face.kps = face.kps + np.array([ox, oy], dtype=face.kps.dtype)
                candidates.append(face)

    deduped = []  # overlapping tiles see the same face twice; keep the best
    for face in sorted(candidates, key=lambda f: f.det_score, reverse=True):
        if all(_iou(face.bbox, kept.bbox) < 0.4 for kept in deduped):
            deduped.append(face)
    return deduped


def _cluster_identities(faces: list) -> List[list]:
    """Greedy identity clustering, best-quality faces first; each cluster's
    first face is its highest-quality anchor."""
    clusters: List[list] = []
    for face in sorted(faces, key=_quality, reverse=True):
        for cluster in clusters:
            if (
                _cosine(cluster[0].normed_embedding, face.normed_embedding)
                >= SAME_CHARACTER_THRESHOLD
            ):
                cluster.append(face)
                break
        else:
            clusters.append([face])
    return clusters


def load_character(path: str, swapper) -> Character:
    """Build a Character from one sheet image using the swapper's analyser."""
    from insightface.app.common import Face

    img = _read_sheet(path)
    faces = _detect(img, swapper)
    if not faces:
        raise ValueError(
            f"no face found on character sheet {path!r} — the swap needs at "
            f"least one clear, mostly-frontal face somewhere on the sheet"
        )

    usable = [
        f for f in faces
        if f.det_score >= MIN_VIEW_DET_SCORE and _height(f) >= ABS_MIN_VIEW_HEIGHT_PX
    ]
    if not usable:
        usable = [max(faces, key=_quality)]

    # The dominant character is the identity cluster with the most total
    # quality — not necessarily the one with the single largest face.
    clusters = _cluster_identities(usable)
    best = max(clusters, key=lambda c: sum(_quality(f) for f in c))
    if len(clusters) > 1:
        print(
            f"[character] {len(clusters) - 1} other identity cluster(s) on the "
            f"sheet ignored (side characters or faces too small/partial to match)"
        )

    anchor = best[0]
    tallest = max(_height(f) for f in best)
    views = [
        f for f in best if f is anchor or _height(f) >= REL_MIN_VIEW_HEIGHT * tallest
    ]
    weights = [_quality(f) for f in views]

    total_w = float(sum(weights))
    for f, w in zip(views, weights):
        x, y = int(f.bbox[0]), int(f.bbox[1])
        print(
            f"[character]   view at ({x},{y}) h={_height(f):.0f}px "
            f"score={f.det_score:.2f} -> {100 * w / total_w:.0f}% of identity"
        )
    if tallest < GOOD_VIEW_HEIGHT_PX:
        print(
            f"[character]   note: best view is only {tallest:.0f}px tall — the "
            f"identity may look generic; add a close-up portrait to the sheet "
            f"for a stronger likeness"
        )

    avg = np.average(
        [f.normed_embedding for f in views], axis=0, weights=np.asarray(weights)
    )
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
            print(f"[character] loading {f}")
            ch = load_character(f, swapper)
            print(f"[character] loaded {ch.name!r} ({ch.num_views} view(s) on sheet)")
            characters.append(ch)
        except ValueError as e:
            print(f"[character] skipping: {e}")
    if not characters:
        raise SystemExit("no usable character sheets — nothing to swap to")
    return characters
