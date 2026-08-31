#!/usr/bin/env python3
"""delulucam avatar mode — full hair + outfit, delulustream-style, fully local.

Instead of pasting the character's face onto your body (face mode), avatar
mode animates the character's PORTRAIT itself with your webcam motion using
LivePortrait: hair, outfit and background come pixel-perfect from your
character sheet, and your head pose, expressions, blinks and lip movement
drive it live. The animated portrait is streamed to a virtual camera.

Runs on Apple Silicon via the MLX port of FasterLivePortrait:
    https://github.com/ivanfioravanti/fasterliveportrait-mlx

Usage (from inside the fasterliveportrait-mlx checkout):
    uv pip install pyvirtualcam
    uv run python /path/to/delulupublic/delulucam/avatar.py mysheet.png

This file is deliberately standalone (no delulucam imports) so it can run
inside the fasterliveportrait-mlx virtualenv without installing delulucam.
"""

import argparse
import os
import platform
import sys
import time

import cv2
import numpy as np

PROFILES = ("quality", "reference", "speed", "turbo", "ultra")


def find_flp_dir(explicit: str) -> str:
    candidates = [
        explicit,
        os.environ.get("DELULUCAM_FLP"),
        os.getcwd(),
        os.path.expanduser("~/fasterliveportrait-mlx"),
    ]
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, "configs", "mlx_infer.yaml")):
            return os.path.abspath(c)
    raise SystemExit(
        "could not find the fasterliveportrait-mlx checkout — clone it "
        "(git clone https://github.com/ivanfioravanti/fasterliveportrait-mlx) "
        "and pass --flp /path/to/it, or run this from inside that directory"
    )


def lmk_bbox(lmk: np.ndarray):
    xs, ys = lmk[..., 0].reshape(-1), lmk[..., 1].reshape(-1)
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _seams(gray: np.ndarray, axis: int, min_score: float = 0.2, min_sep: int = 40):
    """Find collage tile seams: coordinates where adjacent rows/columns differ
    for a large fraction of their length. Two guards separate true seams from
    long content edges (hair lines, figure contours): a real seam is a narrow
    isolated 1px line (its +-5px neighbourhood is quiet, while a soft content
    edge smears into a wide band), and it does not hug the image border."""
    d = np.abs(np.diff(gray, axis=axis))
    frac = (d > 20).mean(axis=1 - axis)
    n = len(frac)
    seams = []
    for i in np.where(frac >= min_score)[0]:
        if not (0.08 * n <= i <= 0.92 * n):
            continue
        lo, hi = max(0, i - 5), min(n - 1, i + 5)
        if max(frac[lo], frac[hi]) >= max(0.05, 0.25 * frac[i]):
            continue
        if seams and i - seams[-1][0] < min_sep:
            if frac[i] > seams[-1][1]:
                seams[-1] = (i, frac[i])
        else:
            seams.append((i, frac[i]))
    return [int(s[0]) + 1 for s in seams]


def find_tile(gray: np.ndarray, fx: float, fy: float):
    """Bounds of the collage tile containing point (fx, fy). Vertical seams are
    detected across the whole sheet, horizontal seams only within the point's
    column, so uneven grids work. A plain portrait yields the whole image."""
    h, w = gray.shape
    vxs = [0] + _seams(gray, axis=1) + [w]
    x1 = max(v for v in vxs if v <= fx)
    x2 = min(v for v in vxs if v > fx)
    hys = [0] + _seams(gray[:, x1:x2], axis=0) + [h]
    y1 = max(v for v in hys if v <= fy)
    y2 = min(v for v in hys if v > fy)
    return x1, y1, x2, y2


def prepare_portrait(img_path: str, pipe, view: int, margins, out_dir: str) -> str:
    """Cut a portrait (hair + shoulders included) around one face of the sheet.

    A character sheet is usually a collage; LivePortrait wants one portrait.
    We find the faces with the pipeline's own analyser, snap each to its
    collage tile via seam detection, rank views by bust-portrait framing, and
    crop within the chosen tile. Margins are fractions of the face height:
    (top, sides, bottom)."""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"could not read character image: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)

    faces = pipe.model_dict["face_analysis"].predict(img)
    if not faces:
        raise SystemExit(
            f"no face found in {img_path} — avatar mode needs a clear portrait "
            f"or a sheet with at least one clear face"
        )

    candidates = []
    for lmk in faces:
        box = lmk_bbox(lmk)
        fh = box[3] - box[1]
        tile = find_tile(gray, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        # Bust-portrait framing: the face should be a substantial but not
        # overwhelming part of its tile — rules out full-body tiles (face is a
        # sliver) and detail crops (face overflows the tile).
        ratio = fh / max(1.0, tile[3] - tile[1])
        framed_well = 0.18 <= ratio <= 0.85
        candidates.append((fh if framed_well else fh * 0.1, box, tile, ratio))
    candidates.sort(key=lambda c: c[0], reverse=True)

    if view >= len(candidates):
        raise SystemExit(f"--view {view} but only {len(candidates)} face(s) found")
    _, (x1, y1, x2, y2), (tx1, ty1, tx2, ty2), ratio = candidates[view]
    fh = y2 - y1
    top, side, bottom = margins

    cx1 = max(tx1, int(x1 - side * fh))
    cx2 = min(tx2, int(x2 + side * fh))
    cy1 = max(ty1, int(y1 - top * fh))
    cy2 = min(ty2, int(y2 + bottom * fh))
    crop = img[cy1:cy2, cx1:cx2]

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "avatar_portrait.png")
    cv2.imwrite(out, crop)
    print(
        f"[avatar] portrait: view {view} of {len(candidates)}, tile "
        f"({tx1},{ty1})-({tx2},{ty2}), face/tile ratio {ratio:.2f}, cropped "
        f"{crop.shape[1]}x{crop.shape[0]} -> {out}"
    )
    print(
        "[avatar] wrong view or framing? try --view 1/2/..., tune --margins, "
        "or pass a dedicated portrait image"
    )
    return out


class Canvas:
    """Letterbox the animated portrait onto a fixed webcam-shaped frame with a
    blurred backdrop made from the portrait itself."""

    def __init__(self, width: int, height: int, portrait_bgr: np.ndarray):
        self.w, self.h = width, height
        bg = cv2.resize(portrait_bgr, (width, height), interpolation=cv2.INTER_AREA)
        bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=25)
        self.bg = (bg * 0.55).astype(np.uint8)  # darken so the avatar pops
        self._geom = None

    def compose(self, frame_bgr: np.ndarray) -> np.ndarray:
        fh, fw = frame_bgr.shape[:2]
        if self._geom is None or self._geom[0] != (fw, fh):
            scale = min(self.w / fw, self.h / fh)
            tw, th = int(fw * scale), int(fh * scale)
            self._geom = ((fw, fh), tw, th, (self.w - tw) // 2, (self.h - th) // 2)
        _, tw, th, ox, oy = self._geom
        out = self.bg.copy()
        out[oy : oy + th, ox : ox + tw] = cv2.resize(
            frame_bgr, (tw, th), interpolation=cv2.INTER_LINEAR
        )
        return out


def parse_args():
    p = argparse.ArgumentParser(
        prog="delulucam-avatar",
        description="Animate your character's portrait with your webcam and "
        "stream it as a virtual camera (full hair + outfit).",
        epilog="For your own characters/likeness only — do not impersonate real people.",
    )
    p.add_argument("character", help="character sheet or portrait image")
    p.add_argument("--flp", default=None, help="path to the fasterliveportrait-mlx checkout")
    p.add_argument("-c", "--camera", type=int, default=0, help="driving camera index")
    p.add_argument("--profile", choices=PROFILES, default="speed",
                   help="MLX speed/quality profile (default speed; try turbo if choppy)")
    p.add_argument("--view", type=int, default=0,
                   help="which face on the sheet to build the portrait around, "
                   "largest first (default 0)")
    p.add_argument("--margins", default="1.0,0.9,2.6",
                   help="portrait crop margins around the face as fractions of face "
                   "height: top,sides,bottom (default 1.0,0.9,2.6)")
    p.add_argument("--no-crop", action="store_true",
                   help="use the character image as-is (it is already a portrait)")
    p.add_argument("--canvas", default="1280x720",
                   help="output size WxH, letterboxed with a blurred backdrop; "
                   "'none' streams the raw portrait size (default 1280x720)")
    p.add_argument("--no-paste-back", action="store_true",
                   help="stream the raw 512px animated face crop instead of the "
                   "full portrait (faster, but loses outfit framing)")
    p.add_argument("--mirror", action="store_true", help="mirror the driving camera")
    p.add_argument("--fps", type=int, default=25, help="virtual camera fps (default 25)")
    p.add_argument("--no-vcam", action="store_true", help="preview only")
    p.add_argument("--no-preview", action="store_true", help="no preview window")
    return p.parse_args()


def main():
    args = parse_args()
    if not (sys.platform == "darwin" and platform.machine() == "arm64"):
        print("[avatar] warning: MLX runs on Apple Silicon Macs; this is unlikely "
              "to work on this machine", file=sys.stderr)

    character = os.path.abspath(args.character)
    flp = find_flp_dir(args.flp)
    os.chdir(flp)  # their configs use repo-relative checkpoint paths
    sys.path.insert(0, flp)
    print(f"[avatar] engine: {flp}")

    from omegaconf import OmegaConf

    from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline
    from src.runtime_assets import ensure_runtime_assets
    from src.utils.mlx_profiles import apply_mlx_profile

    cfg = OmegaConf.load(os.path.join(flp, "configs", "mlx_infer.yaml"))
    ensure_runtime_assets(cfg)  # downloads MLX weights from Hugging Face once
    cfg.infer_params.flag_pasteback = not args.no_paste_back
    apply_mlx_profile(args.profile)
    print(f"[avatar] MLX profile: {args.profile}")

    pipe = FasterLivePortraitPipeline(cfg=cfg, is_animal=False)

    if args.no_crop:
        portrait_path = character
    else:
        margins = tuple(float(v) for v in args.margins.split(","))
        if len(margins) != 3:
            raise SystemExit("--margins wants three numbers: top,sides,bottom")
        portrait_path = prepare_portrait(
            character, pipe, args.view, margins,
            os.path.join(os.path.expanduser("~"), ".delulucam", "portraits"),
        )
    if not pipe.prepare_source(portrait_path, realtime=True):
        raise SystemExit(f"LivePortrait found no usable face in {portrait_path}")
    portrait_bgr = cv2.cvtColor(pipe.src_imgs[0], cv2.COLOR_RGB2BGR)

    canvas = None
    if args.canvas.lower() != "none":
        cw, ch = (int(v) for v in args.canvas.lower().split("x"))
        canvas = Canvas(cw, ch, portrait_bgr)
        out_w, out_h = cw, ch
    elif args.no_paste_back:
        out_w, out_h = 512, 512
    else:
        out_h, out_w = portrait_bgr.shape[:2]

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")

    vcam = None
    if not args.no_vcam:
        import pyvirtualcam
        from pyvirtualcam import PixelFormat

        vcam = pyvirtualcam.Camera(width=out_w, height=out_h, fps=args.fps,
                                   fmt=PixelFormat.BGR)
        print(f"[avatar] streaming {out_w}x{out_h} to {vcam.device}")

    def fit(frame_bgr):
        """Letterbox any frame to the fixed output size."""
        if canvas is not None:
            return canvas.compose(frame_bgr)
        fh, fw = frame_bgr.shape[:2]
        if (fw, fh) == (out_w, out_h):
            return frame_bgr
        scale = min(out_w / fw, out_h / fh)
        tw, th = int(fw * scale), int(fh * scale)
        boxed = np.zeros((out_h, out_w, 3), np.uint8)
        ox, oy = (out_w - tw) // 2, (out_h - th) // 2
        boxed[oy : oy + th, ox : ox + tw] = cv2.resize(frame_bgr, (tw, th))
        return boxed

    idle = fit(portrait_bgr)
    print("[avatar] running — look at the camera with a neutral face for the "
          "first frame (that pose becomes 'rest'); press r to recalibrate, "
          "s to drop out of character (real camera), m to mirror, q to quit")

    first, mirror, in_character = True, args.mirror, True
    times = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[avatar] camera stopped delivering frames")
                break
            if mirror:
                frame = cv2.flip(frame, 1)

            if not in_character:
                out = fit(frame)  # passthrough: the real you, stream stays live
                first = True  # recalibrate rest pose when coming back
            else:
                t0 = time.perf_counter()
                _, out_crop, out_org, _ = pipe.run(
                    frame, pipe.src_imgs[0], pipe.src_infos[0],
                    first_frame=first, realtime=True,
                )
                first = False
                times.append(time.perf_counter() - t0)

                if out_crop is None:
                    out = idle  # no face in the driving frame: hold the still portrait
                else:
                    rgb = out_crop if args.no_paste_back else out_org
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    out = fit(bgr)

            if vcam is not None:
                vcam.send(out)
                vcam.sleep_until_next_frame()
            if not args.no_preview:
                hud = out.copy()
                if times:
                    fps_now = 1.0 / max(1e-6, float(np.mean(times[-30:])))
                    cv2.putText(hud, f"{fps_now:4.1f} fps", (12, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                    cv2.putText(hud, f"{fps_now:4.1f} fps", (12, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 120), 2)
                cv2.imshow(
                    "delulucam avatar (q quit / s in-out of character / "
                    "r recalibrate / m mirror)", hud)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("s"):
                    in_character = not in_character
                    print(f"[avatar] {'in character' if in_character else 'out of character (real camera)'}")
                elif key == ord("r"):
                    first = True
                    print("[avatar] recalibrated rest pose")
                elif key == ord("m"):
                    mirror = not mirror
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if vcam is not None:
            vcam.close()
        cv2.destroyAllWindows()
        if times:
            print(f"[avatar] avg inference: {1000 * float(np.mean(times)):.0f} ms/frame")


if __name__ == "__main__":
    main()
