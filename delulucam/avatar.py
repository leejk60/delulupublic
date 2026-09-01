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

AVATAR_VERSION = "0.6.0"

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


def find_tile(gray: np.ndarray, box, inset: float):
    """Bounds of the collage tile containing the face at `box`. Vertical seams
    are detected across the whole sheet, horizontal seams only within the
    face's column, so uneven grids work. A seam cutting through the face box
    interior is a content edge, not a tile boundary — no collage splits a face
    across tiles — so it is ignored; a seam just outside the box (e.g. right
    under the chin) is a legitimate tile edge and kept. A plain portrait
    yields the whole image."""
    h, w = gray.shape
    x1, y1, x2, y2 = box
    fx, fy = (x1 + x2) / 2, (y1 + y2) / 2
    vxs = [0] + [s for s in _seams(gray, axis=1) if not (x1 + inset < s < x2 - inset)] + [w]
    tx1 = max(v for v in vxs if v <= fx)
    tx2 = min(v for v in vxs if v > fx)
    hys = [0] + [
        s for s in _seams(gray[:, tx1:tx2], axis=0) if not (y1 + inset < s < y2 - inset)
    ] + [h]
    ty1 = max(v for v in hys if v <= fy)
    ty2 = min(v for v in hys if v > fy)
    return tx1, ty1, tx2, ty2


def write_portrait(img: np.ndarray, out_dir: str, max_dim: int) -> str:
    """Save the source portrait, capped to max_dim pixels: the animated face
    crop is 512px regardless, so a huge portrait only slows paste-back."""
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
        print(f"[avatar] portrait downscaled to {img.shape[1]}x{img.shape[0]} "
              f"for speed (--portrait-size to change)")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "avatar_portrait.png")
    cv2.imwrite(out, img)
    return out


def mac_camera_names():
    """Camera names via system_profiler; AVFoundation (and therefore OpenCV)
    indices normally follow this order."""
    if sys.platform != "darwin":
        return []
    import json
    import subprocess

    try:
        out = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"],
            capture_output=True, text=True, timeout=20,
        )
        return [c.get("_name", "?") for c in json.loads(out.stdout).get("SPCameraDataType", [])]
    except Exception:
        return []


def list_cameras(max_index: int = 10):
    names = mac_camera_names()
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        opened = False
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                opened = True
                name = names[i] if i < len(names) else ""
                found.append((i, f"{frame.shape[1]}x{frame.shape[0]}", name))
        cap.release()
        if not opened and found:
            break  # macOS camera indices are contiguous; stop at the first gap
    return found


def prepare_portrait(img_path: str, pipe, view: int, margins, out_dir: str,
                     max_dim: int) -> str:
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
        tile = find_tile(gray, box, inset=0.05 * fh)
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

    out = write_portrait(crop, out_dir, max_dim)
    print(
        f"[avatar] portrait: view {view} of {len(candidates)}, tile "
        f"({tx1},{ty1})-({tx2},{ty2}), face/tile ratio {ratio:.2f}, cropped "
        f"{crop.shape[1]}x{crop.shape[0]} -> {out}"
    )
    print(
        f"[avatar] check the framing:  open {out}\n"
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
    p.add_argument("character", nargs="?", help="character sheet or portrait image")
    p.add_argument("--flp", default=None, help="path to the fasterliveportrait-mlx checkout")
    p.add_argument("-c", "--camera", type=int, default=0, help="driving camera index")
    p.add_argument("--list-cameras", action="store_true",
                   help="probe camera indices and exit (on a Mac mini, index 0 is "
                   "often the iPhone Continuity Camera, not your webcam)")
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
    p.add_argument("--portrait-size", type=int, default=960,
                   help="cap the source portrait's longest side in pixels — bigger "
                   "is slower with no gain in face detail (default 960)")
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
    if args.list_cameras:
        cams = list_cameras()
        if not cams:
            print("no working cameras found (indices 0-9)")
        for idx, res, name in cams:
            print(f"  camera {idx}: {res}  {name}")
        if any("OBS Virtual Camera" in c[2] for c in cams):
            print("note: never pick OBS Virtual Camera as the input — that is "
                  "this app's own OUTPUT (feedback loop); pick your real webcam")
        return
    if not args.character:
        raise SystemExit("give me a character sheet or portrait image "
                         "(or --list-cameras to probe cameras)")
    if not (sys.platform == "darwin" and platform.machine() == "arm64"):
        print("[avatar] warning: MLX runs on Apple Silicon Macs; this is unlikely "
              "to work on this machine", file=sys.stderr)

    character = os.path.abspath(args.character)
    flp = find_flp_dir(args.flp)
    os.chdir(flp)  # their configs use repo-relative checkpoint paths
    sys.path.insert(0, flp)
    print(f"[avatar] engine: {flp}")

    # The profile's MLX tuning flags are env vars read AT IMPORT TIME by the
    # engine's modules — apply them before anything from src.* is imported,
    # or the profile silently has no effect.
    from src.utils.mlx_profiles import apply_mlx_profile

    apply_mlx_profile(args.profile)
    print(f"[avatar] delulucam avatar v{AVATAR_VERSION}")
    print(f"[avatar] MLX profile: {args.profile} "
          f"(warp interval={os.environ.get('FLP_MLX_TEMPORAL_WARP_INTERVAL')}, "
          f"threshold={os.environ.get('FLP_MLX_TEMPORAL_WARP_THRESHOLD')})")

    from omegaconf import OmegaConf

    from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline
    from src.runtime_assets import ensure_runtime_assets

    cfg = OmegaConf.load(os.path.join(flp, "configs", "mlx_infer.yaml"))
    ensure_runtime_assets(cfg)  # downloads MLX weights from Hugging Face once
    cfg.infer_params.flag_pasteback = not args.no_paste_back

    pipe = FasterLivePortraitPipeline(cfg=cfg, is_animal=False)

    portraits_dir = os.path.join(os.path.expanduser("~"), ".delulucam", "portraits")
    if args.no_crop:
        img = cv2.imread(character, cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"could not read character image: {character}")
        if max(img.shape[:2]) > args.portrait_size:
            portrait_path = write_portrait(img, portraits_dir, args.portrait_size)
        else:
            portrait_path = character
    else:
        margins = tuple(float(v) for v in args.margins.split(","))
        if len(margins) != 3:
            raise SystemExit("--margins wants three numbers: top,sides,bottom")
        portrait_path = prepare_portrait(
            character, pipe, args.view, margins, portraits_dir, args.portrait_size,
        )
    if not pipe.prepare_source(portrait_path, realtime=True):
        raise SystemExit(f"LivePortrait found no usable face in {portrait_path}")
    portrait_bgr = cv2.cvtColor(pipe.src_imgs[0], cv2.COLOR_RGB2BGR)

    # The engine SKIPS paste-back in realtime mode (out_org comes back as the
    # untouched source image), so we composite the animated 512px face crop
    # into the full portrait ourselves with the engine's own utility.
    paste_data = None
    if not args.no_paste_back:
        from src.utils.crop import paste_back_numpy

        face_info = pipe.src_infos[0]
        if face_info and isinstance(face_info[0], (list, tuple)):
            face_info = face_info[0]
        mask_ori, m_c2o = face_info[-2], face_info[-1]
        if mask_ori is not None and m_c2o is not None:
            paste_data = (paste_back_numpy, mask_ori, m_c2o)
        else:
            print("[avatar] paste-back data unavailable — streaming the face "
                  "crop instead of the full portrait")

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
    # 720p is plenty for driving-motion capture and cheaper than 1080p+.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Non-square camera input must be face-cropped before motion extraction —
    # without this the engine squashes the whole 16:9 frame into its 256px
    # motion input and extracts almost no motion (mirrors the engine's own
    # maybe_enable_auto_driving_crop).
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if cam_w > 0 and cam_h > 0 and cam_w != cam_h and not cfg.infer_params.flag_crop_driving_video:
        cfg.infer_params.flag_crop_driving_video = True
        print(f"[avatar] driving crop enabled for non-square camera input ({cam_w}x{cam_h})")

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

    first, mirror, in_character, face_seen = True, args.mirror, True, True
    show_driver = True  # 'd' hides the driver picture-in-picture (preview only)
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

                face_seen = out_crop is not None
                if out_crop is None:
                    out = idle  # no face in the driving frame: hold the still portrait
                else:
                    if paste_data is not None:
                        paste, mask_ori, m_c2o = paste_data
                        rgb = np.asarray(
                            paste(out_crop, m_c2o, pipe.src_imgs[0].copy(), mask_ori),
                            dtype=np.uint8,
                        )
                    else:
                        rgb = out_crop
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    out = fit(bgr)

            if vcam is not None:
                vcam.send(out)
                vcam.sleep_until_next_frame()
            if not args.no_preview:
                hud = out.copy()
                if show_driver:
                    # picture-in-picture of what the driving camera sees, so a
                    # wrong -c index or a camera pointed elsewhere is obvious
                    dh, dw = frame.shape[:2]
                    tw = max(64, hud.shape[1] // 5)
                    th = max(36, int(tw * dh / max(1, dw)))
                    if th < hud.shape[0] - 8 and tw < hud.shape[1] - 8:
                        x0, y0 = hud.shape[1] - tw - 8, hud.shape[0] - th - 8
                        hud[y0 : y0 + th, x0 : x0 + tw] = cv2.resize(frame, (tw, th))
                        color = (80, 220, 120) if face_seen else (60, 60, 230)
                        cv2.rectangle(hud, (x0, y0), (x0 + tw, y0 + th), color, 2)
                        label = (f"driver cam {args.camera}: "
                                 f"{'tracking' if face_seen else 'NO FACE'}")
                        cv2.putText(hud, label, (x0, y0 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                        cv2.putText(hud, label, (x0, y0 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                if in_character and not face_seen:
                    msg = (f"no face seen by camera {args.camera} - wrong camera? "
                           f"try --list-cameras")
                    cv2.putText(hud, msg, (12, hud.shape[0] - 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                    cv2.putText(hud, msg, (12, hud.shape[0] - 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 230), 2)
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
                elif key == ord("d"):
                    show_driver = not show_driver
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
