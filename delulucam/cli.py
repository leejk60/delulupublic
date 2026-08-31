"""Command-line entry point for delulucam."""

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="delulucam",
        description=(
            "Local virtual-cam character filter: transforms your webcam feed "
            "into one of your characters (from a character sheet image) and "
            "streams it as a virtual camera."
        ),
        epilog=(
            "This tool is for transforming yourself into your OWN characters or "
            "your own likeness. Do not use it to impersonate real people, and "
            "disclose the filter where honesty about your appearance matters."
        ),
    )
    p.add_argument(
        "characters",
        nargs="*",
        default=["characters"],
        help="character sheet image(s) and/or folder(s) of sheets (default: ./characters)",
    )
    p.add_argument("-c", "--camera", type=int, default=0, help="camera index (default 0)")
    p.add_argument("--list-cameras", action="store_true", help="probe camera indices and exit")
    p.add_argument("--width", type=int, default=1280, help="capture width (default 1280)")
    p.add_argument("--height", type=int, default=720, help="capture height (default 720)")
    p.add_argument("--fps", type=int, default=30, help="target fps (default 30)")
    p.add_argument("--no-vcam", action="store_true", help="preview only, no virtual camera")
    p.add_argument("--no-preview", action="store_true", help="no preview window (headless)")
    p.add_argument("--mirror", action="store_true", help="mirror the image horizontally")
    p.add_argument(
        "--enhance", action="store_true",
        help="enable GFPGAN face enhancement (needs requirements-enhance.txt; costs fps)",
    )
    p.add_argument(
        "--det-size", type=int, default=640,
        help="face detector input size; 320 is ~2x faster, slightly less robust (default 640)",
    )
    p.add_argument(
        "--detect-every", type=int, default=1,
        help="run face detection every N frames, reusing positions between (default 1; "
        "2-3 boosts CPU fps at the cost of slight lag on fast movement)",
    )
    p.add_argument(
        "--max-faces", type=int, default=1,
        help="how many faces in frame to transform, largest first; 0 = all (default 1)",
    )
    p.add_argument("--cpu", action="store_true", help="force CPU inference")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_cameras:
        from .vcam import list_cameras

        cams = list_cameras()
        if not cams:
            print("no working cameras found (indices 0-9)")
        for idx, res in cams:
            print(f"  camera {idx}: {res}")
        return 0

    from . import models
    from .character import load_characters
    from .pipeline import Pipeline, run
    from .swapper import FaceSwapper
    from .vcam import open_capture, open_vcam

    inswapper_path = models.ensure_model("inswapper_128.onnx")
    swapper = FaceSwapper(inswapper_path, det_size=args.det_size, prefer_gpu=not args.cpu)
    characters = load_characters(args.characters, swapper)

    enhancer = None
    if args.enhance:
        from . import enhancer as enhancer_mod

        enhancer = enhancer_mod.try_create(models.ensure_model("GFPGANv1.4.pth"))

    cap = open_capture(args.camera, args.width, args.height, args.fps)
    import cv2

    ok, first = cap.read()
    if not ok or first is None:
        print("camera opened but delivered no frame", file=sys.stderr)
        return 1
    height, width = first.shape[:2]
    print(f"[cli] capturing {width}x{height} from camera {args.camera}")

    vcam = open_vcam(width, height, args.fps, enabled=not args.no_vcam)
    pipeline = Pipeline(
        swapper,
        characters,
        enhancer=enhancer,
        detect_every=args.detect_every,
        max_faces=None if args.max_faces == 0 else args.max_faces,
        mirror=args.mirror,
    )
    run(pipeline, cap, vcam, preview=not args.no_preview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
