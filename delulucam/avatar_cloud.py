#!/usr/bin/env python3
"""delulucam cloud avatar mode — drives a rented GPU's real-time TensorRT
LivePortrait server instead of the local MLX engine (see avatar.py).

Unlike avatar.py, this file needs no MLX/engine imports — it just captures,
sends, receives, and displays — so it runs in delulucam's own venv:

    python delulucam/avatar_cloud.py mycharacter.png --server wss://HOST:8765 --token ...

NOT YET IMPLEMENTED. Scaffold only: matches server/realtime_server.py's wire
protocol so both sides can be built independently, but the actual send/recv
loop is deferred until there's a rented box on the other end to test
latency and error-handling against.
"""

import argparse
import base64
import json
import sys

import cv2


def parse_args():
    p = argparse.ArgumentParser(
        prog="delulucam-avatar-cloud",
        description="Animate your character via a rented GPU's real-time "
        "server instead of local MLX (full hair + outfit, lower latency, "
        "costs cloud GPU time, your feed leaves this machine).",
        epilog="For your own characters/likeness only — do not impersonate real people.",
    )
    p.add_argument("character", help="character portrait or sheet image")
    p.add_argument("--server", required=True, help="wss://host:port of the rented server")
    p.add_argument("--token", default=None,
                   help="auth token (or set DELULUCAM_SERVER_TOKEN)")
    p.add_argument("-c", "--camera", type=int, default=0)
    p.add_argument("--mirror", action="store_true")
    p.add_argument("--no-vcam", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    img = cv2.imread(args.character, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"could not read character image: {args.character}")
    ok, jpeg = cv2.imencode(".jpg", img)
    if not ok:
        raise SystemExit("could not encode character portrait")
    portrait_b64 = base64.b64encode(jpeg.tobytes()).decode("ascii")

    init_msg = json.dumps({"cmd": "init", "portrait_jpeg_b64": portrait_b64,
                           "token": args.token})
    print(f"[avatar-cloud] would connect to {args.server} and send init "
          f"({len(init_msg)} bytes)")

    # TODO once a server exists to test against:
    #   import websockets, asyncio
    #   async with websockets.connect(args.server) as ws:
    #       await ws.send(init_msg)
    #       cap = cv2.VideoCapture(args.camera)
    #       while True:
    #           ok, frame = cap.read()
    #           if args.mirror: frame = cv2.flip(frame, 1)
    #           _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    #           await ws.send(jpeg.tobytes())
    #           out_bytes = await ws.recv()
    #           out = cv2.imdecode(np.frombuffer(out_bytes, np.uint8), cv2.IMREAD_COLOR)
    #           # then vcam.send(out) / cv2.imshow(...), same as avatar.py's loop
    raise NotImplementedError("send/recv loop pending a real server to test against")


if __name__ == "__main__":
    sys.exit(main())
