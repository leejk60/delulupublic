#!/usr/bin/env python3
"""delulucam cloud avatar server — real-time face animation over a websocket.

Runs on a rented NVIDIA GPU box, inside the FasterLivePortrait TensorRT
environment (see Dockerfile in this directory). Accepts one JPEG-encoded
webcam frame per message, returns one JPEG-encoded animated frame, reusing
the same FasterLivePortraitPipeline class the upstream repo's batch API uses
— just called in a streaming loop instead of over a whole uploaded video.

NOT YET IMPLEMENTED. This is a scaffold: the wire protocol and structure are
decided, the actual pipeline wiring is intentionally deferred until there is
a rented GPU to develop and measure against — guessing at TensorRT-specific
behavior without hardware to test on would be worse than not writing it.

Wire protocol (v0, subject to change once real latency numbers exist):
    client -> server: binary websocket message, one JPEG frame
    server -> client: binary websocket message, one JPEG frame (or a small
                       JSON error message on failure)
    First message from a session should instead be a JSON control message:
        {"cmd": "init", "portrait_jpeg_b64": "...", "token": "..."}
    to set the character portrait and authenticate before streaming begins.
"""

import argparse
import asyncio
import json
import os

# TODO: import the pipeline once this runs on the actual GPU box:
# from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline

AUTH_TOKEN = os.environ.get("DELULUCAM_SERVER_TOKEN")


async def handle_session(websocket):
    if not AUTH_TOKEN:
        await websocket.close(code=1011, reason="server has no DELULUCAM_SERVER_TOKEN set")
        return

    init_raw = await websocket.recv()
    try:
        init = json.loads(init_raw)
    except (TypeError, ValueError):
        await websocket.close(code=1002, reason="expected a JSON init message first")
        return
    if init.get("cmd") != "init" or init.get("token") != AUTH_TOKEN:
        await websocket.close(code=1008, reason="bad init or token")
        return

    # TODO: decode init["portrait_jpeg_b64"], run pipe.prepare_source(...)
    # TODO: loop: recv JPEG frame bytes -> pipe.run(...) -> paste-back ->
    #       encode JPEG -> send bytes back. Mirrors avatar.py's local loop,
    #       with cv2.imdecode/imencode replacing the local camera/vcam calls.
    raise NotImplementedError("pipeline wiring pending a real GPU box to test against")


async def main():
    import websockets  # local import: not needed for --help

    p = argparse.ArgumentParser(description="delulucam cloud avatar server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()

    print(f"[server] listening on {args.host}:{args.port}")
    async with websockets.serve(handle_session, args.host, args.port, max_size=None):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
