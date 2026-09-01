# delulucam

A **local virtual-cam character filter**: point it at your webcam (e.g. an
OBSBOT), give it a character sheet image, and it live-transforms your face
into that character and streams the result as a **virtual camera** that any
app — OBS, Zoom, Meet, Discord, TikTok Live Studio — can pick as its webcam.

Think delulustream.com, but running entirely on your own machine: no upload,
no cloud, your camera feed never leaves your computer.

```
OBSBOT cam ──▶ face detect ──▶ swap to character ──▶ (optional enhance) ──▶ virtual cam
                                    ▲
                       your character sheet image(s)
```

## Fair use

This tool is for turning yourself into **your own characters** (or your own
likeness). Don't use it to impersonate real people, and disclose that you're
using an AI filter anywhere honesty about your appearance matters. You are
responsible for complying with the platform rules and laws that apply to you.

## Setup — macOS on Apple Silicon

One-time prerequisites:

```bash
xcode-select --install                 # C compiler (insightface builds from source)
brew install python@3.11               # 3.9–3.11 work best with insightface
brew install --cask obs                # ships the virtual camera driver
```

Then open **OBS once** and click *Start Virtual Camera* a single time — macOS
will ask you to approve the camera system extension in System Settings. After
that, the virtual cam is available system-wide and you never need OBS running
for delulucam to work.

Install delulucam:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The stock `onnxruntime` wheel for Apple Silicon already includes the
**CoreML** execution provider, and delulucam auto-selects it — inference runs
on the Neural Engine/GPU where the ops allow, with CPU fallback. An M-series
Mac handles 720p30 comfortably.

**Camera permission:** on first run macOS prompts that your terminal app
(Terminal, iTerm2, or VS Code — whichever you launch from) wants camera
access. Grant it, or OpenCV will silently deliver no frames. If you denied it
once: System Settings → Privacy & Security → Camera.

**Finding the OBSBOT:** run `python -m delulucam --list-cameras`. Note that
Continuity Camera (your iPhone) often grabs index 0 on Macs, so the OBSBOT
may be index 1 or 2. If OBSBOT Center holds the camera exclusively, quit it
or use the second device the camera exposes.

<details>
<summary>Setup on Windows / Linux</summary>

- **Windows** — install [OBS Studio](https://obsproject.com) once for the
  virtual cam driver, then the same `pip install -r requirements.txt` in a
  Python 3.9–3.11 venv (`.venv\Scripts\activate`).
- **Linux** — `sudo apt install v4l2loopback-dkms`, then
  `sudo modprobe v4l2loopback devices=1 card_label=delulucam exclusive_caps=1`.
</details>

On first run the face-swap model (`inswapper_128.onnx`, ~530 MB with the
detector models) is downloaded to `~/.delulucam/models` and checksum-verified.

## Usage

Drop one or more character sheets into `characters/` (PNG/JPG/WebP), then:

```bash
python -m delulucam                      # uses ./characters, camera 0
python -m delulucam mycharacter.png      # a single sheet
python -m delulucam --list-cameras       # find your OBSBOT's index
python -m delulucam -c 1 --mirror        # pick camera 1, mirror the view
```

Then in OBS/Zoom/etc., select the **virtual camera** (named after the OBS
Virtual Camera / v4l2loopback device) as your webcam.

**Any sheet layout works** — a single portrait, a grid collage, a turnaround
with full-body views, detail crops, any resolution. The loader detects every
face on the sheet, clusters them by identity, picks the dominant character
(stray side-character faces land in their own clusters and are ignored), and
**averages the good views into one identity**: quality-weighted, so crisp
close-ups dominate over partial or tiny views, with all size thresholds
relative to the sheet's own best view rather than fixed pixels. Very large
collages whose faces are too small for one detector pass get a second, tiled
pass automatically. On load it prints exactly which views were used and how
much each contributes — and warns you if the sheet only offers small faces
(add one clear close-up portrait for the strongest likeness). One image file
= one character; multiple files = switchable characters.

### Preview hotkeys

| Key | Action |
|-----|--------|
| `q` / `Esc` | quit |
| `s` | toggle the swap on/off (raw camera passthrough) |
| `[` / `]` | previous / next character |
| `e` | toggle GFPGAN enhancement (if installed) |
| `m` | toggle mirroring |
| `h` | hide/show the help overlay |

The HUD is drawn only on the local preview — the virtual cam gets the clean
feed.

## Performance

The swap runs comfortably in real time on a GPU and adequately on a modern
CPU at 720p. Knobs, roughly in order of impact:

1. **Accelerated inference** — Apple Silicon: nothing to do, CoreML is in the
   stock `onnxruntime` wheel and auto-selected (if a model runs oddly under
   CoreML, `--cpu` is a clean baseline — M-series CPUs cope well). NVIDIA:
   `pip uninstall onnxruntime && pip install onnxruntime-gpu` (needs CUDA +
   cuDNN). Windows AMD/Intel: `pip install onnxruntime-directml`.
2. `--det-size 320` — halves detection cost, fine for a single centered face.
3. `--detect-every 2` (or 3) — reuses face positions between detections;
   big CPU win, slight lag on fast head movement.
4. Lower capture size: `--width 960 --height 540`.

### Optional face enhancement

The raw swap is 128×128 and can look slightly soft on a large frame. GFPGAN
sharpens it at a real fps cost (worth it mostly on GPU):

```bash
pip install -r requirements-enhance.txt
python -m delulucam --enhance          # or toggle live with 'e'
```

## Avatar mode — full hair + outfit (delulustream-style)

Face mode swaps only the face region, so you keep your own hair and clothes.
Sites like delulustream get the *whole* character — hair, outfit, background —
because they don't composite onto your video at all: they animate the
character's image with your motion (on cloud GPUs, hence their latency).
Avatar mode does the same thing locally: it takes a portrait of your
character straight from the sheet and drives it live with your head pose,
expressions, blinks and lip movement via
[LivePortrait](https://github.com/KwaiVGI/LivePortrait), running natively on
Apple Silicon through the
[MLX port of FasterLivePortrait](https://github.com/ivanfioravanti/fasterliveportrait-mlx).

| | face mode (`python -m delulucam`) | avatar mode (`delulucam/avatar.py`) |
|---|---|---|
| face | character's | character's |
| hair / outfit / background | **yours** | **character's, pixel-perfect from the sheet** |
| body language | full — walk, gesture, hold things | talking-head — head, expressions, lips |
| speed | fastest | real-time on M-series (use `--profile speed`/`turbo`) |

### Avatar mode setup (Apple Silicon)

```bash
git clone https://github.com/ivanfioravanti/fasterliveportrait-mlx ~/fasterliveportrait-mlx
cd ~/fasterliveportrait-mlx
brew install ffmpeg uv
uv sync
uv pip install pyvirtualcam
```

Then run the bridge from inside that directory (MLX weights auto-download
from Hugging Face on first run):

```bash
uv run python ~/path/to/delulupublic/delulucam/avatar.py ~/sheets/mycharacter.png
```

The bridge picks the best face on your sheet and crops a portrait around it
(hair and shoulders included) — `--view 1` picks a different face,
`--margins top,sides,bottom` tunes the crop, `--no-crop` uses the image
as-is. **A dedicated waist-up portrait image of your character gives the best
result** — worth generating one alongside your sheet. Output is letterboxed
onto a 1280x720 frame with a blurred backdrop (`--canvas 1920x1080`, or
`none` for the raw portrait size).

Live keys: `q` quit, `s` — **drop in/out of character** (the virtual cam
switches to your real camera and back, stream stays live), `m` mirror, and
`r` — **recalibrate**: your pose at the first frame becomes the character's
rest pose, so start (and re-`r`) while facing the camera with a neutral
expression.

Expectations, honestly: expressions, blinks, lip sync and head turns come
through convincingly; the character's body itself stays in its portrait pose,
so this is a talking-head webcam, not full-body motion capture. That's the
trade for running 100% locally with your feed never leaving the Mac. Model
weights (LivePortrait, InsightFace) are released for non-commercial research
use — check upstream licenses before monetised streaming.

## Cloud avatar mode — rented GPU, lower latency (status: scaffold, not built)

Local avatar mode's MLX engine has a real throughput ceiling on Apple
Silicon today (SPADE generator cost dominates and isn't skippable — see the
project history for how that was measured). The alternative: run the same
LivePortrait architecture's TensorRT build on a rented NVIDIA GPU, where it's
built for real-time, and stream frames to and from it.

This is **not implemented yet** — deliberately: getting a network protocol
right by guessing, with no GPU to measure real latency against, tends to be
wasted work. What exists now is the decided shape:

- `server/realtime_server.py` + `server/Dockerfile` — a websocket server for
  a rented GPU box, built on the maintained `shaoguo/faster_liveportrait:v3`
  TensorRT image. One JPEG frame in, one animated JPEG frame back.
- `delulucam/avatar_cloud.py` — the Mac-side client. Unlike `avatar.py`, it
  needs no MLX/engine imports (just capture, network, virtual cam), so it
  runs in delulucam's own venv rather than inside the `fasterliveportrait-mlx`
  checkout.

To pick this up: rent a GPU (RunPod etc.), deploy `server/`, wire the
send/recv loop in both files against real measured latency, done together
since the wire protocol on both ends must match.

Honest trade-offs versus local avatar mode: your webcam feed and character
leave the Mac and go to a rented server every frame (no longer fully
private), and it costs real, ongoing GPU-rental money for however long you
stream (rough range: under a dollar per hour on budget GPU tiers). In
exchange: TensorRT on real GPU hardware should comfortably clear real-time,
which MLX does not yet for this model.

## OBSBOT tips

OBSBOT cameras show up as normal UVC webcams — find yours with
`--list-cameras`. Two things help a lot:

- In OBSBOT Center, **lock AI tracking to gentle/off** while filtering: the
  swap is most stable when your face isn't being re-framed constantly.
- Lock exposure/white balance if your lighting is steady — identity swaps
  flicker less with a stable image.

If OBSBOT Center or another app has the camera open exclusively, OpenCV may
fail to grab it; close the other app or pick the second device index the
camera exposes.

## Uninstalling

Nothing runs in the background — no daemons, login items, or services. When
the app isn't running it does nothing, so "off" is just quitting it (`q`).

To remove everything it put on your machine:

```bash
rm -rf ~/delulucam                 # the app, its venv, and your character sheets
rm -rf ~/.delulucam                # downloaded face-swap models + portrait crops
rm -rf ~/.insightface              # detector models cache
# avatar mode, if you set it up:
rm -rf ~/fasterliveportrait-mlx
rm -rf ~/.cache/huggingface/hub/models--ivanfioravanti--FasterLivePortrait-MLX-weights
```

Optionally, in System Settings → Privacy & Security → Camera, revoke your
terminal app's camera access. The Homebrew tools (`python@3.11`, `ffmpeg`,
`uv`) and OBS are general-purpose — keep them unless nothing else uses them
(`brew uninstall ffmpeg uv python@3.11 && brew uninstall --cask obs`;
removing OBS also removes the virtual camera driver).

## How it works

- **Detection/recognition**: InsightFace `buffalo_l` pack (downloads
  automatically on first run).
- **Swap**: `inswapper_128.onnx` — takes each detected face in the frame plus
  your character's identity embedding, and re-renders the face region with
  the character's identity while keeping your pose, expression and lighting.
  That's why your smile, blinks and head turns come through as the character.
- **Output**: `pyvirtualcam` pushes BGR frames straight into the OS virtual
  camera device.

`--max-faces 0` swaps everyone in frame (default: just the largest face —
you).
