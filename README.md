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

## Cloud avatar mode — Decart Lucy, real-time, hosted

Local avatar mode's MLX engine has a real throughput ceiling on Apple
Silicon today (SPADE generator cost dominates and isn't skippable — see the
project history for how that was measured). An earlier plan here was to rent
a GPU and self-host a TensorRT server for this; that's no longer necessary —
[Decart](https://decart.ai)'s **Lucy** models do real-time character-reference
transformation as a hosted API, confirmed working (including full-body, on
`lucy-2.5`) against this project's own character sheet.

`delulucam/web/` is a small local browser app — not a Python script, because
Decart's most complete SDK is JavaScript, and running it in a browser avoids
reimplementing WebRTC frame handling ourselves:

- Enter your [Decart API key](https://platform.decart.ai), pick a model
  (`lucy-2.5`/`lucy-2.1` for character reference, `lucy-restyle-2` for style,
  `lucy-vton-3` for outfit-only), upload a reference portrait from your
  character sheet, write a prompt, and pick your OBSBOT as the camera.
- The transformed stream renders live in the page. "Apply changes live"
  pushes a new prompt or reference image into the running session without
  reconnecting.

Run it — either manually, two terminals:

```bash
python3 delulucam/web/serve.py       # terminal 1: serves on http://localhost:8420 and opens it
python3 delulucam/web/vcam_bridge.py # terminal 2: the virtual-camera bridge (needs the venv active)
```

(Plain `file://` often blocks camera access in the browser — serving over
`localhost` sidesteps that.)

**Or with one double-click**, once you've done the manual run at least once
(so `.venv` exists): `delulucam/mac/delulucam.app` starts both of the above
together in a Terminal window, and stops both cleanly when you close it or
hit Ctrl-C. To get an actual installable `.dmg` out of it, run this once
**on your Mac** (it needs `hdiutil`, which only exists there — this can't be
built from a Linux dev environment):

```bash
./delulucam/mac/build_dmg.sh    # writes delulucam-installer.dmg to the repo root
```

It isn't code-signed (that needs a paid Apple Developer account), so the
first launch needs a right-click → Open past Gatekeeper's "unidentified
developer" warning — a one-time approval, same as any indie/open-source Mac
app without a paid cert. After that it opens normally.

**Feeding the output to your virtual camera** — two ways:

*Direct (recommended)* — `delulucam/web/vcam_bridge.py` spawns a real system
virtual camera (the same `pyvirtualcam`/OBS Virtual Camera device used
elsewhere in this project) and feeds it frames the browser page captures
from its own output. No OBS needed at all — any app (Zoom, Meet, Discord,
OBS included) can select the resulting camera directly, the same way you'd
pick any other webcam.

```bash
python3 delulucam/web/vcam_bridge.py   # run alongside serve.py, in delulucam's venv
```

Then in the web page, once connected to Decart, check **"Feed output to a
system virtual camera"** in the sidebar. The page streams JPEG frames to the
bridge over a local WebSocket (`ws://localhost:8421`); the bridge decodes
them and pushes them into the virtual camera, creating the device on first
frame once it knows the real output resolution.

*Fallback (OBS Window Capture)* — if you'd rather not run the extra bridge
process, or you're already living in OBS anyway: open `http://localhost:8420`
in your regular browser (**not** inside OBS's Browser Source — OBS's Browser
Source runs its own embedded browser with separate permissions and reliably
fails to get camera access), then in OBS: **Sources → + → Window Capture** →
select that browser window → crop to just the output video → Start Virtual
Camera. This captures the already-working browser tab's pixels instead of
asking OBS's internal browser to access your camera itself.

**Watermark**: there's no code/SDK flag for this — check
[platform.decart.ai/watermark](https://platform.decart.ai/watermark) in
your Decart account; it's an account-level setting on their platform, not
something this app controls.

Honest trade-offs versus local avatar mode: your webcam feed and reference
image go to Decart's servers every frame (no longer fully private), and
generation is billed by the second while a session is active — check
current pricing in their dashboard. Latency you actually feel also depends
on your network round-trip to wherever they serve realtime sessions from,
on top of their own sub-40ms model latency — worth checking directly if
you're far from their infrastructure.

**Character reference best practices** (from Decart's own docs): a clear,
front-facing, well-lit, head-and-shoulders crop works far better than a
full-body or occluded shot — reuse one of the close-up crops this project
already produced from your character sheet, not the whole collage.

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
