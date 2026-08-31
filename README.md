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

## Setup

Requires Python 3.9–3.11 (insightface wheels are most reliable there).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Virtual camera driver** (one-time, system-wide):

- **Windows / macOS** — install [OBS Studio](https://obsproject.com); it ships
  the virtual camera driver pyvirtualcam uses. On macOS, launch OBS once and
  press *Start Virtual Camera* one time to register the extension.
- **Linux** — `sudo apt install v4l2loopback-dkms`, then
  `sudo modprobe v4l2loopback devices=1 card_label=delulucam exclusive_caps=1`.

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

A character sheet works best with a clear, well-lit, mostly frontal face. If
the sheet shows the character from several angles, delulucam detects all the
views, keeps the ones matching the dominant identity, and **averages them
into one identity** — multi-view sheets give a more stable swap than a single
crop. One image file = one character; multiple files = switchable characters.

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

1. **GPU inference** — NVIDIA: `pip uninstall onnxruntime && pip install
   onnxruntime-gpu` (needs CUDA + cuDNN). Apple Silicon: `pip install
   onnxruntime-silicon` for CoreML. Windows AMD/Intel: `pip install
   onnxruntime-directml`. delulucam auto-picks the best available provider;
   `--cpu` forces CPU.
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
