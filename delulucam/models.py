"""Model file management: locate, download and verify the ONNX/torch weights."""

import hashlib
import os
import sys
import urllib.request

DEFAULT_MODEL_DIR = os.environ.get(
    "DELULUCAM_MODEL_DIR", os.path.join(os.path.expanduser("~"), ".delulucam", "models")
)

MODELS = {
    "inswapper_128.onnx": {
        "url": "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx",
        "sha256": "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af",
    },
    "GFPGANv1.4.pth": {
        "url": "https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.pth",
        "sha256": None,  # verified by successful torch load instead
    },
}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: str) -> None:
    tmp = dest + ".part"

    def report(blocks, block_size, total):
        done = blocks * block_size
        if total > 0:
            pct = min(100.0, done * 100.0 / total)
            sys.stderr.write(f"\r  downloading {os.path.basename(dest)}: {pct:5.1f}%")
        else:
            sys.stderr.write(f"\r  downloading {os.path.basename(dest)}: {done >> 20} MiB")
        sys.stderr.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=report)
    sys.stderr.write("\n")
    os.replace(tmp, dest)


def ensure_model(name: str, model_dir: str = DEFAULT_MODEL_DIR) -> str:
    """Return the local path of a model file, downloading it on first use."""
    if name not in MODELS:
        raise KeyError(f"unknown model: {name}")
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, name)
    spec = MODELS[name]
    if not os.path.exists(path):
        print(f"[models] {name} not found, fetching from {spec['url']}")
        _download(spec["url"], path)
    if spec["sha256"] is not None:
        digest = _sha256(path)
        if digest != spec["sha256"]:
            raise RuntimeError(
                f"{name} failed checksum verification "
                f"(got {digest}, expected {spec['sha256']}). "
                f"Delete {path} and retry, or download it manually."
            )
    return path
