#!/bin/bash
# The actual delulucam startup sequence: venv setup (if needed), then serve.py
# + vcam_bridge.py together, with clean shutdown of both on Ctrl-C / window
# close. Run directly, or via delulucam.app's launcher.

set -e
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && cd .. && pwd)"
cd "$REPO_DIR"

echo "delulucam — starting up from $REPO_DIR"
echo

if [ ! -d ".venv" ]; then
  echo "[setup] no .venv found — creating one (first run only)..."
  PYTHON_BIN="$(command -v python3.11 || command -v python3)"
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate

# Reinstall only if requirements.txt changed since the last successful
# install, so normal launches don't pay the pip-check cost every time.
STAMP=".venv/.requirements.stamp"
if [ ! -f "$STAMP" ] || [ requirements.txt -nt "$STAMP" ]; then
  echo "[setup] installing/updating dependencies..."
  pip install --quiet -r requirements.txt
  touch "$STAMP"
fi

cleanup() {
  echo
  echo "[delulucam] shutting down..."
  [ -n "$SERVE_PID" ] && kill "$SERVE_PID" 2>/dev/null
  [ -n "$BRIDGE_PID" ] && kill "$BRIDGE_PID" 2>/dev/null
  wait 2>/dev/null
  echo "[delulucam] stopped."
}
trap cleanup EXIT INT TERM

echo "[delulucam] starting web app..."
python3 delulucam/web/serve.py &
SERVE_PID=$!

echo "[delulucam] starting virtual-camera bridge..."
python3 delulucam/web/vcam_bridge.py &
BRIDGE_PID=$!

echo
echo "delulucam is running. Press Ctrl-C in this window to stop everything."
echo

wait
