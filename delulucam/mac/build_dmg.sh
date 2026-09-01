#!/bin/bash
# Wraps delulucam.app into a real .dmg — run this ON macOS (it uses hdiutil,
# which only exists there). One-time, or whenever you want a fresh .dmg
# after pulling updates.
#
# Usage: ./delulucam/mac/build_dmg.sh
# Output: ./delulucam-installer.dmg in the repo root.

set -e
MAC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$MAC_DIR/../.." && pwd)"
APP="$MAC_DIR/delulucam.app"
OUT="$REPO_DIR/delulucam-installer.dmg"

if [ ! -d "$APP" ]; then
  echo "error: $APP not found" >&2
  exit 1
fi

STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

# Copy (not move) the .app into a staging folder alongside an Applications
# symlink, so the mounted .dmg shows the familiar "drag app to Applications"
# layout.
cp -R "$APP" "$STAGING/"
ln -s /Applications "$STAGING/Applications"

rm -f "$OUT"
hdiutil create -volname "delulucam" -srcfolder "$STAGING" -ov -format UDZO "$OUT"

echo
echo "Built $OUT"
echo "Note: this isn't code-signed (no Apple Developer certificate involved)."
echo "First launch will need a right-click > Open (or System Settings >"
echo "Privacy & Security > 'Open Anyway') past Gatekeeper's unidentified-"
echo "developer warning -- that's a one-time approval, same as any other"
echo "indie/open-source Mac app without a paid Apple cert."
