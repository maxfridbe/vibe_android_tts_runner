#!/usr/bin/env bash
# Copy cloned Supertonic styles from brainiac onto a connected phone.
#
#   tooling/push_styles.sh                 # every style in the library run
#   tooling/push_styles.sh Dale "Tony Jay" # a selection
#
# Files land in two places on purpose:
#
#   /sdcard/Download/tts-speakers   a normal folder that survives an uninstall.
#                                   Point TTS Runner's backup folder here
#                                   (Speakers -> Files -> Keep a backup folder)
#                                   and everything in it is imported on sight.
#   the app's own library           via run-as, so they show up immediately
#                                   without touching the UI. Debug builds only.
set -euo pipefail
SELF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HOST=${HOST:-brainiac-nvidia}
REMOTE=${REMOTE:-supertonic-experiment/clone_out/library/styles}
PKG=com.maxfridbe.ttsrunner
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

if [ $# -gt 0 ]; then
  for name in "$@"; do scp -q "$HOST:$REMOTE/$name.json" "$STAGE/"; done
else
  scp -q "$HOST:$REMOTE/*.json" "$STAGE/"
fi
echo "fetched $(ls "$STAGE" | wc -l) styles"

adb shell mkdir -p /sdcard/Download/tts-speakers >/dev/null
for f in "$STAGE"/*.json; do
  name=$(basename "$f")
  adb push "$f" "/sdcard/Download/tts-speakers/" >/dev/null
  # the app's files dir is invisible to `adb push` (per-app mount namespace),
  # so stage in /data/local/tmp and let the app's own uid do the copy
  adb push "$f" /data/local/tmp/stage.json >/dev/null
  adb shell "run-as $PKG sh -c 'mkdir -p files/voices && cp /data/local/tmp/stage.json \"files/voices/$name\"'" \
    || echo "  (app-private copy failed for $name — release build? import the folder instead)"
  echo "  $name"
done
adb shell rm -f /data/local/tmp/stage.json >/dev/null 2>&1 || true
echo "done — open Speakers in TTS Runner"
