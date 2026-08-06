#!/usr/bin/env bash
# Drive TTS Runner on a connected device with zero screen taps.
#
#   tooling/adb_test.sh                       # speak a default sentence on CPU
#   tooling/adb_test.sh "some text" cpu save  # render to Music/TTS Runner/*.m4a
#   ADB=/path/to/adb tooling/adb_test.sh
#
# Requires the debug build (run-as needs a debuggable app) and a voice +
# model already present on the device (import once via the UI).
set -euo pipefail

ADB=${ADB:-adb}
PKG=com.maxfridbe.ttsrunner
TEXT=${1:-"The quick brown fox jumps over the lazy dog."}
BACKEND=${2:-cpu}
MODE=${3:-speak}   # speak | save
SAVE=false; [ "$MODE" = "save" ] && SAVE=true

echo "==> version: $($ADB shell dumpsys package $PKG | grep versionName | head -1 | tr -d ' ')"
echo "==> starting job (backend=$BACKEND save=$SAVE): $TEXT"
$ADB shell am start-foreground-service -n $PKG/.TtsService \
    -a com.maxfridbe.ttsrunner.SPEAK \
    --es text "'$TEXT'" --es title adbtest --es backend "$BACKEND" --ez save $SAVE

echo "==> tailing app log (ctrl-c to stop)"
$ADB shell run-as $PKG sh -c "'touch files/debug.log; tail -n 5 -f files/debug.log'"
