#!/usr/bin/env bash
# Clone a voice into a Supertonic style JSON that TTS Runner can import.
#
# Cloning is gradient inversion through the public ONNX weights
# (github.com/Mimocro/supertonic-voice-cloning): the style tensors are
# optimised until the synthesised speaker matches your recording. It needs a
# CUDA GPU with >=6 GB and a few minutes; the phone only ever consumes the
# resulting JSON, which is why this is a desktop step.
#
#   tooling/clone_voice.sh reference.wav "exact transcript of the recording"
#
# The transcript matters: with it, whisper is never needed. TTS Runner's voice
# recorder already knows what you read (it stores the passage as the voice's
# transcript), so export that text with the recording and the pipeline is
# fully deterministic.
set -euo pipefail

REF=${1:?usage: clone_voice.sh <reference.wav> [transcript] [init-style]}
TEXT=${2:-}
INIT=${3:-}
WORK=${WORK:-$HOME/supertonic-experiment}
ITERS=${ITERS:-500}

REPO=$WORK/supertonic-voice-cloning
ASSETS=$WORK/assets/onnx
STYLES=$WORK/assets/voice_styles
OUT=$WORK/clone_out
mkdir -p "$OUT"

[ -d "$REPO" ] || git clone --depth 1 https://github.com/Mimocro/supertonic-voice-cloning.git "$REPO"
[ -f "$ASSETS/vocoder.onnx" ] || { echo "missing ONNX assets in $ASSETS"; exit 1; }

# Their code targets a root-only Docker image and torch 2.8-era inductor.
# Two adjustments make it run in a normal venv on current torch:
#   - speechbrain's ECAPA cache path is hardcoded to /root/.cache
#   - NOCOMPILE=1 skips torch.compile, whose dynamic-shape handling breaks
#     ("vr must not be None for symbol q1") on torch >= 2.13
python3 - "$REPO" <<'PY'
import os, sys
p = os.path.join(sys.argv[1], "src/speaker_encoder.py")
s = open(p).read()
if "/root/.cache" in s:
    s = s.replace('savedir="/root/.cache/speechbrain/ecapa"',
                  'savedir=os.environ.get("SB_CACHE", os.path.expanduser("~/.cache/speechbrain/ecapa"))')
    if not s.startswith("import os"):
        s = "import os\n" + s
    open(p, "w").write(s)
    print("patched speaker_encoder.py for non-root use")
PY

# Warm start from the closest published style unless one was named: the
# optimiser only moves deltas, so a nearer starting speaker is free progress.
if [ -z "$INIT" ]; then
  INIT=$STYLES/F1.json
  echo "init style: $(basename "$INIT") (pass a third argument to choose another)"
fi

export TORCHINDUCTOR_CACHE_DIR=$WORK/.inductor TRITON_CACHE_DIR=$WORK/.triton NOCOMPILE=1
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

NAME=$(basename "${REF%.*}")
ARGS=(--reference "$REF" --onnx-dir "$ASSETS" --init-voice "$INIT"
      --output "$OUT/$NAME.json" --save-at "$ITERS" --batch-size 3 --device cuda)
[ -n "$TEXT" ] && ARGS+=(--ref-text "$TEXT")

cd "$REPO/src"
"${PYTHON:-$WORK/venv/bin/python}" invert.py "${ARGS[@]}"

echo
echo "style written: $OUT/$NAME.$ITERS.json"
echo "copy it to the phone and import it in Voices -> Import (pick the .json)."
