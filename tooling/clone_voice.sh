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
SELF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # resolve before any cd

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

export TORCHINDUCTOR_CACHE_DIR=$WORK/.inductor TRITON_CACHE_DIR=$WORK/.triton NOCOMPILE=1
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

NAME=$(basename "${REF%.*}")
PY=${PYTHON:-$WORK/venv/bin/python}
cd "$REPO/src"

run_one() {  # <init-json> <gpu> <tag>
  local init=$1 gpu=$2 tag=$3
  local args=(--reference "$REF" --onnx-dir "$ASSETS" --init-voice "$init"
              --output "$OUT/$NAME-$tag.json" --save-at "$ITERS"
              --batch-size 3 --device cuda --shutoff 0)
  [ -n "$TEXT" ] && args+=(--ref-text "$TEXT")
  CUDA_VISIBLE_DEVICES=$gpu "$PY" invert.py "${args[@]}" > "$OUT/$NAME-$tag.log" 2>&1
}

# Multi-start, because the starting voice and the run itself both matter more
# than they look: measured on one 3 s reference, held-out speaker similarity
# came out 0.81 from an F3 start and 0.69-0.73 from two F1 starts. The run's
# own reported cosine cannot rank these — it is measured on the probe text
# being fitted — so candidates are scored on held-out sentences below.
GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')
if [ -n "$INIT" ]; then
  STARTS=("$INIT")
else
  STARTS=("$STYLES/F1.json" "$STYLES/F3.json" "$STYLES/M1.json")
fi

# One run per GPU at a time: each needs ~6 GB, and two on a 12 GB card OOM
# partway through ("Tried to allocate 2.00 MiB"), losing the whole run.
NGPU=$(echo "$GPUS" | wc -w)
i=0
while [ $i -lt ${#STARTS[@]} ]; do
  pids=()
  for g in $GPUS; do
    [ $i -lt ${#STARTS[@]} ] || break
    init=${STARTS[$i]}
    tag=$(basename "${init%.json}")
    echo "start $tag on GPU $g"
    run_one "$init" "$g" "$tag" &
    pids+=($!)
    i=$((i + 1))
  done
  for p in "${pids[@]}"; do wait "$p"; done
done

CANDIDATES=()
for init in "${STARTS[@]}"; do
  tag=$(basename "${init%.json}")
  [ -f "$OUT/$NAME-$tag.$ITERS.json" ] && CANDIDATES+=("$OUT/$NAME-$tag.$ITERS.json")
done
[ ${#CANDIDATES[@]} -gt 0 ] || { echo "no candidates produced; see $OUT/*.log"; exit 1; }

echo
echo "scoring candidates on held-out sentences ..."
"$PY" "$SELF/eval_style.py" "$REF" "${CANDIDATES[@]}" | tee "$OUT/$NAME.eval.txt"

BEST=$(tail -1 "$OUT/$NAME.eval.txt" | sed 's/^best: //')
cp "$BEST" "$OUT/$NAME.style.json"
echo
echo "style written: $OUT/$NAME.style.json  (from $(basename "$BEST"))"
echo "copy it to the phone and import it in Voices -> Import (pick the .json)."
