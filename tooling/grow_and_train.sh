#!/usr/bin/env bash
# Grow the inverted-pair bank on both GPUs, then retrain the cloning heads.
#
# The heads and the basis the on-device refine searches both improve with more
# inverted-real-speaker pairs, and the weak voices (Dale/Fry-style character
# audio) most need accent/expressive diversity — so this adds VCTK on one GPU
# and more LibriSpeech on the other, appending to the same bank, then reruns
# the retrain-and-compare once it has grown.
#
#   nohup tooling/grow_and_train.sh > clone_out/overnight/grow.log 2>&1 &
set -u
WORK=${WORK:-$HOME/supertonic-experiment}
O=$WORK/clone_out/overnight
TOOL=$WORK/tooling
PY=$WORK/venv/bin/python
START=$(date +%s)
export TG_TOKEN="7321403954:AAGcZxrFGHVTu_ycYwfzDc070QiZNUV_d4k" TG_CHAT="573950781"
notify() {
  MSG="$1" python3 - >/dev/null 2>&1 <<'PY' || true
import json, os, urllib.request
req = urllib.request.Request(
    f"https://api.telegram.org/bot{os.environ['TG_TOKEN']}/sendMessage",
    data=json.dumps({"chat_id": os.environ["TG_CHAT"], "text": "🧬 cloning | " + os.environ["MSG"]}).encode(),
    headers={"Content-Type": "application/json"})
urllib.request.urlopen(req, timeout=30)
PY
}
count() { ls "$O/inversions/pairs" 2>/dev/null | grep -c '\.npz$' || true; }

BASE=$(count)
notify "grow-and-train started: $BASE pairs now, targeting ~320 with VCTK (accents) on GPU1 and LibriSpeech on GPU0"

setsid nohup $PY "$TOOL/invert_corpus.py" --corpus "$WORK/corpora/LibriSpeech/train-clean-100" \
  --out "$O/inversions" --gpu 0 --skip 250 --max-speakers 80 \
  >> "$O/grow_libri.log" 2>&1 &
L=$!
setsid nohup $PY "$TOOL/invert_corpus.py" --corpus "$WORK/corpora/VCTK" --layout vctk \
  --out "$O/inversions" --gpu 1 --skip 100 --max-speakers 80 \
  >> "$O/grow_vctk.log" 2>&1 &
V=$!

DEADLINE=$((START + 9 * 3600))
while [ "$(count)" -lt 320 ] && [ "$(date +%s)" -lt "$DEADLINE" ]; do
  sleep 300
  { kill -0 "$L" 2>/dev/null || kill -0 "$V" 2>/dev/null; } || break
done
kill -- -"$L" -- -"$V" 2>/dev/null || true
sleep 5
N=$(count)
notify "bank grown to $N pairs (from $BASE); retraining the heads"

# retrain + compare, reusing the update driver's recipe
bash "$TOOL/update_cloner.sh" >> "$O/grow_train.log" 2>&1 || notify "grow-and-train: retrain step failed, see grow_train.log"
touch "$O/DONE_GROW"
