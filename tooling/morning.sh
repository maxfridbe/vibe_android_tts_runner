#!/usr/bin/env bash
# Day-after driver: capacity and diversity, the two levers the overnight run
# left on the table.
#
# The night ended at 0.27 mean on unseen speakers with k=64 and a basis of
# same-corpus (LibriSpeech) styles — flat mean, rising best-case, the shape of
# a saturated basis. So:
#
#   r4  k=128 retrain, right now, with the ~85-pair bank         (GPU0)
#   ..  VCTK 0.92 downloads in parallel; the 3060 finishes its
#       LibriSpeech queue in its own time
#   ..  VCTK speakers embedded into the target bank, then inverted
#       on whichever GPUs are free (accent + style diversity)
#   r5  k=128 retrain on the diversified basis, eval, report
#
#   nohup tooling/morning.sh > clone_out/overnight/morning_driver.log 2>&1 &
set -u
WORK=${WORK:-$HOME/supertonic-experiment}
O=$WORK/clone_out/overnight
TOOL=$WORK/tooling
PY=$WORK/venv/bin/python
CORP=$WORK/corpora
CO=$WORK/clone_out
START=$(date +%s)
export TG_TOKEN="7321403954:AAGcZxrFGHVTu_ycYwfzDc070QiZNUV_d4k" TG_CHAT="573950781"

notify() {
  MSG="$1" python3 - >/dev/null 2>&1 <<'PY' || true
import json, os, urllib.request
req = urllib.request.Request(
    f"https://api.telegram.org/bot{os.environ['TG_TOKEN']}/sendMessage",
    data=json.dumps({"chat_id": os.environ["TG_CHAT"],
                     "text": "🧬 cloning | " + os.environ["MSG"]}).encode(),
    headers={"Content-Type": "application/json"})
urllib.request.urlopen(req, timeout=30)
PY
}
count() { ls "$O/inversions/pairs" 2>/dev/null | grep -c '\.npz$' || true; }

REFS=("$CO/library/refs/Dale.wav" "$CO/library/refs/Stephen Fry.wav"
      "$CO/library/refs/Fireside Narrator.wav"
      "$CO/library/refs/Soothing female british voice.wav")
pick_tc() {
  TCROOT="$CORP/LibriSpeech/test-clean" $PY - <<'PY'
import os, soundfile as sf
root, n = os.environ["TCROOT"], 0
for spk in sorted(os.listdir(root)):
    d = os.path.join(root, spk)
    if not os.path.isdir(d):
        continue
    best = None
    for base, _, names in os.walk(d):
        for f in names:
            if f.endswith(".flac"):
                p = os.path.join(base, f)
                i = sf.info(p)
                dur = i.frames / i.samplerate
                if 8 <= dur <= 15 and (best is None or abs(dur-11) < abs(best[0]-11)):
                    best = (dur, p)
    if best:
        print(best[1]); n += 1
    if n >= 4:
        break
PY
}
mapfile -t TC < <(pick_tc)

run_eval() {  # $1 = checkpoint, $2 = out json tag
  CUDA_VISIBLE_DEVICES=0 $PY "$TOOL/eval_encoder_real.py" \
    --encoders "$O/encoder-$1.pt.best" --refs "${REFS[@]}" "${TC[@]}" \
    --gpu 0 --out "$O/eval_$1.json" > "$O/eval_$1.log" 2>&1
  EV=$O/eval_$1.json python3 - <<'PY'
import json, os
rows = json.load(open(os.environ["EV"]))
m = sum(r["held_out"] for r in rows if r["held_out"]) / max(1, sum(1 for r in rows if r["held_out"]))
tc = [r for r in rows if r["ref"].endswith(".flac")]
mtc = sum(r["held_out"] for r in tc) / max(1, len(tc))
print(f"mean {m:.3f} over {len(rows)} refs; unseen LibriSpeech {mtc:.3f} (r2 0.267, r3 0.274)")
PY
}

notify "day run started: k=128 retrain now, VCTK downloading, then accent-diverse inversions and a final retrain"

# ---- VCTK download, in the background ---------------------------------------
(
  set -e; cd "$CORP"
  [ -d VCTK/wav48_silence_trimmed ] && exit 0
  for i in 1 2 3; do
    curl -sL -C - -o vctk.zip \
      "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip" && break
    sleep 60
  done
  mkdir -p VCTK
  (cd VCTK && (unzip -q ../vctk.zip || $PY -m zipfile -e ../vctk.zip .))
  rm -f vctk.zip
) > "$O/vctk_download.log" 2>&1 &
DL=$!

# ---- r4: k=128 on the current bank (GPU0) -----------------------------------
if [ ! -f "$O/eval_r4.json" ]; then
  cp "$O/inversions/aux_pairs.npz" "$O/aux_snapshot_r4.npz"
  N=$(SNAP=$O/aux_snapshot_r4.npz python3 -c 'import numpy as np, os; print(len(np.load(os.environ["SNAP"], allow_pickle=True)["emb"]))')
  notify "r4 (k=128) training with $N inverted speakers in the basis"
  CUDA_VISIBLE_DEVICES=0 $PY "$TOOL/train_encoder.py" --pairs "$CO/pairs_p1.npz" \
    --targets "$O/bank_real_all.npz" --k 128 --steps 15000 \
    --basis-extra "$O/aux_snapshot_r4.npz" --aux-pairs "$O/aux_snapshot_r4.npz" \
    --w-aux 0.3 --val-every 500 --val-max 96 \
    --out "$O/encoder-r4.pt" > "$O/train_r4.log" 2>&1 \
    || { notify "r4 FAILED: $(tail -n 3 "$O/train_r4.log" | tr '\n' ' ' | tail -c 300)"; exit 1; }
  notify "r4 eval: $(run_eval r4)"
fi

# ---- VCTK ready? ------------------------------------------------------------
VCTK_DEADLINE=$((START + 5 * 3600))
while kill -0 "$DL" 2>/dev/null && [ "$(date +%s)" -lt "$VCTK_DEADLINE" ]; do sleep 120; done
if [ ! -d "$CORP/VCTK/wav48_silence_trimmed" ]; then
  kill "$DL" 2>/dev/null
  notify "VCTK did not arrive by the deadline — stopping here; r4 artefacts are in clone_out/overnight/"
  exit 0
fi

# ---- VCTK into the target bank (GPU0) ---------------------------------------
if [ ! -f "$O/bank_real_all2.npz" ]; then
  CUDA_VISIBLE_DEVICES=0 $PY "$TOOL/embed_corpus.py" \
    --roots "$CORP/VCTK/wav48_silence_trimmed" \
    --out "$O/vctk_bank.npz" --per-speaker 6 --device cuda:0 > "$O/embed_vctk.log" 2>&1 \
    || { notify "VCTK embed FAILED: $(tail -n 3 "$O/embed_vctk.log" | tr '\n' ' ' | tail -c 300)"; exit 1; }
  $PY "$TOOL/merge_banks.py" "$O/bank_real_all2.npz" \
    "$O/bank_real_all.npz" "$O/vctk_bank.npz" >> "$O/embed_vctk.log" 2>&1
  notify "VCTK in the bank: $(grep 'unique rows' "$O/embed_vctk.log" | tail -1)"
fi

# ---- VCTK inversions on whatever is free ------------------------------------
setsid nohup $PY "$TOOL/invert_corpus.py" --corpus "$CORP/VCTK" --layout vctk \
  --out "$O/inversions" --gpu 0 --skip 0 --max-speakers 55 \
  >> "$O/inversions_vctk0.log" 2>&1 &
V0=$!
notify "VCTK inversions started on GPU0"
V1=""
for i in $(seq 1 100); do          # the 3060 joins once its LibriSpeech queue ends
  if ! pgrep -f "[i]nvert_corpus.py.*gpu 1" >/dev/null; then
    setsid nohup $PY "$TOOL/invert_corpus.py" --corpus "$CORP/VCTK" --layout vctk \
      --out "$O/inversions" --gpu 1 --skip 55 --max-speakers 45 \
      >> "$O/inversions_vctk1.log" 2>&1 &
    V1=$!
    notify "3060 finished LibriSpeech and joined VCTK inversions"
    break
  fi
  sleep 180
done

# ---- r5: the diversified basis (GPU0) ---------------------------------------
R5_AT=$((START + 8 * 3600))
while [ "$(date +%s)" -lt "$R5_AT" ] && [ "$(count)" -lt 170 ]; do sleep 300; done
kill -- -"$V0" 2>/dev/null          # GPU1's worker keeps going through r5
sleep 10
cp "$O/inversions/aux_pairs.npz" "$O/aux_snapshot_r5.npz"
N=$(SNAP=$O/aux_snapshot_r5.npz python3 -c 'import numpy as np, os; print(len(np.load(os.environ["SNAP"], allow_pickle=True)["emb"]))')
notify "r5 (k=128, VCTK-diversified) training with $N inverted speakers"
CUDA_VISIBLE_DEVICES=0 $PY "$TOOL/train_encoder.py" --pairs "$CO/pairs_p1.npz" \
  --targets "$O/bank_real_all2.npz" --k 128 --steps 15000 \
  --basis-extra "$O/aux_snapshot_r5.npz" --aux-pairs "$O/aux_snapshot_r5.npz" \
  --w-aux 0.3 --val-every 500 --val-max 96 \
  --out "$O/encoder-r5.pt" > "$O/train_r5.log" 2>&1 \
  || { notify "r5 FAILED: $(tail -n 3 "$O/train_r5.log" | tr '\n' ' ' | tail -c 300)"; exit 1; }
notify "r5 eval: $(run_eval r5). Day run complete — encoders r4/r5, evals, and $(count) total pairs in clone_out/overnight/"
touch "$O/DONE_R5"
