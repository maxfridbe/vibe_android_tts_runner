#!/usr/bin/env bash
# Round 3, using the dead hours after the overnight driver finishes.
#
# Round 2 proved the mechanism (unseen LibriSpeech speakers 0.15 -> 0.27 with
# 40 inverted styles in the basis) and the trajectory says the basis is still
# the ceiling. So: put the idle 5070 back on inversions, wait for ~70 pairs,
# then retrain and eval one more time before morning.
set -u
WORK=${WORK:-$HOME/supertonic-experiment}
O=$WORK/clone_out/overnight
TOOL=$WORK/tooling
PY=$WORK/venv/bin/python
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

# the 5070 rejoins the inversion pool on its own unfinished slice
setsid nohup $PY "$TOOL/invert_corpus.py" \
  --corpus "$WORK/corpora/LibriSpeech/train-clean-100" \
  --out "$O/inversions" --gpu 0 --skip 120 --max-speakers 60 \
  >> "$O/inversions_gpu0.log" 2>&1 &
W=$!

DEADLINE=$(( $(date +%s) + 7200 ))
while [ "$(count)" -lt 70 ] && [ "$(date +%s)" -lt "$DEADLINE" ]; do sleep 120; done
kill -- -"$W" 2>/dev/null
sleep 10

cp "$O/inversions/aux_pairs.npz" "$O/aux_snapshot_r3.npz"
N=$(SNAP=$O/aux_snapshot_r3.npz python3 -c 'import numpy as np, os; print(len(np.load(os.environ["SNAP"], allow_pickle=True)["emb"]))')
notify "round-3 training starting with $N inverted real speakers in the basis"

CUDA_VISIBLE_DEVICES=0 $PY "$TOOL/train_encoder.py" --pairs "$WORK/clone_out/pairs_p1.npz" \
  --targets "$O/bank_real_all.npz" --k 64 --steps 15000 \
  --basis-extra "$O/aux_snapshot_r3.npz" --aux-pairs "$O/aux_snapshot_r3.npz" \
  --w-aux 0.3 --val-every 500 --val-max 96 \
  --out "$O/encoder-r3.pt" > "$O/train_r3.log" 2>&1 \
  || { notify "round-3 FAILED: $(tail -n 3 "$O/train_r3.log" | tr '\n' ' ' | tail -c 300)"; exit 1; }

REFS=("$WORK/clone_out/library/refs/Dale.wav" "$WORK/clone_out/library/refs/Stephen Fry.wav"
      "$WORK/clone_out/library/refs/Fireside Narrator.wav"
      "$WORK/clone_out/library/refs/Soothing female british voice.wav")
mapfile -t TC < <(TCROOT="$WORK/corpora/LibriSpeech/test-clean" $PY - <<'PY'
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
)
CUDA_VISIBLE_DEVICES=0 $PY "$TOOL/eval_encoder_real.py" \
  --encoders "$O/encoder-r3.pt.best" --refs "${REFS[@]}" "${TC[@]}" \
  --gpu 0 --out "$O/eval_r3.json" > "$O/eval_r3.log" 2>&1 \
  || { notify "round-3 eval FAILED: $(tail -n 3 "$O/eval_r3.log" | tr '\n' ' ' | tail -c 300)"; exit 1; }

E3=$(EV=$O/eval_r3.json python3 - <<'PY'
import json, os
rows = json.load(open(os.environ["EV"]))
m = sum(r["held_out"] for r in rows if r["held_out"]) / max(1, sum(1 for r in rows if r["held_out"]))
tc = [r for r in rows if r["ref"].endswith(".flac")]
mtc = sum(r["held_out"] for r in tc) / max(1, len(tc))
print(f"mean held-out cos {m:.3f} over {len(rows)} refs; unseen LibriSpeech mean {mtc:.3f} "
      f"(r1 0.149, r2 0.267)")
PY
)
notify "round-3 eval: $E3. Artefacts: encoder-r3.pt.best, eval_r3.json"
touch "$O/DONE_R3"
