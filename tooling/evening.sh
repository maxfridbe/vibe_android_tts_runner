#!/usr/bin/env bash
# Evening driver: the pivot the listening test forced.
#
# The encoder alone lands in the right neighbourhood, not on the person, so
# tonight builds the two things that don't depend on it: a shippable voice
# pack (Qwen VoiceDesign rolls inverted at desktop quality — option A) and
# the measurement that decides on-device cloning's future — forward-only
# CMA-ES search in the refit basis, the algorithm a phone can actually run.
#
#   nohup tooling/evening.sh > clone_out/overnight/evening_driver.log 2>&1 &
set -u
WORK=${WORK:-$HOME/supertonic-experiment}
O=$WORK/clone_out/overnight
P=$WORK/clone_out/packs
TOOL=$WORK/tooling
PY=$WORK/venv/bin/python
ABM_PY=$HOME/audiobook-maker/venv/bin/python
CO=$WORK/clone_out
START=$(date +%s)
export TG_TOKEN="7321403954:AAGcZxrFGHVTu_ycYwfzDc070QiZNUV_d4k" TG_CHAT="573950781"
mkdir -p "$P"

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

# ---- wait for the day run to release the GPUs -------------------------------
DAY_DEADLINE=$((START + 4 * 3600))
while [ ! -f "$O/DONE_R5" ] && pgrep -f "[m]orning.sh" >/dev/null \
      && [ "$(date +%s)" -lt "$DAY_DEADLINE" ]; do sleep 300; done
notify "evening run: voice-pack build + forward-only inversion experiment starting"

# ---- P1: roll pack candidates (GPU0, Qwen venv) -----------------------------
if [ ! -f "$P/rolls/pack_059.wav" ] && [ "$(ls "$P/rolls" 2>/dev/null | grep -c '\.wav$')" -lt 55 ]; then
  CUDA_VISIBLE_DEVICES=0 $ABM_PY "$TOOL/gen_pack_roll.py" --count 60 \
    --out "$P/rolls" > "$O/pack_roll.log" 2>&1 \
    || { notify "pack roll FAILED: $(tail -n 3 "$O/pack_roll.log" | tr '\n' ' ' | tail -c 300)"; exit 1; }
fi
N=$(ls "$P/rolls" | grep -c '\.wav$')
notify "rolled $N pack candidates; inverting on both GPUs (~3h)"

# wait out any straggling VCTK worker on GPU1
while pgrep -f "[i]nvert_corpus.py.*gpu 1" >/dev/null \
      && [ "$(date +%s)" -lt $((START + 6 * 3600)) ]; do sleep 120; done

# ---- P2: invert the candidates on both GPUs ---------------------------------
setsid nohup $PY "$TOOL/invert_corpus.py" --corpus "$P/rolls" --layout flat \
  --out "$P/inversions" --gpu 0 --skip 0 --max-speakers 35 --min-dur 6 \
  > "$O/pack_inv0.log" 2>&1 &
I0=$!
setsid nohup $PY "$TOOL/invert_corpus.py" --corpus "$P/rolls" --layout flat \
  --out "$P/inversions" --gpu 1 --skip 35 --max-speakers 25 --min-dur 6 \
  > "$O/pack_inv1.log" 2>&1 &
I1=$!
INV_DEADLINE=$((START + 9 * 3600))
while { kill -0 "$I0" 2>/dev/null || kill -0 "$I1" 2>/dev/null; } \
      && [ "$(date +%s)" -lt "$INV_DEADLINE" ]; do sleep 300; done
kill -- -"$I0" -- -"$I1" 2>/dev/null
NP=$(ls "$P/inversions/pairs" 2>/dev/null | grep -c '\.npz$')
notify "pack inversions done: $NP of $N candidates"

# ---- P3: curate into the shippable pack -------------------------------------
$PY "$TOOL/build_packs.py" --inversions "$P/inversions" --rolls "$P/rolls" \
  --out "$P/voicepack_v1" --top 24 --gpu 0 > "$O/build_packs.log" 2>&1 \
  || { notify "pack build FAILED: $(tail -n 3 "$O/build_packs.log" | tr '\n' ' ' | tail -c 300)"; exit 1; }
notify "voice pack built: $(grep 'kept' "$O/build_packs.log" | tail -1); zip at clone_out/packs/voicepack_v1.zip"

# ---- P4: the on-device-inversion measurement --------------------------------
ENC=$(ls "$O"/encoder-r5.pt.best "$O"/encoder-r4.pt.best "$O"/encoder-r2.pt.best 2>/dev/null | head -1)
REFS=("$CO/library/refs/Dale.wav" "$CO/library/refs/Stephen Fry.wav"
      "$CO/library/refs/Fireside Narrator.wav"
      "$CO/library/refs/Soothing female british voice.wav")
CUDA_VISIBLE_DEVICES=0 $PY "$TOOL/cma_polish.py" --refs "${REFS[@]}" \
  --basis-extra "$O/inversions/aux_pairs.npz" --bank "$O/inversions/aux_pairs.npz" \
  ${ENC:+--encoder "$ENC"} --k 128 --iters 30 --pop 10 \
  --out "$O/cma_eval.json" > "$O/cma_eval.log" 2>&1 \
  || { notify "CMA experiment FAILED: $(tail -n 3 "$O/cma_eval.log" | tr '\n' ' ' | tail -c 300)"; exit 1; }
CMA=$(EV=$O/cma_eval.json python3 - <<'PY'
import json, os
rows = json.load(open(os.environ["EV"]))
m0 = sum(r["start_held_out"] for r in rows) / len(rows)
m1 = sum(r["held_out"] for r in rows) / len(rows)
ev = sum(r["evals"] for r in rows) / len(rows)
print(f"start {m0:.3f} -> polished {m1:.3f} over {len(rows)} refs, ~{ev:.0f} evals "
      f"(≈{ev/60:.0f}-{ev/30:.0f} phone-min); " +
      "; ".join(f"{r['ref'].split('.')[0]} {r['held_out']:.2f}" for r in rows))
PY
)
notify "forward-only inversion result: $CMA"
notify "evening run complete. voicepack_v1.zip + cma_eval.json + per-ref polished styles in clone_out/"
touch "$O/DONE_EVENING"
