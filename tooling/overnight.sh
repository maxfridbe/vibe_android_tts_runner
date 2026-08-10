#!/usr/bin/env bash
# Overnight driver for the cloning-encoder experiments (docs/on-device-cloning.md).
#
# One unattended night on the two-GPU host, sequenced so the cards never idle:
#
#   GPU0: basis-ceiling diagnostic -> embed real corpora into a target bank ->
#         retrain the encoder on real targets (round 1) -> eval on real
#         recordings -> retrain again with inverted-style supervision (round 2)
#         once enough Phase-2 pairs exist -> eval again
#   GPU1: invert LibriSpeech speakers into styles all night (invert_corpus.py)
#
# Every milestone and any failure is pushed to Telegram. The script is
# resumable: finished artefacts are detected and skipped.
#
#   nohup tooling/overnight.sh > clone_out/overnight/driver.log 2>&1 &
set -uo pipefail
export WORK=${WORK:-$HOME/supertonic-experiment}
TOOL=$WORK/tooling
export OUT=$WORK/clone_out/overnight
CO=$WORK/clone_out
CORP=$WORK/corpora
PY=$WORK/venv/bin/python
mkdir -p "$OUT" "$CORP"
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

STAGE=start
LOG=$OUT/driver.log
fail() {
  notify "FAILED in $STAGE — $(tail -n 4 "$LOG" 2>/dev/null | tr '\n' ' ' | tail -c 400)"
  exit 1
}

notify "overnight run started ($(date +%H:%M)). Plan: basis-ceiling diagnostic, real-speech target bank, encoder retrain, LibriSpeech inversions on the 3060, second retrain with inverted-style supervision."

# ---- corpora download (network-bound, overlaps the diagnostic) --------------
STAGE=downloads
(
  set -e; cd "$CORP"
  for part in train-clean-100 dev-clean test-clean; do
    [ -d "LibriSpeech/$part" ] && continue
    curl -sL -o "$part.tgz" "https://www.openslr.org/resources/12/$part.tar.gz"
    tar xzf "$part.tgz" && rm "$part.tgz"
  done
) > "$OUT/download.log" 2>&1 &
DL=$!

# ---- basis-ceiling diagnostic (GPU0) ----------------------------------------
STAGE=basis-ceiling; LOG=$OUT/basis_ceiling.log
if [ ! -f "$OUT/basis_ceiling.json" ]; then
  CUDA_VISIBLE_DEVICES=0 $PY "$TOOL/basis_ceiling.py" --pairs "$CO/pairs_p1.npz" \
    --gpu 0 --out "$OUT/basis_ceiling.json" > "$LOG" 2>&1 || fail
fi
CEIL=$(python3 - <<'PY'
import json, os
s = json.load(open(os.path.join(os.environ["OUT"], "basis_ceiling.json")))
print(f"library inversions score {s['mean_orig']:.3f} as-is; "
      f"{s['mean_proj']:.3f} projected onto the preset-simplex basis; "
      f"{s['mean_proj_refit']:.3f} onto a basis refit with real inverted styles "
      f"(n={s['n_voices']}, k={s['k']})")
PY
)
notify "basis ceiling: $CEIL"

# ---- real-speech target bank (GPU0) -----------------------------------------
STAGE=downloads; LOG=$OUT/download.log
wait $DL || fail
STAGE=embed; LOG=$OUT/embed.log
if [ ! -f "$OUT/bank_real_all.npz" ]; then
  CUDA_VISIBLE_DEVICES=0 $PY "$TOOL/embed_corpus.py" \
    --roots "$CORP/LibriSpeech/train-clean-100" "$CORP/LibriSpeech/dev-clean" \
    --out "$OUT/real_bank.npz" --per-speaker 6 --device cuda:0 > "$LOG" 2>&1 || fail
  $PY "$TOOL/merge_banks.py" "$OUT/bank_real_all.npz" \
    "$CO/bank_all.npz" "$OUT/real_bank.npz" >> "$LOG" 2>&1 || fail
fi
notify "real-speech bank ready: $(grep 'wrote.*real_bank' "$LOG" | tail -1); merged: $(grep 'unique rows' "$LOG" | tail -1)"

# ---- Phase-2 inversions, all night on the 3060 ------------------------------
STAGE=inversions
nohup $PY "$TOOL/invert_corpus.py" --corpus "$CORP/LibriSpeech/train-clean-100" \
  --out "$OUT/inversions" --gpu 1 --max-speakers 120 > "$OUT/inversions.log" 2>&1 &
INV=$!
notify "phase-2 inversions started on the 3060 (120 LibriSpeech speakers queued, ~6-8 min each)"

# ---- round 1: retrain on real targets (GPU0) --------------------------------
STAGE=train-r1; LOG=$OUT/train_r1.log
if [ ! -f "$OUT/encoder-r1.pt.best" ]; then
  CUDA_VISIBLE_DEVICES=0 $PY "$TOOL/train_encoder.py" --pairs "$CO/pairs_p1.npz" \
    --targets "$OUT/bank_real_all.npz" --k 64 --steps 20000 \
    --val-every 500 --val-max 96 --out "$OUT/encoder-r1.pt" > "$LOG" 2>&1 || fail
fi
R1=$(MET=$OUT/encoder-r1.pt.metrics.json python3 - <<'PY'
import json, os
h = json.load(open(os.environ["MET"]))
b = max(h, key=lambda r: r["val_cos"])
print(f"best held-out val cos {b['val_cos']:.4f} at step {b['step']} (previous run: 0.3421 on Qwen targets)")
PY
)
notify "round-1 training done: $R1"

# ---- eval round 1 on real recordings ----------------------------------------
STAGE=eval-r1; LOG=$OUT/eval_r1.log
REFS=("$CO/library/refs/Dale.wav" "$CO/library/refs/Stephen Fry.wav"
      "$CO/library/refs/Fireside Narrator.wav"
      "$CO/library/refs/Soothing female british voice.wav")
mapfile -t TC < <(TCROOT="$CORP/LibriSpeech/test-clean" $PY - <<'PY'
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
  --encoders "$OUT/encoder-r1.pt.best" --refs "${REFS[@]}" "${TC[@]}" \
  --gpu 0 --out "$OUT/eval_r1.json" > "$LOG" 2>&1 || fail
E1=$(EV=$OUT/eval_r1.json python3 - <<'PY'
import json, os
rows = json.load(open(os.environ["EV"]))
dale = next((r for r in rows if r["ref"] == "Dale.wav"), None)
m = sum(r["held_out"] for r in rows if r["held_out"]) / max(1, sum(1 for r in rows if r["held_out"]))
out = f"mean held-out cos {m:.3f} over {len(rows)} real recordings"
if dale:
    out += f"; Dale {dale['held_out']:.3f} (was 0.223, desktop-inversion ceiling 0.820)"
print(out)
PY
)
notify "round-1 real-recording eval: $E1"

# ---- round 2: wait for inverted pairs, retrain with supervision -------------
STAGE=wait-pairs
R2_DEADLINE=$((START + 7 * 3600))
count_pairs() { ls "$OUT/inversions/pairs" 2>/dev/null | grep -c '\.npz$' || true; }
while [ "$(count_pairs)" -lt 40 ] && [ "$(date +%s)" -lt "$R2_DEADLINE" ]; do
  sleep 300
done
NP=$(count_pairs)
if [ "$NP" -ge 12 ]; then
  STAGE=train-r2; LOG=$OUT/train_r2.log
  cp "$OUT/inversions/aux_pairs.npz" "$OUT/aux_snapshot.npz"
  notify "round-2 training starting with $NP inverted real speakers (basis refit + supervised coefficients)"
  if [ ! -f "$OUT/encoder-r2.pt.best" ]; then
    CUDA_VISIBLE_DEVICES=0 $PY "$TOOL/train_encoder.py" --pairs "$CO/pairs_p1.npz" \
      --targets "$OUT/bank_real_all.npz" --k 64 --steps 15000 \
      --basis-extra "$OUT/aux_snapshot.npz" --aux-pairs "$OUT/aux_snapshot.npz" \
      --w-aux 0.3 --val-every 500 --val-max 96 \
      --out "$OUT/encoder-r2.pt" > "$LOG" 2>&1 || fail
  fi
  STAGE=eval-r2; LOG=$OUT/eval_r2.log
  CUDA_VISIBLE_DEVICES=0 $PY "$TOOL/eval_encoder_real.py" \
    --encoders "$OUT/encoder-r2.pt.best" --refs "${REFS[@]}" "${TC[@]}" \
    --gpu 0 --out "$OUT/eval_r2.json" > "$LOG" 2>&1 || fail
  E2=$(EV=$OUT/eval_r2.json python3 - <<'PY'
import json, os
rows = json.load(open(os.environ["EV"]))
dale = next((r for r in rows if r["ref"] == "Dale.wav"), None)
m = sum(r["held_out"] for r in rows if r["held_out"]) / max(1, sum(1 for r in rows if r["held_out"]))
out = f"mean held-out cos {m:.3f}"
if dale:
    out += f"; Dale {dale['held_out']:.3f}"
print(out)
PY
)
  notify "round-2 real-recording eval: $E2"
else
  notify "only $NP inverted pairs by the round-2 deadline — skipping round 2, inversions keep running"
fi

# ---- let the inversions run out the night -----------------------------------
STAGE=final
FINAL_DEADLINE=$((START + 11 * 3600))
while kill -0 "$INV" 2>/dev/null && [ "$(date +%s)" -lt "$FINAL_DEADLINE" ]; do
  sleep 600
done
kill "$INV" 2>/dev/null || true
FINAL="pairs banked: $(count_pairs). Artefacts in clone_out/overnight/ (encoder-r1/r2.pt.best, eval_r1/r2.json, basis_ceiling.json, inversions/aux_pairs.npz)."
notify "overnight run complete ($( printf '%dh%02dm' $(( ($(date +%s)-START)/3600 )) $(( (($(date +%s)-START)%3600)/60 )) )). $FINAL"
touch "$OUT/DONE"
