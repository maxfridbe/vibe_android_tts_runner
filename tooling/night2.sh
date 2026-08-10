#!/usr/bin/env bash
# Second-night driver: the Qwen-encoder translation experiment.
#
# Waits for the evening run (voice pack + CMA) to release the GPUs, then runs
# the controlled comparison: extract Qwen clone-prompt features for every
# reference that has an inverted style, train one translation head on those
# features and an identical head on ECAPA embeddings, and score both heads'
# held-out styles against the actual recordings. One variable, one number.
#
#   nohup tooling/night2.sh > clone_out/overnight/night2_driver.log 2>&1 &
set -u
WORK=${WORK:-$HOME/supertonic-experiment}
O=$WORK/clone_out/overnight
P=$WORK/clone_out/packs
TOOL=$WORK/tooling
PY=$WORK/venv/bin/python
ABM_PY=$HOME/audiobook-maker/venv/bin/python
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

while [ ! -f "$O/DONE_EVENING" ] && pgrep -f "[e]vening.sh" >/dev/null \
      && [ "$(date +%s)" -lt $((START + 12 * 3600)) ]; do sleep 600; done
notify "night-2: Qwen-encoder translation experiment starting"

# one pair set: real speakers + designed pack voices
$PY - <<PY || { notify "night-2 pair merge FAILED"; exit 1; }
import numpy as np, os
parts = [p for p in ["$O/inversions/aux_pairs.npz", "$P/inversions/aux_pairs.npz"] if os.path.exists(p)]
ds = [np.load(p, allow_pickle=True) for p in parts]
out = {k: np.concatenate([d[k] for d in ds]) for k in ("emb", "ttl", "dp", "cos", "spk")}
np.savez_compressed("$O/all_pairs.npz", **out)
print(f"{len(out['emb'])} pairs from {len(parts)} sets")
PY

STAGE_LOG=$O/qwen_feats.log
if [ ! -f "$O/qwen_feats.npz" ]; then
  CUDA_VISIBLE_DEVICES=0 $ABM_PY "$TOOL/extract_qwen_features.py" \
    --wavs "$O/inversions/refs" "$P/inversions/refs" \
    --out "$O/qwen_feats.npz" > "$STAGE_LOG" 2>&1 \
    || { notify "feature extraction FAILED: $(tail -n 3 "$STAGE_LOG" | tr '\n' ' ' | tail -c 300)"; exit 1; }
fi
notify "qwen features: $(tail -n 1 "$STAGE_LOG")"

for VARIANT in qwen ecapa; do
  FEATS=""
  [ "$VARIANT" = qwen ] && FEATS="--features $O/qwen_feats.npz"
  CUDA_VISIBLE_DEVICES=0 $PY "$TOOL/train_translation.py" \
    --pairs-npz "$O/all_pairs.npz" $FEATS --k 128 \
    --out "$O/trans_$VARIANT.pt" > "$O/trans_$VARIANT.log" 2>&1 \
    || { notify "translation head ($VARIANT) FAILED: $(tail -n 3 "$O/trans_$VARIANT.log" | tr '\n' ' ' | tail -c 300)"; exit 1; }
  CUDA_VISIBLE_DEVICES=0 $PY "$TOOL/score_styles.py" \
    --styles "$O/trans_${VARIANT}_val_styles" \
    --refs "$O/inversions/refs" "$P/inversions/refs" \
    --out "$O/trans_${VARIANT}_score.json" >> "$O/trans_$VARIANT.log" 2>&1
done

CMP=$(python3 - <<PY
import json
q = json.load(open("$O/trans_qwen_score.json"))
e = json.load(open("$O/trans_ecapa_score.json"))
print(f"qwen-prompt features {q['mean']:.3f} vs ECAPA {e['mean']:.3f} "
      f"held-out cos over {len(q['rows'])}/{len(e['rows'])} voices "
      f"(inversion labels ~0.80, r5-style encoders ~0.27)")
PY
)
notify "translation-head verdict: $CMP"
touch "$O/DONE_NIGHT2"
