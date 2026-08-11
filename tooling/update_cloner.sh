#!/usr/bin/env bash
# Retrain the shipped cloning heads on all pairs banked so far and re-export
# only what beats the current champion.
#
# The labelers kept running after trans_ecapa_s shipped (0.492 true-unseen on
# ~85 pairs); there are ~187 now. This retrains the ECAPA and Qwen heads with
# the same winning recipe (refit basis + manufactured pairs + dropout/noise)
# on the larger set, scores them on the same held-out refs, and prints the
# comparison. export happens in a second step, by hand, only for a head that
# actually improves — no silent regression into the APK.
#
#   nohup tooling/update_cloner.sh > clone_out/overnight/update.log 2>&1 &
set -u
WORK=${WORK:-$HOME/supertonic-experiment}
O=$WORK/clone_out/overnight
TOOL=$WORK/tooling
PY=$WORK/venv/bin/python
ABM_PY=$HOME/audiobook-maker/venv/bin/python
CO=$WORK/clone_out
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
fail() { notify "update FAILED at $1: $(tail -n 3 "$2" 2>/dev/null | tr '\n' ' ' | tail -c 300)"; exit 1; }

# fresh snapshot of every banked pair; and an ECAPA feature file for them
$PY - <<PY || fail snapshot /dev/null
import numpy as np
d = np.load("$O/inversions/aux_pairs.npz", allow_pickle=True)
np.savez_compressed("$O/all_pairs_u.npz", **{k: d[k] for k in ("emb","ttl","dp","cos","spk")})
np.savez_compressed("$O/real_ecapa_u.npz", feat=np.asarray(d["emb"], dtype=np.float32),
                    spk=np.asarray(d["spk"]).astype(str))
print(len(d["emb"]), "pairs")
PY
N=$($PY -c "import numpy as np; print(len(np.load('$O/all_pairs_u.npz')['emb']))")
notify "update: retraining cloning heads on $N pairs (was ~85 for the shipped 0.492)"

# manufacture 3000 (audio, coeff) pairs in the current refit basis, then the
# two feature views of them
CUDA_VISIBLE_DEVICES=0 $PY "$TOOL/gen_style_pairs2.py" --pairs-npz "$O/all_pairs_u.npz" \
  --k 128 --count 3000 --out-dir "$O/mfg_u" > "$O/u_mfg.log" 2>&1 || fail manufacture "$O/u_mfg.log"
CUDA_VISIBLE_DEVICES=0 $ABM_PY "$TOOL/extract_qwen_features.py" \
  --wavs "$O/inversions/refs" "$O/eval_refs" --out "$O/qwen_feats_u.npz" \
  > "$O/u_featreal.log" 2>&1 || fail "real qwen feats" "$O/u_featreal.log"
CUDA_VISIBLE_DEVICES=0 $ABM_PY "$TOOL/extract_qwen_features.py" --wavs "$O/mfg_u" \
  --out "$O/mfg_qwen_u.npz" > "$O/u_mfgq.log" 2>&1 || fail "mfg qwen feats" "$O/u_mfgq.log"
CUDA_VISIBLE_DEVICES=0 $PY "$TOOL/embed_wavs.py" --wavs "$O/mfg_u" \
  --out "$O/mfg_ecapa_u.npz" > "$O/u_mfge.log" 2>&1 || fail "mfg ecapa feats" "$O/u_mfge.log"

$PY - <<PY || fail join /dev/null
import numpy as np
co = np.load("$O/mfg_u/coeffs.npz", allow_pickle=True)
cmap = {s: c for s, c in zip(np.asarray(co["spk"]).astype(str), co["coeff"])}
for feats, out in (("$O/mfg_qwen_u.npz", "$O/extra_qwen_u.npz"),
                   ("$O/mfg_ecapa_u.npz", "$O/extra_ecapa_u.npz")):
    f = np.load(feats, allow_pickle=True)
    s = np.asarray(f["spk"]).astype(str)
    keep = np.array([x in cmap for x in s])
    np.savez_compressed(out, feat=np.asarray(f["feat"])[keep],
                        coeff=np.stack([cmap[x] for x in s[keep]]))
print("joined")
PY

CUDA_VISIBLE_DEVICES=0 $PY "$TOOL/train_translation.py" --pairs-npz "$O/all_pairs_u.npz" \
  --k 128 --dropout 0.1 --feat-noise 0.05 --steps 8000 \
  --extra-pairs "$O/extra_ecapa_u.npz" --out "$O/trans_ecapa_u.pt" \
  > "$O/trans_ecapa_u.log" 2>&1 || fail "ecapa head" "$O/trans_ecapa_u.log"
CUDA_VISIBLE_DEVICES=0 $PY "$TOOL/train_translation.py" --pairs-npz "$O/all_pairs_u.npz" \
  --features "$O/qwen_feats_u.npz" --pca 256 --dropout 0.1 --feat-noise 0.05 --steps 8000 \
  --extra-pairs "$O/extra_qwen_u.npz" --out "$O/trans_qwen_u.pt" \
  > "$O/trans_qwen_u.log" 2>&1 || fail "qwen head" "$O/trans_qwen_u.log"

REFS=("$O/eval_refs/Dale.wav" "$O/eval_refs/Stephen Fry.wav"
      "$O/eval_refs/Fireside Narrator.wav" "$O/eval_refs/Soothing female british voice.wav")
for f in "$O"/eval_refs/*.flac; do REFS+=("$f"); done
for V in ecapa_u qwen_u; do
  F=""; [ "$V" = qwen_u ] && F="--features $O/qwen_feats_u.npz"
  CUDA_VISIBLE_DEVICES=0 $PY "$TOOL/predict_head.py" --head "$O/trans_$V.pt" \
    --refs "${REFS[@]}" $F --out-dir "$O/unseen_$V" > "$O/u_score_$V.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 $PY "$TOOL/score_styles.py" --styles "$O/unseen_$V" \
    --refs "$O/eval_refs" --out "$O/unseen_${V}_score.json" >> "$O/u_score_$V.log" 2>&1
done

RESULT=$(python3 - <<PY
import json
def g(p):
    try: return json.load(open(p))["mean"]
    except Exception: return float("nan")
e = g("$O/unseen_ecapa_u_score.json"); q = g("$O/unseen_qwen_u_score.json")
best = "ecapa" if e >= q else "qwen"
print(f"ecapa_u {e:.3f}, qwen_u {q:.3f} (shipped ecapa 0.492); "
      f"{'IMPROVED — export '+best if max(e,q) > 0.492 else 'no gain over shipped, keeping current'}")
PY
)
notify "update result: $RESULT"
touch "$O/DONE_UPDATE"
