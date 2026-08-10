#!/usr/bin/env bash
# Scale the Qwen-translation idea past its data starvation.
#
# Round one memorised 145 pairs (train loss 0.0000) and the 2048-d Qwen input
# lost to ECAPA-192 purely on conditioning. This round fixes the regime:
# fresh pair snapshot (the labelers kept working), PCA + dropout + feature
# noise on the head, thousands of manufactured (audio, coeff) pairs from the
# refit basis, and a true-unseen eval on voices that are in neither the
# training set nor the basis.
#
#   nohup tooling/qwen_scale.sh > clone_out/overnight/qwen_scale.log 2>&1 &
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
fail() { notify "qwen-scale FAILED at $1: $(tail -n 3 "$2" 2>/dev/null | tr '\n' ' ' | tail -c 300)"; exit 1; }

# GPU0 belongs to this run; the GPU1 labeler keeps growing the pair bank
G0=$(pgrep -f "[i]nvert_corpus.py.*gpu 0" | head -1)
[ -n "$G0" ] && kill -- -"$G0" 2>/dev/null

# ---- fresh snapshot: every pair the labelers have produced so far -----------
$PY - <<PY || fail "pair snapshot" /dev/null
import numpy as np
d = np.load("$O/inversions/aux_pairs.npz", allow_pickle=True)
np.savez_compressed("$O/all_pairs2.npz", **{k: d[k] for k in ("emb","ttl","dp","cos","spk")})
print(f"{len(d['emb'])} real pairs snapshotted")
PY
notify "qwen-scale started: $($PY -c "import numpy as np; print(len(np.load('$O/all_pairs2.npz')['emb']))") real pairs in the snapshot"

# ---- eval refs: 4 library voices + 4 unseen test-clean speakers -------------
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
REFS=("$CO/library/refs/Dale.wav" "$CO/library/refs/Stephen Fry.wav"
      "$CO/library/refs/Fireside Narrator.wav"
      "$CO/library/refs/Soothing female british voice.wav" "${TC[@]}")
mkdir -p "$O/eval_refs"
for r in "${REFS[@]}"; do cp -n "$r" "$O/eval_refs/" 2>/dev/null || true; done

# ---- features for real refs (regrown) and eval refs -------------------------
CUDA_VISIBLE_DEVICES=0 $ABM_PY "$TOOL/extract_qwen_features.py" \
  --wavs "$O/inversions/refs" "$O/eval_refs" \
  --out "$O/qwen_feats2.npz" > "$O/qsc_feats.log" 2>&1 || fail "real-ref features" "$O/qsc_feats.log"

# ---- manufacture supervision in the same basis ------------------------------
CUDA_VISIBLE_DEVICES=0 $PY "$TOOL/gen_style_pairs2.py" --pairs-npz "$O/all_pairs2.npz" \
  --k 128 --count 2000 --out-dir "$O/mfg_wavs" > "$O/qsc_mfg.log" 2>&1 || fail "manufacture" "$O/qsc_mfg.log"
CUDA_VISIBLE_DEVICES=0 $ABM_PY "$TOOL/extract_qwen_features.py" --wavs "$O/mfg_wavs" \
  --out "$O/mfg_qwen.npz" > "$O/qsc_mfgq.log" 2>&1 || fail "mfg qwen features" "$O/qsc_mfgq.log"
CUDA_VISIBLE_DEVICES=0 $PY "$TOOL/embed_wavs.py" --wavs "$O/mfg_wavs" \
  --out "$O/mfg_ecapa.npz" > "$O/qsc_mfge.log" 2>&1 || fail "mfg ecapa features" "$O/qsc_mfge.log"
$PY - <<PY || fail "mfg join" /dev/null
import numpy as np
co = np.load("$O/mfg_wavs/coeffs.npz", allow_pickle=True)
cmap = {s: c for s, c in zip(np.asarray(co["spk"]).astype(str), co["coeff"])}
for feats, out in (("$O/mfg_qwen.npz", "$O/extra_qwen.npz"),
                   ("$O/mfg_ecapa.npz", "$O/extra_ecapa.npz")):
    f = np.load(feats, allow_pickle=True)
    s = np.asarray(f["spk"]).astype(str)
    keep = np.array([x in cmap for x in s])
    np.savez_compressed(out, feat=np.asarray(f["feat"])[keep],
                        coeff=np.stack([cmap[x] for x in s[keep]]))
    print(out, int(keep.sum()))
PY
notify "qwen-scale: manufactured pairs ready ($(ls $O/mfg_wavs | grep -c '\.wav$') wavs); training scaled heads"

# ---- the scaled twins -------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 $PY "$TOOL/train_translation.py" --pairs-npz "$O/all_pairs2.npz" \
  --features "$O/qwen_feats2.npz" --extra-pairs "$O/extra_qwen.npz" \
  --k 128 --pca 256 --dropout 0.1 --feat-noise 0.05 --steps 8000 \
  --out "$O/trans_qwen_s.pt" > "$O/trans_qwen_s.log" 2>&1 || fail "qwen head" "$O/trans_qwen_s.log"
CUDA_VISIBLE_DEVICES=0 $PY "$TOOL/train_translation.py" --pairs-npz "$O/all_pairs2.npz" \
  --extra-pairs "$O/extra_ecapa.npz" \
  --k 128 --dropout 0.1 --feat-noise 0.05 --steps 8000 \
  --out "$O/trans_ecapa_s.pt" > "$O/trans_ecapa_s.log" 2>&1 || fail "ecapa head" "$O/trans_ecapa_s.log"

# ---- score: same-domain val + true-unseen -----------------------------------
for V in qwen_s ecapa_s; do
  CUDA_VISIBLE_DEVICES=0 $PY "$TOOL/score_styles.py" --styles "$O/trans_${V}_val_styles" \
    --refs "$O/inversions/refs" --out "$O/trans_${V}_score.json" > "$O/score_$V.log" 2>&1
  FEATS=""
  [ "$V" = qwen_s ] && FEATS="--features $O/qwen_feats2.npz"
  CUDA_VISIBLE_DEVICES=0 $PY "$TOOL/predict_head.py" --head "$O/trans_$V.pt" \
    --refs "${REFS[@]}" $FEATS --out-dir "$O/unseen_$V" >> "$O/score_$V.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 $PY "$TOOL/score_styles.py" --styles "$O/unseen_$V" \
    --refs "$O/eval_refs" --out "$O/unseen_${V}_score.json" >> "$O/score_$V.log" 2>&1
done

VERDICT=$(python3 - <<PY
import json
def g(p):
    try: return json.load(open(p))["mean"]
    except Exception: return float("nan")
print(f"val: qwen {g('$O/trans_qwen_s_score.json'):.3f} vs ecapa {g('$O/trans_ecapa_s_score.json'):.3f} | "
      f"TRUE-UNSEEN (Dale & co + test-clean): qwen {g('$O/unseen_qwen_s_score.json'):.3f} "
      f"vs ecapa {g('$O/unseen_ecapa_s_score.json'):.3f} "
      f"(round-1: qwen 0.326 vs ecapa 0.524 on val; old encoders 0.27; inversion 0.80)")
PY
)
notify "qwen-scale verdict: $VERDICT"
touch "$O/DONE_QSCALE"
