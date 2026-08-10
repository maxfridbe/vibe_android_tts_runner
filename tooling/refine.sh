#!/usr/bin/env bash
# Refinement round for the translation heads: the three levers left on the
# table after the scaled twins — an ensemble of both representations (qwen
# carried the character voices, ECAPA the in-corpus ones), three times the
# manufactured supervision, and a forward-only CMA polish on top of the best
# head's predictions, which is the shape the phone would actually run.
#
#   nohup tooling/refine.sh > clone_out/overnight/refine_driver.log 2>&1 &
set -u
WORK=${WORK:-$HOME/supertonic-experiment}
O=$WORK/clone_out/overnight
TOOL=$WORK/tooling
PY=$WORK/venv/bin/python
ABM_PY=$HOME/audiobook-maker/venv/bin/python
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
fail() { notify "refine FAILED at $1: $(tail -n 3 "$2" 2>/dev/null | tr '\n' ' ' | tail -c 300)"; exit 1; }

$PY - <<PY || fail "snapshot" /dev/null
import numpy as np
d = np.load("$O/inversions/aux_pairs.npz", allow_pickle=True)
np.savez_compressed("$O/all_pairs3.npz", **{k: d[k] for k in ("emb","ttl","dp","cos","spk")})
np.savez_compressed("$O/real_ecapa.npz", feat=np.asarray(d["emb"], dtype=np.float32),
                    spk=np.asarray(d["spk"]).astype(str))
print(len(d["emb"]), "pairs")
PY
notify "refine: $($PY -c "import numpy as np; print(len(np.load('$O/all_pairs3.npz')['emb']))") real pairs; manufacturing 6000 and training the ensemble"

CUDA_VISIBLE_DEVICES=0 $PY "$TOOL/gen_style_pairs2.py" --pairs-npz "$O/all_pairs3.npz" \
  --k 128 --count 6000 --out-dir "$O/mfg_wavs" > "$O/rf_mfg.log" 2>&1 || fail "manufacture" "$O/rf_mfg.log"
CUDA_VISIBLE_DEVICES=0 $ABM_PY "$TOOL/extract_qwen_features.py" \
  --wavs "$O/inversions/refs" "$O/eval_refs" --out "$O/qwen_feats3.npz" \
  > "$O/rf_featreal.log" 2>&1 || fail "real features" "$O/rf_featreal.log"
CUDA_VISIBLE_DEVICES=0 $ABM_PY "$TOOL/extract_qwen_features.py" --wavs "$O/mfg_wavs" \
  --out "$O/mfg_qwen3.npz" > "$O/rf_featmfgq.log" 2>&1 || fail "mfg qwen features" "$O/rf_featmfgq.log"
CUDA_VISIBLE_DEVICES=0 $PY "$TOOL/embed_wavs.py" --wavs "$O/mfg_wavs" \
  --out "$O/mfg_ecapa3.npz" > "$O/rf_featmfge.log" 2>&1 || fail "mfg ecapa features" "$O/rf_featmfge.log"
CUDA_VISIBLE_DEVICES=0 $PY "$TOOL/embed_wavs.py" --wavs "$O/eval_refs" \
  --out "$O/eval_ecapa.npz" > /dev/null 2>&1

$PY - <<PY || fail "join" /dev/null
import numpy as np
co = np.load("$O/mfg_wavs/coeffs.npz", allow_pickle=True)
cmap = {s: c for s, c in zip(np.asarray(co["spk"]).astype(str), co["coeff"])}
q = np.load("$O/mfg_qwen3.npz", allow_pickle=True)
e = np.load("$O/mfg_ecapa3.npz", allow_pickle=True)
qm = {s: v for s, v in zip(np.asarray(q["spk"]).astype(str), q["feat"])}
em = {s: v for s, v in zip(np.asarray(e["spk"]).astype(str), e["feat"])}
names = [s for s in cmap if s in qm and s in em]
np.savez_compressed("$O/extra_qwen3.npz", feat=np.stack([qm[s] for s in names]),
                    coeff=np.stack([cmap[s] for s in names]))
np.savez_compressed("$O/extra_ecapa3.npz", feat=np.stack([em[s] for s in names]),
                    coeff=np.stack([cmap[s] for s in names]))
np.savez_compressed("$O/extra_ens3.npz",
                    feat=np.stack([np.concatenate([qm[s], em[s]]) for s in names]),
                    coeff=np.stack([cmap[s] for s in names]))
# eval refs: ensemble features for prediction
qe = np.load("$O/qwen_feats3.npz", allow_pickle=True)
ee = np.load("$O/eval_ecapa.npz", allow_pickle=True)
qem = {s: v for s, v in zip(np.asarray(qe["spk"]).astype(str), qe["feat"])}
eem = {s: v for s, v in zip(np.asarray(ee["spk"]).astype(str), ee["feat"])}
evn = [s for s in eem if s in qem]
np.savez_compressed("$O/eval_ens.npz",
                    feat=np.stack([np.concatenate([qem[s], eem[s]]) for s in evn]),
                    spk=np.array(evn))
print(len(names), "mfg triples,", len(evn), "eval ensemble rows")
PY

train() {  # tag, extra args...
  local TAG=$1; shift
  CUDA_VISIBLE_DEVICES=0 $PY "$TOOL/train_translation.py" --pairs-npz "$O/all_pairs3.npz" \
    --k 128 --dropout 0.1 --feat-noise 0.05 --steps 10000 "$@" \
    --out "$O/trans_$TAG.pt" > "$O/trans_$TAG.log" 2>&1 || fail "head $TAG" "$O/trans_$TAG.log"
}
train ecapa_r --extra-pairs "$O/extra_ecapa3.npz"
train qwen_r --features "$O/qwen_feats3.npz" --pca 256 --extra-pairs "$O/extra_qwen3.npz"
train ens_r --features "$O/qwen_feats3.npz" --features2 "$O/real_ecapa.npz" \
  --pca 320 --extra-pairs "$O/extra_ens3.npz"

REFS=("$O/eval_refs/Dale.wav" "$O/eval_refs/Stephen Fry.wav"
      "$O/eval_refs/Fireside Narrator.wav" "$O/eval_refs/Soothing female british voice.wav")
for f in "$O"/eval_refs/*.flac; do REFS+=("$f"); done
for V in ecapa_r qwen_r ens_r; do
  CUDA_VISIBLE_DEVICES=0 $PY "$TOOL/score_styles.py" --styles "$O/trans_${V}_val_styles" \
    --refs "$O/inversions/refs" --out "$O/trans_${V}_score.json" > "$O/rf_score_$V.log" 2>&1
  case $V in
    ecapa_r) F="";;
    qwen_r)  F="--features $O/qwen_feats3.npz";;
    ens_r)   F="--features $O/eval_ens.npz";;
  esac
  CUDA_VISIBLE_DEVICES=0 $PY "$TOOL/predict_head.py" --head "$O/trans_$V.pt" \
    --refs "${REFS[@]}" $F --out-dir "$O/unseen_$V" >> "$O/rf_score_$V.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 $PY "$TOOL/score_styles.py" --styles "$O/unseen_$V" \
    --refs "$O/eval_refs" --out "$O/unseen_${V}_score.json" >> "$O/rf_score_$V.log" 2>&1
done
SCORES=$(python3 - <<PY
import json
def g(p):
    try: return json.load(open(p))["mean"]
    except Exception: return float("nan")
print(" | ".join(f"{v}: val {g(f'$O/trans_{v}_score.json'):.3f} unseen {g(f'$O/unseen_{v}_score.json'):.3f}"
                 for v in ("ecapa_r","qwen_r","ens_r")) +
      " (prev: ecapa 0.49, qwen 0.42 unseen)")
PY
)
notify "refine heads: $SCORES"

BEST=$(python3 - <<PY
import json
best, bv = "ens_r", -1
for v in ("ecapa_r","qwen_r","ens_r"):
    try:
        m = json.load(open(f"$O/unseen_{v}_score.json"))["mean"]
        if m > bv: best, bv = v, m
    except Exception: pass
print(best)
PY
)
CUDA_VISIBLE_DEVICES=0 $PY "$TOOL/cma_polish.py" --refs "${REFS[@]}" \
  --basis-extra "$O/all_pairs3.npz" --bank "$O/all_pairs3.npz" \
  --start-styles "$O/unseen_$BEST" --k 128 --iters 30 --pop 10 \
  --out "$O/cma_refine.json" > "$O/rf_cma.log" 2>&1 || fail "cma polish" "$O/rf_cma.log"
CMA=$(EV=$O/cma_refine.json python3 - <<'PY'
import json, os
rows = json.load(open(os.environ["EV"]))
m0 = sum(r["start_held_out"] for r in rows) / len(rows)
m1 = sum(r["held_out"] for r in rows) / len(rows)
ev = sum(r["evals"] for r in rows) / len(rows)
print(f"start {m0:.3f} -> polished {m1:.3f}, ~{ev:.0f} evals (≈{ev/40:.0f}-{ev/20:.0f} phone-min); "
      + "; ".join(f"{r['ref'].split('.')[0][:12]} {r['held_out']:.2f}" for r in rows))
PY
)
notify "refine polish (seeded by $BEST): $CMA. Everything in clone_out/overnight/"
touch "$O/DONE_REFINE"
