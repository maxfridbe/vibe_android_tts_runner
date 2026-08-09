#!/usr/bin/env bash
# Train style encoders on both GPUs at once and keep the better one.
#
# The two cards are not equal (5070 ~0.40 s/step, 3060 ~0.57 s/step), so
# splitting one run across them with DDP would leave the fast card waiting.
# Running a *different configuration* per card is the better use of the pair:
# same data, same held-out speakers, two points in a hyper-parameter space we
# have no prior for — and the held-out score picks the winner.
#
#   train_sweep.sh --pairs pairs.npz --targets bank.npz --steps 20000
set -euo pipefail
SELF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK=${WORK:-$HOME/supertonic-experiment}
OUT=$WORK/clone_out
PY=${PYTHON:-$WORK/venv/bin/python}

PAIRS=$OUT/pairs_p1.npz
TARGETS=$OUT/bank_all.npz
STEPS=20000
while [ $# -gt 0 ]; do
  case $1 in
    --pairs) PAIRS=$2; shift 2;;
    --targets) TARGETS=$2; shift 2;;
    --steps) STEPS=$2; shift 2;;
    *) echo "unknown arg $1"; exit 1;;
  esac
done

# Config A: the compact basis — fewer coefficients, so less to overfit and a
# smaller ONNX for the phone.
# Config B: a wider basis and gentler duration anchor, in case identity needs
# room the compact basis cannot express.
run() {  # <gpu> <tag> <extra args...>
  local gpu=$1 tag=$2; shift 2
  echo "GPU $gpu: $tag"
  CUDA_VISIBLE_DEVICES=$gpu nohup "$PY" "$SELF/train_encoder.py" \
    --pairs "$PAIRS" --targets "$TARGETS" --steps "$STEPS" \
    --out "$OUT/encoder-$tag.pt" "$@" > "$OUT/train-$tag.log" 2>&1 &
}

run 0 k64  --k 64  --lr 3e-4 --w-dur 0.3
run 1 k128 --k 128 --lr 2e-4 --w-dur 0.15
wait

echo
for tag in k64 k128; do
  best=$("$PY" - "$OUT/encoder-$tag.pt.metrics.json" <<'PY'
import json, sys
h = json.load(open(sys.argv[1]))
b = max(h, key=lambda r: r["val_cos"])
print(f"{b['val_cos']:.4f} at step {b['step']}")
PY
)
  echo "$tag: best held-out cos $best"
done
echo
echo "export the winner:  tooling/export_cloner.py --encoder $OUT/encoder-<tag>.pt.best --out <assets dir>"
