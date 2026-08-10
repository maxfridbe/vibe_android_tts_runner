#!/usr/bin/env bash
# Opportunistic second inversion worker for the overnight run.
#
# The basis-ceiling measurement made inverted real-speaker styles the scarcest
# resource of the night, and the 5070 sits idle between the round-1 eval and
# round-2 training. This waits for that window, runs invert_corpus.py on GPU0
# over a speaker slice disjoint from the 3060 worker (same shuffle seed,
# --skip past its allocation), and gets out of the way before round-2 training
# needs the card: it stops at 38 pairs (the driver triggers round 2 at 40, so
# the count is crossed by the 3060 worker alone), the moment train_r2.log
# appears, or 15 minutes before the driver's round-2 deadline.
set -u
WORK=${WORK:-$HOME/supertonic-experiment}
OUT=$WORK/clone_out/overnight
PY=$WORK/venv/bin/python
START=$(stat -c %Y "$OUT/driver.log")
DEADLINE=$((START + 7 * 3600 - 900))
count() { ls "$OUT/inversions/pairs" 2>/dev/null | grep -c '\.npz$' || true; }

while [ ! -f "$OUT/eval_r1.json" ]; do
  [ "$(date +%s)" -ge "$DEADLINE" ] && exit 0
  pgrep -f tooling/overnight.sh >/dev/null || exit 0
  sleep 60
done
[ "$(date +%s)" -ge "$DEADLINE" ] && exit 0
[ -f "$OUT/train_r2.log" ] && exit 0

setsid nohup $PY "$WORK/tooling/invert_corpus.py" \
  --corpus "$WORK/corpora/LibriSpeech/train-clean-100" \
  --out "$OUT/inversions" --gpu 0 --skip 120 --max-speakers 60 \
  > "$OUT/inversions_gpu0.log" 2>&1 &
W=$!
echo "gpu0 worker pid $W"

while kill -0 "$W" 2>/dev/null; do
  if [ "$(count)" -ge 38 ] || [ "$(date +%s)" -ge "$DEADLINE" ] \
     || [ -f "$OUT/train_r2.log" ] || ! pgrep -f tooling/overnight.sh >/dev/null; then
    kill -- -"$W" 2>/dev/null
    break
  fi
  sleep 15
done
echo "gpu0 worker released the card at $(count) pairs"
