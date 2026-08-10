#!/usr/bin/env python3
"""Score a directory of predicted styles against their reference recordings.

Each <spk>.json in --styles is synthesised on the held-out sentences and
compared (ECAPA cosine) to <spk>.wav found in one of the --refs dirs. The
mean over voices is the number two translation heads get compared on.

    score_styles.py --styles trans_qwen_val_styles --refs inversions/refs packs/inversions/refs
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_style import HELD_OUT, load16k     # noqa: E402
from clone_library import synth              # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--styles", required=True)
    ap.add_argument("--refs", nargs="+", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.environ.setdefault("SB_CACHE", os.path.expanduser("~/.cache/speechbrain/ecapa"))
    from speechbrain.inference.speaker import EncoderClassifier
    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.environ["SB_CACHE"], run_opts={"device": args.device})

    def emb(p):
        return torch.nn.functional.normalize(
            enc.encode_batch(load16k(p, args.device)).squeeze(), dim=-1)

    scratch = os.path.join(args.styles, "score_work")
    os.makedirs(scratch, exist_ok=True)
    rows = []
    for f in sorted(os.listdir(args.styles)):
        if not f.endswith(".json"):
            continue
        spk = f[:-5]
        ref = next((p for r in args.refs for ext in (".wav", ".flac")
                    for p in [os.path.join(r, spk + ext)] if os.path.exists(p)), None)
        if not ref:
            continue
        ref_emb = emb(ref)
        per = []
        for i, text in enumerate(HELD_OUT):
            wav = os.path.join(scratch, f"{spk}_{i}.wav")
            if not os.path.exists(wav) and not synth(os.path.join(args.styles, f), text, wav, args.gpu):
                continue
            per.append(float(torch.dot(ref_emb, emb(wav))))
        if per:
            rows.append({"spk": spk, "cos": round(float(np.mean(per)), 4)})
            print(f"  {spk:<16} {rows[-1]['cos']:.3f}", flush=True)

    mean = float(np.mean([r["cos"] for r in rows])) if rows else float("nan")
    print(f"mean held-out cos {mean:.4f} over {len(rows)} voices")
    if args.out:
        json.dump({"mean": round(mean, 4), "rows": rows}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
