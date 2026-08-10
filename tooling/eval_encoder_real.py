#!/usr/bin/env python3
"""Score a trained cloning encoder the way the phone will use it.

Real recording in -> ECAPA -> encoder -> style -> synthesise sentences the
encoder never trained on -> compare against the recording. This is the number
docs/on-device-cloning.md tracks (the first encoder scored 0.223 on Dale
against the desktop inversion's 0.820), so the HELD_OUT set from eval_style
is kept for comparability — and because three of those four sentences are
also training probes, a second set of genuinely fresh sentences is scored
alongside to catch probe overfitting.

    eval_encoder_real.py --encoders r1.pt.best r2.pt.best \
                         --refs library/refs/Dale.wav corpora/.../1089-134686-0000.flac
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_cloner import StyleHead             # noqa: E402
from eval_style import HELD_OUT, load16k        # noqa: E402
from clone_library import synth, slugify, WORK  # noqa: E402

FRESH = [
    "The museum closes early on the last Friday of every month.",
    "She poured the tea before the kettle had fully boiled.",
    "Our neighbours painted their fence a strange shade of green.",
    "Take the second left after the old railway bridge.",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoders", nargs="+", required=True, help=".pt.best checkpoints")
    ap.add_argument("--refs", nargs="+", required=True, help="real recordings (wav/flac)")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=os.path.join(WORK, "clone_out/overnight/eval_real.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    scratch = os.path.join(os.path.dirname(args.out), "eval_work")
    os.makedirs(scratch, exist_ok=True)

    os.environ.setdefault("SB_CACHE", os.path.expanduser("~/.cache/speechbrain/ecapa"))
    from speechbrain.inference.speaker import EncoderClassifier
    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.environ["SB_CACHE"], run_opts={"device": args.device})

    def emb(p):
        return torch.nn.functional.normalize(
            enc.encode_batch(load16k(p, args.device)).squeeze(), dim=-1)

    results = []
    for ck in args.encoders:
        ckpt = torch.load(ck, map_location="cpu", weights_only=False)
        head = StyleHead(ckpt).eval()
        tag = os.path.basename(ck).replace(".pt.best", "")
        print(f"\n{tag}: step {ckpt.get('step','?')}, trained val cos "
              f"{ckpt.get('val_cos', float('nan')):.4f}")
        for ref in args.refs:
            ref_emb = emb(ref)
            with torch.no_grad():
                ttl, dp = head(ref_emb.cpu()[None])
            slug = slugify(os.path.splitext(os.path.basename(ref))[0])
            style = os.path.join(scratch, f"{tag}.{slug}.json")
            json.dump({
                "style_ttl": {"dims": [1, *ttl.shape[1:]], "data": ttl.reshape(-1).tolist()},
                "style_dp": {"dims": [1, *dp.shape[1:]], "data": dp.reshape(-1).tolist()},
                "metadata": {"source": f"eval_encoder_real {tag}", "reference": ref},
            }, open(style, "w"))

            held, fresh = [], []
            for i, text in enumerate(HELD_OUT + FRESH):
                wav = os.path.join(scratch, f"{tag}.{slug}.{i}.wav")
                if not os.path.exists(wav) and not synth(style, text, wav, args.gpu):
                    continue
                (held if i < len(HELD_OUT) else fresh).append(
                    float(torch.dot(ref_emb, emb(wav))))
            row = {"encoder": tag, "ref": os.path.basename(ref),
                   "held_out": round(float(np.mean(held)), 4) if held else None,
                   "fresh": round(float(np.mean(fresh)), 4) if fresh else None,
                   "style": style}
            results.append(row)
            print(f"  {row['ref']:<34} held-out {row['held_out']}  fresh {row['fresh']}",
                  flush=True)

    json.dump(results, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
