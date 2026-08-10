#!/usr/bin/env python3
"""Apply a trained translation head to reference recordings -> style JSONs.

The true-unseen test: these refs' styles were never in the head's training
set NOR its basis fit, so score_styles.py on the output is the number that
predicts a stranger recording on a phone.

    predict_head.py --head trans_ecapa.pt --refs Dale.wav ... --out-dir styles/
    predict_head.py --head trans_qwen.pt --features eval_feats.npz --refs ...
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_translation import Head          # noqa: E402
from eval_style import load16k              # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--head", required=True)
    ap.add_argument("--refs", nargs="+", required=True)
    ap.add_argument("--features", default=None,
                    help="npz feat/spk for these refs (qwen heads); omit for ECAPA heads")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ck = torch.load(args.head, map_location="cpu", weights_only=False)
    head = Head(int(ck["in_dim"]), int(ck["k"]), dropout=float(ck.get("dropout", 0.0)))
    head.load_state_dict(ck["model"])
    head.eval()
    basis_m, basis_s, basis_b = ck["mean"].numpy(), ck["scale"].numpy(), ck["basis"].numpy()
    split = int(np.prod([int(v) for v in ck["ttl_shape"]]))

    fmap = None
    if args.features:
        f = np.load(args.features, allow_pickle=True)
        fmap = {s: v for s, v in zip(np.asarray(f["spk"]).astype(str),
                                     np.asarray(f["feat"], dtype=np.float32))}
    else:
        os.environ.setdefault("SB_CACHE", os.path.expanduser("~/.cache/speechbrain/ecapa"))
        from speechbrain.inference.speaker import EncoderClassifier
        enc = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.environ["SB_CACHE"], run_opts={"device": args.device})

    for ref in args.refs:
        stem = os.path.splitext(os.path.basename(ref))[0]
        if fmap is not None:
            if stem not in fmap:
                print(f"{stem}: no feature row, skipped")
                continue
            x = fmap[stem]
        else:
            with torch.no_grad():
                e = enc.encode_batch(load16k(ref, args.device)).squeeze()
                x = torch.nn.functional.normalize(e, dim=-1).cpu().numpy()
        if ck.get("pca_comp") is not None:
            x = (x - ck["pca_mean"]) @ ck["pca_comp"].T
        with torch.no_grad():
            c = head(torch.tensor(x[None], dtype=torch.float32)).numpy()[0]
        flat = basis_m + (c * basis_s) @ basis_b
        ttl = flat[:split].reshape([int(v) for v in ck["ttl_shape"]])
        dp = flat[split:].reshape([int(v) for v in ck["dp_shape"]])
        ttl /= (np.linalg.norm(ttl, axis=-1, keepdims=True) + 1e-8)
        dp /= (np.linalg.norm(dp, axis=-1, keepdims=True) + 1e-8)
        json.dump({
            "style_ttl": {"dims": [1, *ttl.shape], "data": ttl.reshape(-1).tolist()},
            "style_dp": {"dims": [1, *dp.shape], "data": dp.reshape(-1).tolist()},
            "metadata": {"source": f"predict_head {os.path.basename(args.head)}",
                         "reference": ref},
        }, open(os.path.join(args.out_dir, stem + ".json"), "w"))
        print(f"{stem}: style written")


if __name__ == "__main__":
    main()
