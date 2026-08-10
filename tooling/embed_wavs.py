#!/usr/bin/env python3
"""ECAPA-embed a directory of wavs into a name-keyed feature npz — the ECAPA
twin of extract_qwen_features.py, so both variants join manufactured coeffs
by the same spk stems.

    embed_wavs.py --wavs mfg_wavs --out mfg_ecapa.npz
"""
import argparse
import os

import numpy as np
import soundfile as sf
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wavs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.environ.setdefault("SB_CACHE", os.path.expanduser("~/.cache/speechbrain/ecapa"))
    from speechbrain.inference.speaker import EncoderClassifier
    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.environ["SB_CACHE"], run_opts={"device": args.device})

    files = []
    for w in args.wavs:
        if os.path.isdir(w):
            files += [os.path.join(w, f) for f in sorted(os.listdir(w))
                      if f.endswith((".wav", ".flac"))]
        else:
            files.append(w)
    F, S = [], []
    for i, p in enumerate(files):
        a, sr = sf.read(p, dtype="float32")
        if a.ndim > 1:
            a = a.mean(1)
        if sr != 16000:
            n = int(len(a) * 16000 / sr)
            a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype("float32")
        with torch.no_grad():
            e = enc.encode_batch(torch.from_numpy(a)[None].to(args.device)).squeeze()
        F.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy().astype("float32"))
        S.append(os.path.splitext(os.path.basename(p))[0])
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(files)}", flush=True)
    np.savez_compressed(args.out, feat=np.stack(F), spk=np.array(S))
    print(f"wrote {args.out}: {len(F)} x {F[0].shape[0]}")


if __name__ == "__main__":
    main()
