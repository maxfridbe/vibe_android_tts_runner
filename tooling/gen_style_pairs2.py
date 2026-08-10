#!/usr/bin/env python3
"""Manufacture (audio, coefficient) supervision in the refit real-voice basis.

The translation heads memorised their 145 real pairs (train loss 0.0000), so
this revives the Phase-1 trick at the point it is now needed: sample
coefficients in the *refit* basis — the one spanned by inverted real speakers,
not the preset simplex — synthesise a sentence, and keep (wav, coeffs). The
labels are exact because we chose them; the wavs then go through whichever
feature extractor is being tested, and a 2048-d input finally has thousands
of examples instead of a hundred.

The basis here must be bit-identical to train_translation's: same pairs npz,
same basis pairs, same k, same replication formula.

    gen_style_pairs2.py --pairs-npz all_pairs.npz --count 2000 --out-dir mfg_wavs
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch

WORK = os.environ.get("WORK", os.path.expanduser("~/supertonic-experiment"))
sys.path.insert(0, os.path.join(WORK, "supertonic-voice-cloning/src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_encoder import StyleBasis  # noqa: E402

TEXTS = [
    "The rainbow is a division of white light into many beautiful colors.",
    "He described the weather, the passing ships, and the strange green light.",
    "A cup of coffee on the desk had long since gone cold.",
    "They followed the narrow path until the trees gave way to open fields.",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs-npz", required=True)
    ap.add_argument("--basis-pairs", default=os.path.join(WORK, "clone_out/pairs_p1.npz"))
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--sigma", type=float, default=0.8, help="sample scale in coeff units")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device(args.device)
    os.environ.setdefault("NOCOMPILE", "1")

    from pipeline import DiffSynth

    a = np.load(args.pairs_npz, allow_pickle=True)
    d = np.load(args.basis_pairs, allow_pickle=True)
    rep = max(1, len(d["ttl"]) // max(len(a["ttl"]), 1))
    basis = StyleBasis(np.concatenate([d["ttl"], np.repeat(a["ttl"], rep, 0)]),
                       np.concatenate([d["dp"], np.repeat(a["dp"], rep, 0)]), args.k, dev)
    synth = DiffSynth(os.path.join(WORK, "assets/onnx"), device=str(dev))
    dp0 = d["dp"][:1].astype(np.float32)
    pt, pl = synth.pad_bounds(TEXTS, "en", dp0, 1.05)
    preps = [synth.prepare(t, "en", pad_text=pt, pad_latent=pl) for t in TEXTS]

    rng = np.random.default_rng(args.seed)
    C, S = [], []
    made = 0
    for i in range(args.count):
        name = f"mfg_{i:05d}"
        path = os.path.join(args.out_dir, name + ".wav")
        c = np.clip(args.sigma * rng.standard_normal(basis.k), -2.5, 2.5).astype(np.float32)
        if not os.path.exists(path):
            with torch.no_grad():
                ttl, dp = basis.decode(torch.tensor(c[None], device=dev))
                w, _, _ = synth.synth(ttl, dp, preps[i % len(preps)])
            w = w.reshape(-1).cpu().numpy()
            if not np.isfinite(w).all() or np.abs(w).max() < 1e-4:
                continue                       # an off-manifold sample that synthesised as silence
            sf.write(path, w, synth.sr)
        C.append(c)
        S.append(name)
        made += 1
        if made % 200 == 0:
            print(f"  {made}/{args.count}", flush=True)

    np.savez_compressed(os.path.join(args.out_dir, "coeffs.npz"),
                        coeff=np.stack(C), spk=np.array(S), k=args.k)
    print(f"{made} manufactured samples in {args.out_dir}")


if __name__ == "__main__":
    main()
