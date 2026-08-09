#!/usr/bin/env python3
"""Phase 1 of on-device cloning (see docs/on-device-cloning.md): manufacture
(speaker embedding -> style) supervision.

Supertonic ships no speaker encoder, but the opposite direction is a cheap
forward pass, so training data can be generated rather than collected: sample
a style, synthesise a sentence, embed it, and store the pair. The sampler
mixes the published styles on the simplex and adds per-row spherical noise —
the same parametrisation the inversion optimiser uses, so the samples respect
the geometry the model was trained on.

    gen_style_pairs.py --count 2000 --out pairs.npz

Output: style_ttl/style_dp tensors, ECAPA embeddings, and the mixing weights,
ready for a PCA fit and a regressor.
"""
import argparse
import json
import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf
import torch

SENTENCES = [
    "The rainbow is a division of white light into many beautiful colors.",
    "Please leave a message after the tone and somebody will return your call.",
]


def load_styles(style_dir):
    ttl, dp, names = [], [], []
    for f in sorted(os.listdir(style_dir)):
        if not f.endswith(".json"):
            continue
        o = json.load(open(os.path.join(style_dir, f)))
        d1 = o["style_ttl"]["dims"]
        d2 = o["style_dp"]["dims"]
        ttl.append(np.array(o["style_ttl"]["data"], dtype=np.float32).reshape(d1[1], d1[2]))
        dp.append(np.array(o["style_dp"]["data"], dtype=np.float32).reshape(d2[1], d2[2]))
        names.append(f[:-5])
    return np.stack(ttl), np.stack(dp), names


def sample_style(ttl_bank, dp_bank, rng, noise=0.08):
    """Dirichlet mix of the presets, then a small rotation of each row.

    Rows are normalised per-row rather than globally because that is how the
    inversion parametrises its deltas; scaling whole tensors instead tends to
    produce styles that synthesise as noise.
    """
    w = rng.dirichlet(np.ones(len(ttl_bank)) * 0.6)
    ttl = np.tensordot(w, ttl_bank, axes=1)
    dp = np.tensordot(w, dp_bank, axes=1)
    for arr in (ttl, dp):
        n = rng.normal(size=arr.shape).astype(np.float32)
        row_norm = np.linalg.norm(arr, axis=-1, keepdims=True) + 1e-8
        n *= row_norm / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8)
        arr += noise * n
    return ttl.astype(np.float32), dp.astype(np.float32), w.astype(np.float32)


def write_style(path, ttl, dp):
    json.dump({
        "style_ttl": {"dims": [1, *ttl.shape], "data": ttl.reshape(-1).tolist()},
        "style_dp": {"dims": [1, *dp.shape], "data": dp.reshape(-1).tolist()},
        "metadata": {"source": "gen_style_pairs"},
    }, open(path, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--out", default="style_pairs.npz")
    ap.add_argument("--work", default=os.path.expanduser("~/supertonic-experiment"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = os.path.join(args.work, "supertonic-voice-cloning/src")
    onnx = os.path.join(args.work, "assets/onnx")
    styles = os.path.join(args.work, "assets/voice_styles")
    python = os.path.join(args.work, "venv/bin/python")

    os.environ.setdefault("SB_CACHE", os.path.expanduser("~/.cache/speechbrain/ecapa"))
    from speechbrain.inference.speaker import EncoderClassifier
    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.environ["SB_CACHE"], run_opts={"device": args.device})

    ttl_bank, dp_bank, names = load_styles(styles)
    print(f"{len(names)} preset styles: {', '.join(names)}")
    rng = np.random.default_rng(args.seed)

    E, T, D, W = [], [], [], []
    tmpdir = tempfile.mkdtemp()
    for i in range(args.count):
        ttl, dp, w = sample_style(ttl_bank, dp_bank, rng)
        sp = os.path.join(tmpdir, "s.json")
        write_style(sp, ttl, dp)
        embs = []
        for text in SENTENCES:
            wav = os.path.join(tmpdir, "o.wav")
            r = subprocess.run([python, "synth_onnx.py", "--onnx-dir", onnx, "--voice", sp,
                                "--text", text, "--out", wav], cwd=src, capture_output=True)
            if r.returncode != 0:
                break
            a, sr = sf.read(wav, dtype="float32")
            if a.ndim > 1:
                a = a.mean(1)
            n = int(len(a) * 16000 / sr)
            a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype("float32")
            with torch.no_grad():
                e = enc.encode_batch(torch.from_numpy(a)[None].to(args.device)).squeeze()
                embs.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        if len(embs) != len(SENTENCES):
            continue                      # a style that fails to synthesise is not training data
        E.append(np.mean(embs, 0)); T.append(ttl); D.append(dp); W.append(w)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{args.count} pairs", flush=True)

    np.savez_compressed(args.out, emb=np.stack(E), ttl=np.stack(T),
                        dp=np.stack(D), weights=np.stack(W), presets=names)
    print(f"wrote {args.out}: {len(E)} pairs")


if __name__ == "__main__":
    main()
