#!/usr/bin/env python3
"""Extract Qwen3-TTS's speaker conditioning for a set of reference wavs.

The style encoder has so far eaten ECAPA embeddings — a *verification*
representation, trained to keep only what separates speakers. Qwen's clone
prompt is a *generation* representation: whatever create_voice_clone_prompt
captures is, demonstrably, enough for Qwen to speak in that voice. This dumps
that representation for every reference so a translation head can be trained
from it to Supertonic style coefficients, against an ECAPA twin as baseline.

Runs in the audiobook-maker venv (the model needs its stack):

    ~/audiobook-maker/venv/bin/python extract_qwen_features.py \\
        --wavs refs_dir [more_dirs...] --out qwen_feats.npz

The prompt object's structure is not documented, so extraction is
introspective: every float array found in it is kept — vectors as-is,
sequences mean-pooled over time — and concatenated into one feature vector.
The first reference's raw structure is written alongside as JSON for
inspection.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_speaker_bank import ABM, MODELS  # noqa: E402


def harvest(obj, path="", out=None, depth=0):
    """Recursively collect float arrays from an arbitrary prompt object."""
    if out is None:
        out = {}
    if depth > 6 or obj is None:
        return out
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            obj = obj.detach().float().cpu().numpy()   # bf16 has no numpy dtype
    except ImportError:
        pass
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind == "f" and obj.size > 1:
            out[path or "root"] = obj.astype(np.float32)
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            harvest(v, f"{path}.{k}" if path else str(k), out, depth + 1)
        return out
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            harvest(v, f"{path}[{i}]", out, depth + 1)
        return out
    if hasattr(obj, "__dict__"):
        for k, v in vars(obj).items():
            if not k.startswith("_"):
                harvest(v, f"{path}.{k}" if path else k, out, depth + 1)
    return out


def pool(arrays):
    """One flat feature vector: 1-D arrays as-is, higher ranks mean-pooled
    over every axis but the last (the time axes of a token sequence)."""
    parts = []
    for name in sorted(arrays):
        a = arrays[name]
        while a.ndim > 1:
            a = a.mean(0)
        parts.append(a.reshape(-1))
    return np.concatenate(parts) if parts else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wavs", nargs="+", required=True, help="dirs of wavs, or wav files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.path.join(MODELS, "Qwen3-TTS-12Hz-1.7B-Base"))
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(ABM, "python"))
    from abm_worker.engine import Engine
    engine = Engine()
    engine.load(args.model)

    wavs = []
    for w in args.wavs:
        if os.path.isdir(w):
            wavs += [os.path.join(w, f) for f in sorted(os.listdir(w))
                     if f.endswith((".wav", ".flac"))]
        elif w.endswith((".wav", ".flac")):
            wavs.append(w)
    print(f"{len(wavs)} references")

    feats, names, mode_used = [], [], None
    peeked = False
    for i, w in enumerate(wavs):
        prompt = None
        for mode, kwargs in (("full", dict(x_vector_only=False)),
                             ("xvec", dict(x_vector_only=True))):
            try:
                prompt = engine.create_clone_prompt(w, None, **kwargs)
                mode_used = mode_used or mode
                break
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
        if prompt is None:
            print(f"  {os.path.basename(w)}: prompt failed ({err})", flush=True)
            continue
        arrays = harvest(prompt)
        if not peeked:
            json.dump({k: list(v.shape) for k, v in arrays.items()},
                      open(args.out + ".structure.json", "w"), indent=1)
            print(f"prompt structure ({mode_used}): "
                  + ", ".join(f"{k}{list(v.shape)}" for k, v in arrays.items()), flush=True)
            peeked = True
        f = pool(arrays)
        if f is None:
            print(f"  {os.path.basename(w)}: no float arrays in prompt", flush=True)
            continue
        feats.append(f)
        names.append(os.path.splitext(os.path.basename(w))[0])
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(wavs)}", flush=True)

    dims = {len(f) for f in feats}
    if len(dims) > 1:      # variable-length refs can change pooled dims; pad out
        d = max(dims)
        feats = [np.pad(f, (0, d - len(f))) for f in feats]
    F = np.stack(feats)
    np.savez_compressed(args.out, feat=F, spk=np.array(names))
    print(f"wrote {args.out}: {F.shape[0]} refs x {F.shape[1]} dims ({mode_used} prompt)")


if __name__ == "__main__":
    main()
