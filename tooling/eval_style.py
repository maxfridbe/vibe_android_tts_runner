#!/usr/bin/env python3
"""Rank Supertonic voice styles by how well they match a reference speaker on
text they were never optimised against.

Why this exists: the inversion loop reports a speaker cosine measured on its
own probe batch — the very text it is fitting. That number can rise while the
voice generalises worse, and it is not comparable between runs. Scoring a few
held-out sentences instead gives a usable selection criterion, and the spread
across sentences is itself informative: a style that swings by 0.05 between
sentences is less trustworthy than one that holds steady.

    eval_style.py reference.wav style_a.json style_b.json ...

Prints mean ± std of the ECAPA cosine per style, best first.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf
import torch

HELD_OUT = [
    "The rainbow is a division of white light into many beautiful colors.",
    "Nobody has volunteered for the posting since, and the lamp is lit by machinery now.",
    "He described the weather, the passing ships, and the strange green light.",
    "Please leave a message after the tone and somebody will return your call.",
]


def load16k(path: str, device: str) -> torch.Tensor:
    """Mono 16 kHz for the speaker encoder; anything non-WAV goes via ffmpeg."""
    if not path.lower().endswith(".wav"):
        tmp = tempfile.mktemp(suffix=".wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                        "-ar", "16000", "-ac", "1", tmp], check=True)
        path = tmp
    w, sr = sf.read(path, dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    if sr != 16000:
        n = int(len(w) * 16000 / sr)
        w = np.interp(np.linspace(0, len(w) - 1, n), np.arange(len(w)), w).astype("float32")
    return torch.from_numpy(w)[None].to(device)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference")
    ap.add_argument("styles", nargs="+")
    ap.add_argument("--work", default=os.path.expanduser("~/supertonic-experiment"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    src = os.path.join(args.work, "supertonic-voice-cloning/src")
    onnx = os.path.join(args.work, "assets/onnx")
    python = os.path.join(args.work, "venv/bin/python")
    if not os.path.isdir(src):
        print(f"cloning repo not found under {args.work}", file=sys.stderr)
        return 1

    os.environ.setdefault("SB_CACHE", os.path.expanduser("~/.cache/speechbrain/ecapa"))
    from speechbrain.inference.speaker import EncoderClassifier
    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.environ["SB_CACHE"],
        run_opts={"device": args.device})

    def emb(p):
        return torch.nn.functional.normalize(
            enc.encode_batch(load16k(p, args.device)).squeeze(), dim=-1)

    ref = emb(args.reference)
    rows = []
    for style in args.styles:
        scores = []
        for text in HELD_OUT:
            out = tempfile.mktemp(suffix=".wav")
            subprocess.run([python, "synth_onnx.py", "--onnx-dir", onnx, "--voice", style,
                            "--text", text, "--out", out],
                           cwd=src, capture_output=True, check=True)
            scores.append(float(torch.dot(ref, emb(out))))
        a = np.array(scores)
        rows.append((a.mean(), a.std(), style, a))

    rows.sort(key=lambda r: -r[0])
    print(f"reference {os.path.basename(args.reference)} | {len(HELD_OUT)} held-out sentences\n")
    for mean, std, style, a in rows:
        print(f"  {mean:.4f} +/- {std:.4f}   {os.path.basename(style)}   {np.round(a, 3).tolist()}")
    print(f"\nbest: {rows[0][2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
