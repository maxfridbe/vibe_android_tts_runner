#!/usr/bin/env python3
"""Measure how much of a desktop inversion the PCA basis can even express.

The encoder can only emit styles inside the span of its PCA basis, and that
basis is fitted on samples from the preset simplex — while inversion is known
to walk *out* of the simplex to reach a real speaker. If projecting a good
inverted style onto the basis destroys its speaker similarity, no amount of
encoder training or target data will ever close the gap: the ceiling is the
basis, and the fix is refitting it on inverted real-speaker styles.

The library of finished inversions (clone_library.py) supplies ground truth:
voices with a reference recording, a best style, and a held-out cosine around
0.8. For each one this synthesises the held-out sentences three ways —

  orig    the inverted style as-is (should reproduce its library score)
  proj    the same style projected through the pairs-only basis
  proj+   projected through a basis refitted with the *other* library voices'
          styles folded in (leave-one-out, so the target style never helps
          span itself) — a preview of the Phase-2 refit

— and reports the ECAPA cosine of each against the reference recording.

    basis_ceiling.py --pairs clone_out/pairs_p1.npz --out ceiling.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_encoder import StyleBasis            # noqa: E402
from eval_style import HELD_OUT, load16k        # noqa: E402
from clone_library import synth, slugify        # noqa: E402

WORK = os.environ.get("WORK", os.path.expanduser("~/supertonic-experiment"))


def load_style(path):
    o = json.load(open(path))
    d1, d2 = o["style_ttl"]["dims"], o["style_dp"]["dims"]
    ttl = np.array(o["style_ttl"]["data"], dtype=np.float32).reshape(d1[1], d1[2])
    dp = np.array(o["style_dp"]["data"], dtype=np.float32).reshape(d2[1], d2[2])
    return ttl, dp


def write_style(path, ttl, dp, note):
    json.dump({
        "style_ttl": {"dims": [1, *ttl.shape], "data": ttl.reshape(-1).tolist()},
        "style_dp": {"dims": [1, *dp.shape], "data": dp.reshape(-1).tolist()},
        "metadata": {"source": note},
    }, open(path, "w"))


def project(basis, ttl, dp, dev):
    t = torch.tensor(ttl[None], device=dev)
    d = torch.tensor(dp[None], device=dev)
    pt, pd = basis.decode(basis.encode(t, d))
    return pt[0].cpu().numpy(), pd[0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default=os.path.join(WORK, "clone_out/pairs_p1.npz"))
    ap.add_argument("--library", default=os.path.join(WORK, "clone_out/library"))
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--out", default=os.path.join(WORK, "clone_out/overnight/basis_ceiling.json"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    scratch = os.path.join(os.path.dirname(args.out), "ceiling_work")
    os.makedirs(scratch, exist_ok=True)
    dev = torch.device(args.device)

    progress = json.load(open(os.path.join(args.library, "progress.json")))
    voices = []
    for name, s in progress.items():
        style = os.path.join(args.library, "styles", slugify(name) + ".json")
        if s.get("stage") == "done" and os.path.exists(style) and os.path.exists(s.get("ref", "")):
            voices.append({"name": name, "ref": s["ref"], "style": style,
                           "lib_cos": s.get("cos")})
    print(f"{len(voices)} finished library voices")
    for v in voices:
        v["ttl"], v["dp"] = load_style(v["style"])

    d = np.load(args.pairs, allow_pickle=True)
    basis = StyleBasis(d["ttl"], d["dp"], args.k, dev)

    os.environ.setdefault("SB_CACHE", os.path.expanduser("~/.cache/speechbrain/ecapa"))
    from speechbrain.inference.speaker import EncoderClassifier
    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.environ["SB_CACHE"], run_opts={"device": args.device})

    def emb(p):
        return torch.nn.functional.normalize(
            enc.encode_batch(load16k(p, args.device)).squeeze(), dim=-1)

    def score(style_path, tag):
        per = []
        for i, text in enumerate(HELD_OUT):
            wav = os.path.join(scratch, f"{tag}_{i}.wav")
            if not os.path.exists(wav) and not synth(style_path, text, wav, args.gpu):
                continue
            per.append(float(torch.dot(ref_emb, emb(wav))))
        return (float(np.mean(per)), float(np.std(per))) if per else (float("nan"), 0.0)

    results = []
    for v in voices:
        ref_emb = emb(v["ref"])
        slug = slugify(v["name"])

        pt, pd = project(basis, v["ttl"], v["dp"], dev)
        pa = os.path.join(scratch, slug + ".projA.json")
        write_style(pa, pt, pd, "pairs-basis projection")

        # leave-one-out refit: the other voices' styles, replicated so they
        # carry ~1/3 of the fit's mass against 1200 simplex samples
        others = [o for o in voices if o["name"] != v["name"]]
        rep = max(1, len(d["ttl"]) // (2 * max(len(others), 1)))
        ttl_fit = np.concatenate([d["ttl"], np.repeat(np.stack([o["ttl"] for o in others]), rep, 0)])
        dp_fit = np.concatenate([d["dp"], np.repeat(np.stack([o["dp"] for o in others]), rep, 0)])
        loo = StyleBasis(ttl_fit, dp_fit, args.k, dev)
        pt, pd = project(loo, v["ttl"], v["dp"], dev)
        pb = os.path.join(scratch, slug + ".projB.json")
        write_style(pb, pt, pd, "leave-one-out refit projection")

        row = {"name": v["name"], "lib_cos": v["lib_cos"]}
        for tag, path in (("orig", v["style"]), ("proj", pa), ("proj_refit", pb)):
            m, s = score(path, f"{slug}.{tag}")
            row[tag], row[tag + "_std"] = round(m, 4), round(s, 4)
        results.append(row)
        print(f"  {v['name']:<30} orig {row['orig']:.3f}  proj {row['proj']:.3f}  "
              f"proj_refit {row['proj_refit']:.3f}", flush=True)

    ok = [r for r in results if not np.isnan(r["orig"])]
    summary = {"k": args.k, "n_voices": len(ok),
               "mean_orig": round(float(np.mean([r["orig"] for r in ok])), 4),
               "mean_proj": round(float(np.mean([r["proj"] for r in ok])), 4),
               "mean_proj_refit": round(float(np.mean([r["proj_refit"] for r in ok])), 4),
               "voices": results}
    json.dump(summary, open(args.out, "w"), indent=1)
    print(f"\nmean over {len(ok)} voices: orig {summary['mean_orig']:.3f} -> "
          f"projected {summary['mean_proj']:.3f} (refit {summary['mean_proj_refit']:.3f})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
