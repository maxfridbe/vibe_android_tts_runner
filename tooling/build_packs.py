#!/usr/bin/env python3
"""Curate inverted pack candidates into a shippable voice pack.

Takes the (style, score) pairs invert_corpus.py produced from the Qwen rolls,
keeps the ones whose held-out cosine clears the bar, and writes a pack
directory the app can import: one style JSON per voice with its description
and score in the metadata, a preview wav per voice, and an index.

    build_packs.py --inversions packs/inversions --rolls packs/rolls \\
                   --out packs/voicepack_v1 --top 24
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clone_library import synth, PREVIEW_TEXT  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inversions", required=True)
    ap.add_argument("--rolls", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-cos", type=float, default=0.72)
    ap.add_argument("--top", type=int, default=24)
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()

    pairs_dir = os.path.join(args.inversions, "pairs")
    cands = []
    for f in sorted(os.listdir(pairs_dir)):
        if not f.endswith(".npz"):
            continue
        d = np.load(os.path.join(pairs_dir, f), allow_pickle=True)
        name = str(d["spk"])
        meta_path = os.path.join(args.rolls, name + ".meta.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        cands.append({"name": name, "cos": float(d["cos"]),
                      "ttl": d["ttl"], "dp": d["dp"],
                      "desc": meta.get("desc", ""), "seed": meta.get("seed")})
    cands.sort(key=lambda c: -c["cos"])
    keep = [c for c in cands if c["cos"] >= args.min_cos][:args.top]
    print(f"{len(cands)} candidates, {len(keep)} kept (cos >= {args.min_cos}, top {args.top})")

    os.makedirs(args.out, exist_ok=True)
    prev_dir = os.path.join(args.out, "previews")
    os.makedirs(prev_dir, exist_ok=True)
    index = []
    for i, c in enumerate(keep):
        label = f"Designed {i + 1:02d}"
        style_path = os.path.join(args.out, f"{c['name']}.json")
        json.dump({
            "style_ttl": {"dims": [1, *c["ttl"].shape], "data": c["ttl"].reshape(-1).tolist()},
            "style_dp": {"dims": [1, *c["dp"].shape], "data": c["dp"].reshape(-1).tolist()},
            "metadata": {"name": label, "description": c["desc"],
                         "held_out_cos": round(c["cos"], 4),
                         "source": "qwen voicedesign -> inversion", "seed": c["seed"]},
        }, open(style_path, "w"))
        synth(style_path, PREVIEW_TEXT, os.path.join(prev_dir, c["name"] + ".wav"), args.gpu)
        index.append({"file": os.path.basename(style_path), "name": label,
                      "description": c["desc"], "cos": round(c["cos"], 4)})
        print(f"  {label}: cos {c['cos']:.3f} — {c['desc']}")
    json.dump(index, open(os.path.join(args.out, "index.json"), "w"), indent=1)
    bundle = shutil.make_archive(args.out, "zip", args.out)
    print(f"pack: {bundle}")


if __name__ == "__main__":
    main()
