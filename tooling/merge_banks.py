#!/usr/bin/env python3
"""Concatenate speaker banks produced by parallel rollers into one target set."""
import sys
import numpy as np

out = sys.argv[1]
parts = [np.load(p, allow_pickle=True) for p in sys.argv[2:]]
emb = np.concatenate([p["emb"] for p in parts])
desc = np.concatenate([p["desc"] for p in parts]) if all("desc" in p for p in parts) else None
# duplicate rolls are possible across seeds; identical embeddings add nothing
_, keep = np.unique(np.round(emb, 5), axis=0, return_index=True)
emb = emb[np.sort(keep)]
off = (emb @ emb.T)[~np.eye(len(emb), dtype=bool)]
print(f"{sum(len(p['emb']) for p in parts)} rolled -> {len(emb)} unique speakers, "
      f"mean pairwise cos {off.mean():.3f}")
np.savez_compressed(out, emb=emb, **({"desc": desc[np.sort(keep)]} if desc is not None else {}))
print(f"wrote {out}")
