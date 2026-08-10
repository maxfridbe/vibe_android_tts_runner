#!/usr/bin/env python3
"""Concatenate speaker banks produced by parallel rollers into one target set."""
import os
import sys

import numpy as np

out = sys.argv[1]
parts = [(p, np.load(p, allow_pickle=True)) for p in sys.argv[2:]]
emb = np.concatenate([d["emb"] for _, d in parts])
# real-corpus banks (embed_corpus.py) carry a speaker id per row so the trainer
# can hold out whole speakers; rolled banks predate that, and each of their
# rows really is its own speaker, so a unique id per row says exactly that
spk = np.concatenate([np.asarray(d["spk"]).astype(str) if "spk" in d else
                      np.array([f"{os.path.basename(p)}:{i}" for i in range(len(d["emb"]))])
                      for p, d in parts])
desc = np.concatenate([d["desc"] for _, d in parts]) if all("desc" in d for _, d in parts) else None
# duplicate rolls are possible across seeds; identical embeddings add nothing
_, keep = np.unique(np.round(emb, 5), axis=0, return_index=True)
keep = np.sort(keep)
emb, spk = emb[keep], spk[keep]
off = (emb @ emb.T)[~np.eye(len(emb), dtype=bool)]
print(f"{sum(len(d['emb']) for _, d in parts)} rolled -> {len(emb)} unique rows, "
      f"{len(np.unique(spk))} speakers, mean pairwise cos {off.mean():.3f}")
np.savez_compressed(out, emb=emb, spk=spk, **({"desc": desc[keep]} if desc is not None else {}))
print(f"wrote {out}")
