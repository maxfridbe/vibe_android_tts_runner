"""Is the style->embedding map smooth enough to invert with a regressor?

Two checks on generated pairs:
  1. PCA on the styles: how many components explain the variation, i.e. how
     small the regression target can be made.
  2. Correlation between distance in style space and distance in speaker
     space. A regressor can only work if nearby styles sound like nearby
     speakers; if that correlation is weak, the map is too chaotic to learn
     from this kind of data.
"""
import sys
import numpy as np

d = np.load(sys.argv[1], allow_pickle=True)
emb, ttl, dp = d["emb"], d["ttl"], d["dp"]
n = len(emb)
X = np.concatenate([ttl.reshape(n, -1), dp.reshape(n, -1)], 1)
print(f"{n} pairs | style dim {X.shape[1]} | embedding dim {emb.shape[1]}")

Xc = X - X.mean(0)
s = np.linalg.svd(Xc, compute_uv=False)
var = s**2 / (s**2).sum()
cum = np.cumsum(var)
for k in (2, 5, 10, 20, min(50, n - 1)):
    if k < len(cum):
        print(f"  PCA {k:3d} comps -> {cum[k-1]*100:5.1f}% of style variance")

# pairwise distances, cosine in speaker space vs euclidean in style space
ds, de = [], []
for i in range(n):
    for j in range(i + 1, n):
        ds.append(np.linalg.norm(X[i] - X[j]))
        de.append(1.0 - float(emb[i] @ emb[j]))
ds, de = np.array(ds), np.array(de)
r = np.corrcoef(ds, de)[0, 1]
rank = np.corrcoef(np.argsort(np.argsort(ds)), np.argsort(np.argsort(de)))[0, 1]
print(f"\nstyle distance vs speaker distance: pearson {r:.3f}, spearman {rank:.3f}")
print(f"speaker cosine spread across samples: {1-de.mean():.3f} mean, "
      f"{1-de.max():.3f} min, {1-de.min():.3f} max")
