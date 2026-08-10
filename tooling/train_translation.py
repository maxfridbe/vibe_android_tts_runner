#!/usr/bin/env python3
"""Train a translation head: speaker features -> Supertonic style coefficients.

The controlled experiment behind "use Qwen's own encoder": identical data,
identical basis, identical head — the only variable is the input
representation. Run once with Qwen clone-prompt features, once with ECAPA
embeddings, and the held-out gap answers whether a generation-trained
representation carries more of a voice than a verification one.

Supervision is direct: every reference here has an inverted style whose
held-out cosine is known, so the head regresses basis coefficients with the
labels' quality as sample weights. Small data (a few hundred pairs), small
head, minutes to train. Judged the only way that matters — synthesise with
the predicted style, embed, compare to the reference — via eval-compatible
JSON.

    train_translation.py --pairs-npz aux_pairs.npz --features qwen_feats.npz \\
                         --basis-pairs pairs_p1.npz --out trans_qwen.pt
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WORK = os.environ.get("WORK", os.path.expanduser("~/supertonic-experiment"))
sys.path.insert(0, os.path.join(WORK, "supertonic-voice-cloning/src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_encoder import StyleBasis  # noqa: E402


class Head(nn.Module):
    def __init__(self, in_dim, k, hidden=512, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, k),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(self.norm(x))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs-npz", required=True,
                    help="aux pairs npz: spk, ttl, dp, cos, emb (invert_corpus aggregate)")
    ap.add_argument("--features", default=None,
                    help="npz with feat+spk from extract_qwen_features; omit to use "
                         "the pairs' own ECAPA embeddings (the baseline twin)")
    ap.add_argument("--features2", default=None,
                    help="second feature npz concatenated onto the first — the "
                         "ensemble: qwen features carried the character voices, "
                         "ECAPA the in-corpus ones, and a head fed both can "
                         "learn which to trust where")
    ap.add_argument("--basis-pairs", default=os.path.join(WORK, "clone_out/pairs_p1.npz"))
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--pca", type=int, default=0,
                    help="project features to this many dims first; a 2048-d input "
                         "on 145 pairs memorises instead of learning")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--feat-noise", type=float, default=0.0,
                    help="gaussian feature noise during training, in units of "
                         "per-dim std — cheap augmentation for tiny pair counts")
    ap.add_argument("--extra-pairs", default=None,
                    help="npz with (feat, coeff) manufactured pairs "
                         "(gen_style_pairs2 + extract) mixed into training")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.manual_seed(0)

    a = np.load(args.pairs_npz, allow_pickle=True)
    spk = np.asarray(a["spk"]).astype(str)
    d = np.load(args.basis_pairs, allow_pickle=True)
    rep = max(1, len(d["ttl"]) // max(len(a["ttl"]), 1))
    basis = StyleBasis(np.concatenate([d["ttl"], np.repeat(a["ttl"], rep, 0)]),
                       np.concatenate([d["dp"], np.repeat(a["dp"], rep, 0)]), args.k, dev)

    def featmap(path):
        f = np.load(path, allow_pickle=True)
        return {s: v for s, v in zip(np.asarray(f["spk"]).astype(str),
                                     np.asarray(f["feat"], dtype=np.float32))}

    if args.features:
        fmap = featmap(args.features)
        f2 = featmap(args.features2) if args.features2 else None
        keep = np.array([s in fmap and (f2 is None or s in f2) for s in spk])
        X = np.stack([np.concatenate([fmap[s]] + ([f2[s]] if f2 else []))
                      for s in spk[keep]])
        print(f"features: {X.shape[1]}-d"
              + (" ensemble" if f2 else " qwen prompt")
              + f", {keep.sum()}/{len(spk)} refs matched")
    else:
        keep = np.ones(len(spk), bool)
        X = np.asarray(a["emb"], dtype=np.float32)
        print(f"features: {X.shape[1]}-d ECAPA (baseline)")

    with torch.no_grad():
        Y = basis.encode(torch.tensor(np.asarray(a["ttl"], dtype=np.float32)[keep], device=dev),
                         torch.tensor(np.asarray(a["dp"], dtype=np.float32)[keep], device=dev))
    W = torch.tensor(np.asarray(a["cos"], dtype=np.float32)[keep], device=dev)  # label quality
    names = spk[keep]

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    n_val = max(8, int(len(X) * args.val_frac))
    vi, ti = perm[:n_val], perm[n_val:]
    print(f"{len(ti)} train / {len(vi)} val refs")

    pca_mean = pca_comp = None
    if args.pca and args.pca < X.shape[1]:
        pca_mean = X[ti].mean(0)
        _, _, Vt = np.linalg.svd(X[ti] - pca_mean, full_matrices=False)
        pca_comp = Vt[:args.pca]
        X = (X - pca_mean) @ pca_comp.T
        print(f"features PCA-projected to {X.shape[1]} dims (fit on train rows)")
    X = torch.tensor(X.astype(np.float32), device=dev)

    # manufactured pairs join training only; val stays real voices
    Xe = Ye = None
    if args.extra_pairs:
        e = np.load(args.extra_pairs, allow_pickle=True)
        fe = np.asarray(e["feat"], dtype=np.float32)
        if pca_comp is not None:
            fe = (fe - pca_mean) @ pca_comp.T
        Xe = torch.tensor(fe.astype(np.float32), device=dev)
        Ye = torch.tensor(np.asarray(e["coeff"], dtype=np.float32), device=dev)
        print(f"+{len(Xe)} manufactured pairs in the training pool")

    XT = torch.cat([X[ti], Xe]) if Xe is not None else X[ti]
    YT = torch.cat([Y[ti], Ye]) if Xe is not None else Y[ti]
    # real labels carry their measured quality; manufactured ones are exact in
    # coeff space but synthetic in audio, so they weigh half
    WT = torch.cat([W[ti], torch.full((len(Xe),), 0.5, device=dev)]) if Xe is not None else W[ti]
    feat_std = XT.std(0).clamp(min=1e-4)

    head = Head(X.shape[1], basis.k, dropout=args.dropout).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    best = (1e9, None)
    for step in range(1, args.steps + 1):
        j = torch.tensor(rng.integers(0, len(XT), size=min(64, len(XT))), device=dev)
        xb = XT[j]
        if args.feat_noise > 0:
            xb = xb + args.feat_noise * feat_std * torch.randn_like(xb)
        loss = (WT[j] * ((head(xb) - YT[j]) ** 2).mean(1)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 200 == 0:
            head.eval()
            with torch.no_grad():
                v = float(((head(X[vi]) - Y[vi]) ** 2).mean())
            head.train()
            if v < best[0]:
                best = (v, {k: t.clone() for k, t in head.state_dict().items()})
            print(f"  step {step}  train {float(loss):.4f}  val mse {v:.4f}", flush=True)

    head.load_state_dict(best[1])
    head.eval()
    torch.save({"model": head.state_dict(), "k": basis.k, "in_dim": X.shape[1],
                "dropout": args.dropout,
                "pca_mean": None if pca_mean is None else pca_mean.astype(np.float32),
                "pca_comp": None if pca_comp is None else pca_comp.astype(np.float32),
                "basis": basis.basis.cpu(), "mean": basis.mean.cpu(),
                "scale": basis.scale.cpu(), "ttl_shape": basis.ttl_shape,
                "dp_shape": basis.dp_shape, "val_mse": best[0],
                "features": args.features or "ecapa",
                "val_spk": names[vi].tolist()}, args.out)
    print(f"wrote {args.out}: val mse {best[0]:.4f} over {len(vi)} held-out refs")
    # styles for the held-out refs, ready for eval_style.py scoring
    outdir = os.path.splitext(args.out)[0] + "_val_styles"
    os.makedirs(outdir, exist_ok=True)
    with torch.no_grad():
        ttl, dp = basis.decode(head(X[vi]))
    for n, t, p in zip(names[vi], ttl, dp):
        json.dump({"style_ttl": {"dims": [1, *t.shape], "data": t.reshape(-1).tolist()},
                   "style_dp": {"dims": [1, *p.shape], "data": p.reshape(-1).tolist()},
                   "metadata": {"source": f"translation head ({args.features or 'ecapa'})"}},
                  open(os.path.join(outdir, f"{n}.json"), "w"))
    print(f"held-out styles in {outdir}")


if __name__ == "__main__":
    main()
