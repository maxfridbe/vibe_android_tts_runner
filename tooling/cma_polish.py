#!/usr/bin/env python3
"""Forward-only voice inversion: what the phone can actually run.

Gradient inversion needs autograd and ~11 GB; a phone has neither. But the
basis work changed the problem: a real voice now lives at some point in a
64-128 coefficient space fitted on inverted real speakers, and searching that
space needs only forward synthesis and an embedding — both of which the app
already runs in ONNX. This prototypes exactly that loop on the desktop so the
numbers (quality reached, evaluations spent) transfer to the phone: one
evaluation here is one short synthesis plus one ECAPA pass, the same
operations at phone speed cost roughly a second each, so 300 evaluations is
five phone-minutes of one-time cloning.

The optimiser is sep-CMA-ES (diagonal covariance) — small, dependency-free,
and directly portable to Kotlin. Starts are free quality: the encoder's
prediction and the nearest inverted-real styles from the shipped bank, best
of them seeding the search.

    cma_polish.py --refs ref1.wav ref2.flac --basis-extra aux_pairs.npz \\
                  --encoder encoder-r5.pt.best --bank aux_pairs.npz
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

WORK = os.environ.get("WORK", os.path.expanduser("~/supertonic-experiment"))
sys.path.insert(0, os.path.join(WORK, "supertonic-voice-cloning/src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_encoder import StyleBasis, Encoder  # noqa: E402
from eval_style import HELD_OUT, load16k       # noqa: E402
from eval_encoder_real import FRESH            # noqa: E402

# Search probes are disjoint from both scoring sets, so the search cannot
# buy its score by overfitting the sentences it is judged on.
SEARCH_PROBES = [
    "A cup of coffee on the desk had long since gone cold.",
    "They followed the narrow path until the trees gave way to open fields.",
]


def sep_cma(f, x0, sigma0, iters, pop, rng, clamp=3.0):
    """Minimise f over R^n with diagonal-covariance CMA-ES.

    Kept deliberately plain — this exact routine is the Kotlin port."""
    n = len(x0)
    mu = pop // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mu_eff = 1.0 / (w ** 2).sum()
    cs = (mu_eff + 2) / (n + mu_eff + 5)
    ds = 1 + cs + 2 * max(0.0, np.sqrt((mu_eff - 1) / (n + 1)) - 1)
    cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
    c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
    cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
    chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

    m, sigma = x0.copy(), sigma0
    C = np.ones(n)
    ps = np.zeros(n)
    pc = np.zeros(n)
    best_x, best_f = x0.copy(), f(x0)
    evals = 1
    for g in range(iters):
        z = rng.standard_normal((pop, n))
        x = np.clip(m + sigma * np.sqrt(C) * z, -clamp, clamp)
        scores = np.array([f(xi) for xi in x])
        evals += pop
        order = np.argsort(scores)
        if scores[order[0]] < best_f:
            best_f, best_x = scores[order[0]], x[order[0]].copy()
        zsel = z[order[:mu]]
        xsel = x[order[:mu]]
        zmean = (w[:, None] * zsel).sum(0)
        m = (w[:, None] * xsel).sum(0)
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * zmean
        sigma *= np.exp((cs / ds) * (np.linalg.norm(ps) / chi_n - 1))
        hs = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (g + 1))) < (1.4 + 2 / (n + 1)) * chi_n)
        pc = (1 - cc) * pc + hs * np.sqrt(cc * (2 - cc) * mu_eff) * np.sqrt(C) * zmean
        C = (1 - c1 - cmu) * C + c1 * pc ** 2 + cmu * (w[:, None] * (np.sqrt(C) * zsel) ** 2).sum(0)
        C = np.maximum(C, 1e-8)
    return best_x, best_f, evals


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refs", nargs="+", required=True)
    ap.add_argument("--pairs", default=os.path.join(WORK, "clone_out/pairs_p1.npz"))
    ap.add_argument("--basis-extra", required=True, help="inverted styles that refit the basis")
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--encoder", default=None, help=".pt.best checkpoint for the starting guess")
    ap.add_argument("--bank", default=None, help="npz with (emb, ttl, dp) for retrieval starts")
    ap.add_argument("--start-styles", default=None,
                    help="dir of <refname>.json styles (a translation head's "
                         "predictions) used as additional starting points")
    ap.add_argument("--iters", type=int, default=30, help="CMA generations")
    ap.add_argument("--pop", type=int, default=10)
    ap.add_argument("--sigma0", type=float, default=0.35)
    ap.add_argument("--l2", type=float, default=0.02, help="prior pulling coeffs to the manifold")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=os.path.join(WORK, "clone_out/overnight/cma_eval.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dev = torch.device(args.device)

    os.environ.setdefault("NOCOMPILE", "1")
    os.environ.setdefault("SB_CACHE", os.path.expanduser("~/.cache/speechbrain/ecapa"))
    from pipeline import DiffSynth
    from speechbrain.inference.speaker import EncoderClassifier

    d = np.load(args.pairs, allow_pickle=True)
    x = np.load(args.basis_extra, allow_pickle=True)
    rep = max(1, len(d["ttl"]) // max(len(x["ttl"]), 1))
    basis = StyleBasis(np.concatenate([d["ttl"], np.repeat(x["ttl"], rep, 0)]),
                       np.concatenate([d["dp"], np.repeat(x["dp"], rep, 0)]), args.k, dev)

    synth = DiffSynth(os.path.join(WORK, "assets/onnx"), device=str(dev))
    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.environ["SB_CACHE"], run_opts={"device": args.device})
    import torchaudio
    resample = torchaudio.transforms.Resample(synth.sr, 16000).to(dev)

    def embed_wav(w):
        e = enc.encode_batch(resample(w)).squeeze()
        return torch.nn.functional.normalize(e, dim=-1)

    def embed_file(p):
        return torch.nn.functional.normalize(
            enc.encode_batch(load16k(p, args.device)).squeeze(), dim=-1)

    dp0 = d["dp"][:1].astype(np.float32)
    pt, pl = synth.pad_bounds(SEARCH_PROBES + HELD_OUT + FRESH, "en", dp0, 1.05)
    preps = {t: synth.prepare(t, "en", pad_text=pt, pad_latent=pl)
             for t in SEARCH_PROBES + HELD_OUT + FRESH}

    def render_cos(coeffs, texts, ref_emb):
        with torch.no_grad():
            ttl, dp = basis.decode(torch.tensor(coeffs[None], dtype=torch.float32, device=dev))
            per = []
            for t in texts:
                w, _, _ = synth.synth(ttl, dp, preps[t])
                if w.dim() == 1:
                    w = w.unsqueeze(0)
                per.append(float(torch.dot(ref_emb, embed_wav(w))))
        return per

    head = None
    if args.encoder:
        ck = torch.load(args.encoder, map_location="cpu", weights_only=False)
        head = Encoder(192, int(ck["k"]))
        head.load_state_dict(ck["model"])
        head.eval()
        hb = StyleBasis  # the checkpoint's own basis differs from ours; decode
        ck_basis = {"basis": ck["basis"], "mean": ck["mean"], "scale": ck["scale"]}

    bank_emb = bank_coeffs = None
    if args.bank:
        b = np.load(args.bank, allow_pickle=True)
        bank_emb = torch.tensor(np.asarray(b["emb"], dtype=np.float32), device=dev)
        with torch.no_grad():
            bank_coeffs = basis.encode(
                torch.tensor(np.asarray(b["ttl"], dtype=np.float32), device=dev),
                torch.tensor(np.asarray(b["dp"], dtype=np.float32), device=dev)).cpu().numpy()

    rng = np.random.default_rng(0)
    results = []
    for ref in args.refs:
        ref_emb = embed_file(ref)
        t0 = time.time()

        starts = [np.zeros(basis.k, dtype=np.float64)]           # the basis mean
        kinds = ["mean"]
        if head is not None:
            with torch.no_grad():
                c = head(ref_emb.cpu()[None]).numpy()[0]
            # project the checkpoint's style through OUR basis for a fair seed
            flat = ck_basis["mean"].numpy() + (c * ck_basis["scale"].numpy()) @ ck_basis["basis"].numpy()
            split = int(np.prod(ck["ttl_shape"]))
            ttl = flat[:split].reshape(1, *[int(v) for v in ck["ttl_shape"]])
            dp = flat[split:].reshape(1, *[int(v) for v in ck["dp_shape"]])
            with torch.no_grad():
                starts.append(basis.encode(
                    torch.tensor(ttl, dtype=torch.float32, device=dev),
                    torch.tensor(dp, dtype=torch.float32, device=dev)).cpu().numpy()[0])
            kinds.append("encoder")
        if bank_emb is not None:
            near = torch.topk(bank_emb @ ref_emb, k=min(3, len(bank_emb))).indices.cpu().numpy()
            starts += [bank_coeffs[i].astype(np.float64) for i in near]
            kinds += [f"nn{j + 1}" for j in range(len(near))]
        if args.start_styles:
            sj = os.path.join(args.start_styles,
                              os.path.splitext(os.path.basename(ref))[0] + ".json")
            if os.path.exists(sj):
                o = json.load(open(sj))
                d1, d2 = o["style_ttl"]["dims"], o["style_dp"]["dims"]
                st = np.array(o["style_ttl"]["data"], dtype=np.float32).reshape(1, d1[1], d1[2])
                sd = np.array(o["style_dp"]["data"], dtype=np.float32).reshape(1, d2[1], d2[2])
                with torch.no_grad():
                    starts.append(basis.encode(
                        torch.tensor(st, device=dev),
                        torch.tensor(sd, device=dev)).cpu().numpy()[0].astype(np.float64))
                kinds.append("head")

        def objective(c):
            cos = np.mean(render_cos(c.astype(np.float32), SEARCH_PROBES[:1], ref_emb))
            return -(cos - args.l2 * float(np.mean(c ** 2)))

        scored = sorted((objective(s), i, s) for i, s in enumerate(starts))
        start_best = scored[0]
        bx, bf, evals = sep_cma(objective, start_best[2], args.sigma0,
                                args.iters, args.pop, rng)
        evals += len(starts)

        held = render_cos(bx.astype(np.float32), HELD_OUT, ref_emb)
        fresh = render_cos(bx.astype(np.float32), FRESH, ref_emb)
        start_held = render_cos(starts[start_best[1]].astype(np.float32), HELD_OUT, ref_emb)
        row = {"ref": os.path.basename(ref),
               "start_kind": kinds[start_best[1]],
               "start_held_out": round(float(np.mean(start_held)), 4),
               "held_out": round(float(np.mean(held)), 4),
               "fresh": round(float(np.mean(fresh)), 4),
               "evals": evals, "minutes_desktop": round((time.time() - t0) / 60, 1)}
        results.append(row)
        print(f"{row['ref']:<36} start({row['start_kind']}) {row['start_held_out']:.3f} "
              f"-> polished {row['held_out']:.3f} (fresh {row['fresh']:.3f}, "
              f"{evals} evals, {row['minutes_desktop']} min)", flush=True)

        ttl_b, dp_b = basis.decode(torch.tensor(bx[None], dtype=torch.float32, device=dev))
        json.dump({
            "style_ttl": {"dims": [1, *ttl_b.shape[1:]], "data": ttl_b.reshape(-1).tolist()},
            "style_dp": {"dims": [1, *dp_b.shape[1:]], "data": dp_b.reshape(-1).tolist()},
            "metadata": {"source": "cma_polish", "reference": ref,
                         "held_out_cos": row["held_out"]},
        }, open(os.path.splitext(args.out)[0] + f".{os.path.basename(ref)}.json", "w"))

    json.dump(results, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
