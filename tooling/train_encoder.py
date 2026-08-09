#!/usr/bin/env python3
"""Train the speaker encoder Supertonic never shipped: audio -> voice style.

The desktop cloner optimises the style tensors for one recording, thousands of
iterations at a time. This trains a network to *predict* them instead, so the
phone runs one forward pass. The trick is that no inverted labels are needed:
the same loss the cloner minimises is differentiable with respect to the
encoder's weights, because the whole Supertonic pipeline is differentiable
(their ONNX->torch interpreter). Per step:

    reference audio --ECAPA--> e ---E---> coeffs --basis--> style
                                                              |
                            Supertonic synth (frozen) <-------+
                                        |
                                      ECAPA
                                        |
                          loss = 1 - cos(emb, e)      (+ duration anchor)

One step costs about one inversion iteration, but the cost is amortised over
the dataset: after training, a new speaker needs no iterations at all.

Predictions live in a PCA basis fitted over sampled styles, which keeps the
output ~64 numbers instead of ~13k and — more importantly — keeps them on the
manifold where styles actually synthesise as speech.

    train_encoder.py --pairs pairs.npz --steps 20000 --out encoder.pt
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WORK = os.path.expanduser("~/supertonic-experiment")
sys.path.insert(0, os.path.join(WORK, "supertonic-voice-cloning/src"))

PROBES = [
    "The rainbow is a division of white light into many beautiful colors.",
    "Please leave a message after the tone and somebody will return your call.",
    "He described the weather, the passing ships, and the strange green light.",
]


class StyleBasis:
    """PCA over sampled styles: coefficients in, on-manifold style out.

    Rows are re-normalised after decoding because the presets live on a
    per-row unit sphere and the synthesiser assumes it.
    """

    def __init__(self, ttl, dp, k, device):
        n = len(ttl)
        X = np.concatenate([ttl.reshape(n, -1), dp.reshape(n, -1)], 1).astype(np.float32)
        self.mean = torch.tensor(X.mean(0), device=device)
        U, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
        k = min(k, Vt.shape[0])
        self.basis = torch.tensor(Vt[:k], device=device)           # (k, D)
        self.scale = torch.tensor(S[:k] / np.sqrt(max(n - 1, 1)), device=device)
        self.k = k
        self.ttl_shape = ttl.shape[1:]
        self.dp_shape = dp.shape[1:]
        self.split = int(np.prod(ttl.shape[1:]))
        var = (S**2 / (S**2).sum()).cumsum()
        print(f"style basis: {k} components, {var[k-1]*100:.1f}% of variance")

    def decode(self, coeffs):
        flat = self.mean + (coeffs * self.scale) @ self.basis
        ttl = flat[:, :self.split].view(-1, *self.ttl_shape)
        dp = flat[:, self.split:].view(-1, *self.dp_shape)
        return F.normalize(ttl, dim=-1), F.normalize(dp, dim=-1)

    def encode(self, ttl, dp):
        n = ttl.shape[0]
        flat = torch.cat([ttl.reshape(n, -1), dp.reshape(n, -1)], 1)
        return ((flat - self.mean) @ self.basis.T) / self.scale


class Encoder(nn.Module):
    """Speaker embedding -> style coefficients. Deliberately small: the heavy
    lifting is ECAPA's, which already discards everything but identity."""

    def __init__(self, in_dim, k, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, k),
        )
        nn.init.zeros_(self.net[-1].weight)      # start at the basis mean, the
        nn.init.zeros_(self.net[-1].bias)        # average voice, not at noise

    def forward(self, e):
        return self.net(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="npz from gen_style_pairs.py (fits the basis)")
    ap.add_argument("--targets", default=None,
                    help="npz of speaker embeddings to fit (gen_speaker_bank.py); "
                         "defaults to the pairs' own embeddings")
    ap.add_argument("--val-frac", type=float, default=0.1, help="speakers held out from training")
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--w-dur", type=float, default=0.3)
    ap.add_argument("--out", default=os.path.join(WORK, "clone_out/encoder.pt"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log-every", type=int, default=25)
    args = ap.parse_args()

    os.environ.setdefault("NOCOMPILE", "1")      # torch>=2.13 inductor breaks on these graphs
    os.environ.setdefault("SB_CACHE", os.path.expanduser("~/.cache/speechbrain/ecapa"))
    dev = torch.device(args.device)

    from pipeline import DiffSynth
    from speaker_encoder import SpeakerEncoder
    import torchaudio

    d = np.load(args.pairs, allow_pickle=True)
    basis = StyleBasis(d["ttl"], d["dp"], args.k, dev)

    # Targets decide what the encoder can reach. The pairs' own embeddings only
    # span the preset simplex; a Qwen-rolled bank covers far more speaker space
    # (measured mean pairwise cosine 0.14 vs 0.27), which is the whole point of
    # using a second model family to generate them.
    tgt = np.load(args.targets, allow_pickle=True)["emb"] if args.targets else d["emb"]
    tgt = np.asarray(tgt, dtype=np.float32)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(tgt))
    n_val = max(4, int(len(tgt) * args.val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    emb_bank = torch.tensor(tgt[train_idx], device=dev)
    val_bank = torch.tensor(tgt[val_idx], device=dev)
    print(f"targets: {len(emb_bank)} train / {len(val_bank)} held-out speakers"
          f" from {args.targets or args.pairs}")
    dp_bank = torch.tensor(d["dp"], device=dev)

    synth = DiffSynth(os.path.join(WORK, "assets/onnx"), device=str(dev))
    spk = SpeakerEncoder(device=str(dev))
    for p in spk.model.mods.parameters():
        p.requires_grad_(False)
    resample = torchaudio.transforms.Resample(synth.sr, 16000).to(dev)

    enc = Encoder(emb_bank.shape[1], basis.k).to(dev)
    opt = torch.optim.AdamW(enc.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    # Probes are prepared once; padding must cover the longest of them.
    # pad_bounds runs the duration predictor through onnxruntime, so it wants
    # numpy here even though everything downstream is torch
    pt, pl = synth.pad_bounds(PROBES, "en", d["dp"][:1].astype(np.float32), 1.05)
    preps = [synth.prepare(t, "en", pad_text=pt, pad_latent=pl) for t in PROBES]

    print(f"training {sum(p.numel() for p in enc.parameters())/1e3:.0f}k params "
          f"for {args.steps} steps, batch {args.batch}")
    t0 = time.time()
    run_cos, run_n = 0.0, 0
    history, best_val = [], -1.0
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(emb_bank), (args.batch,), device=dev)
        target = emb_bank[idx]                                   # what the voice should sound like
        prep = preps[step % len(preps)]

        coeffs = enc(target)
        ttl, dp = basis.decode(coeffs)
        # prepare() builds a batch-1 probe; the style batch has to line up
        prep_b = dict(prep)
        prep_b["text_ids"] = prep["text_ids"].repeat(args.batch, 1)
        prep_b["text_mask"] = prep["text_mask"].repeat(args.batch, 1, 1)
        wav, dur, _ = synth.synth(ttl, dp, prep_b)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        w16 = resample(wav)
        lens = torch.ones(w16.shape[0], device=dev)
        emb = spk.embed(w16, lens)
        cos = F.cosine_similarity(emb, target).mean()

        # anchor tempo to a reference style's duration so the encoder does not
        # buy identity with unbounded slow-down (the failure the cloner's notes
        # call out). Bank targets carry no ground-truth dp, so the preset pool
        # supplies the tempo reference.
        with torch.no_grad():
            dref = dp_bank[torch.randint(0, len(dp_bank), (args.batch,), device=dev)]
            ref_dur = synth.durations(prep_b["text_ids"], dref, prep_b["text_mask"], 1.05)
        loss = (1.0 - cos) + args.w_dur * ((dur / ref_dur.clamp(min=0.1) - 1.0) ** 2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        opt.step()
        sched.step()

        run_cos += float(cos.detach()); run_n += 1
        if step % args.val_every == 0 or step == args.steps:
            enc.eval()
            with torch.no_grad():
                vcos = []
                for j in range(0, len(val_bank), args.batch):
                    vt = val_bank[j:j + args.batch]
                    if len(vt) < args.batch:
                        break
                    vttl, vdp = basis.decode(enc(vt))
                    vw, _, _ = synth.synth(vttl, vdp, prep_b)
                    if vw.dim() == 1:
                        vw = vw.unsqueeze(0)
                    ve = spk.embed(resample(vw), torch.ones(vw.shape[0], device=dev))
                    vcos.append(float(F.cosine_similarity(ve, vt).mean()))
            enc.train()
            v = float(np.mean(vcos)) if vcos else float("nan")
            history.append({"step": step, "train_cos": run_cos / max(run_n, 1), "val_cos": v})
            with open(args.out + ".metrics.json", "w") as f:
                json.dump(history, f, indent=1)
            print(f"  [val] step {step}  held-out cos {v:.4f}", flush=True)
            if v > best_val:
                best_val = v
                torch.save({"model": enc.state_dict(), "k": basis.k,
                            "basis": basis.basis.cpu(), "mean": basis.mean.cpu(),
                            "scale": basis.scale.cpu(), "val_cos": v, "step": step,
                            "ttl_shape": basis.ttl_shape, "dp_shape": basis.dp_shape},
                           args.out + ".best")
        if step % args.log_every == 0:
            el = time.time() - t0
            print(f"step {step:6d}  cos {run_cos/run_n:.4f}  loss {float(loss):.4f}  "
                  f"{el/step*1000:.0f} ms/step  eta {(args.steps-step)*el/step/60:.0f} min", flush=True)
            run_cos, run_n = 0.0, 0
        if step % 500 == 0 or step == args.steps:
            torch.save({"model": enc.state_dict(), "k": basis.k,
                        "basis": basis.basis.cpu(), "mean": basis.mean.cpu(),
                        "scale": basis.scale.cpu(),
                        "ttl_shape": basis.ttl_shape, "dp_shape": basis.dp_shape}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
