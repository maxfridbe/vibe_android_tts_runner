#!/usr/bin/env python3
"""Export a translation head as the phone's style_encoder.onnx.

Same contract as export_cloner.py's StyleHead — 192-d ECAPA embedding in,
style_ttl/style_dp out, basis folded into the graph — but for the
train_translation.py architecture (LayerNorm + MLP), which is the current
true-unseen champion at roughly twice the shipped checkpoint's score.

    export_translation.py --head trans_ecapa_s.pt --out models/cloner
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_translation import Head          # noqa: E402
from export_cloner import inline_weights    # noqa: E402


class TransStyleHead(nn.Module):
    def __init__(self, ck):
        super().__init__()
        self.head = Head(int(ck["in_dim"]), int(ck["k"]), dropout=float(ck.get("dropout", 0.0)))
        self.head.load_state_dict(ck["model"])
        self.head.eval()
        self.ttl_shape = tuple(int(v) for v in ck["ttl_shape"])
        self.dp_shape = tuple(int(v) for v in ck["dp_shape"])
        self.split = int(np.prod(self.ttl_shape))
        self.register_buffer("basis", ck["basis"])
        self.register_buffer("mean", ck["mean"])
        self.register_buffer("scale", ck["scale"])
        # a PCA front (qwen heads) folds into the graph: raw features in
        self.pca = ck.get("pca_comp") is not None
        if self.pca:
            self.register_buffer("pca_mean", torch.tensor(ck["pca_mean"]))
            self.register_buffer("pca_comp", torch.tensor(ck["pca_comp"]))

    def forward(self, emb):
        if self.pca:
            emb = (emb - self.pca_mean) @ self.pca_comp.T
        flat = self.mean + (self.head(emb) * self.scale) @ self.basis
        ttl = flat[:, :self.split].reshape(-1, *self.ttl_shape)
        dp = flat[:, self.split:].reshape(-1, *self.dp_shape)
        ttl = ttl / (ttl.norm(dim=-1, keepdim=True) + 1e-8)
        dp = dp / (dp.norm(dim=-1, keepdim=True) + 1e-8)
        return ttl, dp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--name", default="style_encoder.onnx",
                    help="output graph name (style_encoder_qwen.onnx for the qwen head)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ck = torch.load(args.head, map_location="cpu", weights_only=False)
    model = TransStyleHead(ck).eval()
    in_dim = ck["pca_comp"].shape[1] if ck.get("pca_comp") is not None else int(ck["in_dim"])
    path = os.path.join(args.out, args.name)
    torch.onnx.export(model, torch.zeros(1, in_dim), path,
                      input_names=["embedding"], output_names=["style_ttl", "style_dp"],
                      dynamic_axes={"embedding": {0: "batch"}}, opset_version=args.opset)
    inline_weights(path)
    print(f"{path}  {os.path.getsize(path)/1e6:.1f} MB "
          f"(val mse {ck.get('val_mse', float('nan')):.3f}, features {ck.get('features')})")


if __name__ == "__main__":
    main()
