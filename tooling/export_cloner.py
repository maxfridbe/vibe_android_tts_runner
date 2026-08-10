#!/usr/bin/env python3
"""Export the on-device cloning pair: speaker encoder + style predictor.

Two ONNX graphs go to the phone:

  spk_encoder.onnx   16 kHz mono waveform -> 192-d ECAPA embedding
  style_encoder.onnx 192-d embedding      -> style_ttl [1,50,256] + style_dp [1,8,16]

The second one folds the PCA basis into its final matmul, so the phone never
sees coefficients — it gets a style tensor it can hand straight to the
synthesiser. Both are small enough to ship inside the APK.

    export_cloner.py --encoder encoder.pt.best --out ../apps/tts-runner/app/src/main/assets
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

WORK = os.path.expanduser("~/supertonic-experiment")
sys.path.insert(0, os.path.join(WORK, "supertonic-voice-cloning/src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_encoder import Encoder  # noqa: E402


class StyleHead(nn.Module):
    """Encoder + basis decode as one graph, so the phone gets tensors not coefficients."""

    def __init__(self, ckpt):
        super().__init__()
        self.k = int(ckpt["k"])
        self.ttl_shape = tuple(int(x) for x in ckpt["ttl_shape"])
        self.dp_shape = tuple(int(x) for x in ckpt["dp_shape"])
        self.enc = Encoder(192, self.k)
        self.enc.load_state_dict(ckpt["model"])
        self.register_buffer("basis", ckpt["basis"])
        self.register_buffer("mean", ckpt["mean"])
        self.register_buffer("scale", ckpt["scale"])
        self.split = int(np.prod(self.ttl_shape))

    def forward(self, emb):
        flat = self.mean + (self.enc(emb) * self.scale) @ self.basis
        ttl = flat[:, :self.split].reshape(-1, *self.ttl_shape)
        dp = flat[:, self.split:].reshape(-1, *self.dp_shape)
        ttl = ttl / (ttl.norm(dim=-1, keepdim=True) + 1e-8)
        dp = dp / (dp.norm(dim=-1, keepdim=True) + 1e-8)
        return ttl, dp


class SpkWrapper(nn.Module):
    """ECAPA with its feature front-end, waveform in and unit-norm embedding out."""

    def __init__(self, mods):
        super().__init__()
        self.compute_features = mods.compute_features
        self.mean_var_norm = mods.mean_var_norm
        self.embedding_model = mods.embedding_model

    def forward(self, wav):
        lens = torch.ones(wav.shape[0], device=wav.device)
        feats = self.compute_features(wav)
        feats = self.mean_var_norm(feats, lens)
        emb = self.embedding_model(feats, lens).squeeze(1)
        return emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)


def inline_weights(path: str) -> None:
    """Fold a sibling .onnx.data back into the graph.

    The phone loads these as asset bytes — there is no filesystem next to them
    for ONNX Runtime to find external weights in — so a two-file export would
    load as a graph with no parameters."""
    data = path + ".data"
    if not os.path.exists(data):
        return
    import onnx
    model = onnx.load(path)          # pulls the external tensors in
    onnx.save_model(model, path, save_as_external_data=False)
    os.remove(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default=os.path.join(WORK, "clone_out/encoder.pt.best"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ckpt = torch.load(args.encoder, map_location="cpu", weights_only=False)
    print(f"checkpoint from step {ckpt.get('step','?')}, held-out cos {ckpt.get('val_cos', float('nan')):.4f}")

    head = StyleHead(ckpt).eval()
    style_path = os.path.join(args.out, "style_encoder.onnx")
    torch.onnx.export(
        head, torch.zeros(1, 192), style_path,
        input_names=["embedding"], output_names=["style_ttl", "style_dp"],
        dynamic_axes={"embedding": {0: "batch"}}, opset_version=args.opset)
    inline_weights(style_path)
    print(f"{style_path}  {os.path.getsize(style_path)/1e6:.1f} MB")

    os.environ.setdefault("SB_CACHE", os.path.expanduser("~/.cache/speechbrain/ecapa"))
    from speechbrain.inference.speaker import EncoderClassifier
    sb = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.environ["SB_CACHE"], run_opts={"device": "cpu"})
    spk = SpkWrapper(sb.mods).eval()
    spk_path = os.path.join(args.out, "spk_encoder.onnx")
    # 4 s of 16 kHz audio as the tracing shape; the length axis stays dynamic
    # The legacy exporter cannot trace speechbrain's STFT ("STFT does not
    # currently support complex types"), so this one goes through the dynamo
    # exporter — which is also why the weights need inlining afterwards.
    torch.onnx.export(
        spk, torch.zeros(1, 16000 * 4), spk_path,
        input_names=["wav"], output_names=["embedding"],
        dynamic_axes={"wav": {0: "batch", 1: "samples"}}, opset_version=args.opset)
    inline_weights(spk_path)
    print(f"{spk_path}  {os.path.getsize(spk_path)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
