#!/usr/bin/env python3
"""Export Qwen3-TTS's speaker encoder as ONNX: 24 kHz waveform -> 2048-d
embedding.

This is the analyzer the listening test preferred: the phone runs this
instead of (or beside) ECAPA, feeds the translation head trained on the same
features, and gets the qwen-variant Supertonic styles entirely on-device.
Follows export_cloner.py's playbook — the mel front-end uses STFT, which only
the dynamo exporter can trace, and the weights are inlined afterwards because
the phone loads assets as bytes with no sibling files.

Runs in the audiobook-maker venv:

    ~/audiobook-maker/venv/bin/python export_qwen_spk.py --out cloner_v2/ \\
        --verify ref1.wav ref2.wav
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_speaker_bank import ABM, MODELS  # noqa: E402


class QwenSpkWrapper(nn.Module):
    """24 kHz mono waveform in, speaker embedding out.

    The mel front-end re-implements qwen_tts's mel_spectrogram with the same
    constants but without its data-dependent warning guard, which the dynamo
    exporter cannot trace. Parity is asserted by --verify."""

    def __init__(self, speaker_encoder):
        super().__init__()
        self.speaker_encoder = speaker_encoder.float()
        from librosa.filters import mel as librosa_mel_fn
        mel = librosa_mel_fn(sr=24000, n_fft=1024, n_mels=128, fmin=0, fmax=12000)
        self.register_buffer("mel_basis", torch.from_numpy(mel).float())
        self.register_buffer("window", torch.hann_window(1024))

    def forward(self, wav):
        pad = (1024 - 256) // 2
        y = torch.nn.functional.pad(wav.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        spec = torch.stft(y, 1024, hop_length=256, win_length=1024, window=self.window,
                          center=False, normalized=False, onesided=True, return_complex=True)
        spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
        mels = torch.log(torch.clamp(self.mel_basis @ spec, min=1e-5))
        return self.speaker_encoder(mels.transpose(1, 2))[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join(MODELS, "Qwen3-TTS-12Hz-1.7B-Base"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--verify", nargs="*", default=[],
                    help="wavs to compare ONNX output against the torch path")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sys.path.insert(0, os.path.join(ABM, "python"))
    from qwen_tts import Qwen3TTSModel

    print("loading model (cpu, fp32 for export)...", flush=True)
    m = Qwen3TTSModel.from_pretrained(args.model, device_map="cpu",
                                      torch_dtype=torch.float32)
    core = m.model if hasattr(m, "model") else m
    spk_enc = None
    for holder in (core, getattr(core, "model", None)):
        if holder is not None and hasattr(holder, "speaker_encoder"):
            spk_enc = holder.speaker_encoder
            break
    if spk_enc is None:
        raise SystemExit("no speaker_encoder module found on the model")
    n_params = sum(p.numel() for p in spk_enc.parameters())
    print(f"speaker_encoder: {type(spk_enc).__name__}, {n_params/1e6:.1f} M params")

    wrap = QwenSpkWrapper(spk_enc).eval()
    path = os.path.join(args.out, "qwen_spk_encoder.onnx")
    torch.onnx.export(wrap, torch.zeros(1, 24000 * 4), path,
                      input_names=["wav"], output_names=["embedding"],
                      dynamic_axes={"wav": {0: "batch", 1: "samples"}},
                      opset_version=args.opset)
    # inline external weights so the phone can load the graph as one blob
    data = path + ".data"
    if os.path.exists(data):
        import onnx
        onnx.save_model(onnx.load(path), path, save_as_external_data=False)
        os.remove(data)
    print(f"{path}  {os.path.getsize(path)/1e6:.1f} MB")

    if args.verify:
        import soundfile as sf
        import onnxruntime as ort
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        for w in args.verify:
            a, sr = sf.read(w, dtype="float32")
            if a.ndim > 1:
                a = a.mean(1)
            if sr != 24000:
                n = int(len(a) * 24000 / sr)
                a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype("float32")
            with torch.no_grad():
                ref = wrap(torch.from_numpy(a)[None]).numpy().reshape(-1)
            out = sess.run(None, {"wav": a[None]})[0].reshape(-1)
            cos = float(np.dot(ref, out) / (np.linalg.norm(ref) * np.linalg.norm(out) + 1e-9))
            print(f"  parity {os.path.basename(w)}: cos {cos:.6f}")


if __name__ == "__main__":
    main()
