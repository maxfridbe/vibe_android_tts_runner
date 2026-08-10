#!/usr/bin/env python3
"""Embed a real-speech corpus into speaker targets for the style encoder.

The first trained encoder's autopsy (docs/on-device-cloning.md) was blunt:
every embedding it ever saw came out of a TTS model, and a real recording
landed at 0.22. This closes that gap at the target end: walk a corpus of real
recordings — any layout where the first directory level under a root is the
speaker (LibriSpeech `train-clean-100/<spk>/...`, VCTK
`wav48_silence_trimmed/<spk>/...`) — embed utterances with the same ECAPA the
trainer optimises against, and write a bank `train_encoder.py --targets` can
consume.

Augmentation happens here rather than at training time because the trainer
never touches audio, only embeddings. Each speaker contributes one clean
utterance plus several corrupted ones (noise, synthetic room reverb, gain, an
MP3 round-trip), so the bank covers what a phone microphone in a real room
will produce instead of only studio-clean synthesis.

    embed_corpus.py --roots corpora/LibriSpeech/train-clean-100 --out real_bank.npz

The output keeps a `spk` column so the trainer can hold out whole speakers,
never just utterances of a speaker it trained on.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf
import torch

AUDIO_EXT = (".flac", ".wav", ".ogg")
SR = 16000


def speakers(root):
    """{speaker_id: [audio files]} — first directory level is the speaker."""
    out = {}
    for spk in sorted(os.listdir(root)):
        d = os.path.join(root, spk)
        if not os.path.isdir(d):
            continue
        files = []
        for base, _, names in os.walk(d):
            files += [os.path.join(base, n) for n in names if n.lower().endswith(AUDIO_EXT)]
        if files:
            out[spk] = sorted(files)
    return out


def load(path):
    w, sr = sf.read(path, dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    if sr != SR:
        n = int(len(w) * SR / sr)
        w = np.interp(np.linspace(0, len(w) - 1, n), np.arange(len(w)), w).astype("float32")
    return w


def aug_noise(w, rng):
    snr = rng.uniform(5.0, 25.0)
    n = rng.standard_normal(len(w)).astype("float32")
    n *= np.std(w) / (np.std(n) + 1e-8) * 10 ** (-snr / 20)
    return w + n


def aug_reverb(w, rng):
    """Synthetic room: exponentially decaying noise as the impulse response."""
    from scipy.signal import fftconvolve
    tau = rng.uniform(0.05, 0.25)
    t = np.arange(int(0.4 * SR)) / SR
    rir = rng.standard_normal(len(t)).astype("float32") * np.exp(-t / tau).astype("float32")
    rir[0] = 3.0                                  # direct path dominates
    rir /= np.linalg.norm(rir) + 1e-8
    return fftconvolve(w, rir)[:len(w)].astype("float32")


def aug_gain(w, rng):
    target = rng.uniform(-30.0, -12.0)            # dBFS rms
    rms = np.sqrt(np.mean(w ** 2)) + 1e-8
    return w * (10 ** (target / 20) / rms)


def aug_codec(w, rng):
    """MP3 round-trip at a phone-recorder bitrate."""
    kbps = int(rng.choice([32, 48, 64, 96]))
    raw = tempfile.mktemp(suffix=".wav")
    mp3 = tempfile.mktemp(suffix=".mp3")
    back = tempfile.mktemp(suffix=".wav")
    try:
        sf.write(raw, w, SR)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", raw, "-b:a", f"{kbps}k", mp3],
                       check=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", mp3, "-ar", str(SR), "-ac", "1", back],
                       check=True)
        return load(back)
    finally:
        for p in (raw, mp3, back):
            if os.path.exists(p):
                os.remove(p)


AUGS = {"noise": aug_noise, "reverb": aug_reverb, "gain": aug_gain, "codec": aug_codec}


def corrupt(w, rng):
    kind = rng.choice(["noise", "reverb", "gain", "codec", "noise+reverb"])
    for k in kind.split("+"):
        w = AUGS[k](w, rng)
    peak = np.abs(w).max() + 1e-8
    if peak > 1.0:
        w = w / peak
    return w, kind


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-speaker", type=int, default=6)
    ap.add_argument("--min-dur", type=float, default=3.0)
    ap.add_argument("--max-dur", type=float, default=20.0, help="longer clips are trimmed")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.environ.setdefault("SB_CACHE", os.path.expanduser("~/.cache/speechbrain/ecapa"))
    from speechbrain.inference.speaker import EncoderClassifier
    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.environ["SB_CACHE"], run_opts={"device": args.device})

    rng = np.random.default_rng(args.seed)
    E, S, A = [], [], []
    n_spk = 0
    for root in args.roots:
        spks = speakers(root)
        print(f"{root}: {len(spks)} speakers")
        for spk, files in spks.items():
            picks = []
            for f in rng.permutation(files):
                try:
                    info = sf.info(f)
                except Exception:
                    continue
                if info.frames / info.samplerate >= args.min_dur:
                    picks.append(f)
                if len(picks) >= args.per_speaker:
                    break
            if not picks:
                continue
            n_spk += 1
            for i, f in enumerate(picks):
                w = load(f)[:int(args.max_dur * SR)]
                aug = "clean"
                if i > 0:                          # first utterance stays clean
                    w, aug = corrupt(w, rng)
                with torch.no_grad():
                    e = enc.encode_batch(torch.from_numpy(w)[None].to(args.device)).squeeze()
                    e = torch.nn.functional.normalize(e, dim=-1).cpu().numpy()
                E.append(e.astype("float32"))
                S.append(f"{os.path.basename(root)}/{spk}")
                A.append(aug)
            if n_spk % 25 == 0:
                print(f"  {n_spk} speakers, {len(E)} embeddings", flush=True)

    emb = np.stack(E)
    np.savez_compressed(args.out, emb=emb, spk=np.array(S), aug=np.array(A))
    off = (emb @ emb.T)[~np.eye(len(emb), dtype=bool)]
    print(f"wrote {args.out}: {len(emb)} embeddings from {n_spk} speakers, "
          f"mean pairwise cos {off.mean():.3f}")


if __name__ == "__main__":
    main()
