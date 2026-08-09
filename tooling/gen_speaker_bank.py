#!/usr/bin/env python3
"""Build a bank of diverse speaker embeddings to train the style encoder on.

The encoder learns audio -> Supertonic style by being pushed to match target
speaker embeddings. Where those targets come from decides what it can learn:

  - Styles sampled from Supertonic's own presets only span those ten voices,
    so an encoder trained on them can never reach a speaker outside that span.
  - Qwen3-TTS VoiceDesign invents a *new* voice per seed and description, from
    a different model family entirely. Its speakers are out-of-domain by
    construction, which is exactly the pressure the encoder needs.
  - Real recordings (VCTK, LibriSpeech) close the last gap: microphone, room
    and codec, none of which any synthesiser reproduces.

This script covers the middle source. It runs in two passes because the Qwen
worker and the cloning tools live in different virtualenvs — and keeping them
apart is useful anyway, since the audio is worth retaining for later
augmentation:

    # pass 1, audiobook-maker venv: roll voices
    gen_speaker_bank.py --count 2000 --audio-dir bank_audio

    # pass 2, supertonic venv: embed them
    gen_speaker_bank.py --embed-only --audio-dir bank_audio --out speaker_bank.npz
"""
import argparse
import os
import random
import sys

import numpy as np
import soundfile as sf
import torch

ABM = os.path.expanduser("~/audiobook-maker")
MODELS = os.path.expanduser("~/comfy/ComfyUI/models/TTS")

# Description axes, crossed and sampled: VoiceDesign conditions on the text,
# so the wording is the diversity knob.
AGE = ["young", "middle-aged", "elderly", "twenty-something", "mature"]
GENDER = ["male", "female", "androgynous"]
TIMBRE = ["deep and gravelly", "bright and clear", "warm and smooth", "thin and nasal",
          "breathy", "resonant and full", "raspy", "light and airy", "booming",
          "soft-spoken", "husky", "crisp"]
MANNER = ["speaking calmly", "with an energetic delivery", "in a measured, formal tone",
          "casually", "with warmth", "in a clipped, precise manner", "slowly and deliberately",
          "with a lively cadence"]
TEXTS = [
    "The rainbow is a division of white light into many beautiful colors.",
    "Please leave a message after the tone and somebody will return your call.",
    "He described the weather, the passing ships, and the strange green light.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--out", default=os.path.expanduser("~/supertonic-experiment/clone_out/speaker_bank.npz"))
    ap.add_argument("--audio-dir", default=os.path.expanduser("~/supertonic-experiment/clone_out/bank_audio"))
    ap.add_argument("--model", default=os.path.join(MODELS, "Qwen3-TTS-12Hz-1.7B-VoiceDesign"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embed-only", action="store_true",
                    help="pass 2: embed the wavs already in --audio-dir")
    args = ap.parse_args()

    os.makedirs(args.audio_dir, exist_ok=True)
    if args.embed_only:
        return embed_dir(args)

    sys.path.insert(0, os.path.join(ABM, "python"))
    from abm_worker.engine import Engine                      # noqa: E402

    engine = Engine()
    engine.load(args.model)          # Engine() takes no args; load() pins cuda:0

    rng = random.Random(args.seed)
    embs, descs, seeds = [], [], []
    for i in range(args.count):
        desc = (f"a {rng.choice(AGE)} {rng.choice(GENDER)} voice, "
                f"{rng.choice(TIMBRE)}, {rng.choice(MANNER)}")
        seed = rng.randint(1, 2**30)
        text = TEXTS[i % len(TEXTS)]
        try:
            # qwen-tts wants language names, not ISO codes
            wav, sr = engine.generate_design(text, desc, "english", {"seed": seed})
        except Exception as e:                                 # a bad roll is not fatal
            print(f"  roll {i} failed: {type(e).__name__}: {e}", flush=True)
            continue
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        path = os.path.join(args.audio_dir, f"spk_{i:05d}.wav")
        sf.write(path, wav, sr)
        with open(path + ".txt", "w") as f:
            f.write(desc + "\n" + str(seed) + "\n")
        descs.append(desc)
        seeds.append(seed)
        if (i + 1) % 25 == 0:
            print(f"  rolled {i+1}/{args.count} voices", flush=True)
    print(f"wrote {len(descs)} wavs to {args.audio_dir}; "
          f"now run with --embed-only in the supertonic venv")


def embed_dir(args):
    """Pass 2: ECAPA-embed every wav in the bank directory."""
    os.environ.setdefault("SB_CACHE", os.path.expanduser("~/.cache/speechbrain/ecapa"))
    from speechbrain.inference.speaker import EncoderClassifier

    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.environ["SB_CACHE"], run_opts={"device": args.device})
    wavs = sorted(f for f in os.listdir(args.audio_dir) if f.endswith(".wav"))
    embs, descs = [], []
    for i, name in enumerate(wavs):
        a, sr = sf.read(os.path.join(args.audio_dir, name), dtype="float32")
        if a.ndim > 1:
            a = a.mean(1)
        if sr != 16000:
            n = int(len(a) * 16000 / sr)
            a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype("float32")
        with torch.no_grad():
            e = enc.encode_batch(torch.from_numpy(a)[None].to(args.device)).squeeze()
            embs.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        meta = os.path.join(args.audio_dir, name + ".txt")
        descs.append(open(meta).read().split("\n")[0] if os.path.exists(meta) else "")
        if (i + 1) % 100 == 0:
            print(f"  embedded {i+1}/{len(wavs)}", flush=True)

    E = np.stack(embs)
    off = (E @ E.T)[~np.eye(len(E), dtype=bool)]
    print(f"bank spread: mean pairwise cos {off.mean():.3f} "
          f"(lower is more diverse; Supertonic's ten presets sit near 0.2 from a random speaker)")
    np.savez_compressed(args.out, emb=E, desc=np.array(descs))
    print(f"wrote {args.out}: {len(E)} speakers")


if __name__ == "__main__":
    main()
