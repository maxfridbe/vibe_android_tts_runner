#!/usr/bin/env python3
"""Roll voice-pack candidates: Qwen VoiceDesign voices meant to become
Supertonic styles.

This differs from gen_speaker_bank.py in intent and shape. The bank wanted
raw diversity — thousands of short one-sentence rolls, only their embeddings
kept. A pack candidate is a voice we intend to *ship*, so each roll reads a
fixed ~11 second passage (long enough to invert well, short enough for the
12 GB inversion window), keeps its transcript beside it for the inverter's
probe, and records its description so the curated pack can say what a voice
is supposed to sound like.

Runs in the audiobook-maker venv (Qwen needs its torch):

    ~/audiobook-maker/venv/bin/python gen_pack_roll.py --count 60 --out packs/rolls
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_speaker_bank import ABM, MODELS, AGE, GENDER, TIMBRE, MANNER  # noqa: E402

# VoiceDesign conditions on the wording, so accent and character are just
# more axes of the same knob.
ACCENT = ["", "with a British accent", "with an American accent",
          "with a soft Irish accent", "with an Australian accent",
          "with a Scottish accent"]
CHARACTER = ["", "like a storyteller", "like a news anchor",
             "like a kindly grandparent", "like a wise mentor",
             "with a hint of mischief"]

# One fixed passage for every candidate: identical transcripts make the
# inversions comparable and give the inverter a probe that exactly matches
# the audio. ~150 chars ≈ 10-11 s of speech.
PASSAGE = ("The old lighthouse keeper climbed the spiral stairs every evening "
           "at dusk. He would trim the wick, polish the brass, and watch the "
           "ships pass far below.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--out", required=True, help="directory for <name>.wav/.txt/.meta.json")
    ap.add_argument("--model", default=os.path.join(MODELS, "Qwen3-TTS-12Hz-1.7B-VoiceDesign"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sys.path.insert(0, os.path.join(ABM, "python"))
    from abm_worker.engine import Engine

    engine = Engine()
    engine.load(args.model)

    rng = random.Random(args.seed)
    made = 0
    for i in range(args.count):
        name = f"pack_{i:03d}"
        wav_path = os.path.join(args.out, name + ".wav")
        if os.path.exists(wav_path):
            made += 1
            continue
        bits = [f"a {rng.choice(AGE)} {rng.choice(GENDER)} voice",
                rng.choice(TIMBRE), rng.choice(MANNER)]
        for extra in (rng.choice(ACCENT), rng.choice(CHARACTER)):
            if extra:
                bits.append(extra)
        desc = ", ".join(bits)
        seed = rng.randint(1, 2**30)
        try:
            wav, sr = engine.generate_design(PASSAGE, desc, "english", {"seed": seed})
        except Exception as e:
            print(f"  {name} failed: {type(e).__name__}: {e}", flush=True)
            continue
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        dur = len(wav) / sr
        if not (6.0 <= dur <= 14.0):        # a runaway roll would blow the inversion window
            print(f"  {name}: {dur:.1f}s outside the window, rerolled next run", flush=True)
            continue
        sf.write(wav_path, wav, sr)
        open(os.path.join(args.out, name + ".txt"), "w").write(PASSAGE)
        json.dump({"desc": desc, "seed": seed, "dur": round(dur, 2)},
                  open(os.path.join(args.out, name + ".meta.json"), "w"))
        made += 1
        if made % 10 == 0:
            print(f"  {made}/{args.count} candidates rolled", flush=True)
    print(f"{made} pack candidates in {args.out}")


if __name__ == "__main__":
    main()
