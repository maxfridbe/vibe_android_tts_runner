#!/usr/bin/env python3
"""Phase 2 of on-device cloning: invert real corpus speakers into styles.

Phase 1 manufactured (embedding, style) pairs by sampling styles and
synthesising them — cheap, but every voice in it came out of a TTS model.
This walks a LibriSpeech-layout corpus (`root/<spk>/<chapter>/*.flac` with
`*.trans.txt` transcripts) and runs the existing gradient inversion on one
utterance per speaker, producing the pairs that actually close the domain
gap: a *real microphone recording's* embedding on one side, a style that
synthesises as that speaker on the other.

Each speaker costs minutes of GPU, so this is the overnight job. It follows
clone_library.py's playbook — per-voice output directory (the inverter caches
its latent target by fixed name), nearest-preset start, snapshots competing
in held-out scoring — but sources from a corpus instead of the audiobook
maker's database, takes the single nearest preset start to maximise speaker
count per night, and writes results incrementally so a training run can pick
up whatever exists so far:

    out/pairs/<spk>.npz     one (emb, ttl, dp, cos) record per finished speaker
    out/aux_pairs.npz       atomic aggregate of the above, rebuilt as it grows

    invert_corpus.py --corpus corpora/LibriSpeech/train-clean-100 \
                     --out clone_out/overnight/inversions --gpu 1
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_style import HELD_OUT  # noqa: E402
from clone_library import (Ecapa, synth, preset_bank, rank_presets, say,  # noqa: E402
                           WORK, SRC, ONNX, PRESETS, PY)


def pick_reference(spk_dir, min_dur, max_dur, target=11.5):
    """(flac, transcript) closest to the target duration within bounds.

    Inversion memory grows with the reference — ~11 GB at batch 2 for 12 s —
    so the bounds are a fit-on-a-12GB-card constraint, not a quality one."""
    best = None
    for base, _, names in os.walk(spk_dir):
        trans = {}
        for n in names:
            if n.endswith(".trans.txt"):
                for line in open(os.path.join(base, n)):
                    utt, _, text = line.partition(" ")
                    trans[utt] = text.strip()
        for n in names:
            if not n.endswith(".flac"):
                continue
            utt = n[:-5]
            if utt not in trans:
                continue
            try:
                info = sf.info(os.path.join(base, n))
            except Exception:
                continue
            dur = info.frames / info.samplerate
            if not (min_dur <= dur <= max_dur):
                continue
            d = abs(dur - target)
            if best is None or d < best[0]:
                best = (d, os.path.join(base, n), trans[utt])
    return (best[1], best[2]) if best else (None, None)


def pick_reference_vctk(root, spk, min_dur, max_dur, target=11.0):
    """VCTK clips run 2-6 s — too short to invert alone — so consecutive mic1
    utterances are concatenated until the total lands in the duration window,
    and their (already punctuated) transcripts are joined to match."""
    adir = os.path.join(root, "wav48_silence_trimmed", spk)
    tdir = os.path.join(root, "txt", spk)
    if not (os.path.isdir(adir) and os.path.isdir(tdir)):
        return None, None
    picked, total = [], 0.0
    for f in sorted(os.listdir(adir)):
        if not f.endswith("_mic1.flac"):
            continue
        txt = os.path.join(tdir, f[:-len("_mic1.flac")] + ".txt")
        if not os.path.exists(txt):
            continue
        try:
            info = sf.info(os.path.join(adir, f))
        except Exception:
            continue
        dur = info.frames / info.samplerate
        if total >= target:
            break
        if dur < 1.0 or total + dur > max_dur:
            continue
        picked.append((os.path.join(adir, f), open(txt).read().strip()))
        total += dur
    if total < min_dur:
        return None, None
    return [a for a, _ in picked], " ".join(t for _, t in picked)


def stage_concat(files, out_wav):
    """Join clips with short gaps and hand the inverter one 24 kHz mono wav."""
    parts, sr0 = [], None
    for f in files:
        w, sr = sf.read(f, dtype="float32")
        if w.ndim > 1:
            w = w.mean(1)
        sr0 = sr0 or sr
        parts += [w, np.zeros(int(0.2 * sr), dtype="float32")]
    raw = out_wav + ".raw.wav"
    sf.write(raw, np.concatenate(parts), sr0)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", raw,
                    "-ar", "24000", "-ac", "1", out_wav], check=True)
    os.remove(raw)


def load_style(path):
    o = json.load(open(path))
    d1, d2 = o["style_ttl"]["dims"], o["style_dp"]["dims"]
    return (np.array(o["style_ttl"]["data"], dtype=np.float32).reshape(d1[1], d1[2]),
            np.array(o["style_dp"]["data"], dtype=np.float32).reshape(d2[1], d2[2]))


def rebuild_aggregate(pairs_dir, out):
    """One npz a trainer can consume mid-run; write-then-replace keeps it atomic.

    Tmp names must end in .npz — numpy appends the suffix otherwise and the
    replace would miss. Loads are tolerant because a second worker on the
    other GPU may be mid-write; its pair joins the next rebuild."""
    E, T, D, C, S = [], [], [], [], []
    for f in sorted(os.listdir(pairs_dir)):
        if not f.endswith(".npz"):
            continue
        try:
            d = np.load(os.path.join(pairs_dir, f), allow_pickle=True)
            E.append(d["emb"]); T.append(d["ttl"]); D.append(d["dp"])
            C.append(float(d["cos"])); S.append(str(d["spk"]))
        except Exception:
            continue
    if not E:
        return 0
    tmp = f"{out}.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, emb=np.stack(E), ttl=np.stack(T), dp=np.stack(D),
                        cos=np.array(C, dtype=np.float32), spk=np.array(S))
    os.replace(tmp, out)
    return len(E)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default=os.path.join(WORK, "clone_out/overnight/inversions"))
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--iters", default="300,500")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--max-speakers", type=int, default=150)
    ap.add_argument("--skip", type=int, default=0,
                    help="skip the first N shuffled speakers; a second worker on the "
                         "other GPU takes a disjoint slice with the same --seed")
    ap.add_argument("--layout", choices=["librispeech", "vctk", "flat"], default="librispeech",
                    help="vctk: speakers under wav48_silence_trimmed/, transcripts "
                         "under txt/, short clips concatenated to reach the window; "
                         "flat: <name>.wav next to <name>.txt, one voice per pair "
                         "(rolled pack candidates)")
    ap.add_argument("--min-dur", type=float, default=8.0)
    ap.add_argument("--max-dur", type=float, default=13.0)
    ap.add_argument("--min-cos", type=float, default=0.45,
                    help="inversions below this are noise, not labels")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preset-renders",
                    default=os.path.join(WORK, "clone_out/library/preset_renders"))
    args = ap.parse_args()

    iters = [int(x) for x in args.iters.split(",")]
    refs_dir = os.path.join(args.out, "refs")
    pairs_dir = os.path.join(args.out, "pairs")
    os.makedirs(refs_dir, exist_ok=True)
    os.makedirs(pairs_dir, exist_ok=True)
    agg = os.path.join(args.out, "aux_pairs.npz")

    sroot = (os.path.join(args.corpus, "wav48_silence_trimmed")
             if args.layout == "vctk" else args.corpus)
    if args.layout == "flat":
        spks = [f[:-4] for f in sorted(os.listdir(sroot)) if f.endswith(".wav")
                and os.path.exists(os.path.join(sroot, f[:-4] + ".txt"))]
    else:
        spks = [s for s in sorted(os.listdir(sroot))
                if os.path.isdir(os.path.join(sroot, s))]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(spks)
    spks = spks[args.skip:args.skip + args.max_speakers]
    say(f"{len(spks)} speakers queued from {args.corpus}")

    ecapa = Ecapa()          # CPU on purpose; the GPU belongs to the inversion
    bank = preset_bank(ecapa, args.preset_renders)
    say(f"presets: {', '.join(sorted(k[:-5] for k in bank))}")

    done = kept = 0
    t0 = time.time()
    for spk in spks:
        pair_out = os.path.join(pairs_dir, f"{spk}.npz")
        if os.path.exists(pair_out):
            done += 1; kept += 1
            continue
        ref = os.path.join(refs_dir, f"{spk}.wav")
        if args.layout == "flat":
            src = os.path.join(sroot, spk + ".wav")
            try:
                info = sf.info(src)
            except Exception:
                say(f"{spk}: unreadable wav, skipped")
                continue
            if not (args.min_dur <= info.frames / info.samplerate <= args.max_dur + 1):
                say(f"{spk}: {info.frames / info.samplerate:.1f}s outside the window, skipped")
                continue
            text = open(os.path.join(sroot, spk + ".txt")).read().strip()
            if not os.path.exists(ref):
                subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                                "-ar", "24000", "-ac", "1", ref], check=True)
        elif args.layout == "vctk":
            files, text = pick_reference_vctk(args.corpus, spk,
                                              args.min_dur, args.max_dur)
            if not files:
                say(f"{spk}: not enough usable clips, skipped")
                continue
            if not os.path.exists(ref):
                stage_concat(files, ref)
        else:
            flac, text = pick_reference(os.path.join(args.corpus, spk),
                                        args.min_dur, args.max_dur)
            if not flac:
                say(f"{spk}: no utterance in [{args.min_dur},{args.max_dur}]s, skipped")
                continue
            if not os.path.exists(ref):
                subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", flac,
                                "-ar", "24000", "-ac", "1", ref], check=True)
            # LibriSpeech transcripts are bare uppercase; the phonemizer prefers prose
            text = text.lower().capitalize() + "."

        ref_emb = ecapa(ref)
        preset = rank_presets(ref_emb, bank)[0][1]
        vdir = os.path.join(args.out, spk)
        os.makedirs(vdir, exist_ok=True)
        out_json = os.path.join(vdir, f"{preset[:-5]}.json")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, NOCOMPILE="1",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                   TORCHINDUCTOR_CACHE_DIR=os.path.join(WORK, ".inductor"),
                   TRITON_CACHE_DIR=os.path.join(WORK, ".triton"))
        t1 = time.time()
        snaps = [os.path.join(vdir, f"{preset[:-5]}.{it}.json") for it in iters]
        if not all(os.path.exists(s) for s in snaps):     # a restart re-scores, not re-inverts
            with open(os.path.join(vdir, "invert.log"), "ab") as fh:
                subprocess.run([PY, "invert.py", "--reference", ref, "--onnx-dir", ONNX,
                                "--init-voice", os.path.join(PRESETS, preset),
                                "--output", out_json,
                                "--save-at", ",".join(str(i) for i in iters),
                                "--batch-size", str(args.batch), "--device", "cuda",
                                "--shutoff", "0", "--ref-text", text],
                               cwd=SRC, stdout=fh, stderr=subprocess.STDOUT, env=env)
        snaps = [s for s in snaps if os.path.exists(s)]
        if not snaps:
            say(f"{spk}: inversion produced nothing — see {vdir}/invert.log")
            continue
        scored = []
        for s in snaps:
            per = []
            for i, t in enumerate(HELD_OUT):
                wav = os.path.join(vdir, f"eval_{os.path.basename(s)}_{i}.wav")
                if not os.path.exists(wav) and not synth(s, t, wav, args.gpu):
                    continue
                per.append(float(np.dot(ref_emb.numpy(), ecapa(wav).numpy())))
            if per:
                scored.append((float(np.mean(per)), s))
        if not scored:
            say(f"{spk}: no snapshot scored")
            continue
        scored.sort(reverse=True)
        cos, best = scored[0]
        done += 1
        if cos < args.min_cos:
            say(f"{spk}: best held-out cos {cos:.3f} < {args.min_cos}, label dropped")
            continue
        ttl, dp = load_style(best)
        tmp = os.path.join(args.out, f".{spk}.{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, emb=ref_emb.numpy().astype("float32"),
                            ttl=ttl, dp=dp, cos=np.float32(cos), spk=spk,
                            ref=ref, snapshot=os.path.basename(best))
        os.replace(tmp, pair_out)
        kept += 1
        n = rebuild_aggregate(pairs_dir, agg)
        say(f"{spk}: cos {cos:.3f} from {os.path.basename(best)} "
            f"({(time.time() - t1) / 60:.1f} min) — {n} pairs banked, "
            f"{(time.time() - t0) / 3600:.1f} h elapsed")

    say(f"finished: {kept} pairs kept from {done} completed of {len(spks)} queued")


if __name__ == "__main__":
    main()
