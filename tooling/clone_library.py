#!/usr/bin/env python3
"""Clone every speaker in the audiobook maker's library into Supertonic styles.

The audiobook maker on the GPU host keeps a voice library: reference audio plus
the exact transcript of that audio. That transcript is what makes a good
inversion cheap — no whisper pass, and the mel/latent losses get a probe that
really matches the recording. This walks that library and produces one
Supertonic style JSON per speaker, ready to import in TTS Runner.

Three things it does that a loop over clone_voice.sh would not:

  * a per-voice output directory. invert.py caches `z_ref.npy` and
    `ref_denoised.wav` under the *output* directory by fixed name, so a shared
    output directory would silently hand voice B the latent target of voice A.
  * starts chosen per speaker. The starting style matters (measured spread
    0.69-0.81 held-out on one reference), so instead of a fixed F1/F3/M1 it
    scores all ten shipped presets against the reference first and starts from
    the two nearest. The preset renders are speaker-independent, so they are
    synthesised once and reused for the whole library.
  * snapshots as candidates. Every run saves at two iteration counts and all
    snapshots compete in the held-out scoring, so a run that peaked early is
    not thrown away.

    clone_library.py --gpus 0,1
    clone_library.py --only "Stephen Fry,Tony Jay" --iters 300,600

Progress is written to <out>/progress.json after every event and the script is
resumable: finished runs are detected and skipped.
"""
import argparse
import json
import os
import queue
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import torch

WORK = os.environ.get("WORK", os.path.expanduser("~/supertonic-experiment"))
SRC = os.path.join(WORK, "supertonic-voice-cloning/src")
ONNX = os.path.join(WORK, "assets/onnx")
PRESETS = os.path.join(WORK, "assets/voice_styles")
PY = os.environ.get("PYTHON", os.path.join(WORK, "venv/bin/python"))
DB = os.path.expanduser("~/.config/audiobook_maker/data/app.db")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_style import HELD_OUT, load16k  # noqa: E402

PREVIEW_TEXT = "Hello there. This is how I sound when I read a book to you."

_print_lock = threading.Lock()


def say(msg: str) -> None:
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip() or "voice"


def library(db_path: str):
    """(name, reference, transcript) for every voice with usable reference audio."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = []
    for vid, name, ref, text in con.execute(
            "select id, name, ref_audio_path, ref_text from voices order by created_at"):
        d = os.path.dirname(ref or "")
        # ref.wav is the maker's cleaned 24 kHz cut; the upload is the fallback
        cand = [os.path.join(d, "ref.wav"), ref or ""]
        src = next((p for p in cand if p and os.path.exists(p)), None)
        if not src:
            say(f"skip {name}: no reference audio on disk")
            continue
        rows.append({"id": vid, "name": name, "src": src, "text": (text or "").strip()})
    return rows


MAX_REF = 13.0   # seconds


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", path], capture_output=True, text=True)
    return float(out.stdout.strip() or 0)


def stage(voices, refs_dir, out_dir=None):
    """One mono 24 kHz WAV per voice, so mp3 uploads and odd rates never reach
    the inverter's loader.

    Long references are cut down. Inversion memory grows with the reference —
    a 12 s clip already peaks near 11 GB at batch 2, and the 39-47 s designed
    voices OOM'd outright. The cut keeps audio and transcript consistent by
    deriving both from the same speaking rate: take a text prefix ending on a
    sentence (or word) boundary, then trim the audio to exactly that prefix's
    predicted length, so the mel and latent losses still line up."""
    os.makedirs(refs_dir, exist_ok=True)
    for v in voices:
        dest = os.path.join(refs_dir, slugify(v["name"]) + ".wav")
        if os.path.exists(dest) and duration(dest) <= MAX_REF + 0.5:
            v["ref"] = dest
            continue
        full = duration(v["src"])
        end = None
        if full > MAX_REF:
            text = v["text"]
            if text:
                rate = len(text) / full           # characters per second
                budget = int(MAX_REF * rate)
                sent = max(text.rfind(". ", 0, budget), text.rfind("? ", 0, budget),
                           text.rfind("! ", 0, budget))
                cut = sent + 1 if sent > budget * 0.5 else text.rfind(" ", 0, budget)
                if cut <= 0:
                    cut = budget
                v["text"] = text[:cut].strip()
                end = len(v["text"]) / rate
            else:
                end = MAX_REF
            say(f"{v['name']}: reference {full:.1f}s -> {end:.1f}s "
                f"({len(v['text'])} chars of transcript kept)")
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", v["src"]]
        if end:
            cmd += ["-t", f"{end:.3f}"]
        subprocess.run(cmd + ["-ar", "24000", "-ac", "1", dest], check=True)
        v["ref"] = dest
        # the inverter caches the denoised reference and its latent target under
        # the output directory; a re-cut reference invalidates both
        if out_dir:
            vdir = os.path.join(out_dir, slugify(v["name"]))
            for stale in ("z_ref.npy", "z_ref_recon.wav", "ref_denoised.wav"):
                p = os.path.join(vdir, stale)
                if os.path.exists(p):
                    os.remove(p)
    return voices


class Ecapa:
    """Speaker embeddings, shared by the preset ranking and the final scoring.

    Kept on the CPU on purpose: a 12 s reference already pushes the inversion
    to the edge of a 12 GB card, and half a gigabyte of resident speaker
    encoder was the difference between fitting and an OOM mid-run."""

    def __init__(self, device="cpu"):
        os.environ.setdefault("SB_CACHE", os.path.expanduser("~/.cache/speechbrain/ecapa"))
        from speechbrain.inference.speaker import EncoderClassifier
        self.device = device
        self.lock = threading.Lock()   # one model, two GPU workers calling it
        self.enc = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.environ["SB_CACHE"], run_opts={"device": device})

    def __call__(self, path):
        with self.lock, torch.no_grad():
            e = self.enc.encode_batch(load16k(path, self.device)).squeeze()
        return torch.nn.functional.normalize(e, dim=-1)


def synth(style: str, text: str, out: str, gpu: str = "0") -> bool:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    r = subprocess.run([PY, "synth_onnx.py", "--onnx-dir", ONNX, "--voice", style,
                        "--text", text, "--out", out],
                       cwd=SRC, capture_output=True, env=env)
    return r.returncode == 0 and os.path.exists(out)


def preset_bank(ecapa, cache_dir):
    """Embed each shipped preset once on the held-out sentences. Speaker-
    independent, so the whole library shares this work."""
    os.makedirs(cache_dir, exist_ok=True)
    bank = {}
    for f in sorted(os.listdir(PRESETS)):
        if not f.endswith(".json"):
            continue
        embs = []
        for i, text in enumerate(HELD_OUT):
            wav = os.path.join(cache_dir, f"{f[:-5]}_{i}.wav")
            if not os.path.exists(wav) and not synth(os.path.join(PRESETS, f), text, wav):
                break
            embs.append(ecapa(wav))
        if embs:
            bank[f] = torch.stack(embs)
    return bank


def rank_presets(ref_emb, bank):
    scored = [(float((bank[f] @ ref_emb).mean()), f) for f in bank]
    scored.sort(reverse=True)
    return scored


def invert(voice, preset, gpu, vdir, iters, log, batch):
    """One inversion run. Returns the snapshot paths it produced."""
    tag = preset[:-5]
    out = os.path.join(vdir, f"{tag}.json")
    made = [os.path.join(vdir, f"{tag}.{it}.json") for it in iters]
    if all(os.path.exists(p) for p in made):
        say(f"{voice['name']}: {tag} already done")
        return made
    args = [PY, "invert.py", "--reference", voice["ref"], "--onnx-dir", ONNX,
            "--init-voice", os.path.join(PRESETS, preset), "--output", out,
            "--save-at", ",".join(str(i) for i in iters),
            "--batch-size", str(batch), "--device", "cuda", "--shutoff", "0"]
    if voice["text"]:
        args += ["--ref-text", voice["text"]]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), NOCOMPILE="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               TORCHINDUCTOR_CACHE_DIR=os.path.join(WORK, ".inductor"),
               TRITON_CACHE_DIR=os.path.join(WORK, ".triton"))
    t0 = time.time()
    with open(log, "ab") as fh:
        subprocess.run(args, cwd=SRC, stdout=fh, stderr=subprocess.STDOUT, env=env)
    say(f"{voice['name']}: {tag} on gpu{gpu} took {(time.time() - t0) / 60:.1f} min")
    return [p for p in made if os.path.exists(p)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DB)
    ap.add_argument("--out", default=os.path.join(WORK, "clone_out/library"))
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--iters", default="300,600", help="snapshot iterations; the last one ends the run")
    ap.add_argument("--starts", type=int, default=2, help="how many nearest presets to invert from")
    ap.add_argument("--batch-size", type=int, default=2,
                    help="probe phrases per step; 3 OOMs a 12 GB card on a 12 s reference")
    ap.add_argument("--only", default="", help="comma-separated voice names")
    args = ap.parse_args()

    iters = [int(x) for x in args.iters.split(",")]
    gpus = args.gpus.split(",")
    os.makedirs(args.out, exist_ok=True)
    styles_dir = os.path.join(args.out, "styles")
    prev_dir = os.path.join(args.out, "previews")
    os.makedirs(styles_dir, exist_ok=True)
    os.makedirs(prev_dir, exist_ok=True)

    voices = stage(library(args.db), os.path.join(args.out, "refs"), args.out)
    if args.only:
        want = {n.strip() for n in args.only.split(",")}
        voices = [v for v in voices if v["name"] in want]
    say(f"{len(voices)} voices: {', '.join(v['name'] for v in voices)}")

    ecapa = Ecapa()
    say("rendering the preset bank (once for the whole library) ...")
    bank = preset_bank(ecapa, os.path.join(args.out, "preset_renders"))
    say(f"presets available: {', '.join(sorted(k[:-5] for k in bank))}")

    state = {}
    state_path = os.path.join(args.out, "progress.json")
    if os.path.exists(state_path):
        state = json.load(open(state_path))
    lock = threading.Lock()

    def checkpoint(name, **kv):
        with lock:
            state.setdefault(name, {}).update(kv)
            tmp = state_path + ".tmp"
            json.dump(state, open(tmp, "w"), indent=1)
            os.replace(tmp, state_path)

    for v in voices:
        v["ref_emb"] = ecapa(v["ref"])
        v["starts"] = rank_presets(v["ref_emb"], bank)
        checkpoint(v["name"], stage="queued", ref=v["ref"],
                   preset_rank=[(round(s, 4), f[:-5]) for s, f in v["starts"][:4]])
        say(f"{v['name']}: nearest presets " +
            ", ".join(f"{f[:-5]} {s:.3f}" for s, f in v["starts"][:3]))

    # Every voice's nearest start runs before any voice's second start, so the
    # library is complete and usable after the first pass and only improves
    # after that — better than finishing three voices perfectly while eleven
    # have nothing.
    jobs = queue.Queue()
    for si in range(args.starts):
        for v in voices:
            if si < len(v["starts"]):
                jobs.put((v, si))
    vlocks = {v["name"]: threading.Lock() for v in voices}

    def score_and_export(v, vdir, gpu):
        """Held-out scoring over every snapshot this voice has so far."""
        cands = sorted(os.path.join(vdir, f) for f in os.listdir(vdir)
                       if re.fullmatch(r".+\.\d+\.json", f))
        scores = []
        for c in cands:
            per = []
            for i, text in enumerate(HELD_OUT):
                wav = os.path.join(vdir, f"eval_{os.path.basename(c)}_{i}.wav")
                if not os.path.exists(wav) and not synth(c, text, wav, gpu):
                    continue
                per.append(float(torch.dot(v["ref_emb"], ecapa(wav))))
            if per:
                scores.append((float(np.mean(per)), float(np.std(per)), c))
        if not scores:
            return None
        scores.sort(key=lambda r: -r[0])
        best_cos, best_std, best = scores[0]
        style = json.load(open(best))
        style["metadata"] = {
            "name": v["name"],
            "source": "audiobook maker library",
            "reference": os.path.basename(v["ref"]),
            "held_out_cos": round(best_cos, 4),
            "from": os.path.basename(best),
        }
        dest = os.path.join(styles_dir, slugify(v["name"]) + ".json")
        json.dump(style, open(dest, "w"))
        synth(dest, PREVIEW_TEXT, os.path.join(prev_dir, slugify(v["name"]) + ".wav"), gpu)
        checkpoint(v["name"], stage="done", cos=round(best_cos, 4),
                   spread=round(best_std, 4), best=os.path.basename(best),
                   ranking=[(round(s, 4), os.path.basename(c)) for s, _, c in scores])
        say(f"{v['name']}: held-out cos {best_cos:.4f} +/- {best_std:.4f} "
            f"from {os.path.basename(best)} -> {dest}")
        return best_cos

    def worker(gpu):
        while True:
            try:
                v, si = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                vdir = os.path.join(args.out, slugify(v["name"]))
                os.makedirs(vdir, exist_ok=True)
                log = os.path.join(vdir, "invert.log")
                preset = v["starts"][si][1]
                checkpoint(v["name"], stage=f"inverting from {preset[:-5]}",
                           gpu=gpu, started=time.strftime("%H:%M:%S"))
                made = invert(v, preset, gpu, vdir, iters, log, args.batch_size)
                if not made:
                    say(f"{v['name']}: {preset[:-5]} produced nothing — see {log}")
                    checkpoint(v["name"], stage="failed", failed_start=preset[:-5])
                with vlocks[v["name"]]:
                    score_and_export(v, vdir, gpu)
            except Exception as e:  # a single bad voice must not stop the library
                checkpoint(v["name"], stage="error", error=str(e))
                say(f"{v['name']}: {type(e).__name__}: {e}")
            finally:
                jobs.task_done()

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in gpus]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    done = [(s.get("cos", 0), n) for n, s in state.items() if s.get("stage") == "done"]
    done.sort(reverse=True)
    say(f"finished {len(done)}/{len(voices)} in {(time.time() - t0) / 60:.1f} min")
    for cos, n in done:
        print(f"  {cos:.4f}  {n}")
    bundle = shutil.make_archive(os.path.join(args.out, "styles"), "zip", styles_dir)
    say(f"bundle: {bundle}")


if __name__ == "__main__":
    raise SystemExit(main())
