# On-device cloning encoder

Graphs the app downloads into `filesDir/cloner/` (Settings → On-device voice
cloning), or that you can side-load with `adb`. There are two analyzer
variants — pick one when cloning; the app offers both:

| file | size | shape |
|---|---|---|
| `spk_encoder.onnx` | 84 MB | 16 kHz mono waveform → 192-d ECAPA embedding |
| `style_encoder.onnx` | 8.4 MB | 192-d ECAPA embedding → `style_ttl [1,50,256]`, `style_dp [1,8,16]` |
| `qwen_spk_encoder.onnx` | 49 MB | 24 kHz mono waveform → 2048-d Qwen3-TTS speaker embedding |
| `style_encoder_qwen.onnx` | 9.6 MB | 2048-d Qwen embedding → the same style tensors |

Each pair turns a recording into a Supertonic speaker in two forward passes,
about a second on a phone, where the desktop cloner needs a CUDA GPU and 500
gradient-inversion steps. The ECAPA variant scores higher on speaker cosine;
the Qwen variant was preferred in listening (it carries character voices
better) and is the default in the clone dialog.

The `style_encoder*` heads are supervised regressors trained through the
frozen synthesiser on inverted-real-speaker labels (see
`docs/on-device-cloning.md`). Retrained on 187 inverted pairs: ECAPA head
0.496, Qwen head 0.429 true-unseen ECAPA cosine (against 0.82 for desktop
inversion). The forward-only CMA polish (`tooling/cma_polish.py`) lifts these
to ~0.57 on the phone by searching the same basis — the path to closing the
rest of the gap.

## What this checkpoint is

Trained by `tooling/train_encoder.py` (amortised inversion: the encoder is
trained *through* the frozen synthesiser against an ECAPA speaker loss, so the
thing it optimises is the thing that is measured).

| | |
|---|---|
| training pairs | 1200 style/embedding pairs from `gen_style_pairs.py` |
| targets | 1122 unique speakers rolled with Qwen3-TTS VoiceDesign, mean pairwise cosine 0.223 |
| basis | PCA, k = 64 |
| steps | 20 000 (best checkpoint at 15 250) |
| duration anchor | `--w-dur 0.3` |
| hardware | one RTX 5070, ~340 ms/step, ~1 h 55 m |
| best held-out | 0.3421 |

A second run at k = 128 (lr 2e-4, `--w-dur 0.15`) reached 0.3176 on the same
data, so capacity was not the limit.

## How good it actually is

Measured the way that matters — synthesise held-out sentences with the
predicted style, embed them, compare to the original recording:

| | held-out ECAPA cosine |
|---|---|
| desktop inversion, 500 iters | **0.820** |
| this encoder, real recording | **0.223** |
| this encoder, Qwen-designed speakers | 0.319 / 0.238 / 0.539 |
| random preset (floor) | 0.10–0.20 |

So: a voice in roughly the right family, not the person. The gap turned out to
be the PCA basis, not the target set: projecting a 0.82-cosine desktop
inversion onto this checkpoint's basis destroys it down to 0.225 — the same
number the encoder scores. Later encoders trained with inverted real-speaker
styles folded into the basis roughly double the unseen-speaker score;
`docs/on-device-cloning.md` ("What the second night found") has the
measurements and the path to a shippable replacement for this checkpoint.

## Reproducing

```
tooling/gen_style_pairs.py  --count 1200 --out clone_out/pairs_p1.npz
tooling/gen_speaker_bank.py --count 1500 --audio-dir clone_out/bank_audio
tooling/gen_speaker_bank.py --embed-only --audio-dir clone_out/bank_audio --out clone_out/bank_a.npz
tooling/merge_banks.py clone_out/bank_all.npz clone_out/bank_a.npz clone_out/bank_b.npz
tooling/train_sweep.sh --pairs clone_out/pairs_p1.npz --targets clone_out/bank_all.npz --steps 20000
tooling/export_cloner.py --encoder clone_out/encoder-k64.pt.best --out models/cloner
```

The roll and the embed pass use different virtualenvs (Qwen needs torch with
CUDA; the embed pass needs the Supertonic one), which is why the bank is built
in two commands.
