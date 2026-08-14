# On-device cloning encoder

Graphs the app downloads into `filesDir/cloner/` (Settings → On-device voice
cloning), or that you can side-load with `adb`. There are two analyzer
variants — pick one when cloning; the app offers both:

| file | size | shape |
|---|---|---|
| `qwen_spk_encoder.onnx` | 49 MB | 24 kHz mono waveform → 2048-d Qwen3-TTS speaker embedding |
| `style_encoder_qwen.onnx` | 24 MB | 2048-d Qwen embedding → `style_ttl [1,50,256]`, `style_dp [1,8,16]` |
| `qwen_center.bin` | 8 KB | 2048 × f32 population mean of Qwen speaker features |
| `spk_encoder.onnx` | 84 MB | 16 kHz mono waveform → 192-d ECAPA embedding |
| `style_encoder.onnx` | 8.4 MB | 192-d ECAPA embedding → the same style tensors |
| `style_basis_v3.bin` | 20 MB | k=384 PCA style basis for the on-device refine search |

Each analyzer pair turns a recording into a Supertonic speaker in two forward
passes, about a second on a phone, where the desktop route needs a CUDA GPU
and 500 gradient steps. **Qwen is the shipped direction**: the project's
similarity metric is centered Qwen cosine (cosine of Qwen speaker features
after subtracting `qwen_center.bin`; raw features share a dominant common
component), and the ear has sided with the Qwen judge consistently — even
against gradient inversion. The ECAPA pair remains for comparison.

## Current checkpoints (2026-08-13)

- **`style_encoder_qwen.onnx`** — "fh_d10": PCA-256 input front + 512×2 MLP,
  trained on a **418-pair expressive bank** of inverted real speakers
  (LibriSpeech/VCTK read speech + RAVDESS/Thorsten emotion + EARS whisper)
  plus 3 000 manufactured (audio, coefficient) pairs. Held-out centered-qwen
  **0.524** over 58 voices — above the 0.4905 the desktop inversions score
  on themselves.
- **`style_basis_v3.bin`** — the k=384 basis refit on the same expressive
  bank. Served under a *versioned name* because a k=384 refit is always the
  same byte size and the downloader's cache-buster is size; the app falls
  back to a side-loaded `style_basis.bin` if the v3 name is absent.
- **`qwen_center.bin`** — when present next to `qwen_spk_encoder.onnx`, the
  in-app refine switches its objective to centered-qwen cosine (measured
  mean 0.840 over 18 voices on desktop, vs the 0.4905 goal line).

Depth/width sweeps kept losing to this small head (4 independent runs) —
data coverage, not capacity, is what improves it, and delivery classes absent
from the bank (whisper before, high-energy "bubbly" currently) clone poorly
until inverted examples of the class are added and everything is retrained.

Methodology, training scripts, experiment log (`attempts.md`) and the Rust
`clonevoice` CLI live in
[vibe_supertonic_voice_cloner](https://github.com/maxfridbe/vibe_supertonic_voice_cloner);
`docs/on-device-cloning.md` here covers the app integration.
