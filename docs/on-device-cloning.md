# Cloning a voice on the phone

Today cloning is a desktop step: gradient inversion optimises Supertonic's
style tensors against a reference recording, ~0.4 s per iteration on an RTX
5070 and 500 iterations for a usable voice. That is not portable. Backprop
through the 257 MB flow model is thousands of times more work than the phone
does to *use* a voice, and ONNX Runtime on Android has no autograd at all, so
no amount of quantisation moves it: the constraint is the algorithm, not the
model size.

The way to put cloning on the phone is to stop solving an optimisation
problem there and solve a *regression* problem instead — train the encoder
Supertone never published.

## Why this is trainable

Supertonic has no speaker encoder: nothing maps audio → style. But the
inverse direction is free. Given any style `s`, synthesising speech is one
cheap forward pass, so **we can manufacture supervision**:

```
s  ──Supertonic──▶  waveform  ──speaker encoder──▶  e
                                                    ▲
        train  E: e ─▶ s   on exactly these pairs ──┘
```

Every pair is generated, not collected. There is no licensing question, no
recording effort, and the labels are exact — we know `s` because we chose it.

## The shape of the problem

`style_ttl` is 50x256 and `style_dp` is 8x16: 12 928 numbers. Regressing that
directly is silly, because the styles that sound like real people occupy a
far smaller manifold. Two observations bound it:

- The published styles differ from an arbitrary reference by ECAPA cosine
  0.03–0.22, i.e. they are all "far" — the manifold is not tiny.
- Inversion from different starts lands at 0.80, 0.81 and 0.52 on the same
  reference (the 0.52 being a male start for a female voice). Gender is a
  coarse coordinate; the rest is refinement.

So: fit a PCA basis over a corpus of styles and have the network predict
**basis coefficients**, not raw tensors. This shrinks the output from ~13 k
dims to (say) 64–256, and — more valuable — constrains predictions to the
manifold, which is what stops a regressor from emitting a style that
synthesises noise.

### First measurement (30 generated pairs)

`tooling/gen_style_pairs.py` produces the pairs and `tooling/probe_manifold.py`
asks whether they are learnable at all:

```
PCA  10 comps ->  75.8% of style variance
PCA  20 comps ->  89.1% of style variance
style distance vs speaker distance: pearson 0.479, spearman 0.543
speaker cosine spread: 0.565 mean, 0.244 min, 0.864 max
```

Two things follow. The style target really does compress — 20 components
carry ~90% of the variation in a 12 928-dim tensor, so a regressor predicting
a couple of dozen coefficients is the right shape. And the map is *ordered
but loose*: a Spearman of 0.54 between style distance and speaker distance
means nearby styles do tend to sound like nearby speakers, which is what
makes learning possible, while the scatter says a single embedding will not
pin a style down exactly. That is an argument for the refinement stage below,
not against the encoder.

Caveat on these numbers: 30 samples drawn from the preset simplex. The PCA
figure in particular is optimistic — this data spans ten presets, so of
course a handful of components covers it. Phase 2 data will move it.

## Training data, in two phases

**Phase 1 — synthetic, hours.** Sample styles as Dirichlet-weighted convex
combinations of the ten published styles plus per-row spherical noise (the
parametrisation the inversion itself uses, so it respects the geometry).
Synthesise one sentence each, embed, store `(e, s)`. At ~1.5 s per style on
one GPU, 2 000 pairs is under an hour, and both GPUs halve it.

This phase proves the loop and produces a weak encoder, but it has a ceiling:
the data only spans the simplex of ten presets, while inversion moves freely
outside it. An encoder trained only on this can never reach a voice the
presets do not already surround.

**Phase 2 — real, overnight.** Take a few hundred speakers from a public
corpus (VCTK, LibriSpeech), invert each into a style with the existing
pipeline, and keep `(real audio embedding, inverted style)`. At ~3.5 min per
inversion across two GPUs, 500 speakers is roughly 15 hours — one unattended
night. These pairs are the ones that matter: they teach the map from *real
microphone audio* to style, closing the domain gap that Phase 1 cannot.

Augment both phases with noise, reverb, level and codec variation, because
the phone will feed it a real room, not a clean synthesis.

## The encoder

Front end: a pretrained speaker embedding, not a from-scratch audio network.
ECAPA (the same one the inversion optimises against) gives a 192-d vector
that already discards everything except identity. Feeding the regressor that,
plus a few global prosody statistics (median f0, speaking rate, spectral
tilt) for the duration side, keeps the trainable part small:

```
ECAPA 192 ─┐
           ├─▶ MLP (2-3 layers, ~1 M params) ─▶ 64-256 PCA coeffs ─▶ style
f0/rate/tilt ┘
```

On-device cost: ECAPA-lite or a distilled speaker encoder in ONNX is
15–80 MB and runs in tens of milliseconds on 10 s of audio; the MLP is
negligible. **Cloning becomes a sub-second operation on the phone.**

## Refinement, if the regression is not enough

A learned encoder will land near the speaker, not exactly on them. The
phone can then do a short *gradient-free* polish in the PCA subspace:
CMA-ES over 64 coefficients, each evaluation being one short synthesis
(~0.5 s at the measured RTF 0.5) plus one embedding. Two hundred evaluations
is about two minutes — acceptable for a one-off "make it sound more like me"
button, and it needs no autograd, only the forward pass the app already runs.

Retrieval is the cheap cousin of the same idea: precompute a bank of styles
with their embeddings, ship the PCA coefficients (10 k styles x 64 floats is
2.5 MB), and start from the nearest neighbour. Worth having as the fallback
when the encoder is uncertain.

## How to know it worked

Score against the desktop inversion, which is the ceiling this is trying to
approximate, using `tooling/eval_style.py` on sentences no method optimised
against:

| method | held-out ECAPA cosine |
|---|---|
| published preset (floor) | 0.20 |
| desktop inversion, 500 iters (ceiling) | 0.80–0.81 |
| learned encoder | target ≥ 0.70 |
| encoder + on-device CMA-ES polish | target ≥ 0.75 |

Hold out speakers, never just sentences. And judge by listening as well:
cosine rewards timbre and is largely blind to the prosody `style_dp`
controls, which is exactly where a regressor is most likely to be bland.

## Honest risks

- **The domain gap is the whole game.** If Phase 2 pairs are too few or too
  clean, the encoder will work on synthesised audio and fail on a phone
  recording. This is the first thing to measure, not the last.
- **PCA may be the wrong basis.** If the manifold is curved, a linear basis
  will waste capacity; a small VAE over styles is the fallback.
- **Inverted styles are noisy labels.** Two inversions of the same reference
  differ (0.73 vs 0.69 measured), so the targets carry run-to-run variance
  that caps how sharp the regressor can get. Averaging several inversions per
  speaker, or keeping only the best by held-out score, costs GPU time but
  buys label quality.
