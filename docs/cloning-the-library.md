# Cloning a whole voice library

`tooling/clone_library.py` walks the audiobook maker's voice library on the
GPU host and produces one Supertonic style per speaker. It exists because
looping `clone_voice.sh` over fourteen voices would be wrong in three ways.

**One output directory per voice.** `invert.py` caches the denoised reference
and its latent target (`ref_denoised.wav`, `z_ref.npy`) under the *output*
directory, by fixed name. A shared output directory silently hands voice B the
latent target of voice A — the runs would finish, the numbers would look fine,
and the voices would be quietly wrong.

**The starting style is chosen per speaker.** Where the voice starts matters
(0.69–0.81 held-out on a single reference, depending only on the start), so
instead of a fixed F1/F3/M1 the script renders all ten shipped presets once,
embeds them with ECAPA, and starts each speaker from its two nearest. That
ranking is speaker-independent work done once for the whole library, and it is
informative on its own: Dale's nearest preset scored 0.362 and Minerva's best
was 0.071, which is a fair warning about how far the optimiser has to travel.

**Every snapshot competes.** Each run saves at 300 and 600 iterations and all
snapshots of all starts are scored on held-out sentences, so a run that peaked
early is not thrown away. It happens: Dale and Fireside Narrator both scored
best at 300, Stephen Fry and Douglas Rain at 600.

Passes are ordered so that every voice gets its best start before any voice
gets its second — the library is complete and usable after ~50 minutes and
only improves after that, rather than being three voices perfect and eleven
missing.

## Reference length is the binding constraint

Inversion memory grows with the reference. A 12 s clip peaks near 11 GB at
`--batch-size 2`, which is why the default 3 OOMs a 12 GB card, and the 39–47 s
designed voices OOM'd outright.

References are therefore cut to 13 s — but audio and transcript have to stay
consistent or the mel and latent losses lose their alignment. Both are derived
from the same speaking rate: take a text prefix ending on a sentence (or word)
boundary, then trim the audio to exactly that prefix's predicted length. The
per-voice caches are deleted when a reference is re-cut, since they describe
the old audio.

## Results

Held-out ECAPA cosine against the original recording, first pass:

| speaker | start | cos |
|---|---|---|
| Fireside Narrator | M5 | 0.886 |
| rigckman my mistress (Alan Rickman) | M1 | 0.888 |
| Stephen Fry | M5 | 0.864 |
| magonagoll | F1 | 0.826 |
| Dale | M5 | 0.820 |
| Serverus 1 | M4 | 0.795 |
| Douglas Rain | F2 | 0.704 |
| minerva | F2 | 0.685 |
| Tony Jay | F2 | 0.533 |

For scale, the reference implementation's own example lands at 0.81 on the
same measure. The weak entries share a pattern — their nearest preset was a
poor match to begin with, which is exactly what the second pass is for.
