# Supertonic 3 expression tags

The model card advertises "10 inline tags (e.g. `<laugh>`, `<breath>`, `<sigh>`)"
but never lists them, and nothing in the published code treats them specially:
the preprocessor passes `<` and `>` straight through as code points, exactly
like the `<en>…</en>` language wrapper. So the tags are whatever sequences the
model itself learned, and the only way to find them is to ask it.

## How they were found

Synthesise a carrier sentence with the candidate in it and transcribe the
result (SenseVoice, running on the GPU host):

    "I see. <x> Well then."   →   is "x" spoken as a word, or consumed?

A real tag disappears from the transcript and leaves a vocalisation behind. A
non-tag is simply read out — `<chuckle>` came back as "I see chuckle well
then", `<gasp>` as "I see gasp", and the invented `<xyzzy>` as "I see exci well
then".

## The ten

    <laugh>   <sigh>   <breath>   <cough>   <cry>
    <yawn>    <hmm>    <um>       <tsk>     <kiss>

`<laughter>` and `<breathe>` are also consumed and are almost certainly
spellings of `<laugh>` and `<breath>` — counting them separately would make
twelve, and the model card says ten.

## They are shorter than you expect

This is why they read as "not working". Measured on M1, one tag adds about a
fifth of a second:

| text | duration |
|---|---|
| `I see. Well then.` | 1.80 s |
| `I see. <laugh> Well then.` | 2.00 s |
| `I see. <laugh><laugh> Well then.` | 2.33 s |
| `I see. <laugh> <laugh> <laugh> Well then.` | 2.71 s |

Repeats stack cleanly for `<laugh>`. They do not for every tag: three `<sigh>`
in a row broke down and the model started saying "sigh, sigh".

The same text rendered on the S24 FE and transcribed back gave 1.86 s → 2.72 s,
so the phone path preserves tags exactly — nothing in the app's chunker,
preprocessor or trimming eats them.

## In the app

Talk's tag row carries these ten. A tap inserts one; a long press inserts
three, which is the difference between a breath and an actual laugh.
