# 002. `Timeline.words` is the source of truth; segments are index ranges

**Status:** Load-bearing. Breaking this is a redesign, not a refactor.

## Decision

ASR produces `Timeline.words` exactly once. Nothing downstream may rewrite, reorder or
renormalise it. A `Segment` is a half-open index range `[word_start, word_end)` over that
array — never a copy of the text.

`Segment.text`, `.start` and `.end` exist, but they are *denormalised* for readability when
you open `timeline.json` to debug. The indices are authoritative; `rebuild_derived()`
regenerates the rest.

## Why

The product is audio splicing. Every cut point has to land on a real timestamp in the source
waveform, and every piece of text has to stay bound to the audio it came from. The failure
mode is not a crash — it is a bilingual track where the English translation no longer lines
up with the Russian sentence it follows, discovered only by listening.

Segmentation is the stage most likely to change. It has eight knobs, an optional neural
detector, an optional LLM arbitration pass, and merge/split constraints that interact. If
re-segmenting produced *new text*, every one of those code paths would be an opportunity to
drift out of sync with the audio.

Making segments index ranges removes the possibility structurally. Re-segmenting with
different knobs computes different `[start, end)` pairs over the same immutable array. There
is no code path that can desynchronise audio from text, because there is no code path that
can produce text.

## Alternatives rejected

**Segments own their text.** The obvious model, and what most transcription pipelines do. It
requires every transformation to carry timestamps along correctly, forever, by convention.
Conventions are not enforced by anything.

**Re-run ASR on segment changes.** Correct, and absurd — it makes a slider adjustment cost a
GPU pass over the whole video.

## Consequences

- Re-segmentation is cheap and provably safe.
- The TTS cache keyed on `sha256(text, voice, lang)` composes with this: change a knob, and
  only sentences whose *text* actually changed need re-synthesising.
- The cost is indirection. Reading `timeline.json` means resolving indices, hence the
  denormalised fields — which must be regenerated, never hand-edited.
- `Word` is per-character for zh/ja, since there is no whitespace to split on. "Word count"
  means character count there, and rejoining must not insert spaces.
