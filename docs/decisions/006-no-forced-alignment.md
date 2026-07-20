# 006. No CTC forced-alignment pass

**Status:** Provisional. Chosen deliberately, not yet validated against the case that would
falsify it.

## Decision

Use Whisper's own word timestamps. Do not run a separate wav2vec2/CTC forced-alignment pass
to refine them.

## Why

The original design treated word-level forced alignment as a hard requirement, and it was
the stated reason for choosing WhisperX. That requirement was overstated.

What the product actually needs is **accurate sentence boundaries**, not accurate word
boundaries. Words are used for two things:

1. Deciding *where* a sentence ends — which needs relative ordering and approximate gaps,
   not millisecond precision.
2. Feeding the pause signal in `sbd.py` — again a relative measure.

The cut itself is not placed by the word timestamp at all. `plan.py` calls
`find_quietest_point` and places the splice acoustically, in a window around the boundary.
Per-word CTC precision was being computed and then discarded.

Dropping it also removed the riskiest piece of the alternative design. Mapping Whisper's
punctuated, cased, mixed-script tokens into a CTC vocabulary and back — while guaranteeing
`Word.text` survives byte-identical, per [002](002-words-are-the-source-of-truth.md) — is
fiddly, and it is fiddly in a way that silently corrupts timing when it goes wrong.

It removed a genuine scaling hazard too: `forced_align` is O(frames × targets), so a
40-minute video is on the order of 10^10 cells and needs windowed alignment to avoid running
out of memory. That is real complexity in service of precision nothing consumes.

## What would falsify this

Whisper's word timestamps stretch across non-speech audio. Measured on the first real clip:
9 of 116 words (8%) had implausible durations — one token spanning 8.7 seconds — concentrated
in three segments, producing roughly 13.5 seconds of dead air in a 166-second output. One
segment covered 14.1 seconds for two words.

That is a sentence-boundary error, which is exactly the class of error this decision claims
not to care about. So the decision is provisional.

The intended fix is **VAD, not forced alignment**: clip word timings to silero-detected
speech regions and drop words falling entirely in non-speech. That addresses the observed
failure directly, and it is far less machinery than a CTC pass.

If VAD-clipped timings still leave boundaries audibly wrong, revisit this — and note that
whisperx's own alignment models are plain HF `Wav2Vec2ForCTC` checkpoints, so reintroducing
alignment does **not** mean reintroducing CTranslate2. For Russian,
`jonatasgrosman/wav2vec2-large-xlsr-53-russian` has a native Cyrillic vocabulary, which
removes the romanization step that makes the general case unpleasant.
