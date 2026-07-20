# 001. ASR with timestamps, not a video-understanding LLM

**Status:** Settled.

## Decision

Transcribe with ASR (Whisper `large-v3`). Do not use a video- or audio-understanding LLM to
"watch" the source and produce a transcript.

## Why

The product splices audio. Every sentence boundary becomes a cut in a waveform, so the
transcript has to be *timestamped* and it has to be *faithful*.

A video-understanding model fails on both counts. Its timestamps are coarse and derived
rather than measured, and — decisively — it will invent speech that was never said. A
plausible-sounding hallucination is not a cosmetic error here: it produces a translation
read aloud over audio containing no such sentence.

ASR is the narrower tool and the correct one. The LLM's real job in this pipeline is
translation, where sentence-level context genuinely beats dedicated NMT — see
[007](007-local-llm-not-nmt.md).

## Model choice

**`large-v3`, not `large-v3-turbo`.** Turbo drops decoder layers, which measurably hurts
non-English. Russian, Chinese and Japanese are core targets, so the tradeoff is wrong here
even though turbo is meaningfully faster.

**Not Parakeet or Canary.** Both give better timestamps than Whisper. Both cover 25 European
languages only — no Chinese, no Japanese. Disqualifying regardless of timestamp quality.

## What ASR still gets wrong, and what we do about it

Whisper hallucinates fluent, correctly-formatted, entirely fabricated text over music and
silence — subtitle credits are the classic case, learned from subtitle training data.
Observed here in Russian (`Редактор субтитров ...`) as well as English and Japanese.

Filtering is therefore part of the stage, not an afterthought: known credit patterns, plus a
compression-ratio test that catches a decoder stuck in a repetition loop. See
[005](005-drop-ctranslate2.md) for why the compression test replaced Whisper's own
`no_speech_prob` and `avg_logprob`.

Whisper's *segment* boundaries are also unusable as sentences — they come from
timestamp-token sampling and the 30-second decoding window, not from linguistics, and they
routinely fall nowhere near the punctuation the model itself emitted. Sentences are decided
in `segment.py` from the word array instead. This is unrelated to punctuation quality and
holds even when punctuation is good.
