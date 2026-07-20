# 003. Stage 6 is pure; stage 7 only executes

**Status:** Load-bearing.

## Decision

Stage 6 (`plan.py`) turns a timeline plus pacing knobs into a flat, absolute-timed edit list.
Stage 7 (`render_audio.py`) executes that list and does no timing arithmetic of its own.

The renderer never computes when anything happens. If it needs a number, the plan carries it.

## Why

**The preview feature depends on it.** Pacing knobs — `tts_speed`, `pre_gap`, `post_gap`,
padding — change only the edit list, never the audio content. Because stage 6 is pure and
cheap, adjusting a slider re-runs stages 6 and 7 in milliseconds of numpy, with no GPU pass
and no model load. That is the entire reason the live preview can exist.

**The video renderer gets it for free.** Phase 2 consumes the same `RenderPlan` unchanged:
`source` clips become `ffmpeg trim`, while `tts` and `silence` clips become a freeze frame at
the preceding clip's `src_end` with the translation burned in via `drawtext`. `Clip.label`
already carries the text. If the audio renderer derived its own timings, the video renderer
would have to re-derive them identically — and the two would drift.

## Consequences

- A whole class of desync bug is caught at plan time rather than after a long render.
  `RenderPlan.enforce_invariants()` checks contiguity, non-negative durations, that source
  clip durations match their `src` spans, and that every `tts` clip is anchored to a
  preceding source clip. It is cheap enough to run on every plan, so it runs on every plan.
- The anchoring rule permits `silence` between a source clip and a `tts` clip;
  `freeze_frame_time()` walks back to the nearest source clip, so a `silence → tts → silence`
  run all freezes the same frame. Reading the invariant as "immediately preceded by" is too
  strict and will produce false alarms.
- Stages 6 and 7 always re-run rather than resuming from `stages_done`. They are pure and
  cheap, so caching them would add invalidation logic to save nothing.

## Related

TTS is always synthesised at speed 1.0, with `tts_speed` applied at render time via ffmpeg
`atempo`. Baking speed into the cached wav would make the cache speed-dependent and destroy
the instant preview this decision exists to enable.
