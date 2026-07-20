# Decisions

One file per decision that would otherwise get relitigated. Each records what was chosen,
what it was chosen *over*, and the reasoning — because the alternatives are usually
reasonable, and "why not X" is the part that gets lost.

These are not neutral surveys. They are arguments, written down at the point the decision
was made, so a later reader can tell the difference between a considered trade-off and an
accident.

| # | Decision | Status |
|---|---|---|
| [001](001-asr-not-video-model.md) | ASR with timestamps, not a video-understanding LLM | Settled |
| [002](002-words-are-the-source-of-truth.md) | `Timeline.words` is immutable; segments are index ranges | Load-bearing |
| [003](003-plan-render-split.md) | Stage 6 is pure; stage 7 only executes | Load-bearing |
| [004](004-llm-returns-indices.md) | The boundary arbiter returns indices, never text | Load-bearing |
| [005](005-drop-ctranslate2.md) | Whisper on plain PyTorch, not faster-whisper | Settled 2026-07-20 |
| [006](006-no-forced-alignment.md) | No CTC forced-alignment pass | Provisional |
| [007](007-local-llm-not-nmt.md) | A local LLM, not NLLB/SeamlessM4T | Settled |
| [008](008-sampling-temperature.md) | Classification calls run at temperature 0 | Settled 2026-07-19 |

**Status** means: *Load-bearing* — code depends on this; breaking it is a redesign.
*Settled* — decided with evidence, revisit only with new evidence. *Provisional* — chosen
deliberately but not yet validated against the case that would falsify it.
