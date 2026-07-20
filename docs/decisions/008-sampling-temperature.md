# 008. Classification calls run at temperature 0

**Status:** Settled 2026-07-19, with measurements.

## Decision

Sampling temperature is a property of the *completer*, not a global constant. Translation
runs at 0.2; boundary arbitration runs at 0.0.

## The bug this fixes

`http_completer` originally hardcoded `temperature: 0.2` for every call. Translation is
generative and benefits from a little sampling. Arbitration is classification — the model
picks a subset of indices it was shown — where sampling buys nothing.

Measured against `nvidia/Qwen3.6-27B-NVFP4`, 12 runs per setting on an identical Russian
prompt with known ground truth:

| temperature | perfect | empty reply |
|---|---|---|
| 0.0 | 12/12 | 0/12 — byte-identical every run |
| 0.2 | 7/12 | **5/12** |

## Why the empty reply is the dangerous part

The failure is not an exception. The model returns well-formed JSON: `{"boundaries": []}`.
It is a *success* by every check in the code path.

That matters because the two failure modes are opposite, and neither raises:

- The LLM **errors** → the handler confirms *every* candidate → **over-splits**.
- The LLM returns **empty** → confirms *none* → **under-splits**, running sentences together.

Both produce a playable MP3 and exit code 0. A 42% rate of silently running sentences
together would have surfaced during tuning as "segmentation quality is mediocre" and cost
hours pointed at the wrong component.

## How it was found, which is the transferable part

The first arbiter test asked only about positions that genuinely end sentences, and the
arbiter confirmed all of them. That test cannot distinguish "working" from "failed open",
because the failure path also confirms everything.

Adding **distractors** — mid-sentence positions a competent reader would reject — made the
test discriminating. It then showed precision was excellent (zero false positives across
five distractors) while recall was intermittently zero.

Any test of a component whose failure mode is "return everything" or "return nothing" must
include cases the component is expected to *reject*. Otherwise a passing test and a broken
component look identical.

## Consequences

- Temperature is set per completer, so the `Completer` contract stays a plain
  `(system, user) -> str` that tests can stub with a two-argument function.
- Regression tests assert the outgoing request temperature, and assert that an empty reply
  is distinguishable from an exception.
- Still open: an empty reply is currently treated as "reject every boundary" when it more
  honestly means "no opinion". It should fall back to the fused score, as the exception path
  should. Both are noted in HANDOFF.md.
