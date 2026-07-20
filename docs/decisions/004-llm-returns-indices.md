# 004. The boundary arbiter returns indices, never text

**Status:** Load-bearing.

## Decision

When the LLM arbitrates ambiguous sentence boundaries, every token is explicitly numbered in
the prompt and the model returns a *subset of the indices it was shown*:

```
Transcript tokens (language: Russian):
[0]привет [1]меня [2]зовут [3]анна [4]я [5]живу ...

Does a sentence end at each of these token indices? [1, 3, 6, 9]
```

```json
{"boundaries": [3]}
```

Indices outside the candidate set are discarded on return.

## Why

Two distinct problems, solved by the same move.

**The model cannot corrupt the words.** Per
[002](002-words-are-the-source-of-truth.md), every word is bound to an ASR timestamp. If the
model returned text, it could silently normalise, correct or reorder it — and any of those
would break the binding between text and audio. Returning indices makes that structurally
impossible: there is no channel through which text can be altered.

**The model never has to count.** Asking an LLM whether a sentence ends at "position 137" of
a 200-token list is an arithmetic task, and it is exactly the kind of arithmetic LLMs get
wrong. Showing it `[137]` next to the token turns counting into copying a number it can
already see.

## Consequences

- The arbiter can only ever *narrow* the candidate set produced by the cheap signals. It
  cannot invent boundaries at positions punctuation and pause already settled — enforced by
  intersecting the reply with the candidate set, and covered by a test.
- Prompt cost is higher: every token carries an index prefix. Worth it.
- The same numbering discipline is why arbitration is a classification task, which is what
  makes temperature 0 correct — see [008](008-sampling-temperature.md).
