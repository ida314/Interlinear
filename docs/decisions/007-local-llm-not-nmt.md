# 007. A local LLM, not NLLB or SeamlessM4T

**Status:** Settled.

## Decision

Translate with a local instruction-following LLM over an OpenAI-compatible HTTP API. Do not
use a dedicated neural machine translation model.

## Why

NMT models are architecturally sentence-in, sentence-out. That is a poor fit for a transcript,
where three things matter and none of them survive sentence isolation:

- **Pronoun resolution.** A sentence whose subject was named two sentences ago cannot be
  translated correctly without seeing them. The pipeline passes preceding sentences as
  context precisely for this.
- **Consistent register.** A speaker addressing a camera informally should sound that way for
  the whole video. NMT has no mechanism to hold register across calls.
- **Glossary adherence.** Recurring terms should translate the same way every time.

The usual reason to prefer NMT is low-resource language pairs, where dedicated models still
lead. Every language targeted here — ru, pt, es, fr, de, zh, ja — is high-resource, so that
advantage does not apply.

There is also a practical argument: the translation prompt asks for output "roughly as long
as the original, because it will be read aloud in a pause." That is a length constraint
expressed in natural language. An LLM honours it; an NMT model has no way to receive it.

## Why HTTP rather than in-process

The client speaks plain OpenAI-compatible HTTP, so Ollama, vLLM, SGLang or a hosted endpoint
are interchangeable behind one setting. The translation model is by far the largest component
in the stack, and decoupling it means it can be swapped, restarted or moved to another
machine without touching the pipeline.

It also keeps the model out of the app's process, which matters on a box where Whisper and
Kokoro are already resident.

## Consequences

- Prompts must survive being answered by different models. `_parse_json` tolerates code
  fences and preamble, because instruct-tuned models add them despite being told not to.
- Response format support varies between servers. `response_format: {"type":"json_object"}`
  is verified working on the current vLLM setup, but is not universal; the fallback of
  omitting it and parsing tolerantly also works, and is the portable choice.
- Batching matters more than any other optimisation on bandwidth-bound hardware, since
  translation is embarrassingly parallel and every batch shares a system prompt. Run the
  server with prefix caching enabled.
- A failed translation must not silently pass source text through as the "translation" —
  that yields an MP3 of the source language read aloud in the target voice, with exit code 0.
  Noted as open in HANDOFF.md.
