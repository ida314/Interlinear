"""Stage 4: translate sentences with surrounding context.

A local LLM rather than a dedicated NMT model, because NMT (NLLB, SeamlessM4T) is
architecturally sentence-in/sentence-out — it cannot see the previous sentence to resolve a
pronoun, cannot hold register consistent across a video, and cannot honour a glossary. Every
language we target is high-resource, so the usual reason to prefer NMT does not apply.

Talks OpenAI-compatible HTTP, so Ollama and vLLM are interchangeable. Start on Ollama for
simplicity; move to vLLM when throughput matters, since translation is embarrassingly
parallel and batching is worth far more on a bandwidth-bound machine than any other
optimisation.

This module also hosts the sentence-boundary arbiter, which reuses the same client.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Sequence

from app.config import Settings, settings
from app.models.timeline import Timeline, Word
from app.pipeline.sbd import word_join

# `(system_prompt, user_prompt) -> completion text`
Completer = Callable[[str, str], str]

# Translation is generative and benefits from a little sampling. Arbitration is
# classification, where sampling only costs accuracy: measured against Qwen3.6-27B, 12 runs
# per setting, temperature 0.2 returned a well-formed but *empty* boundary set 5 times out
# of 12, while temperature 0.0 was correct 12/12. An empty set is not an error — it silently
# means "no sentence ends anywhere here", so the job runs sentences together and still exits
# zero.
TRANSLATE_TEMPERATURE = 0.2
ARBITER_TEMPERATURE = 0.0

LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "pt": "Portuguese", "fr": "French", "de": "German",
    "ru": "Russian", "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "it": "Italian",
    "nl": "Dutch", "pl": "Polish", "tr": "Turkish", "ar": "Arabic", "hi": "Hindi",
}

TRANSLATE_SYSTEM = """You translate speech transcripts for language learners.

Rules:
- Translate each numbered sentence into {target}.
- Translate meaning, not words. Use natural spoken {target}.
- Keep each translation roughly as long as the original; it will be read aloud in a pause.
- Do not add explanations, notes, or commentary.
- If a sentence is an incomplete fragment, translate the fragment as-is.
- Return ONLY a JSON object: {{"translations": [{{"id": <int>, "text": "<translation>"}}]}}
- Return exactly one entry per input sentence, with matching ids."""

ARBITER_SYSTEM = """You detect sentence boundaries in speech transcripts.

You receive numbered tokens from an automatic transcript. Punctuation may be missing or
wrong. For each candidate position you are asked about, decide whether a sentence genuinely
ENDS at that token.

Rules:
- Judge only the positions listed. Ignore all others.
- A sentence ends at a complete thought, not at a pause or a filler word.
- Never rewrite, correct, or re-order the tokens.
- Return ONLY a JSON object: {"boundaries": [<token indices that end a sentence>]}"""


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


def http_completer(
    cfg: Settings | None = None, *, temperature: float = TRANSLATE_TEMPERATURE
) -> Completer:
    """OpenAI-compatible chat completion. Works against Ollama and vLLM unchanged.

    `temperature` is set per *completer* rather than per call, so the `Completer` contract
    stays a plain `(system, user) -> str` that a test can stub with a two-argument function.
    """
    cfg = cfg or settings

    def complete(system: str, user: str) -> str:
        import httpx  # noqa: PLC0415

        response = httpx.post(
            f"{cfg.llm_base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {cfg.llm_api_key}"},
            json={
                "model": cfg.llm_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "response_format": {"type": "json_object"},
            },
            timeout=cfg.llm_timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    return complete


def _parse_json(raw: str) -> dict:
    """Tolerate the fences and preamble that instruct-tuned models add despite being asked
    not to."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in model output: {raw[:200]!r}")
    return json.loads(match.group(0))


# --- translation -----------------------------------------------------------------------


def translate_timeline(
    timeline: Timeline,
    *,
    completer: Completer | None = None,
    cfg: Settings | None = None,
    progress: Callable[[float], None] | None = None,
) -> Timeline:
    cfg = cfg or settings
    complete = completer or http_completer(cfg)
    target = language_name(timeline.params.target_lang)

    pending = [s for s in timeline.segments if s.is_translatable]
    batch_size = max(1, cfg.translate_batch_size)
    system = TRANSLATE_SYSTEM.format(target=target)

    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        context = pending[max(0, offset - cfg.translate_context_sentences) : offset]
        try:
            results = _translate_batch(complete, system, batch, context, target)
        except Exception as exc:  # noqa: BLE001 — one bad batch must not kill the job
            timeline.warn("translate", f"Batch failed ({exc}); retrying one at a time.")
            results = _translate_individually(complete, system, batch, target, timeline)

        for seg, text in zip(batch, results):
            seg.translation = text or seg.text
            seg.translation_model = cfg.llm_model
            if not text:
                timeline.warn("translate", "Translation failed; kept source text.", seg.id)

        if progress:
            progress(min(1.0, (offset + len(batch)) / max(1, len(pending))))

    timeline.mark_done("translate")
    return timeline


def _build_prompt(batch, context, target: str) -> str:
    lines = []
    if context:
        lines.append("Preceding context (do not translate):")
        lines += [f"- {s.text}" for s in context]
        lines.append("")
    lines.append(f"Translate these {len(batch)} sentences into {target}:")
    lines += [f"{i}. {seg.text}" for i, seg in enumerate(batch)]
    return "\n".join(lines)


def _translate_batch(complete: Completer, system: str, batch, context, target: str) -> list[str]:
    payload = _parse_json(complete(system, _build_prompt(batch, context, target)))
    entries = payload.get("translations", [])
    by_id = {int(e["id"]): str(e["text"]).strip() for e in entries if "id" in e and "text" in e}
    if len(by_id) != len(batch):
        raise ValueError(f"expected {len(batch)} translations, got {len(by_id)}")
    return [by_id[i] for i in range(len(batch))]


def _translate_individually(
    complete: Completer, system: str, batch, target: str, timeline: Timeline
) -> list[str]:
    """Last resort before giving up on a sentence. One bad sentence must never fail a
    45-minute job."""
    out = []
    for seg in batch:
        try:
            payload = _parse_json(complete(system, _build_prompt([seg], [], target)))
            out.append(str(payload["translations"][0]["text"]).strip())
        except Exception:  # noqa: BLE001
            out.append("")
    return out


# --- sentence boundary arbitration -----------------------------------------------------


def make_arbiter(
    completer: Completer | None = None, cfg: Settings | None = None
) -> Callable[[Sequence[Word], list[int], str], set[int]]:
    """Build the arbiter that `sbd.apply_arbitration` calls for ambiguous boundaries.

    Every token is explicitly numbered in the prompt and the model returns *indices*, never
    text. That matters twice over: the model cannot silently rewrite words that are bound to
    forced-alignment timestamps, and it never has to count — it copies a number it can see.
    Asking an LLM to tally position N in a 200-token list is exactly the sort of arithmetic
    it gets wrong.
    """
    cfg = cfg or settings
    complete = completer or http_completer(cfg, temperature=ARBITER_TEMPERATURE)

    def arbitrate(words: Sequence[Word], candidates: list[int], lang: str) -> set[int]:
        if not candidates:
            return set()
        confirmed: set[int] = set()
        for chunk in _chunk_candidates(candidates, size=25):
            try:
                confirmed |= _arbitrate_chunk(complete, words, chunk, lang)
            except Exception:  # noqa: BLE001
                # Falling back to the fused score is the safe failure: it is what we would
                # have used had arbitration been switched off.
                confirmed |= {i for i in chunk}
        return confirmed

    return arbitrate


def _chunk_candidates(candidates: list[int], size: int) -> list[list[int]]:
    return [candidates[i : i + size] for i in range(0, len(candidates), size)]


def _arbitrate_chunk(
    complete: Completer, words: Sequence[Word], candidates: list[int], lang: str
) -> set[int]:
    lo = max(0, min(candidates) - 20)
    hi = min(len(words), max(candidates) + 21)
    sep = word_join(lang)

    numbered = sep.join(f"[{i}]{words[i].text}" for i in range(lo, hi))
    user = (
        f"Transcript tokens (language: {language_name(lang)}):\n{numbered}\n\n"
        f"Does a sentence end at each of these token indices? {sorted(candidates)}\n"
        "Return the subset where a sentence genuinely ends."
    )
    payload = _parse_json(complete(ARBITER_SYSTEM, user))
    returned = {int(i) for i in payload.get("boundaries", [])}
    # Ignore anything outside what we asked about — the model does not get to invent
    # boundaries at positions the cheap signals already settled.
    return returned & set(candidates)
