"""Stage 2: audio -> timed words.

Deliberately *not* an LLM watching video: a video-understanding model gives worse timestamps
and invents speech that was never said. This is Whisper, run through plain PyTorch
(`transformers`), which is what keeps the project architecture-agnostic — the same code path
runs on CUDA, MPS and CPU, on aarch64 and x86 alike.

That choice replaced faster-whisper/WhisperX, whose CTranslate2 backend ships no aarch64
CUDA wheels: `pip install` succeeds and then silently runs on CPU at roughly 1/20 speed.

Two things this module does that a bare `pipeline(...)` call does not:

* **It keeps only the words.** Whisper's own segment boundaries come from timestamp-token
  sampling and the 30-second decoding window, not from linguistics — they routinely fall
  nowhere near the punctuation the model itself emitted. Sentence boundaries are decided in
  `segment.py` instead, from the word array.
* **It filters hallucinations**, which Whisper produces reliably over music and silence.

Word timings only need to be good enough to *choose* a sentence boundary; the cut itself is
placed acoustically by `plan.py` via `find_quietest_point`. That is why this needs no
separate CTC forced-alignment pass.
"""

from __future__ import annotations

import re
import zlib
from pathlib import Path

from app import device as device_mod
from app.audio import io
from app.config import Settings, settings
from app.models.timeline import SourceInfo, Timeline, Word

# Whisper emits these over silence and music, learned from subtitle training data. They are
# fluent, correctly formatted, and entirely fabricated.
HALLUCINATION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^\s*(thanks?|thank you) for watching",
        r"^\s*subtitles? (by|created|provided)",
        r"^\s*(amara\.org|subscene|opensubtitles)",
        r"^\s*please (subscribe|like and subscribe)",
        r"^\s*字幕(제작|by|志愿者)",
        r"^\s*ご視聴ありがとうございました",
        # Russian subtitle credits. Observed verbatim from large-v3 over non-speech audio,
        # and ru is a core source language here.
        r"^\s*(редактор|корректор)\s+субтитров",
        r"^\s*субтитры\s+(сделал|подготовил|создал)",
        r"^\s*(♪|♫|\[music\]|\(music\))+\s*$",
    )
]

NO_SPEECH_LIMIT = 0.6
LOGPROB_LIMIT = -1.0
REPEAT_WINDOW = 2  # identical text this many segments back is a stuck decoder

# A decoder stuck in a loop produces text that compresses far better than language does.
# This catches the repetition the fixed-size `REPEAT_WINDOW` misses, which matters because
# the loop is usually longer than two segments — "Редактор субтитров А.Синецкая" twenty
# times over is the canonical case.
COMPRESSION_LIMIT = 2.4
COMPRESSION_MIN_CHARS = 40   # short strings compress unpredictably; don't judge them

# A gap this long between words ends the run of speech used for hallucination filtering.
# Only used for grouping — real sentence boundaries are decided in segment.py.
FILTER_GAP = 0.8


class ASRError(RuntimeError):
    pass


def assert_cuda(cfg: Settings) -> None:
    """Fail loudly rather than fall back to CPU.

    Several components in this stack degrade to CPU instead of erroring, which on a long
    job means shipping something that looks like it worked and took twenty times longer than
    it should have. This is the difference between finding out in seconds and finding out
    after a 40-minute job.
    """
    if not cfg.require_cuda:
        return
    resolved = device_mod.resolve(cfg.device)
    if resolved.startswith("cuda"):
        return
    raise ASRError(
        f"BAG_REQUIRE_CUDA is set but the resolved device is {resolved!r} "
        f"(preference {cfg.device!r}). Set BAG_REQUIRE_CUDA=0 to run on CPU deliberately."
    )


def transcribe(
    job_dir: Path,
    source: SourceInfo,
    *,
    job_id: str,
    cfg: Settings | None = None,
    language: str | None = None,
) -> Timeline:
    cfg = cfg or settings
    assert_cuda(cfg)

    audio = io.decode_to_array(Path(job_dir) / source.wav_path, sr=io.ASR_SR)
    result = _run_whisper(audio, cfg=cfg, language=language)

    source.detected_lang = result["language"]
    source.lang_probability = float(result.get("language_probability", 0.0) or 0.0)
    source.lang_source = "user" if language else "detected"

    timeline = Timeline(job_id=job_id, source=source)
    timeline.words = _collect_words(result["chunks"])
    if not timeline.words:
        raise ASRError("No speech found in this video.")
    timeline.mark_done("asr")
    return timeline


def _model_id(name: str) -> str:
    """Accept the bare checkpoint names the config has always used, as well as full HF ids."""
    return name if "/" in name else f"openai/whisper-{name}"


def _run_whisper(audio, *, cfg: Settings, language: str | None) -> dict:
    import torch  # noqa: PLC0415
    from transformers import pipeline  # noqa: PLC0415 — heavy optional dependency

    device = device_mod.resolve(cfg.device)
    dtype = device_mod.resolve_dtype(device)

    asr = pipeline(
        "automatic-speech-recognition",
        model=_model_id(cfg.whisper_model),
        device=device,
        dtype=dtype,
    )

    # Whisper cannot report a language it was told to use, so detection is only meaningful
    # when the caller did not pin one.
    out = asr(
        audio,
        return_timestamps="word",
        generate_kwargs={"language": language, "task": "transcribe"} if language else
                        {"task": "transcribe"},
    )
    detected = language or _detect_language(asr, audio) or ""
    del torch  # imported only to fail early if the wheel is broken
    return {"chunks": out.get("chunks") or [], "language": detected,
            "language_probability": 1.0 if language else 0.0}


def _detect_language(asr, audio) -> str | None:
    """Best-effort: Whisper's language token for the first window.

    Never fatal — an unknown source language only costs `sbd.py` its per-language
    punctuation weighting, which degrades quality rather than correctness.
    """
    try:
        model, processor = asr.model, asr.tokenizer
        features = asr.feature_extractor(
            audio[: io.ASR_SR * 30], sampling_rate=io.ASR_SR, return_tensors="pt"
        ).input_features.to(model.device, model.dtype)
        detected = model.detect_language(features)[0]
        token = processor.convert_ids_to_tokens(int(detected))
        return token.strip("<|>") or None
    except Exception:  # noqa: BLE001
        return None


def _collect_words(chunks: list[dict]) -> list[Word]:
    """Flatten word chunks into the single word array, dropping hallucinated runs."""
    words: list[Word] = []
    recent: list[str] = []

    for run in _group_runs(chunks):
        text = " ".join(w.text for w in run).strip()
        if _is_hallucination(text, recent):
            continue
        recent.append(text)
        recent[:] = recent[-REPEAT_WINDOW:]
        words.extend(run)

    words.sort(key=lambda w: (w.start, w.end))
    return words


def _group_runs(chunks: list[dict]) -> list[list[Word]]:
    """Split the word stream into runs of continuous speech.

    These are *not* sentences — `segment.py` decides those. They exist only so the
    hallucination filter has a span of text to judge, since a filter cannot tell whether a
    single word was invented but can readily tell that a paragraph was.
    """
    runs: list[list[Word]] = []
    current: list[Word] = []

    for chunk in chunks:
        token = (chunk.get("text") or "").strip()
        if not token:
            continue
        stamp = chunk.get("timestamp") or (None, None)
        start, end = stamp[0], stamp[1]

        # Whisper omits a timestamp when audio is cut mid-word. Interpolating a guess would
        # put a cut point in the wrong place, so the token inherits the neighbouring bound
        # instead — a zero-width word is harmless, a misplaced one is not.
        if start is None:
            start = current[-1].end if current else (runs[-1][-1].end if runs else 0.0)
        if end is None:
            end = start

        start, end = float(start), float(max(end, start))
        if current and start - current[-1].end > FILTER_GAP:
            runs.append(current)
            current = []
        current.append(Word(text=token, start=start, end=end))

    if current:
        runs.append(current)
    return runs


def _is_hallucination(text: str, recent: list[str]) -> bool:
    if not text:
        return True
    if any(p.search(text) for p in HALLUCINATION_PATTERNS):
        return True
    if _is_degenerate(text):
        return True
    # A decoder stuck in a loop repeats itself verbatim.
    return text in recent


def _is_degenerate(text: str) -> bool:
    """Text that compresses far better than language does is a decoder stuck in a loop."""
    raw = text.encode("utf-8")
    if len(raw) < COMPRESSION_MIN_CHARS:
        return False
    return len(raw) / max(1, len(zlib.compress(raw))) > COMPRESSION_LIMIT
