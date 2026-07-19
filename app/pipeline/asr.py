"""Stage 2: audio -> aligned words.

Deliberately *not* an LLM watching video. This is ASR with forced alignment: faster-whisper
transcribes, then WhisperX aligns the result against the waveform with a wav2vec2 model to
recover per-word timing. A video-understanding model would give worse timestamps and invent
speech that was never said.

Two things this module does that a naive `whisperx.transcribe()` call does not:

* **It keeps only the words.** WhisperX's segment boundaries come from timestamp-token
  sampling and the 30-second decoding window, not from linguistics — they routinely fall
  nowhere near the punctuation the model itself emitted. Sentence boundaries are decided in
  `segment.py` instead, from the word array.
* **It filters hallucinations**, which Whisper produces reliably over music and silence.
"""

from __future__ import annotations

import re
from pathlib import Path

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
        r"^\s*(♪|♫|\[music\]|\(music\))+\s*$",
    )
]

NO_SPEECH_LIMIT = 0.6
LOGPROB_LIMIT = -1.0
REPEAT_WINDOW = 2  # identical text this many segments back is a stuck decoder


class ASRError(RuntimeError):
    pass


def assert_cuda(cfg: Settings) -> None:
    """Fail loudly rather than fall back to CPU.

    CTranslate2 ships no aarch64 CUDA wheels: `pip install` succeeds and then silently runs
    on CPU, roughly 20x slower. On a batch workload that produces something which looks like
    it works. This check is the difference between finding out in seconds and finding out
    after a 40-minute job.
    """
    if not cfg.require_cuda or not cfg.device.startswith("cuda"):
        return
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise ASRError(
            "CUDA requested but unavailable. On aarch64 this usually means CTranslate2 was "
            "installed from a CPU-only wheel — see docs/DEPLOY.md. Set BAG_REQUIRE_CUDA=0 "
            "to run on CPU deliberately."
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

    import whisperx  # noqa: PLC0415 — heavy optional dependency

    audio_path = str(Path(job_dir) / source.wav_path)
    audio = whisperx.load_audio(audio_path)

    model = whisperx.load_model(
        cfg.whisper_model,
        device=cfg.device,
        compute_type=cfg.compute_type,
        # Silero VAD keeps silence out of the decoder, which removes most hallucinations at
        # source rather than filtering them afterwards.
        vad_options={"vad_onset": 0.5, "vad_offset": 0.363} if cfg.vad_filter else None,
    )
    result = model.transcribe(audio, language=language, batch_size=16)
    detected = result.get("language", language or "")

    align_model, meta = whisperx.load_align_model(language_code=detected, device=cfg.device)
    aligned = whisperx.align(
        result["segments"], align_model, meta, audio, cfg.device, return_char_alignments=False
    )

    source.detected_lang = detected
    source.lang_probability = float(result.get("language_probability", 0.0) or 0.0)
    source.lang_source = "user" if language else "detected"

    timeline = Timeline(job_id=job_id, source=source)
    timeline.words = _collect_words(aligned, result["segments"])
    if not timeline.words:
        raise ASRError("No speech found in this video.")
    timeline.mark_done("asr")
    return timeline


def _collect_words(aligned: dict, raw_segments: list[dict]) -> list[Word]:
    """Flatten to a single word array, dropping hallucinations and unalignable tokens."""
    words: list[Word] = []
    recent: list[str] = []

    for seg in aligned.get("segments", []):
        text = (seg.get("text") or "").strip()
        if _is_hallucination(text, seg, recent):
            continue
        recent.append(text)
        recent[:] = recent[-REPEAT_WINDOW:]

        for w in seg.get("words", []):
            # Tokens containing no characters in the alignment dictionary (bare numerals,
            # currency amounts) come back without timing. Interpolating a guess would put a
            # cut point in the wrong place, so they inherit the neighbouring bounds instead.
            start, end = w.get("start"), w.get("end")
            if start is None or end is None:
                if words:
                    start = end = words[-1].end
                else:
                    continue
            token = (w.get("word") or "").strip()
            if token:
                words.append(
                    Word(text=token, start=float(start), end=float(end), score=w.get("score"))
                )

    words.sort(key=lambda w: (w.start, w.end))
    return words


def _is_hallucination(text: str, seg: dict, recent: list[str]) -> bool:
    if not text:
        return True
    if any(p.search(text) for p in HALLUCINATION_PATTERNS):
        return True
    if float(seg.get("no_speech_prob", 0.0) or 0.0) > NO_SPEECH_LIMIT:
        return True
    if seg.get("avg_logprob") is not None and float(seg["avg_logprob"]) < LOGPROB_LIMIT:
        return True
    # A decoder stuck in a loop repeats itself verbatim.
    return text in recent
