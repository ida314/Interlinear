"""Stage 5: synthesise the translations.

Kokoro-82M by default: Apache-2.0, ~0.5GB, comfortably faster than realtime, and it has a
native speed parameter. XTTS-v2 and F5-TTS sound good but ship non-commercial weights
(CPML / CC-BY-NC), and Coqui is defunct so there is nobody left to license from — worth
avoiding on a system you may want to share.

**Everything is synthesised at speed 1.0.** The `tts_speed` knob is applied at render time
via atempo. That keeps the cache independent of playback speed, so changing the speed slider
is a pure-numpy re-render instead of a GPU pass — which is what makes the live preview
possible.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from app.audio import io
from app.config import Settings, settings
from app.models.timeline import Timeline, TTSClip

# Kokoro voice per target language. Languages absent here need a different engine — notably
# Russian, which Kokoro does not cover.
KOKORO_VOICES = {
    "en": "af_heart", "es": "ef_dora", "fr": "ff_siwis", "it": "if_sara",
    "pt": "pf_dora", "ja": "jf_alpha", "zh": "zf_xiaobei", "hi": "hf_alpha",
}


class TTSError(RuntimeError):
    pass


class TTSEngine(Protocol):
    sample_rate: int

    def voices(self, lang: str) -> list[str]: ...
    def synth(self, text: str, voice: str, lang: str) -> np.ndarray: ...


class KokoroEngine:
    """Kokoro via its PyTorch path.

    Not ONNX Runtime: aarch64 CUDA builds of onnxruntime-gpu are patchy, and several Kokoro
    wrappers quietly downgrade to CPU when they detect aarch64. At 82M parameters the
    PyTorch path is fast enough that there is nothing to gain by fighting that.
    """

    sample_rate = 24000

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._pipelines: dict[str, object] = {}

    def _pipeline(self, lang: str):
        if lang not in self._pipelines:
            from kokoro import KPipeline  # noqa: PLC0415

            self._pipelines[lang] = KPipeline(lang_code=lang[:1] or "a", device=self.device)
        return self._pipelines[lang]

    def voices(self, lang: str) -> list[str]:
        voice = KOKORO_VOICES.get(lang)
        return [voice] if voice else []

    def synth(self, text: str, voice: str, lang: str) -> np.ndarray:
        pipeline = self._pipeline(lang)
        chunks = [audio for _, _, audio in pipeline(text, voice=voice, speed=1.0)]
        if not chunks:
            raise TTSError(f"Kokoro produced no audio for {text[:60]!r}")
        return np.concatenate([np.asarray(c, dtype=np.float32) for c in chunks])


def get_engine(lang: str, cfg: Settings | None = None) -> TTSEngine:
    cfg = cfg or settings
    if lang in KOKORO_VOICES:
        return KokoroEngine(device=cfg.device)
    raise TTSError(
        f"No TTS engine configured for {lang!r}. Kokoro covers "
        f"{', '.join(sorted(KOKORO_VOICES))}; add a Piper or Chatterbox engine for others."
    )


def clip_key(text: str, voice: str, lang: str) -> str:
    """Cache key. Deliberately excludes speed — speed is a render-time concern.

    This is also what makes re-segmentation cheap: change the segmentation knob and only the
    sentences whose text actually changed need re-synthesising.
    """
    return hashlib.sha256(f"{lang}\x00{voice}\x00{text}".encode()).hexdigest()[:16]


def synthesize_timeline(
    timeline: Timeline,
    job_dir: Path,
    *,
    engine: TTSEngine | None = None,
    cfg: Settings | None = None,
    progress: Callable[[float], None] | None = None,
) -> Timeline:
    cfg = cfg or settings
    lang = timeline.params.target_lang
    engine = engine or get_engine(lang, cfg)

    voice = timeline.params.voice
    if voice not in engine.voices(lang):
        available = engine.voices(lang)
        if not available:
            raise TTSError(f"No voice available for {lang!r}")
        voice = available[0]

    tts_dir = Path(job_dir) / "tts"
    tts_dir.mkdir(parents=True, exist_ok=True)

    pending = [s for s in timeline.segments if s.is_translatable and s.translation]
    for i, seg in enumerate(pending):
        key = clip_key(seg.translation, voice, lang)
        path = tts_dir / f"{key}.wav"

        if not path.exists():
            try:
                audio = engine.synth(seg.translation, voice, lang)
            except Exception as exc:  # noqa: BLE001
                timeline.warn("tts", f"Synthesis failed: {exc}", seg.id)
                continue
            io.write_wav(path, audio, engine.sample_rate)
        else:
            audio, _ = io.read_wav(path)

        seg.tts = TTSClip(
            path=f"tts/{path.name}",
            duration=len(audio) / engine.sample_rate,
            voice=voice,
            lang=lang,
            text_sha=key,
        )
        if progress:
            progress((i + 1) / max(1, len(pending)))

    timeline.mark_done("tts")
    return timeline


def prune_cache(job_dir: Path, timeline: Timeline) -> int:
    """Drop synthesised clips no longer referenced after a re-segmentation."""
    tts_dir = Path(job_dir) / "tts"
    if not tts_dir.exists():
        return 0
    live = {s.tts.path.split("/")[-1] for s in timeline.segments if s.tts}
    removed = 0
    for path in tts_dir.glob("*.wav"):
        if path.name not in live:
            path.unlink()
            removed += 1
    return removed
