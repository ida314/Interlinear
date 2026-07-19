"""Core data model for a job.

The central invariant: **`Timeline.words` is the source of truth for text and timing.**
ASR produces it once; nothing downstream is permitted to rewrite it. Sentence segmentation
produces *index ranges* over this array, never new text. That is what keeps every word
bound to its forced-alignment timestamp, and it is why re-segmenting with different knobs
can never desynchronise the audio.

`Segment` therefore stores `word_start`/`word_end` as authoritative, plus denormalised
`text`/`start`/`end` so that `timeline.json` stays readable when you open it to debug.
Call `rebuild_derived()` after changing any index.
"""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field, computed_field

SegmentKind = Literal["speech", "music", "noise", "skip"]

# How a sentence boundary was decided. Recorded per segment so that bad segmentation can be
# diagnosed after the fact, and so the LLM arbitration pass can be evaluated against the
# cheaper signals it is meant to improve on.
BoundarySource = Literal[
    "punctuation",    # terminal punctuation from ASR
    "sat",            # wtpsplit/SaT textual boundary probability
    "pause",          # acoustic gap between words
    "fused",          # combined score crossed threshold
    "llm",            # LLM arbitration overrode the fused score
    "forced_max",     # no good boundary found; split to respect max duration/length
    "eof",            # end of media
]


class Word(BaseModel):
    """One aligned token. For zh/ja this is a single character — WhisperX aligns CJK
    per character since there is no whitespace to tokenise on."""

    text: str
    start: float
    end: float
    score: float | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


class BoundaryInfo(BaseModel):
    """Provenance for the sentence end that closes a segment."""

    source: BoundarySource
    score: float | None = None          # fused confidence, 0..1
    sat_prob: float | None = None       # textual model's raw probability
    pause_after: float | None = None    # seconds of silence following the last word
    has_terminal_punct: bool = False
    llm_reviewed: bool = False
    llm_agreed: bool | None = None      # None when the LLM never looked at this boundary


class TTSClip(BaseModel):
    """A synthesised translation.

    Always synthesised at speed 1.0. The `tts_speed` knob is applied as an atempo filter at
    render time, so changing playback speed never requires re-running the GPU.
    """

    path: str            # relative to the job directory
    duration: float      # at speed 1.0
    voice: str
    lang: str
    text_sha: str        # sha256 of (text, voice, lang) — the cache key


class Segment(BaseModel):
    """A sentence: a half-open index range `[word_start, word_end)` over `Timeline.words`."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    index: int
    word_start: int
    word_end: int

    # Denormalised from the word range for readability. Authoritative source is the indices.
    text: str = ""
    start: float = 0.0
    end: float = 0.0

    kind: SegmentKind = "speech"
    boundary: BoundaryInfo | None = None
    speaker: str | None = None          # reserved for diarisation; unused in v1

    # ASR confidence, carried through so the hallucination filter can act on it.
    no_speech_prob: float | None = None
    avg_logprob: float | None = None

    translation: str | None = None
    translation_model: str | None = None
    tts: TTSClip | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def word_count(self) -> int:
        return self.word_end - self.word_start

    def words_from(self, words: list[Word]) -> list[Word]:
        return words[self.word_start : self.word_end]

    def rebuild_derived(self, words: list[Word], *, join: str = " ") -> None:
        """Recompute text/start/end from the authoritative index range.

        `join` is "" for languages without inter-word spacing (zh/ja), where each Word is a
        character.
        """
        span = self.words_from(words)
        if not span:
            self.text, self.start, self.end = "", 0.0, 0.0
            return
        self.text = join.join(w.text for w in span).strip()
        self.start = span[0].start
        self.end = span[-1].end

    @property
    def is_translatable(self) -> bool:
        return self.kind == "speech" and bool(self.text.strip())


class SourceInfo(BaseModel):
    url: str
    video_id: str = ""
    title: str = ""
    uploader: str | None = None
    duration: float = 0.0

    detected_lang: str = ""
    lang_probability: float = 0.0
    lang_source: Literal["detected", "user", "captions"] = "detected"

    audio_path: str = ""            # original download, kept for provenance
    wav_path: str = ""              # 24kHz mono s16 — the render source of truth
    video_path: str | None = None   # only fetched when video output is requested


class RenderParams(BaseModel):
    """Every user-facing knob.

    Knobs marked *cheap* affect only stages 6-7 (pure numpy, no GPU), so they can be
    re-applied instantly and drive the live preview. Knobs marked *expensive* change
    sentence boundaries or synthesis and require re-running GPU stages.
    """

    target_lang: str = "en"                        # expensive (re-translate)

    # --- Segmentation (expensive) ---
    segmentation_mode: Literal["sentence", "words"] = "sentence"
    max_words_per_chunk: int = 12                  # only used when mode == "words"
    min_words: int = 2                             # drop shorter fragments ("Yeah.")
    min_duration: float = 1.2                      # merge forward below this
    max_duration: float = 15.0                     # force a split above this
    max_words: int = 40                            # force a split above this
    never_merge_across_gap: float = 2.0            # a long pause always ends a sentence

    # --- Sentence boundary detection (expensive) ---
    sat_threshold: float = 0.5                     # SaT probability to call a boundary
    pause_weight: float = 0.35                     # weight of the acoustic gap in the fusion
    pause_saturation: float = 0.5                  # gap (s) treated as maximal evidence
    llm_arbitration: bool = True                   # LLM resolves only ambiguous boundaries
    llm_gray_zone: tuple[float, float] = (0.35, 0.65)  # fused scores sent to the LLM

    # --- Voice (expensive: re-synthesise) ---
    voice: str = "af_heart"

    # --- Pacing (cheap: re-render only) ---
    tts_speed: float = 1.0
    pre_gap: float = 0.25          # silence between source sentence and translation
    post_gap: float = 0.35         # silence after translation before source resumes
    head_pad: float = 0.10         # audio kept before the first word of a sentence
    tail_pad: float = 0.15         # audio kept after the last word of a sentence
    splice_fade: float = 0.010     # fade applied at every cut
    hard_splice_fade: float = 0.025  # wider fade when no quiet seam exists
    keep_source_gaps: bool = True
    max_source_gap: float = 1.5    # compress original silence longer than this
    match_loudness: bool = True    # normalise TTS to the source's integrated loudness

    @property
    def is_cjk_target(self) -> bool:
        return self.target_lang in {"zh", "ja", "ko"}


class JobWarning(BaseModel):
    stage: str
    message: str
    segment_id: str | None = None


class Timeline(BaseModel):
    schema_version: int = 1
    job_id: str
    source: SourceInfo
    params: RenderParams = Field(default_factory=RenderParams)

    words: list[Word] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)

    stages_done: list[str] = Field(default_factory=list)
    warnings: list[JobWarning] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def speech_segment_count(self) -> int:
        return sum(1 for s in self.segments if s.kind == "speech")

    def word_join(self) -> str:
        """Inter-word separator for the *source* language."""
        return "" if self.source.detected_lang in {"zh", "ja", "th", "lo", "my"} else " "

    def rebuild_all_derived(self) -> None:
        join = self.word_join()
        for i, seg in enumerate(self.segments):
            seg.index = i
            seg.rebuild_derived(self.words, join=join)

    def mark_done(self, stage: str) -> None:
        if stage not in self.stages_done:
            self.stages_done.append(stage)

    def warn(self, stage: str, message: str, segment_id: str | None = None) -> None:
        self.warnings.append(JobWarning(stage=stage, message=message, segment_id=segment_id))
