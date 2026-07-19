"""The edit list — the contract shared by the audio and video renderers.

Stage 6 (`pipeline/plan.py`) is a *pure function* that turns a `Timeline` plus
`RenderParams` into a `RenderPlan`. Stage 7 merely executes it. Neither renderer re-derives
timing.

Two properties fall out of that split and both matter:

1. **The video renderer is nearly free to add.** A `tts`/`silence` clip is always preceded
   by a `source` clip, so the frame to freeze on is unambiguous — it is the frame at the
   previous clip's `src_end`. `enforce_invariants()` guarantees this.
2. **Pacing knobs never touch the GPU.** Re-planning and re-rendering is pure numpy, which
   is what makes the live preview possible.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ClipKind = Literal["source", "tts", "silence"]


class Clip(BaseModel):
    kind: ClipKind
    out_start: float          # position in the output timeline
    duration: float

    # kind == "source": the half-open range to take from SourceInfo.wav_path
    src_start: float | None = None
    src_end: float | None = None

    # kind == "tts": the synthesised wav, relative to the job directory
    path: str | None = None

    segment_id: str | None = None
    fade_in: float = 0.0
    fade_out: float = 0.0

    # Translation text — burned in as subtitles by the video renderer, and emitted as an
    # .lrc sidecar by the audio renderer. Free to carry, annoying to reconstruct later.
    label: str | None = None

    @property
    def out_end(self) -> float:
        return self.out_start + self.duration


class PlanStats(BaseModel):
    source_duration: float = 0.0
    output_duration: float = 0.0
    translated_segments: int = 0
    skipped_segments: int = 0
    hard_splices: int = 0     # cuts made with no quiet seam available

    @property
    def expansion_ratio(self) -> float:
        return self.output_duration / self.source_duration if self.source_duration else 0.0


class RenderPlan(BaseModel):
    schema_version: int = 1
    job_id: str
    sample_rate: int = 24000
    total_duration: float = 0.0
    clips: list[Clip] = Field(default_factory=list)
    stats: PlanStats = Field(default_factory=PlanStats)

    def enforce_invariants(self, *, tol: float = 1e-6) -> None:
        """Validate the properties both renderers depend on. Raises ValueError.

        Cheap enough to run on every plan; catches an entire class of desync bug at the
        point it is introduced rather than after a 40-minute render.
        """
        if not self.clips:
            raise ValueError("plan has no clips")

        cursor = 0.0
        prev: Clip | None = None
        for i, clip in enumerate(self.clips):
            if clip.duration < 0:
                raise ValueError(f"clip {i}: negative duration {clip.duration}")
            if abs(clip.out_start - cursor) > tol:
                raise ValueError(
                    f"clip {i}: out_start {clip.out_start:.6f} != expected {cursor:.6f} "
                    "(clips must be contiguous)"
                )

            if clip.kind == "source":
                if clip.src_start is None or clip.src_end is None:
                    raise ValueError(f"clip {i}: source clip missing src range")
                if clip.src_end < clip.src_start:
                    raise ValueError(f"clip {i}: inverted src range")
                span = clip.src_end - clip.src_start
                if abs(span - clip.duration) > 1e-3:
                    raise ValueError(
                        f"clip {i}: duration {clip.duration:.6f} != src span {span:.6f}"
                    )
            elif clip.kind == "tts":
                if not clip.path:
                    raise ValueError(f"clip {i}: tts clip missing path")
                # The video renderer freezes the frame at the preceding source clip's
                # src_end. Without that anchor it has nothing to freeze on.
                if prev is None or (prev.kind != "source" and prev.kind != "silence"):
                    raise ValueError(f"clip {i}: tts clip not anchored to a source clip")

            cursor += clip.duration
            prev = clip

        if abs(cursor - self.total_duration) > 1e-3:
            raise ValueError(
                f"total_duration {self.total_duration:.6f} != sum of clips {cursor:.6f}"
            )

    def freeze_frame_time(self, index: int) -> float | None:
        """Source time of the frame the video renderer should hold for clip `index`.

        Walks back to the nearest source clip, so a `silence → tts → silence` run all
        freezes on the same frame.
        """
        if self.clips[index].kind == "source":
            return None
        for j in range(index - 1, -1, -1):
            if self.clips[j].kind == "source":
                return self.clips[j].src_end
        return 0.0
