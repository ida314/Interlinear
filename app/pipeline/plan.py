"""Stage 6: Timeline + RenderParams -> RenderPlan.

Pure function. No GPU, no I/O beyond the already-loaded waveform. Every pacing knob is
applied here, which is what makes re-planning cheap enough to drive a live preview.

The hard part is choosing where to cut. See `_boundary_cut`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.audio.dsp import SILENCE_RMS, find_quietest_point, rms_envelope
from app.models.plan import Clip, PlanStats, RenderPlan
from app.models.timeline import RenderParams, Segment, Timeline

# A gap must exceed head_pad + tail_pad by at least this much before we treat it as genuine
# silence and simply trim. Below it, we go looking for a quiet seam instead.
SEAM_MARGIN = 0.10

# Never eat more than this fraction of an adjacent word when hunting for a seam. Forced
# alignment is good but not exact, so a little intrusion is expected and desirable; half a
# word is where it starts being audible.
MAX_WORD_INTRUSION = 0.5
MAX_INTRUSION_SEC = 0.08


@dataclass
class Cut:
    """Where one segment's audio starts and ends in source time."""

    src_in: float
    src_out: float
    hard_in: bool = False   # no quiet seam at the start boundary — widen the fade
    hard_out: bool = False  # ditto at the end boundary


def _intrusion(word_duration: float) -> float:
    return min(MAX_INTRUSION_SEC, MAX_WORD_INTRUSION * word_duration)


def _carries_content(audio: np.ndarray, sr: int, start: float, end: float) -> bool:
    """Does a long non-speech span hold anything worth keeping?

    Long gaps are kept on the assumption that they are music or sound effects. But dead air
    is also a long gap, and preserving several seconds of it — on top of the pause we insert
    for the translation — makes the output drag badly. Measuring is cheap and settles it:
    quiet spans are dropped, spans with energy in them are content and survive.
    """
    a, b = int(max(0.0, start) * sr), int(end * sr)
    span = audio[a:b]
    if len(span) == 0:
        return False
    rms, _ = rms_envelope(span, sr, frame_ms=50.0, hop_ms=25.0)
    # A high percentile, not the mean: a short sound effect inside an otherwise silent span
    # still makes the span worth keeping.
    return float(np.percentile(rms, 90)) > SILENCE_RMS


def _boundary_cut(
    audio: np.ndarray,
    sr: int,
    left_end: float,
    left_dur: float,
    right_start: float,
    right_dur: float,
    params: RenderParams,
) -> tuple[float, float, bool]:
    """Decide the cut between two consecutive sentences.

    Returns (src_out_of_left, src_in_of_right, is_hard_splice).

    Two regimes:

    * **Real silence between them.** Trim to `tail_pad` / `head_pad` and let the inserted
      translation provide the pacing. The two cut points differ; the dead air between is
      dropped by the caller unless it is long enough to count as content.
    * **No silence** — the common case in fast speech, and the one that actually determines
      whether the output sounds professional. Search the seam for the lowest-energy instant
      and cut there, so the splice lands in a stop consonant rather than mid-vowel. Both
      sides share the one cut point, so the audio remains continuous across it.
    """
    gap = right_start - left_end
    if gap >= params.tail_pad + params.head_pad + SEAM_MARGIN:
        return left_end + params.tail_pad, right_start - params.head_pad, False

    lo = left_end - _intrusion(left_dur)
    hi = right_start + _intrusion(right_dur)
    if hi <= lo:
        # Words overlap (or abut exactly). Nothing to search; cut at the reported edge.
        cut = max(0.0, left_end)
        return cut, cut, True

    centre = (left_end + right_start) / 2.0
    cut, rms = find_quietest_point(audio, sr, lo, hi, prefer_centre=centre)
    return cut, cut, rms > SILENCE_RMS


def compute_cuts(
    timeline: Timeline, audio: np.ndarray, sr: int, params: RenderParams | None = None
) -> list[Cut]:
    """Per-segment source ranges, with boundaries shared between neighbours."""
    p = params or timeline.params
    words, segs = timeline.words, timeline.segments
    if not segs:
        return []

    cuts = [Cut(src_in=0.0, src_out=0.0) for _ in segs]
    media_end = timeline.source.duration or (len(audio) / sr)

    # Leading edge.
    first_word = words[segs[0].word_start]
    cuts[0].src_in = max(0.0, first_word.start - p.head_pad)

    # Internal boundaries — each shared by the segment on either side.
    for i in range(len(segs) - 1):
        left = words[segs[i].word_end - 1]
        right = words[segs[i + 1].word_start]
        out_cut, in_cut, hard = _boundary_cut(
            audio, sr, left.end, left.duration, right.start, right.duration, p
        )
        cuts[i].src_out = out_cut
        cuts[i].hard_out = hard
        cuts[i + 1].src_in = in_cut
        cuts[i + 1].hard_in = hard

    # Trailing edge.
    last_word = words[segs[-1].word_end - 1]
    cuts[-1].src_out = min(media_end, last_word.end + p.tail_pad)

    # A degenerate segment (or a bad boundary) must never produce an inverted range.
    for c in cuts:
        if c.src_out < c.src_in:
            c.src_out = c.src_in
    return cuts


def build_plan(
    timeline: Timeline,
    audio: np.ndarray,
    sr: int,
    params: RenderParams | None = None,
    *,
    segment_ids: set[str] | None = None,
) -> RenderPlan:
    """Build the edit list.

    `segment_ids` restricts the plan to a subset of segments — used by the preview endpoint
    to render a two-or-three-sentence excerpt in milliseconds.
    """
    p = params or timeline.params
    segs = timeline.segments
    plan = RenderPlan(job_id=timeline.job_id, sample_rate=sr)
    stats = PlanStats(source_duration=timeline.source.duration or len(audio) / sr)

    if not segs:
        plan.stats = stats
        return plan

    cuts = compute_cuts(timeline, audio, sr, p)
    selected = [
        (seg, cut)
        for seg, cut in zip(segs, cuts)
        if segment_ids is None or seg.id in segment_ids
    ]
    if not selected:
        plan.stats = stats
        return plan

    cursor = 0.0
    media_end = stats.source_duration
    prev_src_end: float | None = None

    def emit(clip: Clip) -> None:
        nonlocal cursor
        if clip.duration <= 0:
            return
        clip.out_start = cursor
        cursor += clip.duration
        plan.clips.append(clip)

    for seg, cut in selected:
        # The span between the previous segment and this one. Short ones are silence made
        # redundant by the translation we are inserting. Long ones are kept only if they
        # actually contain something — music or effects, not dead air.
        bridge_start = 0.0 if prev_src_end is None else prev_src_end
        bridge_len = cut.src_in - bridge_start
        if bridge_len > p.max_source_gap and _carries_content(audio, sr, bridge_start, cut.src_in):
            emit(
                Clip(
                    kind="source",
                    out_start=0.0,
                    duration=bridge_len,
                    src_start=bridge_start,
                    src_end=cut.src_in,
                    fade_in=p.splice_fade if prev_src_end is not None else 0.0,
                    fade_out=p.splice_fade,
                )
            )

        fade_in_len = p.hard_splice_fade if cut.hard_in else p.splice_fade
        fade_out_len = p.hard_splice_fade if cut.hard_out else p.splice_fade
        if cut.hard_out:
            stats.hard_splices += 1

        emit(
            Clip(
                kind="source",
                out_start=0.0,
                duration=cut.src_out - cut.src_in,
                src_start=cut.src_in,
                src_end=cut.src_out,
                segment_id=seg.id,
                fade_in=fade_in_len,
                fade_out=fade_out_len,
            )
        )
        prev_src_end = cut.src_out

        if not _should_translate(seg):
            stats.skipped_segments += 1
            continue

        assert seg.tts is not None  # guaranteed by _should_translate
        emit(Clip(kind="silence", out_start=0.0, duration=p.pre_gap))
        emit(
            Clip(
                kind="tts",
                out_start=0.0,
                duration=seg.tts.duration / max(p.tts_speed, 0.01),
                path=seg.tts.path,
                segment_id=seg.id,
                label=seg.translation,
                fade_in=p.splice_fade,
                fade_out=p.splice_fade,
            )
        )
        emit(Clip(kind="silence", out_start=0.0, duration=p.post_gap))
        stats.translated_segments += 1

    # Outro after the last sentence — kept on the same terms as any other long span.
    if (
        segment_ids is None
        and prev_src_end is not None
        and media_end - prev_src_end > 0.05
        and _carries_content(audio, sr, prev_src_end, media_end)
    ):
        emit(
            Clip(
                kind="source",
                out_start=0.0,
                duration=media_end - prev_src_end,
                src_start=prev_src_end,
                src_end=media_end,
                fade_in=p.splice_fade,
            )
        )

    plan.total_duration = cursor
    stats.output_duration = cursor
    plan.stats = stats
    plan.enforce_invariants()
    return plan


def _should_translate(seg: Segment) -> bool:
    return seg.kind == "speech" and seg.tts is not None and bool(seg.translation)
