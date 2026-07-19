"""Stage 3: turn the flat word array into sentences.

Boundary *detection* lives in `sbd.py`; this module applies the practical constraints on top
— sentences that are too short to be worth translating get merged, ones too long to sit
comfortably in a single pause get split, and the results are marked up with provenance.

Two modes, per the `segmentation_mode` knob:

* ``sentence`` — real sentence ends from the fused signals in `sbd.py`.
* ``words`` — fixed-size chunks of roughly `max_words_per_chunk`, snapped to the nearest
  real pause. Useful when ASR punctuation is unreliable, and for learners who want shorter
  units than a natural sentence.

Nothing here rewrites text. Segments are index ranges over `Timeline.words`, so
re-segmenting with different knobs can never desynchronise audio from timestamps.
"""

from __future__ import annotations

from typing import Sequence

from app.models.timeline import BoundaryInfo, RenderParams, Segment, Timeline, Word
from app.pipeline import sbd
from app.pipeline.sbd import Arbiter, BoundaryCandidate, BoundaryDetector

# Above this fused score a boundary is treated as settled: short segments closed by one are
# real sentences, not fragments to be absorbed into a neighbour.
CONFIDENT_BOUNDARY = 0.8


def segment_timeline(
    timeline: Timeline,
    *,
    detector: BoundaryDetector | None = None,
    arbiter: Arbiter | None = None,
) -> Timeline:
    """Populate `timeline.segments`. Idempotent — safe to re-run with new knobs."""
    words, p = timeline.words, timeline.params
    if not words:
        timeline.segments = []
        return timeline

    lang = timeline.source.detected_lang
    if detector is None and lang in sbd.LOW_PUNCT_RECALL_LANGS:
        # Whisper routinely omits terminal marks in these scripts, so punctuation-only
        # segmentation silently produces enormous run-on segments rather than failing.
        timeline.warn(
            "segment",
            f"No textual boundary model available and {lang!r} punctuation is unreliable; "
            "sentences will be split mainly on pauses. Install the 'gpu' extras for SaT.",
        )
    candidates = sbd.score_boundaries(words, lang, p, detector)

    if p.segmentation_mode == "words":
        splits = _word_count_splits(words, p)
    else:
        if arbiter is not None and p.llm_arbitration:
            candidates = sbd.apply_arbitration(words, candidates, p, arbiter, lang)
        splits = [c.word_index for c in candidates if c.score >= p.sat_threshold]

    splits = _ensure_terminal(splits, len(words))
    splits = _enforce_hard_pauses(splits, words, p)

    by_index = {c.word_index: c for c in candidates}
    scores = {i: c.score for i, c in by_index.items()}

    ranges = _splits_to_ranges(splits)
    ranges = _merge_short(ranges, words, scores, p)
    ranges = _split_long(ranges, words, candidates, p)

    timeline.segments = [
        _make_segment(i, start, end, by_index.get(end - 1), p, words)
        for i, (start, end) in enumerate(ranges)
    ]
    timeline.rebuild_all_derived()
    _mark_unusable(timeline)
    return timeline


# --- boundary selection ----------------------------------------------------------------


def _ensure_terminal(splits: list[int], n_words: int) -> list[int]:
    out = sorted({s for s in splits if 0 <= s < n_words})
    if not out or out[-1] != n_words - 1:
        out.append(n_words - 1)
    return out


def _enforce_hard_pauses(splits: list[int], words: Sequence[Word], p: RenderParams) -> list[int]:
    """A long silence always ends a sentence, whatever the text model thought.

    Speakers do not pause for two seconds mid-clause; when they appear to, the transcript is
    usually wrong. This also stops a missed boundary from producing one enormous run-on
    segment covering an entire pause.
    """
    forced = {
        i
        for i in range(len(words) - 1)
        if words[i + 1].start - words[i].end >= p.never_merge_across_gap
    }
    return sorted(set(splits) | forced)


def _word_count_splits(words: Sequence[Word], p: RenderParams) -> list[int]:
    """Fixed-size chunks, nudged to the largest nearby pause.

    Snapping matters: a blind split every N words lands mid-phrase about as often as not,
    whereas the biggest gap in a small search window is almost always a real breath.
    """
    target = max(1, p.max_words_per_chunk)
    window = max(1, target // 3)
    splits: list[int] = []
    cursor = 0
    while cursor + target < len(words):
        ideal = cursor + target - 1
        lo, hi = max(cursor, ideal - window), min(len(words) - 2, ideal + window)
        best = max(
            range(lo, hi + 1),
            key=lambda i: words[i + 1].start - words[i].end,
            default=ideal,
        )
        splits.append(best)
        cursor = best + 1
    return splits


def _splits_to_ranges(splits: Sequence[int]) -> list[tuple[int, int]]:
    ranges, start = [], 0
    for s in splits:
        ranges.append((start, s + 1))
        start = s + 1
    return [r for r in ranges if r[1] > r[0]]


# --- constraints -----------------------------------------------------------------------


def _duration(words: Sequence[Word], rng: tuple[int, int]) -> float:
    return words[rng[1] - 1].end - words[rng[0]].start


def _merge_short(
    ranges: list[tuple[int, int]],
    words: Sequence[Word],
    scores: dict[int, float],
    p: RenderParams,
) -> list[tuple[int, int]]:
    """Absorb fragments into their neighbour.

    Only *ambiguous* boundaries are candidates for merging. A short segment closed by a
    confident boundary — an unmistakable full stop, say — is a real sentence and is left
    alone; "Hello there." is brief but complete, and gluing it onto the next sentence would
    be worse than translating it on its own. Genuinely trivial utterances are handled later
    by `_mark_unusable`, which suppresses the translation while keeping the audio.
    """
    if len(ranges) <= 1:
        return ranges

    def mergeable(rng: tuple[int, int]) -> bool:
        short = (rng[1] - rng[0]) < p.min_words or _duration(words, rng) < p.min_duration
        return short and scores.get(rng[1] - 1, 0.0) < CONFIDENT_BOUNDARY

    out: list[tuple[int, int]] = []
    for rng in ranges:
        if not out:
            out.append(rng)
            continue

        prev = out[-1]
        gap = words[rng[0]].start - words[prev[1] - 1].end
        if mergeable(prev) and gap < p.never_merge_across_gap:
            out[-1] = (prev[0], rng[1])
        else:
            out.append(rng)

    # The tail has no successor to absorb it, so fold it backwards if the pause allows.
    # Its own closing boundary is always end-of-media, so confidence has to be judged on the
    # boundary that separates it from its predecessor instead.
    if len(out) > 1:
        last, prev = out[-1], out[-2]
        gap = words[last[0]].start - words[prev[1] - 1].end
        short = (last[1] - last[0]) < p.min_words or _duration(words, last) < p.min_duration
        weak_split = scores.get(prev[1] - 1, 0.0) < CONFIDENT_BOUNDARY
        if short and weak_split and gap < p.never_merge_across_gap:
            out[-2:] = [(prev[0], last[1])]
    return out


def _split_long(
    ranges: list[tuple[int, int]],
    words: Sequence[Word],
    candidates: Sequence[BoundaryCandidate],
    p: RenderParams,
) -> list[tuple[int, int]]:
    """Break up anything too long to sit inside a single pause.

    Prefers the best-scoring internal boundary; falls back to the longest pause. Recurses so
    a very long run-on is divided repeatedly rather than merely halved.
    """
    by_index = {c.word_index: c.score for c in candidates}
    out: list[tuple[int, int]] = []
    queue = list(ranges)

    while queue:
        rng = queue.pop(0)
        start, end = rng
        too_long = _duration(words, rng) > p.max_duration or (end - start) > p.max_words
        if not too_long or (end - start) < 2 * p.min_words:
            out.append(rng)
            continue

        interior = range(start + p.min_words - 1, end - p.min_words)
        if not interior:
            out.append(rng)
            continue

        pivot = max(
            interior,
            key=lambda i: (by_index.get(i, 0.0), words[i + 1].start - words[i].end),
        )
        queue.insert(0, (pivot + 1, end))
        queue.insert(0, (start, pivot + 1))

    return sorted(out)


# --- assembly --------------------------------------------------------------------------


def _make_segment(
    index: int,
    start: int,
    end: int,
    candidate: BoundaryCandidate | None,
    p: RenderParams,
    words: Sequence[Word],
) -> Segment:
    boundary = None
    if candidate is not None:
        boundary = BoundaryInfo(
            source=candidate.source,
            score=candidate.score,
            sat_prob=candidate.sat_prob,
            pause_after=candidate.pause_after,
            has_terminal_punct=candidate.has_terminal_punct,
            llm_reviewed=candidate.source == "llm",
        )
    return Segment(index=index, word_start=start, word_end=end, boundary=boundary)


def _mark_unusable(timeline: Timeline) -> None:
    """Flag fragments that survived merging as skip.

    They keep their source audio in the output — only the translation is suppressed.
    Translating "Yeah." into its own pause interrupts the flow for no benefit.
    """
    p = timeline.params
    for seg in timeline.segments:
        if not seg.text.strip():
            seg.kind = "skip"
        elif seg.word_count < p.min_words and seg.duration < p.min_duration:
            seg.kind = "skip"
