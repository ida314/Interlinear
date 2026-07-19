"""Invariants for the edit list.

These are the highest-value tests in the project: a violation here means the output audio
desyncs from the source, or that the phase-2 video renderer has no frame to freeze on. They
run in milliseconds with no GPU.
"""

from __future__ import annotations

import pytest

from app.audio.dsp import SILENCE_RMS, rms_envelope
from app.models.timeline import RenderParams
from app.pipeline.plan import build_plan, compute_cuts
from tests.conftest import SR, add_music, build_timeline


# --- Structural invariants -------------------------------------------------------------


def test_plan_is_contiguous_and_totals_match(simple_timeline):
    tl, audio = simple_timeline
    plan = build_plan(tl, audio, SR)
    plan.enforce_invariants()  # raises on any gap, overlap, or bad source range

    assert plan.clips
    assert plan.total_duration == pytest.approx(sum(c.duration for c in plan.clips))


def test_every_tts_clip_is_anchored_to_a_source_clip(simple_timeline):
    """The video renderer freezes the frame at the preceding source clip's src_end. Without
    that anchor there is nothing to hold on screen."""
    tl, audio = simple_timeline
    plan = build_plan(tl, audio, SR)

    for i, clip in enumerate(plan.clips):
        if clip.kind == "tts":
            assert plan.freeze_frame_time(i) is not None


def test_source_clips_advance_monotonically(simple_timeline):
    tl, audio = simple_timeline
    plan = build_plan(tl, audio, SR)

    src_clips = [c for c in plan.clips if c.kind == "source"]
    for a, b in zip(src_clips, src_clips[1:]):
        assert b.src_start >= a.src_end - 1e-9, "source audio must never play out of order"


def test_each_sentence_produces_source_then_translation(simple_timeline):
    tl, audio = simple_timeline
    plan = build_plan(tl, audio, SR)

    kinds = [c.kind for c in plan.clips]
    # The core pattern: source, gap, translation, gap — repeated per sentence.
    assert "".join(k[0] for k in kinds).count("sts") == 3  # source→silence→tts, x3 sentences
    assert plan.stats.translated_segments == 3


# --- Cut point behaviour ---------------------------------------------------------------


def test_cuts_never_fall_inside_a_word(tight_timeline):
    """The whole reason word-level timestamps are a hard requirement."""
    tl, audio = tight_timeline
    cuts = compute_cuts(tl, audio, SR)

    for cut in cuts:
        for w in tl.words:
            # A cut may intrude slightly into a word (alignment is not exact) but must never
            # land deep inside one.
            for t in (cut.src_in, cut.src_out):
                if w.start < t < w.end:
                    into = min(t - w.start, w.end - t)
                    assert into <= 0.5 * w.duration + 1e-6, (
                        f"cut at {t:.3f} is {into:.3f}s inside word {w.text!r}"
                    )


def test_quiet_seam_is_found_when_no_silence_exists(tight_timeline):
    """With words butted together, the cut should still land in the inter-word trough."""
    tl, audio = tight_timeline
    cuts = compute_cuts(tl, audio, SR)

    boundary = cuts[0].src_out
    assert cuts[1].src_in == boundary, "both sides must share one cut point to stay continuous"

    # The 20ms trough between "mundo" (ends 2.00) and "Como" (starts 2.02).
    assert 1.99 <= boundary <= 2.03

    rms, _ = rms_envelope(audio[int((boundary - 0.005) * SR) : int((boundary + 0.005) * SR)], SR)
    assert float(rms.min()) < SILENCE_RMS


def test_generous_silence_is_trimmed_to_the_pads(simple_timeline):
    """When there is real silence, trim to head/tail pad rather than hunting for a seam."""
    tl, audio = simple_timeline
    p = tl.params
    cuts = compute_cuts(tl, audio, SR)

    # "mundo" ends at 2.0; "Como" starts at 3.0 — a full second of silence.
    assert cuts[0].src_out == pytest.approx(2.0 + p.tail_pad)
    assert cuts[1].src_in == pytest.approx(3.0 - p.head_pad)


def test_hard_splice_widens_the_fade():
    """Continuous speech with no trough: accept the cut but smear it, since a slight blur is
    less objectionable than a click."""
    tl, audio = build_timeline(
        [
            [("uno", 1.0, 1.5)],
            [("dos", 1.5, 2.0)],  # abuts exactly — no seam at all
        ]
    )
    plan = build_plan(tl, audio, SR)

    assert plan.stats.hard_splices >= 1
    first_source = next(c for c in plan.clips if c.kind == "source")
    assert first_source.fade_out == tl.params.hard_splice_fade


# --- Knobs -----------------------------------------------------------------------------


def test_pacing_knobs_change_duration_without_touching_segments(simple_timeline):
    """Pacing is applied entirely at plan time, so the preview can re-render instantly."""
    tl, audio = simple_timeline
    base = build_plan(tl, audio, SR)
    roomy = build_plan(tl, audio, SR, RenderParams(pre_gap=1.0, post_gap=1.0))

    assert roomy.total_duration > base.total_duration
    assert roomy.stats.translated_segments == base.stats.translated_segments


def test_tts_speed_shortens_translation_clips(simple_timeline):
    tl, audio = simple_timeline
    slow = build_plan(tl, audio, SR, RenderParams(tts_speed=1.0))
    fast = build_plan(tl, audio, SR, RenderParams(tts_speed=2.0))

    slow_tts = [c.duration for c in slow.clips if c.kind == "tts"]
    fast_tts = [c.duration for c in fast.clips if c.kind == "tts"]
    assert fast_tts == pytest.approx([d / 2 for d in slow_tts])


def test_untranslated_segments_still_emit_their_audio():
    """Music and filtered hallucinations pass through — the source audio is never dropped
    just because we chose not to translate it."""
    tl, audio = build_timeline(
        [[("la", 1.0, 1.4)], [("hola", 3.0, 3.5)]],
        translated=True,
    )
    tl.segments[0].kind = "music"
    plan = build_plan(tl, audio, SR)

    assert plan.stats.translated_segments == 1
    assert plan.stats.skipped_segments == 1
    assert any(c.segment_id == "seg000" and c.kind == "source" for c in plan.clips)
    assert not any(c.segment_id == "seg000" and c.kind == "tts" for c in plan.clips)


def test_musical_interlude_is_preserved_as_content():
    """A long span with audio in it is content and must survive."""
    tl, audio = build_timeline([[("uno", 1.0, 1.5)], [("dos", 12.0, 12.5)]])
    audio = add_music(audio, 2.0, 11.5)
    plan = build_plan(tl, audio, SR)

    bridges = [c for c in plan.clips if c.kind == "source" and c.segment_id is None]
    assert any(c.duration > 5.0 for c in bridges)


def test_long_dead_air_is_dropped_rather_than_preserved():
    """The same span with nothing in it is just silence. Keeping it — on top of the pause we
    already insert for the translation — makes the output drag."""
    tl, audio = build_timeline([[("uno", 1.0, 1.5)], [("dos", 12.0, 12.5)]])
    plan = build_plan(tl, audio, SR)

    bridges = [c for c in plan.clips if c.kind == "source" and c.segment_id is None]
    assert not any(c.duration > 5.0 for c in bridges)
    assert plan.total_duration < 12.0


def test_preview_subset_renders_only_requested_segments(simple_timeline):
    """Backs the live-knob preview: a two-sentence excerpt, planned in microseconds."""
    tl, audio = simple_timeline
    plan = build_plan(tl, audio, SR, segment_ids={"seg001"})
    plan.enforce_invariants()

    assert plan.stats.translated_segments == 1
    assert plan.total_duration < 10.0


# --- Regression guards -----------------------------------------------------------------


def test_empty_timeline_does_not_explode():
    tl, audio = build_timeline([[("x", 1.0, 1.2)]])
    tl.segments = []
    plan = build_plan(tl, audio, SR)
    assert plan.clips == []


def test_invariant_checker_actually_catches_a_desync(simple_timeline):
    tl, audio = simple_timeline
    plan = build_plan(tl, audio, SR)
    plan.clips[1].duration += 0.5  # simulate a bug that lengthens a clip

    with pytest.raises(ValueError, match="contiguous"):
        plan.enforce_invariants()
