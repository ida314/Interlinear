"""Stage invalidation and timeline persistence.

The invalidation rules are what make knob changes cheap. Getting them wrong is expensive in
both directions: too eager and every slider drag triggers a GPU pass, too lazy and the
output silently stops matching the settings shown in the UI.
"""

from __future__ import annotations

from app.models.timeline import RenderParams, SourceInfo, Timeline, Word
from app.pipeline.runner import invalidated_from, load, save


def test_pacing_knobs_invalidate_nothing():
    """These feed the live preview, so they must never trigger a model run."""
    base = RenderParams()
    for field, value in [
        ("tts_speed", 1.4), ("pre_gap", 0.6), ("post_gap", 0.1), ("head_pad", 0.2),
        ("tail_pad", 0.3), ("max_source_gap", 3.0), ("splice_fade", 0.02),
        ("match_loudness", False),
    ]:
        assert invalidated_from(base, base.model_copy(update={field: value})) is None, field


def test_segmentation_knobs_invalidate_from_segment():
    base = RenderParams()
    for field, value in [
        ("segmentation_mode", "words"), ("max_words_per_chunk", 6), ("min_words", 4),
        ("sat_threshold", 0.7), ("pause_weight", 0.5), ("llm_arbitration", False),
    ]:
        assert invalidated_from(base, base.model_copy(update={field: value})) == "segment", field


def test_target_language_invalidates_from_translate():
    base = RenderParams()
    assert invalidated_from(base, base.model_copy(update={"target_lang": "fr"})) == "translate"


def test_voice_invalidates_only_tts():
    base = RenderParams()
    assert invalidated_from(base, base.model_copy(update={"voice": "am_michael"})) == "tts"


def test_earliest_stage_wins_when_several_knobs_change():
    base = RenderParams()
    changed = base.model_copy(update={"voice": "am_michael", "min_words": 5, "target_lang": "de"})
    assert invalidated_from(base, changed) == "segment"


def test_identical_params_invalidate_nothing():
    base = RenderParams()
    assert invalidated_from(base, base.model_copy()) is None


def test_timeline_round_trips_through_disk(tmp_path):
    tl = Timeline(
        job_id="abc",
        source=SourceInfo(url="u", detected_lang="es", duration=10.0),
        words=[Word(text="hola", start=0.0, end=0.5)],
    )
    tl.mark_done("asr")
    tl.warn("asr", "something odd")
    save(tl, tmp_path)

    loaded = load(tmp_path)
    assert loaded is not None
    assert loaded.job_id == "abc"
    assert loaded.words[0].text == "hola"
    assert loaded.stages_done == ["asr"]
    assert loaded.warnings[0].message == "something odd"


def test_loading_a_missing_timeline_returns_none(tmp_path):
    assert load(tmp_path) is None


def test_mark_done_is_idempotent():
    tl = Timeline(job_id="a", source=SourceInfo(url="u"))
    tl.mark_done("asr")
    tl.mark_done("asr")
    assert tl.stages_done == ["asr"]
