"""Sentence segmentation.

Uses stub detectors and arbiters throughout, so the constraint logic is tested independently
of whether the SaT model is installed.
"""

from __future__ import annotations

import pytest

from app.models.timeline import RenderParams, SourceInfo, Timeline, Word
from app.pipeline import sbd
from app.pipeline.sbd import PunctuationPauseDetector, has_terminal_punct
from app.pipeline.segment import segment_timeline


def make_timeline(
    tokens: list[tuple[str, float, float]], *, lang: str = "es", **knobs
) -> Timeline:
    words = [Word(text=t, start=s, end=e) for t, s, e in tokens]
    return Timeline(
        job_id="t",
        source=SourceInfo(url="u", detected_lang=lang, duration=words[-1].end + 1 if words else 1),
        params=RenderParams(**knobs),
        words=words,
    )


LONG_PAUSE = 2.5  # exceeds the default never_merge_across_gap of 2.0s


def evenly(texts: list[str], *, dur: float = 0.4, gap: float = 0.05, start: float = 0.0):
    """Lay tokens out end to end. Prefix a token with '|' to put a long pause before it."""
    out, t = [], start
    for text in texts:
        if text.startswith("|"):
            text, t = text[1:], t + LONG_PAUSE
        out.append((text, t, t + dur))
        t += dur + gap
    return out


class ScriptedDetector:
    """Returns preset probabilities so fusion can be tested without a model."""

    def __init__(self, probs: list[float]):
        self.probs = probs

    def probabilities(self, words, lang):
        return self.probs + [0.0] * (len(words) - len(self.probs))


# --- punctuation heuristics ------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("mundo.", True), ("mundo!", True), ("¿qué?", True), ("世界。", True),
        ("mundo", False), ("Dr.", False), ("etc.", False), ("J.", False),
        ('mundo."', True), ("3.", True),
    ],
)
def test_terminal_punctuation_detection(text, expected):
    assert has_terminal_punct(text) is expected


# --- sentence mode ---------------------------------------------------------------------


def test_punctuation_splits_sentences():
    tl = make_timeline(evenly(["Hola", "mundo.", "Como", "estas?", "Bien", "gracias."]))
    segment_timeline(tl)

    assert [s.text for s in tl.segments] == ["Hola mundo.", "Como estas?", "Bien gracias."]


def test_long_pause_forces_a_boundary_without_punctuation():
    """Speakers do not pause for two seconds mid-clause; when the transcript says they did,
    the punctuation is what is wrong."""
    tl = make_timeline(evenly(["uno", "dos", "|tres", "cuatro"]), min_words=1, min_duration=0.0)
    segment_timeline(tl)

    assert len(tl.segments) == 2
    assert tl.segments[0].text == "uno dos"


def test_pause_alone_does_not_split_mid_sentence():
    """A short breath is not a sentence end — otherwise every hesitation becomes a pause for
    a translation."""
    tl = make_timeline(evenly(["uno", "dos", "tres", "cuatro", "cinco."], gap=0.25))
    segment_timeline(tl)

    assert len(tl.segments) == 1


def test_fusion_lets_pause_reinforce_a_weak_text_signal():
    words = make_timeline(evenly(["uno", "dos", "|tres"])).words
    p = RenderParams(pause_weight=0.35, pause_saturation=0.5)

    weak = sbd.score_boundaries(words, "es", p, ScriptedDetector([0.0, 0.45, 0.0]))
    assert weak[1].score > weak[0].score
    assert weak[1].pause_after > 0.9


# --- constraints -----------------------------------------------------------------------


def test_ambiguous_short_fragment_merges_forward():
    """A weak boundary that leaves a stub behind should be dissolved, not kept."""
    tl = make_timeline(evenly(["uno", "dos", "tres", "cuatro", "cinco."]), min_duration=1.2)
    segment_timeline(tl, detector=ScriptedDetector([0.0, 0.55, 0.0, 0.0, 1.0]))

    assert len(tl.segments) == 1


def test_short_sentence_with_a_clear_full_stop_is_not_swallowed():
    """"Hola mundo." is brief but complete. Merging it into the next sentence because it
    fell under min_duration would be worse than translating it alone."""
    tl = make_timeline(evenly(["Hola", "mundo.", "Como", "estas?"]), min_duration=1.2)
    segment_timeline(tl)

    assert [s.text for s in tl.segments] == ["Hola mundo.", "Como estas?"]


def test_fragment_isolated_by_pauses_is_kept_but_not_translated():
    """"Yeah." surrounded by silence cannot be merged, so it is marked skip — its audio still
    plays, it just does not earn a translation pause."""
    tl = make_timeline(evenly(["Hola", "mundo.", "|Si.", "|Adios", "amigo."]), min_words=2)
    segment_timeline(tl)

    fragment = next(s for s in tl.segments if s.text == "Si.")
    assert fragment.kind == "skip"
    assert all(s.kind == "speech" for s in tl.segments if s.text != "Si.")


def test_run_on_segment_is_split_at_the_best_internal_boundary():
    tl = make_timeline(evenly([f"w{i}" for i in range(40)]), max_words=12, min_words=2)
    segment_timeline(tl)

    assert len(tl.segments) >= 4
    assert all(s.word_count <= 12 for s in tl.segments)


def test_max_duration_forces_a_split():
    tl = make_timeline(evenly([f"w{i}" for i in range(30)], dur=1.0), max_duration=8.0, max_words=99)
    segment_timeline(tl)

    assert all(s.duration <= 8.0 + 1.5 for s in tl.segments)


# --- words mode (the segmentation_mode knob) -------------------------------------------


def test_words_mode_produces_fixed_size_chunks():
    tl = make_timeline(
        evenly([f"w{i}" for i in range(30)]), segmentation_mode="words", max_words_per_chunk=6
    )
    segment_timeline(tl)

    assert len(tl.segments) >= 4
    assert all(2 <= s.word_count <= 9 for s in tl.segments)


def test_words_mode_snaps_chunks_to_real_pauses():
    """A blind split every N words lands mid-phrase; snapping to the nearest breath does
    not."""
    tokens = evenly(["a", "b", "c", "d", "|e", "f", "g", "h", "i", "j"])
    tl = make_timeline(tokens, segmentation_mode="words", max_words_per_chunk=5, min_words=1)
    segment_timeline(tl)

    # The pause before "e" is the only real gap, so the first chunk should end at "d".
    assert tl.segments[0].text == "a b c d"


# --- Russian ---------------------------------------------------------------------------


def test_russian_sentences_split_on_punctuation():
    tl = make_timeline(
        evenly(["Привет,", "меня", "зовут", "Анна.", "Ты", "готов?"]), lang="ru"
    )
    segment_timeline(tl)

    assert [s.text for s in tl.segments] == ["Привет, меня зовут Анна.", "Ты готов?"]


@pytest.mark.parametrize(
    "token,expected",
    [
        ("г.", False),      # год / город — extremely common mid-sentence
        ("гг.", False),     # годы
        ("т.д.", False),    # и так далее
        ("т.е.", False),    # то есть
        ("ул.", False),     # улица
        ("см.", False),     # смотри / сантиметр
        ("млн.", False),
        ("проф.", False),
        ("А.", False),      # initial in a name
        ("Анна.", True),
        ("хорошо.", True),
        ("готов?", True),
    ],
)
def test_russian_abbreviations_are_not_sentence_ends(token, expected):
    """Russian writes these mid-sentence constantly. Treating "в 1995 г." as a boundary
    would chop a clause in half and hand the translator a fragment."""
    assert has_terminal_punct(token) is expected


def test_russian_abbreviation_does_not_split_a_clause():
    tl = make_timeline(
        evenly(["Это", "было", "в", "1995", "г.", "и", "продолжалось", "долго."]), lang="ru"
    )
    segment_timeline(tl)

    assert len(tl.segments) == 1


def test_russian_is_treated_as_a_strong_punctuation_language():
    """Whisper punctuates Russian well — it hears the prosody. So a terminal mark should be
    close to decisive on its own."""
    assert "ru" in sbd.STRONG_PUNCT_LANGS

    words = make_timeline(evenly(["Привет,", "мир."]), lang="ru").words
    candidates = sbd.score_boundaries(words, "ru", RenderParams())
    assert candidates[1].score > 0.9


def test_russian_words_join_with_spaces():
    assert sbd.word_join("ru") == " "


# --- CJK -------------------------------------------------------------------------------


def test_cjk_words_are_characters_and_join_without_spaces():
    """WhisperX aligns zh/ja per character. Text must reassemble without spaces, or every
    translation prompt is malformed."""
    tl = make_timeline(evenly(list("你好世界。再见了。")), lang="zh", min_words=2)
    segment_timeline(tl)

    assert tl.segments[0].text == "你好世界。"
    assert tl.segments[1].text == "再见了。"


# --- LLM arbitration -------------------------------------------------------------------


def test_only_ambiguous_boundaries_reach_the_arbiter():
    """The LLM is expensive; it should see the handful of genuinely unclear positions, not
    every word."""
    words = make_timeline(evenly(["uno", "dos", "tres", "cuatro."])).words
    p = RenderParams(llm_gray_zone=(0.35, 0.65))
    candidates = sbd.score_boundaries(words, "es", p, ScriptedDetector([0.1, 0.5, 0.9, 1.0]))

    assert sbd.ambiguous_indices(candidates, p.llm_gray_zone) == [1]


def test_arbiter_decision_moves_the_score_across_the_threshold():
    words = make_timeline(evenly(["uno", "dos", "tres", "cuatro."])).words
    p = RenderParams(llm_gray_zone=(0.35, 0.65), sat_threshold=0.5)
    candidates = sbd.score_boundaries(words, "es", p, ScriptedDetector([0.1, 0.5, 0.9, 1.0]))

    confirmed = sbd.apply_arbitration(words, list(candidates), p, lambda w, i, l: set(i), "es")
    assert confirmed[1].score >= p.sat_threshold
    assert confirmed[1].source == "llm"

    rejected = sbd.apply_arbitration(words, list(candidates), p, lambda w, i, l: set(), "es")
    assert rejected[1].score < p.sat_threshold


def test_arbiter_is_skipped_entirely_when_disabled():
    tl = make_timeline(evenly(["uno", "dos.", "tres", "cuatro."]), llm_arbitration=False)
    calls = []

    def spy(words, indices, lang):
        calls.append(indices)
        return set(indices)

    segment_timeline(tl, arbiter=spy)
    assert calls == []


# --- robustness ------------------------------------------------------------------------


def test_empty_word_list_is_safe():
    tl = make_timeline([])
    tl.words = []
    segment_timeline(tl)
    assert tl.segments == []


def test_segments_tile_the_word_array_exactly():
    """The invariant that keeps text bound to timestamps: every word belongs to exactly one
    segment, in order, with nothing dropped or duplicated."""
    tl = make_timeline(evenly(["Hola", "mundo.", "|Como", "estas?", "Bien.", "|Adios."]))
    segment_timeline(tl)

    covered = [i for s in tl.segments for i in range(s.word_start, s.word_end)]
    assert covered == list(range(len(tl.words)))


def test_resegmenting_with_new_knobs_is_idempotent():
    tl = make_timeline(evenly([f"w{i}" for i in range(20)]))
    segment_timeline(tl)
    first = [(s.word_start, s.word_end) for s in segment_timeline(tl).segments]
    second = [(s.word_start, s.word_end) for s in segment_timeline(tl).segments]
    assert first == second


def test_fallback_detector_needs_no_optional_dependencies():
    assert PunctuationPauseDetector().probabilities(
        [Word(text="Hola.", start=0, end=1)], "es"
    ) == [1.0]
