"""Word assembly and hallucination filtering.

None of this loads a model: the logic worth testing is what happens to Whisper's output
after it arrives, which is pure. The real decoder is exercised by `-m gpu` tests.
"""

from __future__ import annotations

import pytest

from app.pipeline.asr import (
    ASRError,
    _collect_words,
    _group_runs,
    _is_degenerate,
    _is_hallucination,
    _model_id,
    assert_cuda,
)

# Verbatim from large-v3 over non-speech audio. Russian subtitle credits are a real
# hallucination class for the ru source workload, not a hypothetical one.
RU_CREDITS = "Редактор субтитров А.Синецкая Корректор А.Синецкая " * 6


def chunk(text: str, start, end) -> dict:
    return {"text": text, "timestamp": (start, end)}


def test_words_carry_their_timestamps_through():
    words = _collect_words([chunk("привет", 0.0, 0.5), chunk("мир", 0.6, 1.0)])
    assert [(w.text, w.start, w.end) for w in words] == [
        ("привет", 0.0, 0.5),
        ("мир", 0.6, 1.0),
    ]


def test_a_long_gap_starts_a_new_run():
    runs = _group_runs([chunk("a", 0.0, 0.5), chunk("b", 0.6, 1.0), chunk("c", 5.0, 5.4)])
    assert [[w.text for w in r] for r in runs] == [["a", "b"], ["c"]]


def test_missing_timestamp_inherits_rather_than_interpolates():
    """A guessed timestamp puts a cut point in the wrong place; a zero-width word does not."""
    runs = _group_runs([chunk("a", 0.0, 0.5), chunk("b", None, None)])
    b = runs[0][1]
    assert (b.start, b.end) == (0.5, 0.5)


def test_missing_timestamp_on_the_very_first_word_is_anchored_at_zero():
    runs = _group_runs([chunk("a", None, None)])
    assert (runs[0][0].start, runs[0][0].end) == (0.0, 0.0)


def test_end_before_start_is_clamped():
    runs = _group_runs([chunk("a", 2.0, 1.0)])
    assert runs[0][0].end >= runs[0][0].start


def test_blank_chunks_are_dropped():
    runs = _group_runs([chunk("  ", 0.0, 0.5), chunk("a", 0.6, 1.0)])
    assert [[w.text for w in r] for r in runs] == [["a"]]


@pytest.mark.parametrize(
    "text",
    [
        "Thanks for watching!",
        "Subtitles by the Amara.org community",
        "Please subscribe to my channel",
        "ご視聴ありがとうございました",
        "Редактор субтитров А.Синецкая",
        "Корректор субтитров А.Синецкая",
    ],
)
def test_known_credit_hallucinations_are_dropped(text: str):
    assert _is_hallucination(text, [])


def test_real_speech_survives_the_filter():
    assert not _is_hallucination("Привет, меня зовут Анна, и сегодня я расскажу о городе.", [])


def test_a_stuck_decoder_looping_is_caught_by_compression():
    """The repeat window only looks two runs back; a long loop needs a different signal."""
    assert _is_degenerate(RU_CREDITS)
    assert not _is_degenerate("Привет, меня зовут Анна, и сегодня я расскажу вам о городе.")


def test_short_strings_are_not_judged_on_compression():
    """Short text compresses unpredictably, so the ratio means nothing there."""
    assert not _is_degenerate("да да да")


def test_verbatim_repeat_of_a_recent_run_is_dropped():
    assert _is_hallucination("one two three", ["one two three"])


def test_hallucinated_runs_do_not_reach_the_word_array():
    words = _collect_words([
        chunk("Привет", 0.0, 0.5),
        chunk("Thanks", 5.0, 5.4),
        chunk("for", 5.5, 5.8),
        chunk("watching", 5.9, 6.3),
    ])
    assert [w.text for w in words] == ["Привет"]


def test_words_come_back_in_time_order():
    words = _collect_words([chunk("b", 5.0, 5.4), chunk("a", 0.0, 0.5)])
    assert [w.text for w in words] == ["a", "b"]


@pytest.mark.parametrize(
    "configured, expected",
    [
        ("large-v3", "openai/whisper-large-v3"),
        ("openai/whisper-large-v3", "openai/whisper-large-v3"),
        ("some-org/custom-whisper", "some-org/custom-whisper"),
    ],
)
def test_bare_checkpoint_names_still_resolve(configured: str, expected: str):
    """BAG_WHISPER_MODEL has always held a bare name; it must keep working."""
    assert _model_id(configured) == expected


def test_require_cuda_refuses_to_run_slowly_in_silence(monkeypatch):
    from app.config import Settings

    monkeypatch.setattr("app.device.resolve", lambda pref="auto": "cpu")
    with pytest.raises(ASRError, match="BAG_REQUIRE_CUDA"):
        assert_cuda(Settings(require_cuda=True, device="auto"))


def test_require_cuda_off_allows_cpu(monkeypatch):
    from app.config import Settings

    monkeypatch.setattr("app.device.resolve", lambda pref="auto": "cpu")
    assert_cuda(Settings(require_cuda=False, device="auto"))
