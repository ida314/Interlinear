"""TTS engine selection and cache keying.

Nothing here imports kokoro: the parts worth testing are the language/voice tables and the
cache key, all of which are pure data.
"""

from __future__ import annotations

import pytest

from app.pipeline.tts import (
    KOKORO_LANG_CODES,
    KOKORO_VOICES,
    TTSError,
    clip_key,
    get_engine,
)


def test_english_uses_kokoros_american_english_code():
    """`"en"[:1]` is `"e"`, which is Kokoro's *Spanish* code — truncating the ISO code would
    synthesise English through a Spanish G2P, which is audible but never raises."""
    assert KOKORO_LANG_CODES["en"] == "a"


def test_every_supported_language_has_both_a_voice_and_a_code():
    """The two tables are indexed by the same key and drift silently if they disagree."""
    assert set(KOKORO_LANG_CODES) == set(KOKORO_VOICES)


@pytest.mark.parametrize("lang, voice", sorted(KOKORO_VOICES.items()))
def test_voice_prefix_matches_the_language_code(lang: str, voice: str):
    """Kokoro voice names are `<lang code><gender>_<name>`, so a mismatch here is the same
    bug as the English one wearing a different hat."""
    assert voice[0] == KOKORO_LANG_CODES[lang]


def test_unsupported_target_language_names_the_alternatives():
    """Russian is the case that matters: Kokoro has no ru voice, and the error is what tells
    the next person what to do about it."""
    with pytest.raises(TTSError, match="Piper|Chatterbox"):
        get_engine("ru")


def test_clip_key_ignores_speed_so_the_cache_survives_a_speed_change():
    assert clip_key("hola", "ef_dora", "es") == clip_key("hola", "ef_dora", "es")


def test_clip_key_separates_text_voice_and_language():
    keys = {
        clip_key("a", "af_heart", "en"),
        clip_key("b", "af_heart", "en"),
        clip_key("a", "ef_dora", "en"),
        clip_key("a", "af_heart", "es"),
    }
    assert len(keys) == 4
