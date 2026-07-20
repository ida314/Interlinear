"""Translation and boundary arbitration, driven by a stub completer.

The failure paths matter more than the happy path here: a single malformed batch must never
take down a 45-minute job.
"""

from __future__ import annotations

import json

import pytest

from app.models.timeline import RenderParams, Segment, SourceInfo, Timeline, Word
from app.pipeline.translate import (
    ARBITER_TEMPERATURE,
    TRANSLATE_TEMPERATURE,
    _parse_json,
    http_completer,
    make_arbiter,
    translate_timeline,
)


def make_timeline(texts: list[str], **knobs) -> Timeline:
    words = [Word(text=t, start=i, end=i + 0.5) for i, t in enumerate(texts)]
    tl = Timeline(
        job_id="t",
        source=SourceInfo(url="u", detected_lang="es", duration=len(texts) + 1),
        params=RenderParams(**knobs),
        words=words,
        segments=[
            Segment(id=f"s{i}", index=i, word_start=i, word_end=i + 1) for i in range(len(texts))
        ],
    )
    tl.rebuild_all_derived()
    return tl


def canned(mapping: dict[str, str] | None = None, *, fail_times: int = 0):
    """A completer that answers translation prompts, optionally failing the first N calls."""
    state = {"calls": 0, "prompts": []}

    def complete(system: str, user: str) -> str:
        state["calls"] += 1
        state["prompts"].append(user)
        if state["calls"] <= fail_times:
            raise RuntimeError("model unavailable")
        ids = [int(line.split(".")[0]) for line in user.splitlines() if line[:1].isdigit()]
        texts = [line.split(". ", 1)[1] for line in user.splitlines() if line[:1].isdigit()]
        return json.dumps({
            "translations": [
                {"id": i, "text": (mapping or {}).get(t, f"EN:{t}")}
                for i, t in zip(ids, texts)
            ]
        })

    complete.state = state  # type: ignore[attr-defined]
    return complete


# --- parsing ---------------------------------------------------------------------------


def test_parses_json_wrapped_in_fences_and_preamble():
    """Instruct-tuned models add these despite being told not to."""
    raw = 'Sure! Here you go:\n```json\n{"translations": [{"id": 0, "text": "Hi"}]}\n```'
    assert _parse_json(raw)["translations"][0]["text"] == "Hi"


def test_unparseable_output_raises():
    with pytest.raises(ValueError, match="no JSON object"):
        _parse_json("I cannot help with that.")


# --- translation -----------------------------------------------------------------------


def test_translates_every_segment():
    tl = make_timeline(["hola", "mundo", "adios"])
    translate_timeline(tl, completer=canned())

    assert [s.translation for s in tl.segments] == ["EN:hola", "EN:mundo", "EN:adios"]
    assert "translate" in tl.stages_done


def test_batches_respect_the_configured_size():
    from app.config import Settings

    tl = make_timeline([f"w{i}" for i in range(10)])
    completer = canned()
    translate_timeline(tl, completer=completer, cfg=Settings(translate_batch_size=4))

    assert completer.state["calls"] == 3  # 4 + 4 + 2


def test_preceding_context_is_supplied_for_pronoun_resolution():
    """The reason to use an LLM over sentence-level NMT at all."""
    from app.config import Settings

    tl = make_timeline([f"w{i}" for i in range(6)])
    completer = canned()
    translate_timeline(
        tl, completer=completer, cfg=Settings(translate_batch_size=3, translate_context_sentences=2)
    )

    assert "Preceding context" in completer.state["prompts"][1]
    assert "w1" in completer.state["prompts"][1]


def test_batch_failure_falls_back_to_single_sentences():
    tl = make_timeline(["uno", "dos"])
    translate_timeline(tl, completer=canned(fail_times=1))

    assert all(s.translation for s in tl.segments)
    assert any("retrying one at a time" in w.message for w in tl.warnings)


def test_total_failure_keeps_source_text_rather_than_killing_the_job():
    tl = make_timeline(["uno", "dos"])

    def always_fails(system: str, user: str) -> str:
        raise RuntimeError("down")

    translate_timeline(tl, completer=always_fails)

    assert [s.translation for s in tl.segments] == ["uno", "dos"]
    assert tl.warnings


def test_count_mismatch_is_caught_and_retried():
    """A model that drops a sentence silently would otherwise shift every translation onto
    the wrong audio."""
    calls = {"n": 0}

    def drops_one(system: str, user: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps({"translations": [{"id": 0, "text": "only one"}]})
        ids = [int(line.split(".")[0]) for line in user.splitlines() if line[:1].isdigit()]
        return json.dumps({"translations": [{"id": i, "text": f"ok{i}"} for i in ids]})

    tl = make_timeline(["uno", "dos", "tres"])
    translate_timeline(tl, completer=drops_one)

    assert all(s.translation for s in tl.segments)
    assert tl.segments[0].translation != "only one"


def test_skipped_segments_are_not_translated():
    tl = make_timeline(["hola", "eh", "adios"])
    tl.segments[1].kind = "skip"
    completer = canned()
    translate_timeline(tl, completer=completer)

    assert tl.segments[1].translation is None
    assert "eh" not in completer.state["prompts"][0]


# --- arbitration -----------------------------------------------------------------------


def test_arbiter_numbers_tokens_so_the_model_never_has_to_count():
    """Asking an LLM for 'position 137' invites miscounting; showing it [137] does not."""
    captured = {}

    def complete(system: str, user: str) -> str:
        captured["user"] = user
        return json.dumps({"boundaries": [1]})

    words = [Word(text=t, start=i, end=i + 0.5) for i, t in enumerate(["uno", "dos", "tres"])]
    result = make_arbiter(complete)(words, [1], "es")

    assert result == {1}
    assert "[1]dos" in captured["user"]


def test_arbiter_cannot_invent_boundaries_it_was_not_asked_about():
    """Positions the cheap signals already settled must stay settled."""

    def overreaches(system: str, user: str) -> str:
        return json.dumps({"boundaries": [0, 1, 2, 99]})

    words = [Word(text=t, start=i, end=i + 0.5) for i, t in enumerate(["uno", "dos", "tres"])]
    assert make_arbiter(overreaches)(words, [1], "es") == {1}


def test_arbiter_failure_defers_to_the_fused_score():
    """The safe failure is behaving as though arbitration were switched off."""

    def broken(system: str, user: str) -> str:
        raise RuntimeError("timeout")

    words = [Word(text=t, start=i, end=i + 0.5) for i, t in enumerate(["uno", "dos"])]
    assert make_arbiter(broken)(words, [0, 1], "es") == {0, 1}


def test_arbiter_with_no_candidates_makes_no_call():
    def explodes(system: str, user: str) -> str:
        raise AssertionError("should not be called")

    assert make_arbiter(explodes)([], [], "es") == set()


def test_arbitration_is_sampled_deterministically(monkeypatch):
    """Arbitration is classification, so it must not sample.

    Measured against Qwen3.6-27B: at temperature 0.2 the model returned a well-formed but
    *empty* boundary set in 5 of 12 runs, versus 0 of 12 at temperature 0.0. An empty set is
    not an error — it silently means "no sentence ends here", so the job runs sentences
    together and still exits zero.
    """
    sent = []

    class FakeResponse:
        def raise_for_status(self) -> None: ...

        def json(self) -> dict:
            return {"choices": [{"message": {"content": '{"boundaries": []}'}}]}

    def fake_post(url, *, headers, json, timeout):  # noqa: A002
        sent.append(json)
        return FakeResponse()

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)

    words = [Word(text=t, start=i, end=i + 0.5) for i, t in enumerate(["uno", "dos"])]
    make_arbiter()(words, [0], "es")
    translate_timeline(make_timeline(["uno"]), completer=http_completer())

    # The stub reply is not a valid translation payload, so the translation stage retries
    # sentence by sentence — hence "every call after the first", not a fixed call count.
    arbiter_call, *translation_calls = [payload["temperature"] for payload in sent]
    assert arbiter_call == ARBITER_TEMPERATURE == 0.0
    assert translation_calls and set(translation_calls) == {TRANSLATE_TEMPERATURE}


def test_empty_arbiter_reply_is_distinguishable_from_failure():
    """The two failure modes are opposite, so a test that only asks about true sentence
    ends cannot tell them apart: an exception confirms *every* candidate (over-splitting),
    while an empty reply confirms *none* (under-splitting)."""
    words = [Word(text=t, start=i, end=i + 0.5) for i, t in enumerate(["uno", "dos", "tres"])]

    def empty(system: str, user: str) -> str:
        return json.dumps({"boundaries": []})

    def broken(system: str, user: str) -> str:
        raise RuntimeError("timeout")

    assert make_arbiter(empty)(words, [0, 1], "es") == set()
    assert make_arbiter(broken)(words, [0, 1], "es") == {0, 1}
