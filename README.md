# Bilingual Audio Generator

Turn a video into bilingual audio: every spoken sentence is followed by an AI-generated
translation of that sentence, with adjustable pacing. Built for language learning, and to
run entirely on local models.

```
python -m app.pipeline.runner "https://youtube.com/watch?v=..." --target en --speed 1.1
```

## How it works

```
1 fetch      URL             -> audio                  yt-dlp + ffmpeg
2 asr        audio           -> aligned words           WhisperX          [GPU]
3 segment    words           -> sentences               SaT + pause + LLM
4 translate  sentences       -> translations            local LLM         [GPU]
5 tts        translations    -> speech clips            Kokoro            [GPU]
6 plan       timeline+knobs  -> edit list               pure CPU
7 render     edit list       -> mp3                     numpy + ffmpeg
```

Two design decisions carry most of the weight:

**`Timeline.words` is the single source of truth.** ASR produces it once; nothing downstream
may rewrite it. Segmentation produces *index ranges* over that array, never new text. This
is what keeps every word bound to its forced-alignment timestamp, and why re-segmenting with
different knobs can never desynchronise the audio.

**Stage 6 is pure and separate from stage 7.** Stage 6 turns the timeline into a flat,
absolute-timed edit list; stage 7 just executes it. Consequences: pacing knobs never touch
the GPU (so the live preview is instant), and the phase-2 video renderer consumes the same
plan rather than re-deriving timing.

## Why ASR and not a video model

The transcription step is ASR with forced alignment, not an LLM "watching" the video. A
video-understanding model gives worse timestamps and invents speech that was never said.
Word-level timing is a hard requirement here — it is what lets the splicer avoid cutting
mid-word. The LLM's real job is translation, where sentence-level context genuinely beats
dedicated NMT.

## Knobs

Cheap knobs re-run only stages 6-7 (pure numpy, milliseconds). Expensive ones re-run GPU
stages, though the TTS cache — keyed on `sha256(text, voice, lang)` — absorbs most of the
cost of re-segmentation.

| Knob | Default | Cost |
|---|---|---|
| `tts_speed` | 1.0 | cheap |
| `pre_gap` / `post_gap` | 0.25 / 0.35 | cheap |
| `head_pad` / `tail_pad` | 0.10 / 0.15 | cheap |
| `max_source_gap` | 1.5 | cheap |
| `segmentation_mode` | `sentence` \| `words` | re-segment |
| `max_words_per_chunk` | 12 | re-segment |
| `target_lang` | `en` | re-translate |
| `voice` | `af_heart` | re-synthesise |

TTS is always synthesised at speed 1.0; `tts_speed` is applied at render time via `atempo`.
That is deliberate — it keeps the cache speed-independent.

## Development

The pure stages (models, dsp, segment, plan, render) have no GPU or network dependencies and
run anywhere:

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev,web]'
.venv/bin/pytest            # 131 tests, ~3s
```

Run the service with two processes — the worker is deliberately separate from the API so
CUDA stays out of a threaded ASGI server and restarting the web layer never kills a job:

```bash
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
.venv/bin/python -m app.worker
```

GPU and network tests are opt-in: `pytest -m gpu`, `pytest -m network`.

For deployment on DGX Spark / aarch64 — including the CTranslate2 source build, which is the
one genuinely hard part — see [docs/DEPLOY.md](docs/DEPLOY.md).

## Language notes

Primary target is **Russian → English**. Two Russian specifics are already handled:

- Whisper punctuates Russian well — it has prosodic access, so it hears a question's rising
  intonation where a text-only model cannot. `ru` is therefore in `STRONG_PUNCT_LANGS` and a
  terminal mark is close to decisive on its own.
- Russian abbreviations (`г.`, `гг.`, `т.д.`, `ул.`, `см.`…) end in a period mid-sentence.
  Without the guard in `_ABBREVIATIONS`, *"известен с 1995 г. и очень популярен"* splits at
  `г.` and the translator receives a subjectless fragment.

Russian works as a **source** language but not yet as a bilingual **target** — Kokoro has no
Russian voice. That needs a Piper or Chatterbox engine behind the existing `TTSEngine`
protocol; `get_engine()` fails loudly rather than silently substituting.

For zh/ja, WhisperX aligns per *character* (no whitespace to tokenise on), so "word count"
means character count and text must rejoin without spaces. Handled, and tested.

## Status

Picking this up mid-stream? Start with **[HANDOFF.md](HANDOFF.md)** — current state,
invariants that must not be broken, and what to do next.

All seven stages are implemented, plus the SQLite job queue, worker, web UI, live preview
and HTTP API.

Verified end to end on synthetic audio for both Russian and Spanish: segment → translate →
TTS → plan → render → MP3, with stubs standing in for the ASR, LLM and TTS models.

**Nothing has run on real hardware yet.** Stages 2, 4 and 5 need the GPU extras, so
`asr.py`, `tts.py`, the SaT detector and the LLM client have never actually executed —
expect real bugs on first contact.

Not yet built: the phase-2 video renderer, the Dockerfile, and Russian as a bilingual
*target* (Kokoro has no Russian voice).

**The next step is to run a real video and listen to the output.** The splice quality and
gap defaults are the difference between usable and grating, and no test can tell you whether
they are right.
