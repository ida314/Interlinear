# Handoff

Point a new session at this file. It covers where the project stands, what must not be
broken, and what to do next.

**State:** Runs end to end on real hardware. A 105s Russian clip produces a 166s ru→en
bilingual MP3 — Whisper, a local LLM and Kokoro all executing for real. 169 tests pass in
~2s with no GPU and no network.

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev,web]'
.venv/bin/pytest                                    # 169 passed
docker start vllm-qwen36-27b-nvfp4                  # translation backend, ~4 min
.venv/bin/python -m app.pipeline.runner <url> --target en --language ru
```

---

## What this is

Paste a video URL, get an MP3 where every spoken sentence is followed by an AI translation
of that sentence, with adjustable pacing. Language-learning tool, all local models, running
on an NVIDIA DGX Spark.

The owner is **studying Russian** — ru→en is the real workload, not a generic test case.
Prefer Russian examples when testing. Other languages of interest: pt, es, zh, ja, fr, de.

**You are probably running on the DGX Spark itself.** Check `hostname` — if it is `gx10`,
this is the target machine, not a dev laptop. There is no SSH step.

---

## Invariants — do not break these

Load-bearing, easy to violate accidentally, expensive to discover later. Full reasoning in
[docs/decisions/](docs/decisions/).

**1. `Timeline.words` is the single source of truth.** ASR produces it once. Nothing
downstream may rewrite, reorder or renormalise it. Segments are *index ranges*
(`word_start`/`word_end`) over that array. The denormalised `text`/`start`/`end` on `Segment`
are a debugging convenience — the indices are authoritative, and `rebuild_derived()`
regenerates the rest. ([002](docs/decisions/002-words-are-the-source-of-truth.md))

**2. Stage 6 (`plan.py`) is pure and separate from stage 7 (`render_audio.py`).** Stage 6
turns a timeline into a flat edit list; stage 7 only executes it. Never let the renderer
compute its own timings. Pacing knobs never touch the GPU — that is the entire preview
feature — and the future video renderer consumes the same plan.
([003](docs/decisions/003-plan-render-split.md))

**3. TTS is always synthesised at speed 1.0.** `tts_speed` is applied at render time via
ffmpeg `atempo`. Baking speed into the cached wav would make the cache speed-dependent and
kill the instant preview.

**4. Every non-source clip is anchored to a source clip.** The video renderer freezes the
frame at the preceding source clip's `src_end`. `RenderPlan.enforce_invariants()` checks
this — keep calling it. Note `silence` may intervene: `freeze_frame_time()` walks back to the
nearest source clip, so "immediately preceded by" is *too strict* a reading and will produce
false alarms.

**5. The LLM returns indices, never text.** In boundary arbitration
(`translate.make_arbiter`), tokens are numbered in the prompt and the model returns a subset
of the indices it was asked about. It cannot rewrite timestamped words, and it never has to
count. ([004](docs/decisions/004-llm-returns-indices.md))

---

## Decisions already made — don't relitigate

See [docs/decisions/](docs/decisions/) for the arguments. Summary:

| Decision | Why |
|---|---|
| ASR, not a video-understanding LLM | A VLM gives worse timestamps and invents speech. |
| Whisper `large-v3`, not `-turbo` | Turbo's dropped decoder layers measurably hurt non-English; ru/zh/ja are core. |
| Not Parakeet/Canary | Better timestamps, but 25 European languages only — no zh/ja. |
| **Whisper on plain PyTorch, not faster-whisper** | CTranslate2 has no aarch64 CUDA wheels and fails *silently* onto CPU. Removing the dependency was cheaper than the day-long source build it required. |
| **No CTC forced-alignment pass** | Only sentence boundaries need accuracy; the cut is placed acoustically by `find_quietest_point`. Provisional — see below. |
| Local LLM, not NLLB/SeamlessM4T | NMT is sentence-in/sentence-out: no context, no glossary, no consistent register. |
| Kokoro, not XTTS-v2 / F5-TTS | Those ship non-commercial weights (CPML / CC-BY-NC) and Coqui is defunct. |
| SQLite + one worker, not Celery | Two daemons to solve distribution problems that don't exist for one user. |
| Vanilla JS, not HTMX/React | The page already needed custom JS for preview sliders. |
| Ignore Whisper's segment boundaries | They come from timestamp-token sampling and the 30s window, not linguistics. |

---

## Layout

```
app/
  config.py              settings (env-driven, BAG_ prefix)
  device.py              auto -> cuda -> mps -> cpu; torch imported lazily
  db.py                  sqlite connect + migrate
  main.py                FastAPI: pages, api, preview, rerender, download
  worker.py              job loop — run as its own process
  models/
    timeline.py          Word, Segment, TTSClip, SourceInfo, RenderParams, Timeline
    plan.py              Clip, RenderPlan, enforce_invariants()
  audio/
    dsp.py               rms_envelope, find_quietest_point, fades, loudness
    io.py                ffmpeg decode/encode/atempo, wav read/write
  pipeline/
    runner.py            STAGES + STAGE_DEPS + run() + CLI      <- start reading here
    fetch.py             1. yt-dlp
    asr.py               2. Whisper via transformers  [GPU-accel]
    sbd.py               3a. boundary detection (fusion)
    segment.py           3b. constraints, sentence/words modes
    translate.py         4. LLM + boundary arbiter    [GPU-accel]
    tts.py               5. Kokoro                    [GPU-accel]
    plan.py              6. cut points -> edit list  (pure)
    render_audio.py      7. edit list -> mp3         (pure)
    preview.py           excerpt renderer + source cache
  jobs/
    schema.sql, store.py claim/heartbeat/cancel/reap
  web/templates/         base, index, job, _status
docs/DEPLOY.md           DGX Spark / aarch64 specifics and vLLM setup
docs/decisions/          why things are the way they are
```

Tests: `test_segment` 45, `test_api` 23, `test_asr` 23, `test_jobs` 18, `test_plan` 16,
`test_translate` 15, `test_tts` 13, `test_runner` 9, `test_render` 7. Total 169.

---

## The reference job

`data/jobs/mvp1` — regenerate with:

```bash
BAG_LLM_BASE_URL=http://localhost:8000/v1 BAG_LLM_MODEL=nvidia/Qwen3.6-27B-NVFP4 \
.venv/bin/python -m app.pipeline.runner \
  "https://www.youtube.com/watch?v=lyCrsZr3ur0" --job-id mvp1 --target en --language ru
```

105s Russian → 21 segments, 16 translated, 1 warning (`wtpsplit` not installed), 166s MP3.
Translations are fluent and no segment fell back to source text. Use it as the tuning
baseline.

---

## Next steps, in order

**1. VAD-clip the word timings.** This is the one real quality defect and the highest-value
fix. Whisper's word timestamps stretch across non-speech: 9 of 116 words (8%) in the
reference job have implausible durations — one token spans 8.7s — concentrated in three
segments, leaving ~13.5s of dead air before translations. Segment 0 covers 14.1s for two
words.

Plan: `app/pipeline/vad.py` using the `silero-vad` pip package (not `torch.hub`, which needs
network at first call). Clip each word's `[start, end]` to detected speech regions and drop
words falling entirely outside them. Degrade gracefully — warn and continue — the way
`runner._load_detector` does. See
[006](docs/decisions/006-no-forced-alignment.md), which this may falsify.

**2. Listen to the output.** Still true, still matters more than any remaining code. The gap
defaults (`pre_gap=0.25`, `post_gap=0.35`) and splice quality decide whether this is usable
or grating. Tune via the preview endpoint, which needs no GPU.

**3. Drive the web UI end to end.** It launches and `/healthz` reports
`{"ok":true,"device":"cuda"}`, but it has never run a real job. **CLI jobs are not registered
in SQLite**, so `data/jobs/mvp1` does not appear in the UI — the worker registers jobs, the
CLI does not. Decide whether the CLI should register too, or whether the UI should discover
jobs on disk. The live preview sliders are the most demo-worthy feature in the project and
have never been seen working.

**4. Process more than one video.** Everything known about real-world behaviour comes from a
single clip. Try one with music, one conversational, one longer than 10 minutes.

**5. Fail loud on translation failure.** `translate.py` currently does
`seg.translation = text or seg.text`, so an unreachable LLM yields an MP3 of Russian read
aloud in an American accent, with exit code 0. Add a strict mode, mark failed segments so TTS
skips them, and add a circuit breaker — if the first few batches all fail, stop rather than
grinding through 400 sentences to produce something useless.

**6. Let the arbiter abstain.** An empty reply currently means "reject every boundary"
(under-split) while an exception means "confirm every candidate" (over-split). Neither is
"no opinion". Both should fall back to the fused score.
([008](docs/decisions/008-sampling-temperature.md))

**7. Speed up ASR.** ~1.04x realtime for `large-v3` is slow for this hardware. Batched or
chunked decoding is the obvious lever and has not been tried.

**8. Then** the video renderer, and Piper for Russian as a *target* language.

---

## Known gaps and traps

- **`wtpsplit` is not installed**, so segmentation uses the punctuation+pause fallback.
  Acceptable for Russian (`STRONG_PUNCT_LANGS`), *not* for zh/ja, which genuinely need SaT.
  It may fight `transformers` on version pins; the fallback degrades gracefully, so deferring
  it is a legitimate choice.
- **Whisper's punctuation recall is not guaranteed even for Russian.** On the reference clip
  it punctuated cleanly for half the video and then emitted bare lowercase runs. The pause
  signal carried those segments.
- **Single-word utterances are marked `skip` on purpose** by `segment._mark_unusable` —
  source audio is kept, only the translation is suppressed. Not a bug; tune `min_words` /
  `min_duration` if you disagree.
- **`torch` gets clobbered easily.** `kokoro` and `wtpsplit` will pull a CPU-only torch from
  PyPI over your CUDA build. Install torch first, and re-check
  `torch.cuda.is_available()` after every subsequent `pip install`.
- **yt-dlp needs a JS runtime** for YouTube or it fails with a bare HTTP 403. `fetch.py`
  auto-detects deno/node/bun.
- **`cfg.tts_engine` is read nowhere.** Dead setting; wire it up or delete it.
- **Russian as a bilingual *target* still raises** — Kokoro has no Russian voice.
  `get_engine("ru")` fails loudly rather than substituting. Needs Piper behind the existing
  `TTSEngine` protocol.
- **A test whose component can "fail open" needs distractor cases.** The arbiter bug in
  [008](docs/decisions/008-sampling-temperature.md) hid behind a test that only asked about
  true positives. Applies to anything returning a subset.

---

## Conventions

- Commits are authored as the owner. **No `Co-Authored-By` trailers, no Claude attribution.**
- Small, single-purpose commits; conventional-ish prefixes; the body carries the *why*.
- Tests ship in the same commit as the code they cover.
- Comments explain why, not what, and match the surrounding density.
- Heavy imports stay function-local, so the pure stages import without torch.
- GPU and network markers exist (`pytest -m gpu`) but nothing carries them yet.
