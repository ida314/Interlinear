# Handoff

Point a new session at this file. It covers where the project stands, what must not be
broken, and what to do next.

**State:** Milestone A + B complete. 131 tests pass in ~3s with no GPU and no network.
Nothing has ever run on real hardware — stages 2, 4 and 5 have only been exercised against
stubs.

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev,web]'
.venv/bin/pytest                                    # 131 passed
.venv/bin/python -m app.pipeline.runner <url>       # needs GPU extras
```

---

## What this is

Paste a video URL, get an MP3 where every spoken sentence is followed by an AI translation
of that sentence, with adjustable pacing. Language-learning tool, all local models, targeted
at an NVIDIA DGX Spark.

The owner is **studying Russian** — ru→en is the real workload, not a generic test case.
Prefer Russian examples when testing. Other languages of interest: pt, es, zh, ja, fr, de.

---

## Invariants — do not break these

These are load-bearing. Each one is easy to violate accidentally and expensive to discover
later.

**1. `Timeline.words` is the single source of truth.** ASR produces it once. Nothing
downstream may rewrite, reorder, or renormalise it. Segments are *index ranges*
(`word_start`/`word_end`) over that array. This is what keeps every word bound to its
forced-alignment timestamp, and it is why re-segmenting can never desync audio from text.
The denormalised `text`/`start`/`end` on `Segment` are a debugging convenience — the indices
are authoritative, and `rebuild_derived()` regenerates the rest.

**2. Stage 6 (`plan.py`) is pure and separate from stage 7 (`render_audio.py`).** Stage 6
turns a timeline into a flat edit list; stage 7 only executes it. Never let the renderer
compute its own timings. Two things depend on this: pacing knobs never touch the GPU (which
is the entire preview feature), and the future video renderer consumes the same plan instead
of re-deriving it.

**3. TTS is always synthesised at speed 1.0.** `tts_speed` is applied at render time via
ffmpeg `atempo`. Baking speed into the cached wav would make the cache speed-dependent and
kill the instant preview.

**4. Every non-source clip is preceded by a source clip.** The video renderer freezes the
frame at the previous clip's `src_end`; without that anchor it has nothing to hold.
`RenderPlan.enforce_invariants()` checks this — keep calling it.

**5. The LLM returns indices, never text.** In boundary arbitration
(`translate.make_arbiter`), tokens are numbered in the prompt and the model returns a subset
of the indices it was asked about. It cannot rewrite timestamped words, and it never has to
count.

---

## Decisions already made — don't relitigate

| Decision | Why |
|---|---|
| ASR, not a video-understanding LLM | Word-level forced alignment is a hard requirement for splicing. A VLM gives worse timestamps and invents speech. |
| Whisper `large-v3`, not `-turbo` | Turbo's dropped decoder layers measurably hurt non-English; ru/zh/ja are core. |
| Not Parakeet/Canary | Better timestamps, but 25 European languages only — no zh/ja. |
| Local LLM, not NLLB/SeamlessM4T | NMT is sentence-in/sentence-out: no context, no glossary, no consistent register. |
| Kokoro, not XTTS-v2 / F5-TTS | Those ship non-commercial weights (CPML / CC-BY-NC) and Coqui is defunct. |
| SQLite + one worker, not Celery | Two daemons to solve distribution problems that don't exist for one user. |
| Vanilla JS, not HTMX/React | The page already needed custom JS for preview sliders. |
| Ignore WhisperX's segment boundaries | They come from timestamp-token sampling and the 30s window, not linguistics — they fall nowhere near real sentence ends, even when punctuation is good. |

---

## Layout

```
app/
  config.py              settings (env-driven, BAG_ prefix)
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
    asr.py               2. WhisperX            [GPU]
    sbd.py               3a. boundary detection (fusion)
    segment.py           3b. constraints, sentence/words modes
    translate.py         4. LLM + boundary arbiter [GPU]
    tts.py               5. Kokoro              [GPU]
    plan.py              6. cut points -> edit list  (pure)
    render_audio.py      7. edit list -> mp3         (pure)
    preview.py           excerpt renderer + source cache
  jobs/
    schema.sql, store.py claim/heartbeat/cancel/reap
  web/templates/         base, index, job, _status
docs/DEPLOY.md           aarch64 / DGX Spark specifics — read before deploying
```

Tests: `test_plan` 16, `test_render` 7, `test_segment` 45, `test_translate` 13,
`test_runner` 9, `test_jobs` 18, `test_api` 23.

---

## Not built yet

- **`render_video.py`** — phase 2. Consumes the existing `RenderPlan` unchanged: `source`
  clips become `ffmpeg trim`, `tts`/`silence` clips become a freeze frame at the previous
  clip's `src_end` with the translation burned in via `drawtext`. `Clip.label` already
  carries the text.
- **`Dockerfile` / `docker-compose.yml`** — referenced by `docs/DEPLOY.md` but not written.
  Needs the CTranslate2 source-build layer.
- **`app/gpu.py`** — mentioned in early planning. Probably unnecessary: 128GB unified memory
  means no load/unload choreography is needed. The one thing worth keeping is
  `asr.assert_cuda()`, which already exists.
- **Russian as a *target* language** — deferred deliberately. Kokoro has no Russian voice, so
  `get_engine("ru")` raises. Needs a Piper or Chatterbox engine behind the existing
  `TTSEngine` protocol. ru as a *source* works fine.
- **Diarization** — out of scope. `Segment.speaker` exists so pyannote can drop in later
  without a schema migration.
- **systemd units** for api + worker.

---

## Never actually executed

Written against real APIs, but no line of these has run:

- `fetch.py` — yt-dlp isn't installed on the dev laptop
- `asr.py` — needs CUDA + the CTranslate2 build
- `sbd.SaTDetector` — needs `wtpsplit`; the punctuation+pause fallback is what the tests use
- `tts.KokoroEngine` — needs `kokoro`
- `translate.http_completer` — needs Ollama or vLLM running

Expect real bugs on first contact. The stubs verify the surrounding logic, not these calls.

---

## Environment

**Dev laptop:** Apple Silicon / Asahi Fedora, aarch64, no CUDA. Python 3.14 system
interpreter with no pip — use `.venv`. Torch and WhisperX have no 3.14 wheels; the pure
stages don't need them.

**Target:** DGX Spark at hostname `gx10` (Tailscale). GB10 Grace Blackwell, 128GB unified,
~273GB/s, aarch64, CUDA 13 / sm_121.

**SSH is not yet working.** `~/.ssh/id_ed25519_gx10` exists and `~/.ssh/config` maps
`gx10 → User dylan`, but the pubkey is not installed on the far end
(`Permission denied (password)`). Owner was given two options: `ssh-copy-id` to their own
account, or a separate unprivileged `claude` user. Check which they chose.

### The two things that will bite on first deploy

**1. CTranslate2 has no aarch64 CUDA wheels.** `pip install faster-whisper` succeeds and
then silently runs on CPU at ~1/20 speed. Must be built from source with
`-DCMAKE_CUDA_ARCHITECTURES="...;120;121;121-virtual"`, C++17 flags, and a Thrust compat
shim for CUDA 13. Working references: `rappdw/transcribe-dgx`, `atripathy86/transcribe`.
Budget a day. Full detail in `docs/DEPLOY.md`.

**2. Silent CPU fallback is the failure mode to fear.** CTranslate2 and Chatterbox both
degrade rather than error. Keep `BAG_REQUIRE_CUDA=1`; `asr.assert_cuda()` is the guard.

Not problems, don't chase: PyTorch aarch64+CUDA works with stock `cu128`/`cu130` wheels
(sm_121 is binary-compatible with sm_120), and the `(8.0) - (12.0)` capability warning on
GB10 is cosmetic.

---

## Suggested next steps, in order

1. **Get SSH working**, then check whether the DGX OS image already has the CUDA dev headers
   and compiler toolchain. Installing them needs root — that step belongs to the owner.
2. **Build CTranslate2** in a container off `nvcr.io/nvidia/pytorch:25.10-py3`. This is the
   one genuinely hard part.
3. **Run one short Russian video end to end** via the CLI, not the web UI — fewer moving
   parts when something breaks.
4. **Listen to the output.** This matters more than any remaining code. The gap defaults
   (`pre_gap=0.25`, `post_gap=0.35`) and splice quality are the difference between usable and
   grating, and no test can tell you whether they're right. Tune via the preview endpoint,
   which needs no GPU.
5. **Benchmark the trivial segmentation baseline.** "Just split on Whisper's punctuation" is
   a serious baseline for pt/ru/es/fr/de, not a strawman — Whisper punctuates those well
   because it has prosodic access. If SaT + fusion doesn't clearly beat it there, the
   complexity isn't earning its place. (zh/ja are the opposite case and genuinely need SaT.)
6. **Then** the video renderer, and Ollama → vLLM if translation throughput matters. A large
   MoE beats a dense model on this hardware: decode cost tracks *active* params while quality
   tracks *total* params, and the machine is bandwidth-bound.

---

## Conventions

- Commits are authored as the owner. **No `Co-Authored-By` trailers, no Claude attribution.**
- Small, single-purpose commits; conventional-ish prefixes; the body carries the *why*.
- Tests ship in the same commit as the code they cover.
- Comments explain why, not what, and match the surrounding density.
- GPU and network tests are opt-in: `pytest -m gpu`, `pytest -m network`.
