# Deploying on DGX Spark (GB10, aarch64)

Target: NVIDIA DGX Spark — GB10 Grace Blackwell, 128GB unified LPDDR5X, ~273GB/s, aarch64,
CUDA 13 / sm_121.

Two properties of this machine drive every decision below.

**128GB unified memory means no model juggling.** Whisper, the translation model and Kokoro
all stay resident. There is no need for the usual load/unload/`empty_cache()` choreography,
and adding it would be pure complexity.

**273GB/s is roughly a quarter of a 4090's bandwidth.** The machine is bandwidth-bound, not
capacity-bound. Single-stream decode is slow; concurrency is nearly free. Translation is
embarrassingly parallel, so **batching is worth more than every other optimisation put
together.**

---

## The one genuinely hard part: CTranslate2

`faster-whisper` depends on CTranslate2, which **ships no aarch64 CUDA wheels**. `pip
install` succeeds, and then either throws at runtime or silently runs on CPU at roughly 1/20
speed. Build it from source:

```bash
cmake -DCMAKE_CUDA_ARCHITECTURES="75;80;86;87;89;90;100;120;121;121-virtual" \
      -DCMAKE_CXX_STANDARD=17 -DCMAKE_CUDA_STANDARD=17 \
      -DWITH_CUDA=ON -DWITH_CUDNN=ON ..
```

Two CUDA 13 gotchas that will bite:

- Thrust 2.x removed `thrust::unary_function` / `binary_function`, which CTranslate2 still
  references. Needs a compat shim after `#include "misc/wrap_thrust.hpp"`.
- The C++17 flags above are required; CTranslate2 defaults lower and the bundled CCCL will
  not compile.

**Read [rappdw/transcribe-dgx](https://github.com/rappdw/transcribe-dgx) before starting.**
It is WhisperX large-v3 on this exact hardware with the sm_121 patches already solved.
[atripathy86/transcribe](https://github.com/atripathy86/transcribe) ships precompiled CUDA 13
aarch64 CT2 binaries. Budget a day; this is a Docker layer, not a redesign.

## Silent CPU fallback is the failure mode to fear

CTranslate2 and Chatterbox both degrade to CPU rather than erroring. On a batch workload you
can ship something that appears to work and is 20x slower than it should be.

`app/pipeline/asr.py:assert_cuda()` exists for this. Keep `BAG_REQUIRE_CUDA=1` in
production. It is the difference between finding out in seconds and finding out after a
40-minute job.

## Things that are NOT problems

- **PyTorch.** sm_121 is binary-compatible with sm_120. Stock aarch64 wheels from
  `download.pytorch.org/whl/cu128` (or `cu130`) work. Ignore the custom "sm_121 wheels"
  circulating on HuggingFace — they solve a problem you do not have.
- **The capability warning.** `Minimum and Maximum cuda capability supported by this version
  of PyTorch is (8.0) - (12.0)` appears on GB10 and is cosmetic. Don't chase it.
- **Kokoro.** Runs fine — but use the **PyTorch path, not ONNX Runtime**. `onnxruntime-gpu`
  aarch64 CUDA support is patchy and several Kokoro wrappers silently downgrade to CPU when
  they detect aarch64. At 82M params the PyTorch path is fast regardless. For a Japanese
  target voice, MeCab needs `unidic-lite` plus a symlink into `site-packages/unidic/dicdir`.

## Base image

`nvcr.io/nvidia/pytorch:25.10-py3` or newer — matches host CUDA 13 and ships `compute_120`
PTX that JITs to sm_121, which removes most of the workarounds above. For vLLM,
`nvcr.io/nvidia/vllm:26.03-py3` is proven on GB10.

No NGC container ships a CUDA-enabled CTranslate2. That stays your one custom build layer.

---

## Translation backend: start with Ollama, move to vLLM

`app/config.py` speaks OpenAI-compatible HTTP, so the two are interchangeable —
`BAG_LLM_BASE_URL` is the only change.

**Start:** Ollama + `qwen3:14b` (strongest on zh/ja). ~20-25 tok/s single-stream here.
Fine for getting the pipeline working.

**Then:** vLLM with a large MoE. This is counterintuitive and worth internalising:

| Model | Single-stream | Concurrency 256 |
|---|---|---|
| Dense 49B NVFP4 | 5.8 tok/s | 695 tok/s |
| gpt-oss-120b MXFP4 | **33.5 tok/s** | **862 tok/s** |
| Dense 70B Q4 | ~2.7-5 tok/s | — |

A 120B MoE beats a dense 49B on *both* axes because only ~3.6B params are active per token:
decode cost tracks active parameters while quality tracks total parameters. **The 128GB
exists to hold total params; the 273GB/s only has to move active ones.** Dense 70B is a trap
here.

Run vLLM with `--enable-prefix-caching` — every translation batch shares the same system
prompt, so the win is large.

Treat published "70B at 35-45 tok/s on DGX Spark" claims as wrong: 40GB at 273GB/s caps out
near 6.8 tok/s. Those numbers confuse prefill with decode, or batch aggregate with
single-stream.

---

## ASR model choice

Stay on **large-v3**, not `large-v3-turbo` — turbo's dropped decoder layers measurably hurt
non-English, and ru/zh/ja are core targets here.

**Parakeet/Canary are disqualified** despite better timestamps: 25 European languages only,
no Chinese, no Japanese.

## Sentence boundaries

Whisper's built-in punctuation is *better* than dedicated restoration models for
pt/ru/es/fr/de, because it has prosodic access — it hears the rising intonation of a
question where a text-only model sees only words. `app/pipeline/sbd.py` weights it
accordingly (`STRONG_PUNCT_LANGS`).

For **zh/ja it frequently omits punctuation entirely**. That is a recall problem, not a
precision one: when a mark appears it is still trustworthy. Noisy-OR fusion handles the
asymmetry, but these languages genuinely need the SaT model — punctuation alone will run
sentences together. `segment_timeline()` emits a warning when the fallback is used there.

Two things worth measuring before trusting any of this:

1. **Benchmark the trivial baseline.** "Just split on Whisper's punctuation" is a serious
   baseline for pt/ru/es/fr/de, not a strawman. If SaT plus fusion does not clearly beat it,
   the complexity is not earning its place for those languages.
2. **Do not use casing as a feature.** large-v3's capitalization SER is ~60 — it is noise.

Whisper's *segment* boundaries remain unusable regardless of punctuation quality: they come
from timestamp-token sampling and the 30s window, so they routinely fall nowhere near the
punctuation the model itself emitted. Always re-split from the word array.
