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

## There is no build step

This used to be the hard part. `faster-whisper` depends on CTranslate2, which ships no
aarch64 CUDA wheels — `pip install` succeeds and then silently runs on CPU at roughly 1/20
speed, and fixing it meant a from-source CMake build with a Thrust compat shim for CUDA 13.

**That dependency is gone.** Whisper now runs through plain PyTorch (`transformers`), so
installation is stock wheels on every platform. Verified on this hardware:
`torch 2.13.0+cu130` from the standard aarch64 index, CUDA available, GB10 detected, no
compilation.

```bash
python3 -m venv .venv
.venv/bin/pip install -U pip wheel setuptools
.venv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
.venv/bin/python -c "import torch; print(torch.cuda.is_available())"   # must print True
.venv/bin/pip install -e '.[dev,web]'
```

Install torch **first and alone**, then re-check `torch.cuda.is_available()` after every
subsequent `pip install`. `kokoro` and `wtpsplit` will pull a CPU-only torch from PyPI and
silently replace the CUDA build if given the chance — the same silent-degradation failure
the CTranslate2 problem used to cause, arriving by a different route.

A host venv is now the recommended layout. The container existed to hold the build; with
nothing left to build, a venv gives faster edit-run cycles. Keep the LLM server in Docker.

## Silent CPU fallback is still the failure mode to fear

Components in this stack degrade to CPU rather than erroring, so you can ship something that
appears to work and is 20x slower than it should be.

`app/pipeline/asr.py:assert_cuda()` guards this — keep `BAG_REQUIRE_CUDA=1` in production.
It now checks the *resolved* device rather than merely asking torch whether CUDA exists.

`BAG_DEVICE` defaults to `auto`, which walks cuda → mps → cpu. Set it explicitly to pin.

## Things that are NOT problems

- **PyTorch.** sm_121 is binary-compatible with sm_120. Stock aarch64 wheels from
  `download.pytorch.org/whl/cu130` (or `cu128`) work — confirmed. Ignore the custom
  "sm_121 wheels" circulating on HuggingFace; they solve a problem you do not have.
- **The capability warning.** `Minimum and Maximum cuda capability supported by this version
  of PyTorch is (8.0) - (12.0)` appears on GB10 and is cosmetic. Don't chase it.
- **Kokoro.** Runs fine — but use the **PyTorch path, not ONNX Runtime**. `onnxruntime-gpu`
  aarch64 CUDA support is patchy and several Kokoro wrappers silently downgrade to CPU when
  they detect aarch64. At 82M params the PyTorch path is fast regardless. For a Japanese
  target voice, MeCab needs `unidic-lite` plus a symlink into `site-packages/unidic/dicdir`.
- **yt-dlp.** Needs a JavaScript runtime for YouTube extraction or it fails with a bare
  HTTP 403. `fetch.py` auto-detects deno, node or bun; install one if none is present.

## Base image

If you do containerise, `nvcr.io/nvidia/pytorch:25.10-py3` or newer matches host CUDA 13 and
ships `compute_120` PTX that JITs to sm_121. For vLLM, `nvcr.io/nvidia/vllm:26.03-py3` is
proven on GB10.

---

## Translation backend

`app/config.py` speaks OpenAI-compatible HTTP, so any server that honours it works —
`BAG_LLM_BASE_URL` is the only change.

**What is actually running on this box** is vLLM serving `nvidia/Qwen3.6-27B-NVFP4`. The
working invocation is preserved in a container named `vllm-qwen36-27b-nvfp4`, so:

```bash
docker start vllm-qwen36-27b-nvfp4      # ~4 min cold start
curl -s localhost:8000/v1/models        # wait for this to answer
```

```
BAG_LLM_BASE_URL=http://localhost:8000/v1
BAG_LLM_MODEL=nvidia/Qwen3.6-27B-NVFP4
```

Flags that matter on GB10, recoverable with `docker inspect`: `VLLM_NVFP4_GEMM_BACKEND=marlin`,
`TORCH_CUDA_ARCH_LIST=12.1a`, `--gpu-memory-utilization 0.5`, `--kv-cache-dtype fp8`,
`--enforce-eager`, `--enable-prefix-caching`, and `enable_thinking: false` — reasoning traces
would wreck JSON parsing.

Measured: ru→en translation is fluent, and `response_format: {"type":"json_object"}` is
accepted (as is omitting it). Single-stream decode is **~10 tok/s** for this dense 27B, which
is the bandwidth-bound behaviour predicted below — batching is worth more than anything else.

**Where to go next:** a large MoE. This is counterintuitive and worth internalising:

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

Observed caveat: on the first real Russian clip, `large-v3` punctuated and capitalised the
first half cleanly and then dropped both for a stretch, emitting bare lowercase runs. So
`STRONG_PUNCT_LANGS` is right about punctuation *precision* but should not be read as a
promise of *recall*, even for Russian. The pause signal carried those segments.

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
