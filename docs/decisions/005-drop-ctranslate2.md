# 005. Whisper on plain PyTorch, not faster-whisper

**Status:** Settled 2026-07-20.

## Decision

Run Whisper through `transformers` on plain PyTorch. Do not depend on `faster-whisper` or
`whisperx`, both of which reach CTranslate2.

## The problem this solves

CTranslate2 ships no aarch64 CUDA wheels. This is not a build failure you notice — `pip
install faster-whisper` **succeeds**, and the resulting install silently runs on CPU at
roughly 1/20 speed. On a long job you ship something that looks like it worked.

Verified rather than assumed: the current `ctranslate2-4.8.1` aarch64 wheel bundles no CUDA
libraries at all and contains zero cuBLAS symbols.

The documented fix was a from-source CMake build pinning
`CMAKE_CUDA_ARCHITECTURES="…;120;121;121-virtual"`, forcing C++17, and patching a Thrust
compat shim because Thrust 2.x removed `thrust::unary_function`, which CTranslate2 still
references. Estimated at a day of work, and it was the single largest remaining item in the
deployment notes.

## Why we removed the dependency instead of fixing it

That build buys speed on exactly one architecture and costs the project portability
everywhere else. The project is meant to run on the deployment box, on a laptop, and on
whatever a future contributor has.

`transformers` runs the same Whisper weights on plain PyTorch, so one code path covers CUDA,
MPS and CPU across aarch64 and x86. Torch itself was never the problem — `torch 2.13.0+cu130`
installs from stock aarch64 wheels and detects the GB10 immediately, with no compilation.

The pointed version: **the hardest task in the deployment plan turned out to be deletable
rather than solvable.** A day of build engineering was avoided by removing the thing that
required it.

## What it costs

faster-whisper is genuinely faster — roughly 3-4x — where CT2 CUDA wheels exist, i.e. x86
Linux. Measured here, the transformers path runs at about 1.04x realtime for `large-v3`,
which is slower than this hardware should manage; batched/chunked decoding is the obvious
lever and has not been pulled yet.

It also cost the two hallucination signals the old path exposed. `no_speech_prob` and
`avg_logprob` are not cheaply available from the HF pipeline, so a compression-ratio test
replaced them — text that compresses far better than language does indicates a decoder stuck
in a loop. That turned out to be *stronger* than what it replaced, since it catches long
repetition loops that the old fixed two-segment repeat window always missed.

An opt-in `faster-whisper` backend remains reasonable for x86 CUDA deployments. It must stay
opt-in, and must never be required.

## Consequences

- No source build on any platform. `docs/DEPLOY.md` lost its largest section.
- `assert_cuda()` was rewritten. It used to promise CT2-CPU-fallback detection while only
  checking `torch.cuda.is_available()` — never the same question. It now checks the resolved
  device.
- The wav2vec2 forced-alignment pass went with it. See [006](006-no-forced-alignment.md).
