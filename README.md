# RTX 3090 GPU Unlock: reproducible cross-model inference benchmarks

Community benchmark project comparing **the same GeForce RTX 3090 before and after applying [Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock)**, with matched model weights, power limits, CUDA backend, and inference settings.

**Status:** GPU unlock initialization and GEMM speedup have been independently observed. The 10–15-model CUDA GGUF suite and the cold-boot stock comparison are being collected. Results will be added as verified—unmeasured models are **not** represented as tested.

## Research question

Does unlocking restricted FP16/BF16 compute throughput on an RTX 3090 materially improve actual LLM inference? Does the answer vary with model size, architecture, mixture-of-experts routing, quantization, prompt prefill and token generation?

An earlier hybrid two-NVIDIA-GPU Maya experiment showed approximately +62% sustained FP16/BF16 GEMM at the same 300 W power limit, but less than 1.3% improvement in 8K/32K prefill latency. That configuration involves different workloads and other bottlenecks. Those observations are preliminary context, not substitutes for the new single-GPU model suite.

## Experimental controls

- GPU: RTX 3090 24 GiB, 300 W cap, exclusively selected with CUDA_VISIBLE_DEVICES=0.
- Inference engine: identical dedicated llama.cpp CUDA backend for both configurations. Source commit, compiler, binary SHA256, and tested flags recorded.
- Tests: 512 and 2048 prompt tokens and 128 generated tokens, each with three repetitions after warmup; mean and standard deviation in tokens/second.
- Models: 10–15 GGUF files from approximately 1B through 35B, including Qwen3.6-35B-A3B, Qwen3.8-27B, and Phi-3 Mini. Exact weight SHA256 hashes in model manifest.
- Isolation: production NVIDIA inference paused during benchmarks; Intel inference remains on separate GPUs.
- Safety: immediate hardware temperature/power-brake/GPU-fault stops and a sustained-software-throttle detector.
- Phases: unlocked first (currently active), then a **verified full chassis-off** into the preserved stock kernel, followed by stock. Same corpus and settings.

## Reproduction

See [methodology](docs/METHODOLOGY.md) and [runner](scripts/bench_suite.py). Model weights are NOT included. Download the indicated exact quantization from its original distributor, verify SHA256, and prepare a local manifest from [example](docs/manifest.local.example.json). Never mix CUDA and Vulkan, or 3090-only and dual-GPU numbers, in a matched comparison.

## Data integrity

Raw benchmark JSONL, command flags, inventory, telemetry, failure records, checkpoints, and hashes are collected privately, then sanitized for public result tables. Failed model loads and out-of-memory results are recorded explicitly. Model weights, proprietary firmware, private machine addresses, account identifiers, and client data are not published.

**Warning:** GPU Unlock uses third-party low-level firmware initialization. A test boot may hang and require independent chassis power controls. Do not use without a verified stock rollback image and out-of-band recovery.

Original scripts in this repository use the MIT license; third-party models and GPU Unlock retain their own licenses.
