# RTX 3090 GPU Unlock: reproducible cross-model inference benchmarks

Community benchmark project comparing **the same GeForce RTX 3090 before and after applying [Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock)**, with matched model weights, power limits, CUDA backend, and inference settings.

**Results, October 28, 2026:** The stock and unlocked suites both completed **15 of 15 models**, with **45 matched test cases** (512-token prefill, 2048-token prefill, and 128-token generation) and three timing samples per case. Same RTX 3090, model SHA-256 hashes, CUDA benchmark binary and 300 W power limit.

**The unlock's approximately 62% FP16/BF16 GEMM advantage did not produce a comparable benefit in quantized GGUF inference.** The geometric mean unlocked/stock throughput change across 15 models was:

| Workload | Mean unlocked gain |
|---|---:|
| 512-token prefill | **−1.22%** |
| 2048-token prefill | **+0.21%** |
| 128-token generation | **−0.28%** |

This is a controlled single-card inference result, not a claim about FP16 training, unquantized inference, other engines or architectures. Thirteen of 45 paired cases had more than 5% within-phase timing variation; the 512-token aggregate is especially sensitive to noisy short runs.

**Read the measurements:**

- [Analysis and complete model-by-model speedup table](results/analysis.md)
- [45-case paired comparison, Markdown](results/comparison.md) and [CSV](results/comparison.csv)
- [Stock throughput table](results/stock-summary.md) and [unlocked throughput table](results/unlocked-summary.md)
- [Stock raw results](results/stock.jsonl) and [unlocked raw results](results/unlocked.jsonl)
- [15 model SHA-256 hashes, filenames, quantizations and source references](results/models.json)
- [Stock environment](results/stock-environment.json), [unlocked environment](results/unlocked-environment.json), and [paired aggregate statistics](results/paired-aggregates.json)
- [Stock GPU telemetry](results/stock-telemetry-summary.json), [unlocked GPU telemetry](results/unlocked-telemetry-summary.json), and [evidence digests](results/stock-evidence-hashes.json)

Model weights and proprietary GPU firmware are deliberately excluded.

**Post-test state verified:** After a full chassis-off boot into stock Linux 7.0.0-31, the two experimental GPU Unlock 1.1.0 packages were purged. The unlock module is absent from the loaded modules, DKMS and the rebuilt stock initramfs. NVIDIA driver 595.99.02 and the original 420 W/250 W board limits remain in service. Both Maya and Ornith returned successful READY completions after the cleanup; the Intel B70 driver bindings are unchanged. See the [restoration verification](results/analysis.md#post-test-restoration).


## Research question

Does unlocking restricted FP16/BF16 compute throughput on an RTX 3090 materially improve actual LLM inference? Does the answer vary with model size, architecture, mixture-of-experts routing, quantization, prompt prefill and token generation?

An earlier hybrid two-NVIDIA-GPU Maya experiment showed approximately +62% sustained FP16/BF16 GEMM at the same 300 W power limit, but less than 1.3% improvement in 8K/32K prefill latency. That configuration involves different workloads and other bottlenecks. Those observations are preliminary context, not substitutes for the new single-GPU model suite.

## Experimental controls

- GPU: RTX 3090 24 GiB, 300 W cap, exclusively selected with CUDA_VISIBLE_DEVICES=0.
- Inference engine: identical dedicated llama.cpp CUDA backend for both configurations. Source commit, compiler, binary SHA256, and tested flags recorded.
- Tests: 512 and 2048 prompt tokens and 128 generated tokens, each with three repetitions after warmup; mean and standard deviation in tokens/second.
- Models: 15 GGUF files from approximately 1B through 35B, including Qwen3.6-35B-A3B, Qwen3.8-27B, and Phi-3 Mini. Exact weight SHA256 hashes in model manifest.
- Isolation: production NVIDIA inference paused during benchmarks; Intel inference remains on separate GPUs.
- Safety: immediate hardware temperature/power-brake/GPU-fault stops and a sustained-software-throttle detector.
- Phases completed: unlocked first, followed by a verified **full chassis-off** and the preserved stock kernel. Same model corpus, CUDA binary and benchmark settings.

## Reproduction

See [methodology](docs/METHODOLOGY.md) and [runner](scripts/bench_suite.py). Model weights are NOT included. Download the indicated exact quantization from its original distributor, verify SHA256, and prepare a local manifest from [example](docs/manifest.local.example.json). Never mix CUDA and Vulkan, or 3090-only and dual-GPU numbers, in a matched comparison.

## Data integrity

Raw benchmark JSONL, command flags, inventory, telemetry, failure records, checkpoints, and hashes are collected privately, then sanitized for public result tables. Failed model loads and out-of-memory results are recorded explicitly. Model weights, proprietary firmware, private machine addresses, account identifiers, and client data are not published.

**Warning:** GPU Unlock uses third-party low-level firmware initialization. A test boot may hang and require independent chassis power controls. Do not use without a verified stock rollback image and out-of-band recovery.

Original scripts in this repository use the MIT license; third-party models and GPU Unlock retain their own licenses.
