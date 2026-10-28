# Experiment specification

Independent variable: stock NVIDIA RTX 3090 firmware initialization versus GPU Unlock Ampere initialization on the SAME physical card.

Controls: 300 W cap, 24 GiB VRAM, NVIDIA driver 595.99.02, Linux 7.0.0-31-generic, CUDA Toolkit 12.8, pinned llama.cpp source commit 6a1a922, compute capability 8.6, host RAM/CPU, model SHA256, quantization, GPU layer count, thread count, batch sizes, and benchmark token lengths.

The CUDA benchmark binary must be byte-identical across both phases. Each model isolates CUDA device 0 and hides the RTX 2080 with CUDA_VISIBLE_DEVICES. The Intel B70s remain on an independent inference stack.

llama-bench tests:
- Prompt processing: pp512 and pp2048.
- Token generation: tg128.
- Three repetitions per test and built-in warmup.
- 16 CPU threads, batch 512, microbatch 512, KV cache f16, FlashAttention on.
- CUDA0, split-mode none, all model layers to GPU (-ngl 99), 300 W cap.

Record raw avg_ts/stddev_ts/samples_ts, plus model load failures and environment state. Report tokens/s and paired unlocked/stock percentage. Mark model size, quantization and actual GPU offloading.

Synthetic throughput alone cannot establish model answer quality. Functional generation tests must be reported separately.

## Safety

Monitor temperature, board power, clock limiting reasons, GPU/PCIe errors, and host memory. Immediate stop at 85 C, hardware thermal/power-brake event, new GPU/PCIe error, or less than 8 GiB available memory. Stop for sustained software throttling; a brief low-temperature software slowdown flag alone is not proof of unsafe heating. Restore original 420 W GPU limit and restart production inference after the run, even on error.

## Publication

Do not conflate FP16 GEMM throughput with model inference tokens/s. Do not compare different quant files or GPU-offload patterns as matched. Publish failure outcomes; do not silently drop failed models. Do not upload model weights or firmware blobs.

## Completed paired dataset, October 28, 2026

Both 15-model phases passed all 45 synthetic throughput test cases. Models were tested in the same order; the unlocked phase preceded a genuine chassis-off power cycle and stock boot, after which the same dedicated CUDA benchmark was run. The stock boot had no GPU Unlock kernel selector or loaded module. All published model and binary hashes match between phases.

The dataset is a paired single-run comparison, not an alternating ABBA study. Some short-context cases had coefficients of variation above 5% and should not be treated as reproducible gains or losses without additional repetitions. Aggregated results are model-weighted geometric means of unlocked/stock throughput ratios. See results/analysis.md for all values and caveats.
