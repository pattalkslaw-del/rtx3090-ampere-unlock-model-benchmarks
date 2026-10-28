# RTX 3090 GPU Unlock: 15-model paired results

October 28, 2026. Same RTX 3090, same llama.cpp CUDA binary, same 15 GGUF files, and the same 300 W board-power cap. All 15 model files completed three CUDA benchmark cases in both stock and unlocked modes. Each case used three timed samples after built-in warmup. The stock boot used the original NVIDIA firmware initialization with no unlock module loaded.

## Aggregate throughput change

| Case | Geometric mean (15 models) | Median (15 models) | Range |
|---|---:|---:|---:|
| 512-token prefill | -1.22% | -0.52% | -7.83% to +2.37% |
| 2048-token prefill | +0.21% | +0.30% | -0.73% to +1.13% |
| 128-token decode | -0.28% | -0.12% | -0.94% to +0.13% |

Positive percentages mean unlocked ran faster; negative values mean stock ran faster. The geometric mean weights each model equally, rather than letting a 1B model dominate by absolute tokens/second.

## Per-model throughput change (unlocked relative to stock)

| Model | Parameters | Quantization | pp512 | pp2048 | tg128 |
|---|---:|---|---:|---:|---:|
| llama3.2-1b-q8 | 1.2B | Q8_0 | +0.40% | +0.81% | -0.20% |
| llama3.2-3b-q4 | 3.2B | Q4_K_M | -7.83% | +0.20% | -0.82% |
| phi3-mini-q4 | 3.8B | Q4_K_M | -0.03% | +0.30% | -0.43% |
| gemma3-4b-q4 | 3.9B | Q4_K_M | -1.36% | -0.73% | -0.83% |
| qwen3-4b-q4 | 4.0B | Q4_K_M | -0.66% | +0.45% | -0.94% |
| mistral-7b-q4 | 7.2B | Q4_K_M | -1.00% | +0.44% | -0.08% |
| qwen3-8b-q4 | 8.2B | Q4_K_M | +0.10% | +0.44% | -0.05% |
| ornith1-9b-q8 | 9.0B | Q8_0 | -0.15% | -0.28% | +0.11% |
| qwen3-14b-q4 | 14.8B | Q4_K_M | -4.28% | +1.13% | +0.13% |
| qwen3.6-27b-q4 | 26.9B | Q4_K_XL | -0.52% | -0.09% | -0.41% |
| qwen3.8-27b-iq3 | 26.9B | IQ3_S | -2.35% | +0.03% | -0.11% |
| qwen3.8-27b-q4 | 27.3B | Q4_K_XL | +2.37% | +0.15% | -0.10% |
| qwen3.6-35b-a3b-q4 | 34.7B | Q4_K_M | +0.75% | +0.52% | -0.12% |
| ornith1-35b-q4 | 34.7B | Q4_K_M | -3.60% | -0.52% | -0.24% |
| ornith1.5-35b-q4 | 35.5B | Q4_K_M | +0.29% | +0.40% | -0.04% |

This measures quantized GGUF inference throughput, not FP16/BF16 GEMM throughput or model answer quality. A separate direct 8192-matrix GEMM test measured approximately 62% more FP16/BF16 TFLOPS after unlock, but a large dense-arithmetic improvement does not imply the same improvement for memory-bound or quantized inference kernels.

## Variability and interpretation

13 of 45 paired benchmark cases had a coefficient of variation above 5% in at least one phase. Those cases are marked in the machine-readable aggregate JSON and should not support strong conclusions from a single short run. This experiment is one unlocked-then-stock pass, not an alternating ABBA trial. Three benchmark samples do not establish confidence intervals, causal attribution among compute, memory, and offload bottlenecks, or long-term reliability.

The 2080 Super was excluded through CUDA_VISIBLE_DEVICES=0, and both Intel B70s remained assigned to a separate inference service. All model files have published SHA-256 digests. The exact CUDA binary hash and runtime flags are preserved in both environment files.

## Files

- comparison.csv: matched stock/unlocked tokens per second, per test, and percentage change.
- comparison.md: complete 45-case Markdown table.
- unlocked.jsonl and stock.jsonl: sample-level timings, model hashes, and execution metadata.
- unlocked-environment.json and stock-environment.json: runtime controls and binary hash.
- models.json: 15 model filenames, quantizations, checksums, and source references.
- paired-aggregates.json: per-case aggregate calculations and high-variability flags.
- unlocked-telemetry-summary.json and stock-telemetry-summary.json: sampled GPU temperatures and clock-limit flags.

No model weights, firmware binaries, client data, or machine access credentials are included.

## Post-test restoration

Following the 15-model stock test, the experimental packages gpu-unlock-dkms 1.1.0 and gpu-unlock-firmware 1.1.0 were purged. The command removed only those two packages. The module is absent from lsmod, DKMS status and the rebuilt initramfs. The preserved original 7.0.0-31 stock recovery initramfs still matched SHA-256 eaf34084c9f38943fe74f923e4a1652a532573814ec2c811b0b79a54874802cb, and GRUB syntax passed validation.

Production NVIDIA driver 595.99.02 remains installed, RTX 3090 power limit 420 W and RTX 2080 Super 250 W. Both Intel B70 GPUs remain bound to xe. Maya and Ornith were active, their model endpoints returned HTTP 200, and both passed an independent READY completion after restoration.

No extra reboot was required after package removal because the host had already completed a verified full chassis-off stock boot with no unlock selector and no loaded unlock module.
