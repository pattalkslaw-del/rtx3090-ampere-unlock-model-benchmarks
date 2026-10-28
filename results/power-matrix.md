# RTX 3090 four-way power and firmware comparison

Same RTX 3090, identical CUDA llama-bench binary, 15 identically hashed GGUF models, three timing samples per case. This is one sequential comparison, not a randomized repeated study. A power limit is a ceiling, not measured power consumption.

## Geometric-mean throughput changes across 15 models

| Workload | Unlock effect 300 W | Unlock effect 420 W | Stock: 420 vs 300 W | Unlocked: 420 vs 300 W |
|---|---:|---:|---:|---:|
| 512-token prefill | -1.22% | -0.37% | +7.89% | +8.81% |
| 2048-token prefill | +0.21% | +0.67% | +10.78% | +11.29% |
| 128-token generation | -0.28% | -0.04% | +5.47% | +5.72% |

Positive values mean the second condition is faster. The model-weighted geometric mean weights each model equally. High variability (>5% within-phase coefficient of variation) appears in 14 of 45 cases across one or more conditions.

## Per-model throughput and unlock effect at 420 W

| Model | Workload | Stock 300 tok/s | Unlock 300 tok/s | Stock 420 tok/s | Unlock 420 tok/s | 420 W unlock gain | Variability |
|---|---|---:|---:|---:|---:|---:|---|
| gemma3-4b-q4 | pp512 | 8636.50 | 8519.27 | 8554.25 | 8113.75 | -5.15% | >5% CV |
| gemma3-4b-q4 | pp2048 | 9003.34 | 8937.26 | 9809.75 | 9860.31 | +0.52% |  |
| gemma3-4b-q4 | tg128 | 191.58 | 189.99 | 202.64 | 202.68 | +0.02% |  |
| llama3.2-1b-q8 | pp512 | 25423.17 | 25526.03 | 26114.91 | 25577.96 | -2.06% | >5% CV |
| llama3.2-1b-q8 | pp2048 | 26367.18 | 26579.58 | 28690.15 | 28986.76 | +1.03% |  |
| llama3.2-1b-q8 | tg128 | 456.26 | 455.32 | 467.43 | 466.32 | -0.24% |  |
| llama3.2-3b-q4 | pp512 | 9630.95 | 8876.57 | 10310.83 | 9181.59 | -10.95% | >5% CV |
| llama3.2-3b-q4 | pp2048 | 9839.29 | 9858.87 | 10687.96 | 10836.69 | +1.39% |  |
| llama3.2-3b-q4 | tg128 | 261.61 | 259.45 | 277.51 | 276.28 | -0.44% |  |
| mistral-7b-q4 | pp512 | 4927.51 | 4878.44 | 5397.21 | 5403.91 | +0.12% | >5% CV |
| mistral-7b-q4 | pp2048 | 4832.46 | 4853.84 | 5381.93 | 5430.18 | +0.90% |  |
| mistral-7b-q4 | tg128 | 148.00 | 147.88 | 157.99 | 157.95 | -0.03% |  |
| ornith1-35b-q4 | pp512 | 3211.08 | 3095.46 | 3523.74 | 3549.16 | +0.72% | >5% CV |
| ornith1-35b-q4 | pp2048 | 3265.36 | 3248.45 | 3625.50 | 3647.15 | +0.60% |  |
| ornith1-35b-q4 | tg128 | 171.68 | 171.28 | 179.80 | 180.30 | +0.28% |  |
| ornith1-9b-q8 | pp512 | 4021.24 | 4015.26 | 4128.94 | 4459.52 | +8.01% | >5% CV |
| ornith1-9b-q8 | pp2048 | 4061.71 | 4050.36 | 4589.03 | 4587.43 | -0.04% |  |
| ornith1-9b-q8 | tg128 | 85.00 | 85.09 | 86.93 | 86.94 | +0.02% |  |
| ornith1.5-35b-q4 | pp512 | 3216.80 | 3226.26 | 3535.74 | 3416.33 | -3.38% | >5% CV |
| ornith1.5-35b-q4 | pp2048 | 3255.45 | 3268.34 | 3613.49 | 3637.98 | +0.68% |  |
| ornith1.5-35b-q4 | tg128 | 165.32 | 165.25 | 173.40 | 174.09 | +0.40% |  |
| phi3-mini-q4 | pp512 | 8260.74 | 8258.65 | 8990.56 | 8957.54 | -0.37% | >5% CV |
| phi3-mini-q4 | pp2048 | 7816.14 | 7839.24 | 8580.17 | 8571.40 | -0.10% |  |
| phi3-mini-q4 | tg128 | 226.79 | 225.81 | 238.19 | 237.02 | -0.49% |  |
| qwen3-14b-q4 | pp512 | 2823.80 | 2702.91 | 3151.83 | 3160.98 | +0.29% | >5% CV |
| qwen3-14b-q4 | pp2048 | 2738.53 | 2769.52 | 3076.48 | 3116.71 | +1.31% |  |
| qwen3-14b-q4 | tg128 | 78.54 | 78.64 | 84.00 | 84.02 | +0.02% |  |
| qwen3-4b-q4 | pp512 | 7959.16 | 7906.95 | 8498.87 | 8542.44 | +0.51% | >5% CV |
| qwen3-4b-q4 | pp2048 | 7876.18 | 7911.49 | 8635.11 | 8755.26 | +1.39% |  |
| qwen3-4b-q4 | tg128 | 201.58 | 199.68 | 213.70 | 213.06 | -0.30% |  |
| qwen3-8b-q4 | pp512 | 4756.49 | 4761.22 | 4892.89 | 5234.67 | +6.99% | >5% CV |
| qwen3-8b-q4 | pp2048 | 4690.63 | 4711.13 | 5221.61 | 5261.59 | +0.77% |  |
| qwen3-8b-q4 | tg128 | 134.17 | 134.11 | 142.97 | 142.95 | -0.01% |  |
| qwen3.6-27b-q4 | pp512 | 1277.23 | 1270.61 | 1433.58 | 1437.83 | +0.30% |  |
| qwen3.6-27b-q4 | pp2048 | 1287.47 | 1286.32 | 1438.89 | 1445.13 | +0.43% |  |
| qwen3.6-27b-q4 | tg128 | 39.69 | 39.52 | 41.91 | 41.91 | +0.01% |  |
| qwen3.6-35b-a3b-q4 | pp512 | 3311.90 | 3336.73 | 3660.30 | 3673.18 | +0.35% | >5% CV |
| qwen3.6-35b-a3b-q4 | pp2048 | 3356.47 | 3374.00 | 3731.06 | 3756.13 | +0.67% |  |
| qwen3.6-35b-a3b-q4 | tg128 | 154.35 | 154.16 | 159.71 | 159.70 | -0.01% |  |
| qwen3.8-27b-iq3 | pp512 | 1259.70 | 1230.10 | 1401.46 | 1405.71 | +0.30% | >5% CV |
| qwen3.8-27b-iq3 | pp2048 | 1292.23 | 1292.60 | 1437.92 | 1442.65 | +0.33% |  |
| qwen3.8-27b-iq3 | tg128 | 44.27 | 44.22 | 49.18 | 49.28 | +0.20% |  |
| qwen3.8-27b-q4 | pp512 | 1247.89 | 1277.46 | 1431.97 | 1434.12 | +0.15% | >5% CV |
| qwen3.8-27b-q4 | pp2048 | 1290.33 | 1292.29 | 1441.99 | 1445.35 | +0.23% |  |
| qwen3.8-27b-q4 | tg128 | 39.76 | 39.72 | 41.63 | 41.62 | -0.02% |  |

## Scope and limitations

- These are quantized CUDA GGUF inference throughput tests, not full-precision GEMM, training throughput, answer accuracy, or hybrid multi-GPU Maya inference.
- GPU offload, CUDA backend, model SHA-256, binary SHA-256, execution flags and GPU power cap were validated for each paired condition.
- Three samples cannot establish reliable confidence intervals for small differences, especially pp512.
- The run order was unlocked 300 W, stock 300 W, stock 420 W, unlocked 420 W. Environmental drift and benchmark order are confounders.
- See stock-420.jsonl and unlocked-420.jsonl and the previous 300 W files for all individual timing samples. Thermal summaries distinguish sampled power draw from configured caps.
