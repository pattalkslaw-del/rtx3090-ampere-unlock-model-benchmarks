# RTX 3090 unlocked phase, 300 W (stock comparison pending)

Measured with the same pinned CUDA benchmark; tokens/sec are **not** a model-quality score.

| Model | Quant | Parameters | pp512 tok/s | pp2048 tok/s | tg128 tok/s | Status |
|---|---|---:|---:|---:|---:|---|
| llama3.2-1b-q8 | Q8_0 | 1.24 B | 25526.03 | 26579.58 | 455.32 | ok |
| llama3.2-3b-q4 | Q4_K_M | 3.21 B | 8876.57 | 9858.87 | 259.45 | ok |
| phi3-mini-q4 | Q4_K_M | 3.82 B | 8258.65 | 7839.24 | 225.81 | ok |
| gemma3-4b-q4 | Q4_K_M | 3.88 B | 8519.27 | 8937.26 | 189.99 | ok |
| qwen3-4b-q4 | Q4_K_M | 4.02 B | 7906.95 | 7911.49 | 199.68 | ok |
| mistral-7b-q4 | Q4_K_M | 7.25 B | 4878.44 | 4853.84 | 147.88 | ok |
| qwen3-8b-q4 | Q4_K_M | 8.19 B | 4761.22 | 4711.13 | 134.11 | ok |
| ornith1-9b-q8 | Q8_0 | 8.95 B | 4015.26 | 4050.36 | 85.09 | ok |
| qwen3-14b-q4 | Q4_K_M | 14.77 B | 2702.91 | 2769.52 | 78.64 | ok |
| qwen3.6-27b-q4 | Q4_K_XL | 26.9 B | 1270.61 | 1286.32 | 39.52 | ok |
| qwen3.8-27b-iq3 | IQ3_S | 26.9 B | 1230.10 | 1292.60 | 44.22 | ok |
| qwen3.8-27b-q4 | Q4_K_XL | 27.32 B | 1277.46 | 1292.29 | 39.72 | ok |
| ornith1-35b-q4 | Q4_K_M | 34.66 B | 3095.46 | 3248.45 | 171.28 | ok |
| qwen3.6-35b-a3b-q4 | Q4_K_M | 34.66 B | 3336.73 | 3374.00 | 154.16 | ok |
| ornith1.5-35b-q4 | Q4_K_M | 35.51 B | 3226.26 | 3268.34 | 165.25 | ok |

See the corresponding JSONL file for per-run samples, standard deviations, SHA256 hashes, and failure details.
