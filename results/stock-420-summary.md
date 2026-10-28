# RTX 3090 stock CUDA inference at 420 W

Measured with the same pinned CUDA benchmark; tokens/sec are **not** a model-quality score.

| Model | Quant | Parameters | pp512 tok/s | pp2048 tok/s | tg128 tok/s | Status |
|---|---|---:|---:|---:|---:|---|
| llama3.2-1b-q8 | Q8_0 | 1.24 B | 26114.91 | 28690.15 | 467.43 | ok |
| llama3.2-3b-q4 | Q4_K_M | 3.21 B | 10310.83 | 10687.96 | 277.51 | ok |
| phi3-mini-q4 | Q4_K_M | 3.82 B | 8990.56 | 8580.17 | 238.19 | ok |
| gemma3-4b-q4 | Q4_K_M | 3.88 B | 8554.25 | 9809.75 | 202.64 | ok |
| qwen3-4b-q4 | Q4_K_M | 4.02 B | 8498.87 | 8635.11 | 213.70 | ok |
| mistral-7b-q4 | Q4_K_M | 7.25 B | 5397.21 | 5381.93 | 157.99 | ok |
| qwen3-8b-q4 | Q4_K_M | 8.19 B | 4892.89 | 5221.61 | 142.97 | ok |
| ornith1-9b-q8 | Q8_0 | 8.95 B | 4128.94 | 4589.03 | 86.93 | ok |
| qwen3-14b-q4 | Q4_K_M | 14.77 B | 3151.83 | 3076.48 | 84.00 | ok |
| qwen3.6-27b-q4 | Q4_K_XL | 26.9 B | 1433.58 | 1438.89 | 41.91 | ok |
| qwen3.8-27b-iq3 | IQ3_S | 26.9 B | 1401.46 | 1437.92 | 49.18 | ok |
| qwen3.8-27b-q4 | Q4_K_XL | 27.32 B | 1431.97 | 1441.99 | 41.63 | ok |
| ornith1-35b-q4 | Q4_K_M | 34.66 B | 3523.74 | 3625.50 | 179.80 | ok |
| qwen3.6-35b-a3b-q4 | Q4_K_M | 34.66 B | 3660.30 | 3731.06 | 159.71 | ok |
| ornith1.5-35b-q4 | Q4_K_M | 35.51 B | 3535.74 | 3613.49 | 173.40 | ok |

See the corresponding JSONL file for per-run samples, standard deviations, SHA256 hashes, and failure details.
