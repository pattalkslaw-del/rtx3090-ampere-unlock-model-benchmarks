# RTX 3090 unlocked CUDA inference at 420 W

Measured with the same pinned CUDA benchmark; tokens/sec are **not** a model-quality score.

| Model | Quant | Parameters | pp512 tok/s | pp2048 tok/s | tg128 tok/s | Status |
|---|---|---:|---:|---:|---:|---|
| llama3.2-1b-q8 | Q8_0 | 1.24 B | 25577.96 | 28986.76 | 466.32 | ok |
| llama3.2-3b-q4 | Q4_K_M | 3.21 B | 9181.59 | 10836.69 | 276.28 | ok |
| phi3-mini-q4 | Q4_K_M | 3.82 B | 8957.54 | 8571.40 | 237.02 | ok |
| gemma3-4b-q4 | Q4_K_M | 3.88 B | 8113.75 | 9860.31 | 202.68 | ok |
| qwen3-4b-q4 | Q4_K_M | 4.02 B | 8542.44 | 8755.26 | 213.06 | ok |
| mistral-7b-q4 | Q4_K_M | 7.25 B | 5403.91 | 5430.18 | 157.95 | ok |
| qwen3-8b-q4 | Q4_K_M | 8.19 B | 5234.67 | 5261.59 | 142.95 | ok |
| ornith1-9b-q8 | Q8_0 | 8.95 B | 4459.52 | 4587.43 | 86.94 | ok |
| qwen3-14b-q4 | Q4_K_M | 14.77 B | 3160.98 | 3116.71 | 84.02 | ok |
| qwen3.6-27b-q4 | Q4_K_XL | 26.9 B | 1437.83 | 1445.13 | 41.91 | ok |
| qwen3.8-27b-iq3 | IQ3_S | 26.9 B | 1405.71 | 1442.65 | 49.28 | ok |
| qwen3.8-27b-q4 | Q4_K_XL | 27.32 B | 1434.12 | 1445.35 | 41.62 | ok |
| ornith1-35b-q4 | Q4_K_M | 34.66 B | 3549.16 | 3647.15 | 180.30 | ok |
| qwen3.6-35b-a3b-q4 | Q4_K_M | 34.66 B | 3673.18 | 3756.13 | 159.70 | ok |
| ornith1.5-35b-q4 | Q4_K_M | 35.51 B | 3416.33 | 3637.98 | 174.09 | ok |

See the corresponding JSONL file for per-run samples, standard deviations, SHA256 hashes, and failure details.
