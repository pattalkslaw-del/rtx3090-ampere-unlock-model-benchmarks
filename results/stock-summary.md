# RTX 3090 stock phase, 300 W

Measured with the same pinned CUDA benchmark; tokens/sec are **not** a model-quality score.

| Model | Quant | Parameters | pp512 tok/s | pp2048 tok/s | tg128 tok/s | Status |
|---|---|---:|---:|---:|---:|---|
| llama3.2-1b-q8 | Q8_0 | 1.24 B | 25423.17 | 26367.18 | 456.26 | ok |
| llama3.2-3b-q4 | Q4_K_M | 3.21 B | 9630.95 | 9839.29 | 261.61 | ok |
| phi3-mini-q4 | Q4_K_M | 3.82 B | 8260.74 | 7816.14 | 226.79 | ok |
| gemma3-4b-q4 | Q4_K_M | 3.88 B | 8636.50 | 9003.34 | 191.58 | ok |
| qwen3-4b-q4 | Q4_K_M | 4.02 B | 7959.16 | 7876.18 | 201.58 | ok |
| mistral-7b-q4 | Q4_K_M | 7.25 B | 4927.51 | 4832.46 | 148.00 | ok |
| qwen3-8b-q4 | Q4_K_M | 8.19 B | 4756.49 | 4690.63 | 134.17 | ok |
| ornith1-9b-q8 | Q8_0 | 8.95 B | 4021.24 | 4061.71 | 85.00 | ok |
| qwen3-14b-q4 | Q4_K_M | 14.77 B | 2823.80 | 2738.53 | 78.54 | ok |
| qwen3.6-27b-q4 | Q4_K_XL | 26.9 B | 1277.23 | 1287.47 | 39.69 | ok |
| qwen3.8-27b-iq3 | IQ3_S | 26.9 B | 1259.70 | 1292.23 | 44.27 | ok |
| qwen3.8-27b-q4 | Q4_K_XL | 27.32 B | 1247.89 | 1290.33 | 39.76 | ok |
| ornith1-35b-q4 | Q4_K_M | 34.66 B | 3211.08 | 3265.36 | 171.68 | ok |
| qwen3.6-35b-a3b-q4 | Q4_K_M | 34.66 B | 3311.90 | 3356.47 | 154.35 | ok |
| ornith1.5-35b-q4 | Q4_K_M | 35.51 B | 3216.80 | 3255.45 | 165.32 | ok |

See the corresponding JSONL file for per-run samples, standard deviations, SHA256 hashes, and failure details.
