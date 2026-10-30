# GPT-OSS-120B on four heterogeneous GPUs: CUDA + Intel SYCL

Experiment date: 2026-10-29/30 US Central. Host: ASRock ROMED8-2T / EPYC 7532 / 256 GiB DRAM, RTX 3090 24 GiB + RTX 2080 SUPER 8 GiB + 2x Intel Arc Pro B70 32 GiB. Weights: GPT-OSS-120B MXFP4 GGUF (~60 GB), read-only from network storage. Runtime: llama.cpp source commit `52ffa02`, hybrid executable built with `GGML_CUDA=ON`, `GGML_SYCL=ON`, Intel oneAPI compiler 2026.1 and CUDA toolkit. This experiment made no verified upstream inference kernel changes: the effective modifications were *combined-backend build configuration, environment, device assignment and benchmarking scripts*.

## Primary results

| Workload | Hybrid prompt tok/s | Hybrid decode tok/s | Outcome |
|---|---:|---:|---|
| Short-input 8K allocation, 512 output | 525.0 | 85.9 | success |
| Repetitive ~83 KB input, 32K allocation | 586.8 | 71.3 | success |
| Repetitive ~229 KB input, 64K allocation | 519.6 | 52.5 | success |
| Same ~229 KB input, 128K allocation | 518.9 | 52.5 | success; not 128K occupied |
| Varied ~220 KB input / 64K allocation | 420.9 | 44.4 | success; 53,961 input tokens measured by model tokenizer |
| Identical varied input repeated | 421.1 | 44.4 | success; normalized generated model output identical |
| Varied ~500 KB input / 128K allocation | 296.6 | 26.4 | success; **122,918 input tokens**, tokenizer measured |
| Varied ~220 KB input / 64K; 2,048 max output | 420.6 | 44.3 | success |

Prior comparison: RTX 3090 + RTX 2080 SUPER with 36 MoE experts resident on CPU, the same earlier 229 KB input and 512 generation limit, achieved 149.1 prefill / 30.8 decode tok/s. Four-device hybrid on that input obtained 519.8 prefill / 52.6 decode tok/s (71% higher decoder speed). These are *single-run observed timings*, not a controlled distribution-level statistical comparison. Actual input-token count for that 229 KB prompt measured later at 36,797 tokens (the tokenizer call did not include every possible chat-template token).

## Placement findings

Device enumeration confirms CUDA0 3090, CUDA1 2080 SUPER, SYCL0 and SYCL1 B70. Verbose trace records model, KV cache and compute buffers on all four backends (CUDA0 ~16.5 GiB model buffer, CUDA1 ~4.9 GiB, SYCL0 ~19.8 GiB, SYCL1 ~18.7 GiB on short run). This shows **real multi-backend layer placement and computation**. It does NOT establish individually routed MoE experts executing across backends; there is no verified expert-level scheduler patch. Do not describe this as dynamic CUDA/SYCL per-expert execution.

## Repeatability / correctness

Two identical 53,961-token-input runs with seed=42, temp=0, top-k=1 yielded byte-identical normalized model generation (SHA-256 both `ea506deb2d0e977851e976ef711ceae7c47a7af36d482bf1216d51be48c23310`). Full terminal stdout hashes differ because they include non-model text / timings. This is an internal reproducibility check, **not numerical-logit parity with CPU or CUDA-only backends**. The previously observed CPU-vs-Intel first-token reasoning divergence remains unproven as a defect.

## Limitations and next work

Native 128K context, limit 131,072 tokens inclusive of output. The near-full-context input (122,918) leaves 8,154 tokens theoretical capacity before chat-template overhead and answer. Some generated answers entered the reasoning channel and were cutoff by the output limit. Larger factual/semantic correctness suites, multirun variance, stress/concurrency and token-level logits parity still require more work. An attempted two-device CUDA0+SYCL0 run failed because CUDA0 tried to allocate ~26.3 GiB on the 24 GiB card; four-device split succeeded.

This is an **engineering proof-of-concept success**, not production certification. No Maya or Ornith services were restarted for these tests.

## Reproduction

With Intel oneAPI environment loaded, configure separate CMake build `-DGGML_CUDA=ON -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_DNN=OFF -DGGML_CUDA_FA=ON`. Link CUDA toolkit and Intel MKL/SYCL/Level Zero packages. `llama-cli --list-devices` should show `CUDA0,CUDA1,SYCL0,SYCL1`. Use the script collection in this folder and supply your own model path. Do not hardcode infrastructure credentials. CLI examples: `--device CUDA0,CUDA1,SYCL0,SYCL1 --n-gpu-layers all --ctx-size 131072 --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on --temp 0 --top-k 1`. Observe GPU VRAM during load. Performance is highly input-dependent.

## Final acceptance checks (2026-10-30)

- Token occupancy measured with `llama-tokenize --show-count`: 53,961 on varied 64K dataset, 122,918 on varied 128K dataset, and 36,797 on earlier repeated workload.
- Normalized generated output of the 53,961-token repeated test matched exactly, hash `ea506deb2d0e977851e976ef711ceae7c47a7af36d482bf1216d51be48c23310`.
- Verbose backend trace confirms model allocations and KV/compute buffers on CUDA0, CUDA1, SYCL0, SYCL1; this supports distributed multi-device execution, **not** a per-expert CUDA/SYCL router.
- Three identical 512-token accuracy probes completed exit-code 0, all produced the correct arithmetic values (42,824 and 2,281.5) and correctly reasoned that ordinary power-of-attorney agency ends at death. However, all three stopped while still generating reasoning-channel text, without a final answer. These are therefore **partial reasoning checks**, not fully passing final-answer accuracy checks. A larger completion budget and independent scoring are required.
- Important: the 128K near-full-input test contained 122,918 tokens, leaving 8,154 native-context tokens *before accounting for chat framing*. It did not consume every single native-context token.