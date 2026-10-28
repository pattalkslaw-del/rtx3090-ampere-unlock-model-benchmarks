# I unlocked an RTX 3090's tensor performance. Then I tested 15 AI models at 300 W and 420 W.

**LawLab GPU evaluation | October 28, 2026 | Linux, CUDA, RTX 3090 24 GB**

**The result:** GPU Unlock produced a 61.9% FP16 and 61.5% BF16 throughput improvement in a sustained dense-matrix benchmark at 300 W. Yet across 15 quantized language models in llama.cpp, the gain at the RTX 3090's original 420 W board limit was just **+0.67% for 2,048-token prompt processing** and **-0.04% for 128-token generation**. Giving the stock GPU 420 W instead of 300 W improved the same workloads by **10.78% and 5.47%**, respectively.

We tested the same card in four configurations, preserved all 15 model weights and the benchmark executable between runs, and published the measurements. The important lesson is that a verified hardware compute improvement can have little effect on the workload you care about.

**Repository:** https://github.com/pattalkslaw-del/rtx3090-ampere-unlock-model-benchmarks

## 1. What were we testing?

The open-source [Zagitta GPU Unlock project](https://github.com/Zagitta/gpu-unlock) changes certain NVIDIA consumer GPU tensor-math instruction issue-rate settings. It uses a Linux kernel module early in boot, before the normal NVIDIA driver binds the card. It does not require replacing the production NVIDIA driver or flashing a new VBIOS.

Our test used pinned GPU Unlock 1.1.0 from [source commit 6081407b312aa3139a2bed62bdda354f4ffddc1d](https://github.com/Zagitta/gpu-unlock/commit/6081407b312aa3139a2bed62bdda354f4ffddc1d). It targeted the RTX 3090 only. The RTX 2080 Super and the two Intel Arc Pro B70 cards were left out of the unlock. The unsigned experimental module tainted the test kernel, and Secure Boot was disabled on this host.

To understand why this is interesting, consider two different measurements:

- **TFLOPS**, or trillions of floating-point operations per second, measures how quickly a GPU performs certain mathematical operations. We used sustained matrix multiplication, often called GEMM, to measure a specific FP16/BF16 compute path.
- **Tokens per second** measures how quickly a language model processes or generates text. That speed depends on model architecture, GPU memory traffic, numerical precision, CUDA kernels, caching and other work besides the raw tensor calculations.

FP16 and BF16 are 16-bit floating-point formats. Q4, Q8 and IQ3 model files use quantized weights, often stored in roughly four, eight or three bits per value. Quantization reduces the memory needed to store a model, but it also changes which GPU kernels do the work. Faster floating-point math cannot accelerate kernels that do not use the accelerated path.

We wanted to know whether the large synthetic compute gain survived contact with the models we run.

## 2. The test machine

| Component | Configuration |
|---|---|
| Host | LawLab, Ubuntu-based Linux server |
| Processor and memory | AMD EPYC 7532, 32 cores; 256 GB system RAM |
| Experimental GPU | NVIDIA GeForce RTX 3090, 24 GB, PCI 0000:01:00.0 |
| Original board power limit | 420 W on this particular RTX 3090 |
| Other NVIDIA card | RTX 2080 Super, 250 W limit; hidden from the dedicated CUDA benchmarks |
| Other accelerators | Two Intel Arc Pro B70 cards, left on the normal xe driver |
| Kernel | Linux 7.0.0-31-generic |
| Production NVIDIA driver | 595.99.02, unchanged through the tests |
| Benchmark software | llama.cpp CUDA backend, source commit 6a1a922, CUDA Toolkit 12.8, SM 8.6 |
| Exact benchmark executable SHA-256 | 4e663f6611fa9192a0898c47f1340bbd41f590437680082d1da33f78be84e118 |

A second machine, Render-1, supervised the test, recorded recovery status and preserved results independently of LawLab. The RTX 3090 was selected as CUDA device 0; the other NVIDIA card was excluded through the device visibility settings. The Intel cards remained in their own inference stack.

The operating environment also ran Maya, a production NVIDIA inference service, and Ornith, an Intel-based service. Maya was stopped during the dedicated NVIDIA benchmark runs and restarted afterward. Ornith remained independent. The safety controller restored the original RTX 3090 power limit after each test phase, including failure paths.

## 3. Experimental design: four configurations

The most important control was using the same physical GPU, model files and CUDA benchmark binary each time.

| Phase | Firmware initialization | RTX 3090 power cap |
|---|---|---:|
| A | GPU Unlock active | 300 W |
| B | Stock | 300 W |
| C | Stock | 420 W |
| D | GPU Unlock active | 420 W |

The phases ran in the order shown. The stock/unlocked transitions used a one-time experimental GRUB entry and a supervised full chassis-off recovery. The production NVIDIA driver remained unchanged.

**What did we hold constant?** The SHA-256 of every GGUF model file, the SHA-256 of the CUDA executable, the card, GPU device selection, all-GPU-layer request, thread count, KV precision, batch settings, test lengths and three measured repetitions per case. Our comparison script rejected mismatches rather than quietly comparing two different configurations.

For each of the 15 models we ran:

- **pp512:** prompt processing with 512 input tokens, measured in input tokens per second.
- **pp2048:** prompt processing with 2,048 input tokens, measured the same way.
- **tg128:** generation of 128 output tokens, measured in output tokens per second.

A warmup preceded three measured samples for each benchmark case. The entire experiment contains **60 model-condition runs**, **180 model-condition workload cases**, and **540 recorded timing samples**. There were no failed model loads.

The dedicated command followed this pattern; substitute one of the exact files listed in [the model manifest](../results/models.json):

~~~bash
CUDA_VISIBLE_DEVICES=0 llama-bench \
  -m /path/to/model.gguf \
  -dev CUDA0 -sm none -mg 0 -ngl 99 \
  -fa on -t 16 -b 512 -ub 512 \
  -ctk f16 -ctv f16 \
  -p 512,2048 -n 128 -r 3 -o jsonl
~~~

The 99-layer GPU offload setting, single-GPU selection, 16 CPU threads, FlashAttention, F16 KV cache, 512 batch and microbatch, and test lengths were identical throughout. Results contain the actual reported backend and offload settings for verification.

These short synthetic workloads are not a substitute for long-conversation serving benchmarks. In particular, we did not perform a dedicated llama-bench test with an already populated 32K or 100K KV cache.

## 4. The dense compute test: GPU Unlock works

Before the 15-model suite we tested the RTX 3090 using sustained 8,192-by-8,192 matrix multiplication, with numerical correctness checks and roughly 17-second measurement windows. Both configurations were capped at 300 W.

| Dense GEMM benchmark | Stock | GPU Unlock | Throughput gain |
|---|---:|---:|---:|
| FP16 inputs | 68.84 TFLOPS | 111.42 TFLOPS | **+61.9%** |
| BF16 inputs | 68.21 TFLOPS | 110.17 TFLOPS | **+61.5%** |

The unlocked kernel log confirmed the initialization stages and reported the GPU as unlocked. The improvement was substantial, and the sustained tests passed their numeric-correctness and safety checks. This GEMM comparison was not a large multi-day repeatability study.

That result deserves its own credit. The remaining question is how much of a language model's time is spent in the accelerated compute path.

We did **not** repeat this particular GEMM test at 420 W, so the GEMM figures above apply to 300 W only.

## 5. Four-way results: power matters much more for this workload

The table reports the geometric mean of the throughput ratios across all 15 models. Each model receives equal weight. A positive percentage means the second configuration is faster. A negative number means it is slower.

| Workload | Unlock versus stock, 300 W | Unlock versus stock, 420 W | Stock 420 W versus stock 300 W | Unlocked 420 W versus unlocked 300 W |
|---|---:|---:|---:|---:|
| 512-token prefill | -1.22% | -0.37% | +7.89% | +8.81% |
| 2,048-token prefill | +0.21% | **+0.67%** | **+10.78%** | +11.29% |
| 128-token generation | -0.28% | **-0.04%** | **+5.47%** | +5.72% |

The formula used for any matched comparison is:

- First, divide the second condition's tokens/sec by the first condition's tokens/sec for each model.
- Take the geometric mean of the 15 ratios.
- Subtract one and multiply by 100 to get the percentage change.

This keeps a 1B model with tens of thousands of prefill tokens/sec from dominating a 35B model running at a much lower absolute rate.

**The clearest signal is the power-limit comparison.** The stock 3090 gained 10.78% on pp2048 when allowed 420 W instead of 300 W. In the same pp2048 test, adding GPU Unlock at 420 W gained only 0.67%.

The pp512 column needs more skepticism. Short tasks are sensitive to scheduling, startup effects and other timing noise. At least one condition had a coefficient of variation over 5% in **14 of the 45 model/workload cases**, and every such flagged case was in pp512. For example, the pp512 result for Llama 3.2 3B showed -10.95% at 420 W, while Qwen 3 8B showed +6.99%. Their high variability makes those eye-catching figures poor evidence of a dependable gain or loss.

The pp2048 and tg128 results were much steadier, and those two columns carry more weight in our practical assessment.

## 6. All 15 models: the 420 W measurements

This table gives the direct stock-versus-unlocked numbers at the original 420 W limit. It lists quantization, prefill throughput, generation throughput and their relative change. These are the measured mean tokens/sec values; the [CSV](../results/power-matrix.csv) has the full 300 W and 420 W matrix and per-condition variation.

| Model | Quantization | pp2048 stock | pp2048 unlocked | Prefill change | tg128 stock | tg128 unlocked | Decode change |
|---|---|---:|---:|---:|---:|---:|---:|
| Llama 3.2 1B | Q8_0 | 28,690.1 | 28,986.8 | +1.03% | 467.43 | 466.32 | -0.24% |
| Llama 3.2 3B | Q4_K_M | 10,688.0 | 10,836.7 | +1.39% | 277.51 | 276.28 | -0.44% |
| Phi-3 Mini | Q4_K_M | 8,580.2 | 8,571.4 | -0.10% | 238.19 | 237.02 | -0.49% |
| Gemma 3 4B | Q4_K_M | 9,809.7 | 9,860.3 | +0.52% | 202.64 | 202.68 | +0.02% |
| Qwen 3 4B | Q4_K_M | 8,635.1 | 8,755.3 | +1.39% | 213.70 | 213.06 | -0.30% |
| Mistral 7B | Q4_K_M | 5,381.9 | 5,430.2 | +0.90% | 157.99 | 157.95 | -0.03% |
| Qwen 3 8B | Q4_K_M | 5,221.6 | 5,261.6 | +0.77% | 142.97 | 142.95 | -0.01% |
| Ornith 1.0 9B | Q8_0 | 4,589.0 | 4,587.4 | -0.04% | 86.93 | 86.94 | +0.02% |
| Qwen 3 14B | Q4_K_M | 3,076.5 | 3,116.7 | +1.31% | 84.00 | 84.02 | +0.02% |
| Qwen 3.6 27B | Q4_K_XL | 1,438.9 | 1,445.1 | +0.43% | 41.91 | 41.91 | +0.01% |
| Qwen 3.8 27B | IQ3_S | 1,437.9 | 1,442.7 | +0.33% | 49.18 | 49.28 | +0.20% |
| Qwen 3.8 27B | Q4_K_XL | 1,442.0 | 1,445.4 | +0.23% | 41.63 | 41.62 | -0.02% |
| Ornith 1.0 35B | Q4_K_M | 3,625.5 | 3,647.2 | +0.60% | 179.80 | 180.30 | +0.28% |
| Qwen 3.6 35B-A3B | Q4_K_M | 3,731.1 | 3,756.1 | +0.67% | 159.71 | 159.70 | -0.01% |
| Ornith 1.5 35B-A3B | Q4_K_M | 3,613.5 | 3,638.0 | +0.68% | 173.40 | 174.09 | +0.40% |

This table also shows why model architecture matters. A 35B model with sparse expert activation can behave differently from a dense 27B model. The number in a model name is not enough to predict throughput or the amount of active computation per token.

## 7. A separate test on the real Maya inference service

The llama.cpp suite used only the RTX 3090. We also ran a separate matched production-style test on Maya using DeepSeek V4.1 Flash, a CPU-assisted, dual-NVIDIA configuration. That test was at 300 W and used synthetic prompts at approximately 8K and 32K input lengths.

| Maya metric | Stock | Unlocked | Change |
|---|---:|---:|---:|
| 8K time to first token | 24.526 s | 24.228 s | 1.21% faster |
| 8K end-to-end completion | 31.203 s | 30.917 s | 0.92% faster |
| 32K time to first token | 74.611 s | 74.437 s | 0.23% faster |
| 32K end-to-end completion | 81.202 s | 81.112 s | 0.11% faster |
| 32K decode speed | 19.3 tok/s | 18.5 tok/s | 4.15% slower |

Maya's offload, CPU work, second GPU and PCIe traffic create different constraints from the single-card GGUF test. We did not isolate those components individually. Still, this independent end-to-end experiment also failed to show a practical improvement from the unlock for the tested configuration.

The predefined Maya retention gate required at least a 10% time-to-first-token or end-to-end improvement at both prompt lengths, without an unacceptable decode regression. It failed that gate.

## 8. Power, temperature and recovery

The values below are the observed peak GPU board-power samples and peak GPU core temperatures during the four dedicated GGUF phases. The configured cap is a ceiling, not a guarantee that every model continuously draws that wattage.

| Condition | RTX 3090 cap | Peak sampled draw | Peak core temperature | Brief software thermal flags | Hardware thermal / power-brake flags |
|---|---:|---:|---:|---:|---:|
| Stock, 300 W | 300 W | 303.33 W | 66 C | 0 | 0 / 0 |
| Unlocked, 300 W | 300 W | 299.95 W | 67 C | 4 | 0 / 0 |
| Stock, 420 W | 420 W | 419.42 W | 72 C | 4 | 0 / 0 |
| Unlocked, 420 W | 420 W | 420.32 W | 74 C | 4 | 0 / 0 |

A transient reported software slowdown is not the same thing as the card reaching a dangerous core temperature. The safety controller distinguished isolated flags from sustained throttling and stopped on hardware thermal or power-brake events, GPU/PCIe faults, a core temperature of 85 C or higher, or critically low available host memory. Neither full 420 W run had a safety stop. GPU memory-junction and hotspot measurements were not available in the captured telemetry.

Getting back to stock was part of the experiment, not an afterthought. The RTX 3090 unlock was activated through a one-time boot entry; the original stock boot entry remained the default. After the tests, we removed the experimental menu entry, performed a complete chassis shutdown and powered back on through the motherboard's BMC. We verified the new stock boot ID and kernel, then removed the two installed unlock packages.

The original NVIDIA 595.99.02 driver was preserved. The post-cleanup audit showed no GPU Unlock module, DKMS entry, installed package or leftover initramfs component. Both the NVIDIA and Intel inference services returned live READY completions. No extra reboot was needed after the package purge.

## 9. How I interpret the result

GPU Unlock worked on the RTX 3090's targeted floating-point compute path. Our dense GEMM test proved that with a 61% to 62% throughput gain.

The quantized GGUF inference tests likely spent much of their time somewhere else. Candidate constraints include quantized matrix kernels that do not use the accelerated BF16/FP16 path, GPU memory bandwidth, and decoding's repeated movement of model weights. Our experiment measures the final effect but does not profile and assign an exact percentage to each bottleneck.

The biggest practical gain we observed was from allowing this particular RTX 3090 to use its original 420 W board limit rather than holding it to 300 W. The card drew more power and delivered about 10.8% higher stock pp2048 throughput, along with 5.5% higher generation throughput. We did not measure wall-plug energy per generated token, so we cannot say whether the higher limit is more energy-efficient.

**The conclusion applies to the precise workloads we tested.** It would be a mistake to declare GPU Unlock useless for every LLM workload. A [separate community report](https://github.com/Zagitta/gpu-unlock/issues/158) using two RTX 3090s, a different configuration and an already populated KV cache reported increasing prefill gains at 32K and 100K context depths. Our dedicated llama-bench cases did not test those cache depths. That report is independent evidence, not a result from LawLab, and the figures cannot be pooled.

## 10. What an engineer should challenge or repeat

This is a carefully controlled single-host comparison, but it is not a randomized laboratory study. The main limits are:

1. **One sequential pass:** order was unlocked 300 W, stock 300 W, stock 420 W, unlocked 420 W. Thermal history, ambient temperature and execution order could influence small differences.
2. **Three repetitions:** enough to catch gross errors and measure variability, but insufficient for tight confidence intervals around sub-1% gains. Fourteen pp512 cases exceeded 5% coefficient of variation in at least one condition.
3. **Short-context GGUF tests:** pp512, pp2048 and tg128 with no dedicated long KV-cache depth sweeps. Long-context FlashAttention could behave differently.
4. **One model execution stack:** a pinned llama.cpp CUDA build using quantized GGUF files, single-card offload, F16 KV cache and fixed batch sizes. FP16/BF16 model inference, other kernels, training, alternate engines and multi-card tensor parallelism were not tested in the four-way suite.
5. **Power is not energy:** caps and sampled peaks were recorded; synchronized joules per token, average card draw per test, junction temperatures and wall-power efficiency were not.

The next engineering-grade study should randomize or alternate stock/unlocked conditions with more repetitions, instrument per-kernel compute and memory traffic, test longer context depths, and include unquantized BF16/FP16 models. Run every phase with its own controlled recovery plan. Long-context tests are the first gap I would close.

## 11. Data and reproduction

Everything needed to audit the measured results is in the public repository. We published:

- [The complete four-way matrix](../results/power-matrix.md) and [CSV](../results/power-matrix.csv), covering every model and test case.
- [Raw stock 300 W](../results/stock.jsonl), [unlocked 300 W](../results/unlocked.jsonl), [stock 420 W](../results/stock-420.jsonl), and [unlocked 420 W](../results/unlocked-420.jsonl) timing records.
- [All 15 model filenames, quantizations and SHA-256 hashes](../results/models.json).
- [Environment and binary SHA-256 records](../results/stock-420-environment.json), plus the [benchmark runner](../scripts/bench_suite.py) and [comparison script](../scripts/compare_power_matrix.py).
- [GPU telemetry summaries](../results/stock-420-telemetry-summary.json) and [restoration audit](../results/stock-restoration-420w-20261028.md).

We did not publish NVIDIA firmware binaries, proprietary model weights or private machine credentials. The upstream GPU Unlock project and the tested model weights have their own licenses.

**Bottom line:** On this single RTX 3090, GPU Unlock made dense floating-point matrix math dramatically faster. The 15-model quantized GGUF suite saw little incremental benefit at 300 W or 420 W. The card's normal 420 W power budget mattered considerably more for this workload. Future tests with a populated long-context cache may produce a different result.

**Repository:** https://github.com/pattalkslaw-del/rtx3090-ampere-unlock-model-benchmarks
