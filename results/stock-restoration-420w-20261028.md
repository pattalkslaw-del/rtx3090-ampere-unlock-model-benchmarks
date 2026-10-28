# Final stock restoration after the 420 W benchmark

**Completed October 28, 2026.** The four-condition RTX 3090 inference study is published, and the test machine has returned to its verified stock configuration.

After the unlocked 420 W suite, the one-time experimental GRUB entry was removed. The owner then used the motherboard's BMC to power the machine back on following a full chassis shutdown. Independent network monitoring confirmed a fresh boot into Linux 7.0.0-31, using the saved stock recovery default.

The stock boot had no GPU Unlock kernel selector and no loaded GPU Unlock module. A read-only APT simulation listed only gpu-unlock-dkms and gpu-unlock-firmware for removal. Both packages, version 1.1.0, were purged. Checks after removal confirmed:

- No GPU Unlock package in the package database and no GPU Unlock entry in DKMS.
- No GPU Unlock module in loaded kernel modules or the regenerated kernel initramfs for either installed 7.0.0-31 or 7.0.0-38.
- The separately preserved stock recovery initramfs retained its original SHA-256 checksum, and GRUB syntax/default checks passed.
- NVIDIA driver 595.99.02 remained unchanged. RTX 3090 and RTX 2080 Super board power limits remained 420 W and 250 W, with NVIDIA persistence mode enabled.
- Both Intel Arc Pro B70 cards remained bound to the xe driver.
- Both production inference services returned HTTP 200 on their model endpoints and passed independent live READY completions after the package purge.
- All 15 model files, the CUDA benchmark binary and the raw measurement datasets were preserved.

This was a real chassis-off return to stock, not a warm reboot. An additional restart after package removal was unnecessary because the system was already running the verified stock kernel and the modified firmware state had been cleared.

See [the four-condition throughput comparison](power-matrix.md), [the paired CSV](power-matrix.csv) and the published raw datasets for the performance findings. The unlock supplied less than 1% model-weighted geometric-mean throughput improvement at 420 W in each of the three quantized GGUF tests; increasing the RTX 3090's power cap from 300 W to 420 W yielded much larger gains.
