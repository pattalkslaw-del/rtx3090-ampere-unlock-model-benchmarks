#!/usr/bin/env bash
set -uo pipefail
cd /home/patrick/inference-lab/gpt-oss-120b-parity
O=logs/four-gpu-validation
mkdir -p "$O"
BIN=build-cuda-sycl/bin/llama-cli
MODEL=/mnt/nlf925/models/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf
printf 'START %s\n' "$(date -Is)" >> "$O/status.log"
"$BIN" --list-devices > "$O/devices.txt" 2>&1
printf 'LIST_DEVICES_EXIT=%s\n' "$?" >> "$O/status.log"
for profile in all4 cuda_plus_b70; do
  if [ "$profile" = all4 ]; then devices=CUDA0,CUDA1,SYCL0,SYCL1; else devices=CUDA0,SYCL0; fi
  printf 'BEGIN %s %s\n' "$profile" "$(date -Is)" >> "$O/status.log"
  timeout 600 "$BIN" --model "$MODEL" --device "$devices" --n-gpu-layers all --ctx-size 8192 --threads 32 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --no-warmup --single-turn --simple-io --seed 42 --temp 0 --top-k 1 --predict 128 --prompt 'Explain in five sentences how Mixture-of-Experts inference benefits from heterogeneous GPU placement.' > "$O/$profile.stdout" 2> "$O/$profile.stderr"
  rc=$?
  printf 'END %s %s EXIT=%s\n' "$profile" "$(date -Is)" "$rc" >> "$O/status.log"
  if [ "$rc" -ne 0 ]; then break; fi
done
printf 'DONE %s\n' "$(date -Is)" >> "$O/status.log"