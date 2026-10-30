#!/usr/bin/env bash
set -o pipefail
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
cd /home/patrick/inference-lab/gpt-oss-120b-parity
O=logs/hybrid-sweep-20261029
mkdir -p "$O"
BIN=build-cuda-sycl/bin/llama-cli
MODEL=/mnt/nlf925/models/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf
printf 'TEST START %s\n' "$(date -Is)" >> "$O/STATUS.md"
"$BIN" --list-devices > "$O/devices.txt" 2>&1
nvidia-smi -q > "$O/nvidia-before.txt" 2>&1
command -v xpu-smi >/dev/null && xpu-smi discovery > "$O/intel-before.txt" 2>&1
python3 - <<'PY'
from pathlib import Path
root=Path('logs/hybrid-sweep-20261029')
s=Path('long-prompt-extended.txt').read_text()
for label,n in [('8k',13000),('32k',83000),('64k',229000),('128k',229000)]:
 (root/f'{label}-input.txt').write_text(s[:n]+'\nProvide a technical summary of latency, memory bandwidth, MoE routing, and the tradeoffs of four GPU inference.\n')
PY
for entry in '8k 8192' '32k 32768' '64k 65536' '128k 131072'; do
 read -r tag ctx <<< "$entry"
 printf '\nBEGIN %s context=%s %s\n' "$tag" "$ctx" "$(date -Is)" >> "$O/STATUS.md"
 timeout 1300 "$BIN" --model "$MODEL" --device CUDA0,CUDA1,SYCL0,SYCL1 --n-gpu-layers all --ctx-size "$ctx" --threads 32 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --batch-size 1024 --ubatch-size 256 --no-warmup --single-turn --simple-io --seed 42 --temp 0 --top-k 1 --predict 512 --file "$O/$tag-input.txt" > "$O/$tag.stdout" 2> "$O/$tag.stderr"
 rc=$?
 echo "END $tag exit=$rc $(date -Is)" >> "$O/STATUS.md"
 grep -E 'Prompt:|Generation:' "$O/$tag.stdout" | tail -1 >> "$O/STATUS.md" || true
 nvidia-smi --query-gpu=name,memory.used,power.draw,temperature.gpu --format=csv,noheader > "$O/$tag-nvidia-after.csv" 2>&1
 if [ "$rc" -ne 0 ]; then echo "HALTED after failure; no automatic service restoration" >> "$O/STATUS.md"; break; fi
done
echo "SWEEP ENDED $(date -Is)" >> "$O/STATUS.md"