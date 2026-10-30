#!/usr/bin/env bash
set -o pipefail
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
cd /home/patrick/inference-lab/gpt-oss-120b-parity
O=logs/hybrid-validation-2
mkdir -p "$O"
BIN=build-cuda-sycl/bin/llama-cli
MODEL=/mnt/nlf925/models/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf
python3 - <<'PY'
from pathlib import Path
import random,hashlib
p=Path('logs/hybrid-validation-2')
r=random.Random(19377)
terms=['CUDA attention','SYCL expert placement','KV cache','quantization error','PCIe bandwidth','token routing','numerical parity','scheduler synchronization','GPU occupancy','memory allocator','host offload','context scaling','flash attention','decode throughput','prefill capacity']
rows=[]
for i in range(4000):
 a,b,c=r.sample(terms,3)
 datum=hashlib.sha256(f'{i}:{a}:{b}:{c}'.encode()).hexdigest()[:20]
 rows.append(f'Record {i:04d}: measurement key {datum}. The {a} experiment varied {b} under controlled {c}. Sampled register {r.randint(10000,99999)} with offset {r.randint(1,9999)} and checksum {r.getrandbits(64):016x}. Validate attribution and do not infer speedups without matched inputs.\n')
text=''.join(rows)
for label,n in [('unique64',220000),('unique128',500000)]:
 prompt=text[:n]+'\nFind and summarize the operational risks of mixed CUDA and SYCL inference.\n'
 (p/f'{label}.txt').write_text(prompt)
 print(label,len(prompt),file=open(p/'input-stats.txt','a'))
PY
for spec in 'unique64 65536 512' 'unique64repeat 65536 512' 'unique128 131072 512' 'longdecode 65536 2048'; do
 read -r tag ctx predict <<< "$spec"
 input=unique64
 if [ "$tag" = unique128 ]; then input=unique128; fi
 echo "BEGIN $tag $(date -Is)" >> "$O/STATUS.md"
 timeout 1500 "$BIN" --model "$MODEL" --device CUDA0,CUDA1,SYCL0,SYCL1 --n-gpu-layers all --ctx-size "$ctx" --threads 32 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --batch-size 1024 --ubatch-size 256 --no-warmup --single-turn --simple-io --seed 42 --temp 0 --top-k 1 --predict "$predict" --file "$O/$input.txt" > "$O/$tag.stdout" 2> "$O/$tag.stderr"
 rc=$?
 echo "END $tag exit=$rc $(date -Is)" >> "$O/STATUS.md"
 grep -E 'Prompt:|Generation:' "$O/$tag.stdout" | tail -1 >> "$O/STATUS.md" || true
 sha256sum "$O/$tag.stdout" >> "$O/hashes.sha256"
 if [ "$rc" -ne 0 ]; then echo 'HALT ON FAILURE' >> "$O/STATUS.md"; break; fi
done
printf 'COMPLETE %s\n' "$(date -Is)" >> "$O/STATUS.md"