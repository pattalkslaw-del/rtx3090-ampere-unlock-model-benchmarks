#!/usr/bin/env bash
set -o pipefail
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
cd /home/patrick/inference-lab/gpt-oss-120b-parity
O=logs/final-gates
mkdir -p "$O"
MODEL=/mnt/nlf925/models/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf
BIN=build-cuda-sycl/bin/llama-cli
TOK=build-cuda-sycl/bin/llama-tokenize
echo 'FINAL FOUR GATES' > "$O/STATUS.md"
echo "START $(date -Is)" >> "$O/STATUS.md"
# Gate 1: quantify actual input occupancy; record native max.
for input in logs/hybrid-validation-2/unique64.txt logs/hybrid-validation-2/unique128.txt logs/cuda-sustained/input.txt; do
 name=$(basename "$input")
 echo "TOKEN_COUNT_BEGIN $name $(date -Is)" >> "$O/STATUS.md"
 timeout 240 "$TOK" -m "$MODEL" -f "$input" --show-count > "$O/token-$name.stdout" 2> "$O/token-$name.stderr"
 echo "TOKEN_COUNT_END $name exit=$?" >> "$O/STATUS.md"
 tail -2 "$O/token-$name.stdout" >> "$O/STATUS.md"
done
# Gate 2: actual identical model text comparison (exclude variable timing / shell framing)
python3 - <<'PY'
from pathlib import Path
import hashlib,difflib
p=Path('logs/hybrid-validation-2')
def normalize(x):
 s=(p/(x+'.stdout')).read_text(errors='replace')
 a=s.find('[Start thinking]');b=s.find('[ Prompt:')
 return s[a:b].strip() if a>=0 and b>a else s
x=normalize('unique64');y=normalize('unique64repeat')
f=Path('logs/final-gates/model-text-parity.txt')
f.write_text(f'NORMALIZED_IDENTICAL={x==y}\nFIRST_HASH={hashlib.sha256(x.encode()).hexdigest()}\nSECOND_HASH={hashlib.sha256(y.encode()).hexdigest()}\n'+''.join(list(difflib.unified_diff(x.splitlines(True),y.splitlines(True)))[:60]))
PY
echo 'PARITY' >> "$O/STATUS.md"
head -4 "$O/model-text-parity.txt" >> "$O/STATUS.md"
# Gate 3: inference with backend tracing to investigate actual placement
printf 'PLACEMENT_START %s\n' "$(date -Is)" >> "$O/STATUS.md"
timeout 480 "$BIN" --log-verbose --model "$MODEL" --device CUDA0,CUDA1,SYCL0,SYCL1 --n-gpu-layers all --ctx-size 8192 --threads 32 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --no-warmup --single-turn --simple-io --seed 42 --temp 0 --top-k 1 --predict 128 --prompt 'Explain the technical distinction between distributed layer placement and explicit heterogeneous expert routing.' > "$O/placement.stdout" 2> "$O/placement.stderr"
echo "PLACEMENT_END exit=$? $(date -Is)" >> "$O/STATUS.md"
grep -Ei 'assign|offload|layer|cuda|sycl|buffer' "$O/placement.stderr" | tail -40 > "$O/placement-evidence.txt" || true
# Gate 4: independent accuracy control and repeated output
for n in 1 2 3; do
 echo "ACCURACY_START_$n $(date -Is)" >> "$O/STATUS.md"
 timeout 480 "$BIN" --model "$MODEL" --device CUDA0,CUDA1,SYCL0,SYCL1 --n-gpu-layers all --ctx-size 8192 --threads 32 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --no-warmup --single-turn --simple-io --seed 42 --temp 0 --top-k 1 --predict 512 --prompt 'Answer all three: (1) compute 137*289 + 4187 - 956 (2) An attorney power of attorney agent acts under a POA; does their authority survive the principal death? (3) Correctly compute 27% of 8450. Show the intermediate arithmetic and identify uncertainty or jurisdictional assumptions.' > "$O/accuracy-$n.stdout" 2> "$O/accuracy-$n.stderr"
 echo "ACCURACY_END_$n exit=$? $(date -Is)" >> "$O/STATUS.md"
 grep -E 'Prompt:|Generation:' "$O/accuracy-$n.stdout" | tail -1 >> "$O/STATUS.md" || true
done
printf 'DONE %s\n' "$(date -Is)" >> "$O/STATUS.md"