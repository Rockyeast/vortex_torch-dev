#!/usr/bin/env bash
# usage (inside box.sh): prof.sh <run id> <arm> [<arm> ...]   arms: dense, or a code copy under /w (Top-P fast path)
set -uo pipefail
export HOME=/w/tmp/h TVM_FFI_CACHE_DIR=/w/cache/ap/tvm-ffi VORTEX_FAST_MIN_MAJOR=9
cd /w
RUN=$1; shift
VXJ=$(cat tmp/vx.json)
for ARM in "$@"; do
  EXTRA=(--vortex-config "$VXJ"); VX=$ARM
  [ "$ARM" = dense ] && { EXTRA=(); VX=vxq; }
  export PYTHONPATH=/w/pyd:/w/$VX:/w/sl3:/root/Megatron-LM
  TAG=$RUN-$ARM; LOG=logs/prof-$TAG.log
  python3 -m sglang.launch_server --model-path /w/model --port 30500 --host 127.0.0.1 --mem-fraction-static 0.5 --page-size 16 \
    --attention-backend flashinfer --cuda-graph-backend-prefill disabled --context-length 32768 "${EXTRA[@]}" > "$LOG" 2>&1 &
  S=$!
  for i in $(seq 1 300); do curl -sf http://127.0.0.1:30500/health >/dev/null && break; kill -0 $S 2>/dev/null || break; sleep 2; done
  python3 tmp/prof.py 30500 "$TAG" 2>&1 | grep -a '^#####\|^   \|Error\|Traceback\|no trace'
  kill $S 2>/dev/null; wait $S 2>/dev/null
  echo "errors in log: $(grep -acE 'illegal memory|Traceback' "$LOG")"; sleep 4
done
