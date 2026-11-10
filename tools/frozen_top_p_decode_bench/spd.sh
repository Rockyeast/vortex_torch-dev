#!/usr/bin/env bash
# usage (inside box2.sh): spd.sh <run id> <arm> [<arm> ...]   arms: dense topp topk128 toppold
# One server per arm on the card box2.sh picked, then spd.py against it. The server always stops before the next arm.
set -uo pipefail
export HOME=/w/tmp/h TVM_FFI_CACHE_DIR=/w/cache/ap/tvm-ffi
cd /w
RUN=$1; shift
VXJ=$(cat tmp/vx.json)
BASE_PP=/w/pyd:%s:/w/sl3:/root/Megatron-LM
for ARM in "$@"; do
  unset VORTEX_TEST_FIXED_K; export VORTEX_FAST_MIN_MAJOR=9
  EXTRA=(--vortex-config "$VXJ")
  case "$ARM" in
    dense)   VX=vxq; EXTRA=() ;;
    topp)    VX=vxq ;;
    topk128) VX=vxk; export VORTEX_TEST_FIXED_K=128 ;;
    toppold) VX=vxp; export VORTEX_FAST_MIN_MAJOR=10 ;;
    *) VX=$ARM ;;   # any other code copy under /w, Top-P, fast path
  esac
  export PYTHONPATH=$(printf "$BASE_PP" "/w/$VX")
  TAG=$RUN-$ARM; LOG=logs/spd-$TAG.log
  echo "##### $TAG  $(date -u +%H:%M:%S) UTC  code=$VX"
  python3 -m sglang.launch_server --model-path /w/model --port 30500 --host 127.0.0.1 --mem-fraction-static ${MEMFRAC:-0.5} --page-size 16 \
    --attention-backend flashinfer --cuda-graph-backend-prefill disabled --context-length 32768 "${EXTRA[@]}" > "$LOG" 2>&1 &
  S=$!
  for i in $(seq 1 300); do curl -sf http://127.0.0.1:30500/health >/dev/null && break; kill -0 $S 2>/dev/null || break; sleep 2; done
  if curl -sf http://127.0.0.1:30500/health >/dev/null; then
    python3 tmp/spd.py 30500 "$TAG" 2>&1 | grep -a '^{\|^skip\|Error\|Traceback\|assert'
  else
    echo "server did not start"; tail -5 "$LOG"
  fi
  kill $S 2>/dev/null; wait $S 2>/dev/null
  echo "errors in log: $(grep -acE 'illegal memory|Traceback' "$LOG")"
  sleep 5
done
echo "##### done $(date -u +%H:%M:%S) UTC"
