#!/usr/bin/env bash
# usage (inside box.sh): [SO=16] aime.sh <arm> [<arm> ...]   arm = dense, or a code copy under /w (Top-P 0.9 fast path); AIME24+25, 16 samples each
# Same client and sampling as PROTOCOL.md Tier 2. 128 concurrent, mem fraction 0.8 so no request is retracted.
set -uo pipefail
export HOME=/w/tmp/h TVM_FFI_CACHE_DIR=/w/cache/ap/tvm-ffi VORTEX_FAST_MIN_MAJOR=9
cd /w
VXJ=$(cat tmp/vx.json)
SO=${SO:-0}; SFX=$([ "$SO" = 0 ] || echo _s$SO)
for ARM in "$@"; do
  VX=$ARM; EXTRA=(--vortex-config "$VXJ")
  [ "$ARM" = dense ] && { VX=vxq; EXTRA=(); }
  export PYTHONPATH=/w/pyd:/w/$VX:/w/sl3:/root/Megatron-LM
  LOG=logs/aime-spd-$ARM$SFX.log
  python3 -m sglang.launch_server --model-path /w/model --port 30500 --host 127.0.0.1 --mem-fraction-static 0.8 --page-size 16 \
    --attention-backend flashinfer --cuda-graph-backend-prefill disabled --context-length 32768 "${EXTRA[@]}" > "$LOG" 2>&1 &
  S=$!
  for i in $(seq 1 300); do curl -sf http://127.0.0.1:30500/health >/dev/null && break; kill -0 $S 2>/dev/null || break; sleep 2; done
  echo "=== $ARM seed offset $SO $(date -u +%H:%M:%S) UTC"
  PYTHONPATH=/w/pyd python3 aime_eval.py --port 30500 --n 16 --arm spd_$ARM$SFX --seed-offset $SO --conc 128 2>&1 | grep -a '^{\|Error\|Traceback'
  echo "retracted: $(grep -ac -i 'retract' "$LOG")  errors: $(grep -acE 'illegal memory|Traceback' "$LOG")  max token usage: $(grep -ao 'token usage: [0-9.]*' "$LOG" | sort -t' ' -k3 -n | tail -1)"
  kill $S 2>/dev/null; wait $S 2>/dev/null; sleep 5
done
echo "=== done $(date -u +%H:%M:%S) UTC"
