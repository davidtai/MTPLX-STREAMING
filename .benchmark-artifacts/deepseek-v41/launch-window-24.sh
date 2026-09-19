#!/bin/bash
# Window 24: DSpark-direct decode @1K (bench), K28 parity, prefill_best vs no-K28 @16K, served DSpark-direct 1K.
WT=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41
MODEL=/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4
OUT=$WT/docs/deepseek-v41/receipts/gpu-windows/window-24
mkdir -p "$OUT"
cd "$WT"
for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8080/health | grep -q '"ok":true' && break; sleep 10; done
echo "[launcher] $(date -u +%FT%TZ) agent healthy; starting window 24"
export GPU_WINDOW_CHILD_RSS_CAP_BYTES=$((85*1024*1024*1024))
exec bash scripts/deepseek_v41/gpu_window.sh bash -c "
export PYTHONPATH=$WT DSV41_MODEL=$MODEL MTPLX_DSV41_PREFILL_CHUNK=1024
PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3
F='transformers\]|█|╗|╝|║|╭|╰|│'
echo '[step 1] dspark-direct decode @1024 (control, stack_a)'; \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 1024 --decode-tokens 256 --decode-mode dspark --dspark-depth 3 --arms control stack_a --memory-limit-gib 82 --out $OUT/dspark-1024.json 2>&1 | grep -vE \"\$F\" | tail -24
echo '[step 2] K28 fused softmax parity'; MTPLX_GPU_PARITY=1 MTPLX_PARITY_RECEIPT=$OUT/k28-parity.json \$PY -m pytest tests/models/test_deepseek_v41_fused_softmax.py::test_fused_softmax_parity_gpu -q -s -p no:cacheprovider > $OUT/k28-parity.log 2>&1; tail -4 $OUT/k28-parity.log; head -c 600 $OUT/k28-parity.json 2>/dev/null; echo
echo '[step 3] prefill_best vs prefill_best_nok28 @16384 (60 GiB)'; \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 16384 --decode-tokens 8 --max-kv 16640 --arms prefill_best_nok28 prefill_best --prefill-stage-timing --memory-limit-gib 60 --out $OUT/prefill-16384-best.json 2>&1 | grep -vE \"\$F\" | tail -12
echo '[step 4] served DSpark-direct 1K x3'; MTPLX_SERVE_STAGE_TIMING=1 MTPLX_SERVE_STAGE_TIMING_RECEIPT=$OUT/served-dspark-1k-stage DSV41_CONTEXTS=1024 DSV41_SERVE_EXTRA_ARGS='--load-mtp --generation-mode dspark --depth 3' DSV41_RECEIPT_DIR=$OUT/dspark-1k bash scripts/deepseek_v41/served_cell_bench.sh 2>&1 | grep -vE \"\$F\" | tail -14
echo '[window 24 done]'
"
