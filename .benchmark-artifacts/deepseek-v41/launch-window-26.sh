#!/bin/bash
# Window 26: K29 parity; 1K arms stack_a/stack_b/decode_attn_kernel/selected_keys; dspark with stack_b + kernel; 16K prefill_lean_sel vs prefill_lean; served AR 16K at 70 GiB.
WT=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41
MODEL=/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4
OUT=$WT/docs/deepseek-v41/receipts/gpu-windows/window-26
mkdir -p "$OUT"
cd "$WT"
for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8080/health | grep -q '"ok":true' && break; sleep 10; done
echo "[launcher] $(date -u +%FT%TZ) agent healthy; starting window 26"
export GPU_WINDOW_CHILD_RSS_CAP_BYTES=$((88*1024*1024*1024))
exec bash scripts/deepseek_v41/gpu_window.sh bash -c "
export PYTHONPATH=$WT DSV41_MODEL=$MODEL MTPLX_DSV41_PREFILL_CHUNK=1024
PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3
F='transformers\]|█|╗|╝|║|╭|╰|│'
echo '[step 1] K29 decode attention parity'; MTPLX_GPU_PARITY=1 MTPLX_PARITY_RECEIPT=$OUT/k29-parity.json \$PY -m pytest tests/models/test_deepseek_v41_decode_attn_kernel.py::test_decode_attn_parity_gpu -q -s -p no:cacheprovider > $OUT/k29-parity.log 2>&1; tail -3 $OUT/k29-parity.log; head -c 500 $OUT/k29-parity.json 2>/dev/null; echo
echo '[step 2] 1K arms: stack_a stack_b decode_attn_kernel selected_keys'; \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 1024 --decode-tokens 256 --arms stack_a stack_b decode_attn_kernel selected_keys --memory-limit-gib 82 --out $OUT/ab-1024-k29-k30.json 2>&1 | grep -vE \"\$F\" | tail -12
echo '[step 3] dspark @1024 with stack_b (verify rows through K30)'; \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 1024 --decode-tokens 256 --decode-mode dspark --dspark-depth 3 --arms stack_b --stage-timing --memory-limit-gib 82 --out $OUT/dspark-1024-stackb.json 2>&1 | grep -vE \"\$F\" | tail -10
echo '[step 4] 16K prefill_lean_sel vs prefill_lean (60 GiB)'; \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 16384 --decode-tokens 8 --max-kv 16640 --arms prefill_lean prefill_lean_sel --prefill-stage-timing --memory-limit-gib 60 --out $OUT/prefill-16384-sel.json 2>&1 | grep -vE \"\$F\" | tail -10
echo '[step 5] served AR 16K x1 (70 GiB, KV 17664)'; DSV41_CONTEXTS=16384 DSV41_SEEDS=20260829 DSV41_MEMORY_LIMIT_GIB=70 DSV41_RECEIPT_DIR=$OUT/ar-16k bash scripts/deepseek_v41/served_cell_bench.sh 2>&1 | grep -vE \"\$F\" | tail -10
echo '[window 26 done]'
"
