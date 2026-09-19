#!/bin/bash
# Window 27: K29 re-gate (precise exp + split-K); decode_attn_kernel arm w/ engagement; DSpark with kernels + single-barrier verify; AR with-mtp reprice A/B; served DSpark 1K.
WT=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41
MODEL=/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4
OUT=$WT/docs/deepseek-v41/receipts/gpu-windows/window-27
mkdir -p "$OUT"
cd "$WT"
for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8080/health | grep -q '"ok":true' && break; sleep 10; done
echo "[launcher] $(date -u +%FT%TZ) agent healthy; starting window 27"
export GPU_WINDOW_CHILD_RSS_CAP_BYTES=$((88*1024*1024*1024))
exec bash scripts/deepseek_v41/gpu_window.sh bash -c "
export PYTHONPATH=$WT DSV41_MODEL=$MODEL MTPLX_DSV41_PREFILL_CHUNK=1024
PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3
F='transformers\]|█|╗|╝|║|╭|╰|│'
echo '[step 1] K29 parity re-gate'; MTPLX_GPU_PARITY=1 MTPLX_PARITY_RECEIPT=$OUT/k29-parity.json \$PY -m pytest tests/models/test_deepseek_v41_decode_attn_kernel.py::test_decode_attn_parity_gpu -q -s -p no:cacheprovider > $OUT/k29-parity.log 2>&1; tail -2 $OUT/k29-parity.log; \$PY -c \"import json;d=json.load(open('$OUT/k29-parity.json'));print('all_passed',d.get('all_passed'));print({k:(v.get('passed'),round(v.get('max_abs_d',0),8),v.get('argmax_mismatch')) for k,v in (d.get('arms') or {}).items()})\" 2>/dev/null
echo '[step 2] 1K arms: stack_a decode_attn_kernel'; \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 1024 --decode-tokens 256 --arms stack_a decode_attn_kernel --memory-limit-gib 82 --out $OUT/ab-1024-k29.json 2>&1 | grep -vE \"\$F\" | tail -8
echo '[step 3] dspark @1024 stack_a (K29+K30 armed, single-barrier verify)'; \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 1024 --decode-tokens 256 --decode-mode dspark --dspark-depth 3 --arms stack_a --stage-timing --memory-limit-gib 82 --out $OUT/dspark-1024.json 2>&1 | grep -vE \"\$F\" | tail -10
echo '[step 4] AR stack_a --with-mtp --no-reprice vs --with-mtp'; \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 1024 --decode-tokens 256 --decode-mode ar --with-mtp --no-reprice --arms stack_a --memory-limit-gib 82 --out $OUT/ar-1024-withmtp-noreprice.json 2>&1 | grep -vE \"\$F\" | tail -4; \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 1024 --decode-tokens 256 --decode-mode ar --with-mtp --arms stack_a --memory-limit-gib 82 --out $OUT/ar-1024-withmtp.json 2>&1 | grep -vE \"\$F\" | tail -4
echo '[step 5] served DSpark-direct 1K x3'; DSV41_CONTEXTS=1024 DSV41_SERVE_EXTRA_ARGS='--load-mtp --generation-mode dspark --depth 3' DSV41_RECEIPT_DIR=$OUT/dspark-1k bash scripts/deepseek_v41/served_cell_bench.sh 2>&1 | grep -vE \"\$F\" | tail -8
echo '[window 27 done]'
"
