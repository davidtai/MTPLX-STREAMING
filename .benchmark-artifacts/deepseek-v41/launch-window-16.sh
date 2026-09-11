#!/bin/bash
# Window 16: served AR vs MTP (256 tokens) under the stack_a levers, fast-path B, fixed layer-major @16K, stage timing under stack_a.
WT=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41
MODEL=/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4
OUT=$WT/docs/deepseek-v41/receipts/gpu-windows/window-16
mkdir -p "$OUT"
cd "$WT"
for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8080/health | grep -q '"ok":true' && break; sleep 10; done
echo "[launcher] $(date -u +%FT%TZ) agent healthy; starting window 16"
export GPU_WINDOW_CHILD_RSS_CAP_BYTES=$((85*1024*1024*1024))
exec bash scripts/deepseek_v41/gpu_window.sh bash -c "
export PYTHONPATH=$WT DSV41_MODEL=$MODEL MTPLX_DSV41_PREFILL_CHUNK=1024
PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3
F='transformers\]|█|╗|╝|║|╭|╰|│'
echo '[step 1] serve AR 256 tok under stack_a levers'; MTPLX_DSV41_HEAD_MODE=bf16 MTPLX_DSV41_SINKHORN_METAL=1 MTPLX_DSV41_ATTN_COMPILE=1 DSV41_MAX_TOKENS=256 bash scripts/deepseek_v41/serve_health.sh 2>&1 | grep -vE \"\$F\" | tail -14
echo '[step 2] serve MTP 256 tok under stack_a levers'; MTPLX_DSV41_HEAD_MODE=bf16 MTPLX_DSV41_SINKHORN_METAL=1 MTPLX_DSV41_ATTN_COMPILE=1 DSV41_MAX_TOKENS=256 DSV41_SERVE_EXTRA_ARGS='--generation-mode mtp' bash scripts/deepseek_v41/serve_health.sh 2>&1 | grep -vE \"\$F\" | tail -30
echo '[step 3] fast-path B @1024'; \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 1024 --decode-tokens 256 --arms control switch_fastpath_b --memory-limit-gib 82 --out $OUT/ab-1024-fastpath-b.json 2>&1 | grep -vE \"\$F\" | tail -12
echo '[step 4] layer-major @16384 (60 GiB plan)'; \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 16384 --decode-tokens 32 --max-kv 16640 --arms control layer_major --memory-limit-gib 60 --out $OUT/ab-16384-k16.json 2>&1 | grep -vE \"\$F\" | tail -12
echo '[step 5] stage timing under stack_a'; \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 1024 --decode-tokens 64 --arms stack_a --stage-timing --memory-limit-gib 82 --out $OUT/stage-timing-stack-a.json 2>&1 | grep -vE \"\$F\" | tail -30
echo '[window 16 done]'
"
