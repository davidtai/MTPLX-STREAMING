#!/bin/bash
# Window 15: stacked arm (head_bf16 + switch fast-path OFF + sinkhorn + attn_compile) vs head_bf16 vs control @1K.
WT=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41
MODEL=/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4
OUT=$WT/docs/deepseek-v41/receipts/gpu-windows/window-15
mkdir -p "$OUT"
cd "$WT"
for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8080/health | grep -q '"ok":true' && break; sleep 10; done
echo "[launcher] $(date -u +%FT%TZ) agent healthy; starting window 15"
export GPU_WINDOW_CHILD_RSS_CAP_BYTES=$((85*1024*1024*1024))
exec bash scripts/deepseek_v41/gpu_window.sh bash -c "
export PYTHONPATH=$WT DSV41_MODEL=$MODEL MTPLX_DSV41_PREFILL_CHUNK=1024
PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3
F='transformers\]|█|╗|╝|║|╭|╰|│'
echo '[step 1] env arms @1024 (attn compile, stack)'; \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 1024 --decode-tokens 256 --arms control head_bf16 attn_compile stack_a --syncs 16 --memory-limit-gib 82 --out $OUT/ab-1024-stack.json 2>&1 | grep -vE \"\$F\" | tail -30
echo '[step 2] stage timing under head_bf16'; MTPLX_DSV41_HEAD_MODE=bf16 \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 1024 --decode-tokens 64 --arms head_bf16 --stage-timing --memory-limit-gib 82 --out $OUT/stage-timing-head-bf16.json 2>&1 | grep -vE \"\$F\" | tail -40
echo '[window 15 done]'
"
