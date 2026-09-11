#!/bin/bash
# Window 14: head (K21) + switch fast-path (K23) arms @1K, 16K layer-major at a 60 GiB plan, MTP serve re-check.
WT=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41
MODEL=/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4
OUT=$WT/docs/deepseek-v41/receipts/gpu-windows/window-14
mkdir -p "$OUT"
cd "$WT"
for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8080/health | grep -q '"ok":true' && break; sleep 10; done
echo "[launcher] $(date -u +%FT%TZ) agent healthy; starting window 14"
export GPU_WINDOW_CHILD_RSS_CAP_BYTES=$((85*1024*1024*1024))
exec bash scripts/deepseek_v41/gpu_window.sh bash -c "
export PYTHONPATH=$WT DSV41_MODEL=$MODEL MTPLX_DSV41_PREFILL_CHUNK=1024
PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3
F='transformers\]|█|╗|╝|║|╭|╰|│'
echo '[step 1] env arms @1024 (head, switch fast-path)'; \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 1024 --decode-tokens 256 --arms control head_bf16 head_mxfp8 switch_fastpath sinkhorn_metal --syncs 16 --memory-limit-gib 82 --out $OUT/ab-1024-head-switch.json 2>&1 | grep -vE \"\$F\" | tail -40
echo '[step 2] K16 A/B @16384 (60 GiB plan)'; \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 16384 --decode-tokens 32 --max-kv 16640 --arms control layer_major --memory-limit-gib 60 --out $OUT/ab-16384-k16.json 2>&1 | grep -vE \"\$F\" | tail -40
echo '[step 3] serve_health MTP'; DSV41_MAX_TOKENS=64 DSV41_SERVE_EXTRA_ARGS='--generation-mode mtp' bash scripts/deepseek_v41/serve_health.sh 2>&1 | grep -vE \"\$F\" | tail -30
echo '[window 14 done]'
"
