#!/bin/bash
# Window 19: device_route + stack_a arms @1K; served MTP smoke (64 tok); 16K prefill stage timing (chunk-major vs layer-major).
WT=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41
MODEL=/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4
OUT=$WT/docs/deepseek-v41/receipts/gpu-windows/window-19
mkdir -p "$OUT"
cd "$WT"
for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8080/health | grep -q '"ok":true' && break; sleep 10; done
echo "[launcher] $(date -u +%FT%TZ) agent healthy; starting window 19"
export GPU_WINDOW_CHILD_RSS_CAP_BYTES=$((85*1024*1024*1024))
exec bash scripts/deepseek_v41/gpu_window.sh bash -c "
export PYTHONPATH=$WT DSV41_MODEL=$MODEL MTPLX_DSV41_PREFILL_CHUNK=1024
PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3
F='transformers\]|█|╗|╝|║|╭|╰|│'
echo '[step 1] env arms @1024 (device route, stack)'; \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 1024 --decode-tokens 256 --arms control device_route stack_a --syncs 16 --memory-limit-gib 82 --out $OUT/ab-1024-device-route.json 2>&1 | grep -vE \"\$F\" | tail -24
echo '[step 2] serve MTP smoke 64 tok'; DSV41_MAX_TOKENS=64 DSV41_SERVE_EXTRA_ARGS='--generation-mode mtp' bash scripts/deepseek_v41/serve_health.sh 2>&1 | grep -vE \"\$F\" | tail -24
echo '[step 3] 16K prefill stage timing (60 GiB plan)'; \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 16384 --decode-tokens 0 --max-kv 16640 --arms control layer_major --prefill-stage-timing --memory-limit-gib 60 --out $OUT/prefill-16384-stage-timing.json 2>&1 | grep -vE \"\$F\" | tail -40
echo '[window 19 done]'
"
