#!/bin/bash
# Window 33: W80 window ring A/B on the cell — AR cell16k vs cell16k_ring (stage timing), then DSpark cell16k_ring; 60 GiB plan.
WT=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41
MODEL=/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4
OUT=$WT/docs/deepseek-v41/receipts/gpu-windows/window-33
mkdir -p "$OUT"
cd "$WT"
for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8080/health | grep -q '"ok":true' && break; sleep 10; done
echo "[launcher] $(date -u +%FT%TZ) agent healthy; starting window 33"
export GPU_WINDOW_CHILD_RSS_CAP_BYTES=$((96*1024*1024*1024))
exec bash scripts/deepseek_v41/gpu_window.sh bash -c "
export PYTHONPATH=$WT DSV41_MODEL=$MODEL MTPLX_DSV41_PREFILL_CHUNK=1024
PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3
F='transformers\]|█|╗|╝|║|╭|╰|│'
echo '[step 1] AR cell16k_ring @16384, decode 256, stage timing, 60 GiB (vs window-30 cell16k 2.24 tok/s)'; MTPLX_ROUTE_STAGE_PROBE=1 \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 16384 --decode-tokens 256 --max-kv 17408 --arms cell16k_ring --stage-timing --memory-limit-gib 60 --out $OUT/ar-16k-cell16k-ring.json 2>&1 | grep -vE \"\$F\" | tail -16
echo '[step 2] DSpark cell16k_ring @16384, decode 256, stage timing, 60 GiB'; MTPLX_ROUTE_STAGE_PROBE=1 \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 16384 --decode-tokens 256 --max-kv 17408 --decode-mode dspark --dspark-depth 3 --arms cell16k_ring --stage-timing --memory-limit-gib 60 --out $OUT/dspark-16k-cell16k-ring.json 2>&1 | grep -vE \"\$F\" | tail -24
echo '[window 33 done]'
"
