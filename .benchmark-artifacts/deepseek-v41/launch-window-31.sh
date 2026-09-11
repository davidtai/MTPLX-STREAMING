#!/bin/bash
# Window 31: (1) W78 Metal decode-attention op bisect at T=1K/4K/16K (selected, no-selected-keys, no-compile); (2) DSpark cell16k on the cell with stage timing + divergence classification (W77).
WT=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41
MODEL=/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4
OUT=$WT/docs/deepseek-v41/receipts/gpu-windows/window-31
mkdir -p "$OUT"
cd "$WT"
for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8080/health | grep -q '"ok":true' && break; sleep 10; done
echo "[launcher] $(date -u +%FT%TZ) agent healthy; starting window 31"
export GPU_WINDOW_CHILD_RSS_CAP_BYTES=$((96*1024*1024*1024))
exec bash scripts/deepseek_v41/gpu_window.sh bash -c "
export PYTHONPATH=$WT DSV41_MODEL=$MODEL MTPLX_DSV41_PREFILL_CHUNK=1024
PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3
F='transformers\]|█|╗|╝|║|╭|╰|│'
echo '[step 1] W78 Metal attention bisect, selected (cell16k) path'; nice -n 19 \$PY scripts/deepseek_v41/metal_decode_attn_bisect.py --gpu --T 1024 4096 16384 --iters 30 --warmup 5 --out $OUT/w78-bisect-selected.json 2>&1 | grep -vE \"\$F\" | tail -60
echo '[step 1b] W78 bisect, --no-selected-keys'; nice -n 19 \$PY scripts/deepseek_v41/metal_decode_attn_bisect.py --gpu --T 1024 4096 16384 --iters 30 --warmup 5 --no-selected-keys --out $OUT/w78-bisect-noselect.json 2>&1 | grep -vE \"\$F\" | tail -40
echo '[step 1c] W78 bisect, --compare-no-compile'; nice -n 19 \$PY scripts/deepseek_v41/metal_decode_attn_bisect.py --gpu --T 1024 4096 16384 --iters 30 --warmup 5 --compare-no-compile --out $OUT/w78-bisect-nocompile.json 2>&1 | grep -vE \"\$F\" | tail -40
echo '[step 2] DSpark cell16k @16384, decode 256, stage timing, 60 GiB plan (divergence classified, not fatal)'; MTPLX_ROUTE_STAGE_PROBE=1 \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 16384 --decode-tokens 256 --max-kv 17408 --decode-mode dspark --dspark-depth 3 --arms cell16k --stage-timing --memory-limit-gib 60 --out $OUT/dspark-16k-cell16k.json 2>&1 | grep -vE \"\$F\" | tail -24
echo '[window 31 done]'
"
