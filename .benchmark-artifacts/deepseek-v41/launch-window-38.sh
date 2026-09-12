#!/bin/bash
# Window 38: UNFENCED five-pass attribution of the current runner on the 16K cell (W94): full / switch-stub+barrier / attn-stub / switch-stub-no-barrier / small-stages floor.
# Gives the true split of the ~460 ms token (SSD vs 40 syncs vs attention vs small stages) = the baseline the W95 rebuild is measured against.
WT=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41
MODEL=/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4
OUT=$WT/docs/deepseek-v41/receipts/gpu-windows/window-38
IDS=$WT/docs/deepseek-v41/receipts/gpu-windows/window-28b/ar-16k/prompt-ids-deepseek-v41.json
mkdir -p "$OUT"
cd "$WT"
for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8080/health | grep -q '"ok":true' && break; sleep 10; done
echo "[launcher] $(date -u +%FT%TZ) agent healthy; starting window 38"
export GPU_WINDOW_CHILD_RSS_CAP_BYTES=$((96*1024*1024*1024))
exec bash scripts/deepseek_v41/gpu_window.sh bash -c "
export PYTHONPATH=$WT DSV41_MODEL=$MODEL MTPLX_DSV41_PREFILL_CHUNK=1024
PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3
F='transformers\]|█|╗|╝|║|╭|╰|│'
echo '[step 1] W94 unfenced five-pass attribution, cell16k_ring @16384, 64 steps/pass, 60 GiB'; nice -n 19 \$PY scripts/deepseek_v41/metal_decode_attn_bisect.py --in-model --unfenced --gpu --model $MODEL --arms cell16k_ring --context-tokens 16384 --memory-limit-gib 60 --max-kv 17408 --in-model-steps 64 --utilization --prompt-ids-file $IDS --prompt-seed 20260829 --out $OUT/unfenced-attribution.json 2>&1 | grep -vE \"\$F\" | tail -40
echo '[window 38 done]'
"
