#!/bin/bash
# Window 34 (the 16K cell, 60 GiB, bench now on the profile's 48 transient slots):
# 1 W78 in-model attention census (full / expert-stub / attn-stub); 2 DSpark cell16k_ring_draft depth 3 (W81 batched verify + K33 draft compile);
# 3 DSpark cell16k_ring_draft depth 5 (W82: draft length up to 5); 4 AR cell16k_ring_pinned (W64+W71 barrier-free route); 5 AR cell16k_ring at an 80 GiB plan (hit-rate lever).
WT=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41
MODEL=/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4
OUT=$WT/docs/deepseek-v41/receipts/gpu-windows/window-34
IDS=$WT/docs/deepseek-v41/receipts/gpu-windows/window-28b/ar-16k/prompt-ids-deepseek-v41.json
mkdir -p "$OUT"
cd "$WT"
for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8080/health | grep -q '"ok":true' && break; sleep 10; done
echo "[launcher] $(date -u +%FT%TZ) agent healthy; starting window 34"
export GPU_WINDOW_CHILD_RSS_CAP_BYTES=$((96*1024*1024*1024))
exec bash scripts/deepseek_v41/gpu_window.sh bash -c "
export PYTHONPATH=$WT DSV41_MODEL=$MODEL MTPLX_DSV41_PREFILL_CHUNK=1024
PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3
AB=scripts/deepseek_v41/ab_decode_env_levers.py
F='transformers\]|█|╗|╝|║|╭|╰|│'
echo '[step 1] W78 in-model attention census: full / expert-stub / attn-stub, cell16k, 30 steps'; nice -n 19 \$PY scripts/deepseek_v41/metal_decode_attn_bisect.py --in-model --gpu --model $MODEL --arms cell16k --context-tokens 16384 --memory-limit-gib 60 --max-kv 17408 --in-model-steps 30 --prompt-ids-file $IDS --prompt-seed 20260829 --out $OUT/w78-in-model.json 2>&1 | grep -vE \"\$F\" | tail -40
echo '[step 2] DSpark cell16k_ring_draft depth 3 @16384, decode 256, stage timing, 60 GiB'; MTPLX_ROUTE_STAGE_PROBE=1 \$PY \$AB --context-tokens 16384 --decode-tokens 256 --max-kv 17408 --decode-mode dspark --dspark-depth 3 --arms cell16k_ring_draft --stage-timing --memory-limit-gib 60 --out $OUT/dspark-d3-ring-draft.json 2>&1 | grep -vE \"\$F\" | tail -26
echo '[step 3] DSpark cell16k_ring_draft depth 5 @16384, decode 256, stage timing, 60 GiB'; MTPLX_ROUTE_STAGE_PROBE=1 \$PY \$AB --context-tokens 16384 --decode-tokens 256 --max-kv 17408 --decode-mode dspark --dspark-depth 5 --arms cell16k_ring_draft --stage-timing --memory-limit-gib 60 --out $OUT/dspark-d5-ring-draft.json 2>&1 | grep -vE \"\$F\" | tail -26
echo '[step 4] AR cell16k_ring_pinned @16384, decode 256, stage timing, 60 GiB (parity vs sha 2d39390f5686)'; MTPLX_ROUTE_STAGE_PROBE=1 \$PY \$AB --context-tokens 16384 --decode-tokens 256 --max-kv 17408 --arms cell16k_ring_pinned --stage-timing --memory-limit-gib 60 --out $OUT/ar-ring-pinned.json 2>&1 | grep -vE \"\$F\" | tail -18
echo '[step 5] AR cell16k_ring @16384, decode 256, 80 GiB plan (hit-rate lever)'; \$PY \$AB --context-tokens 16384 --decode-tokens 256 --max-kv 17408 --arms cell16k_ring --memory-limit-gib 80 --out $OUT/ar-ring-80g.json 2>&1 | grep -vE \"\$F\" | tail -10
echo '[window 34 done]'
"
