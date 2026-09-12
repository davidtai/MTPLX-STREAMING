#!/bin/bash
# Window 36 (RUNNER): each reviewed lever alone vs the window-33 cell16k_ring reference (2.33 tok/s, sha 2d39390f5686), with macmon utilization
# sampling in every receipt; then the DVFS discriminator (in-model 16K, cooldown 180 s vs 0) and the in-situ census at 1K.
WT=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41
MODEL=/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4
OUT=$WT/docs/deepseek-v41/receipts/gpu-windows/window-36
IDS=$WT/docs/deepseek-v41/receipts/gpu-windows/window-28b/ar-16k/prompt-ids-deepseek-v41.json
mkdir -p "$OUT"
cd "$WT"
for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8080/health | grep -q '"ok":true' && break; sleep 10; done
echo "[launcher] $(date -u +%FT%TZ) agent healthy; starting window 36"
export GPU_WINDOW_CHILD_RSS_CAP_BYTES=$((96*1024*1024*1024))
exec bash scripts/deepseek_v41/gpu_window.sh bash -c "
export PYTHONPATH=$WT DSV41_MODEL=$MODEL MTPLX_DSV41_PREFILL_CHUNK=1024
PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3
AB=scripts/deepseek_v41/ab_decode_env_levers.py
BI=scripts/deepseek_v41/metal_decode_attn_bisect.py
C='--context-tokens 16384 --decode-tokens 256 --max-kv 17408 --stage-timing --memory-limit-gib 60 --utilization'
F='transformers\]|█|╗|╝|║|╭|╰|│'
echo '[step 1] AR cell16k_ring @16384 (reference re-run WITH utilization sampling)'; MTPLX_ROUTE_STAGE_PROBE=1 \$PY \$AB \$C --arms cell16k_ring --out $OUT/ar-ring-ref.json 2>&1 | grep -vE \"\$F\" | tail -22
echo '[step 2] AR cell16k_ring_stable @16384 (W90: ≤1% dispatch cut; swa_only is the control; sha 2d39390f5686)'; MTPLX_ROUTE_STAGE_PROBE=1 \$PY \$AB \$C --arms cell16k_ring_stable --out $OUT/ar-ring-stable.json 2>&1 | grep -vE \"\$F\" | tail -22
echo '[step 3] in-model 16K census, cooldown 0 (DVFS/thermal discriminator A)'; nice -n 19 \$PY \$BI --in-model --gpu --model $MODEL --arms cell16k --context-tokens 16384 --memory-limit-gib 60 --max-kv 17408 --in-model-steps 30 --prompt-ids-file $IDS --prompt-seed 20260829 --utilization --cooldown-s 0 --out $OUT/in-model-16k-cool0.json 2>&1 | grep -vE \"\$F\" | tail -26
echo '[step 4] in-model 16K census, cooldown 180 s (discriminator B)'; nice -n 19 \$PY \$BI --in-model --gpu --model $MODEL --arms cell16k --context-tokens 16384 --memory-limit-gib 60 --max-kv 17408 --in-model-steps 30 --prompt-ids-file $IDS --prompt-seed 20260829 --utilization --cooldown-s 180 --out $OUT/in-model-16k-cool180.json 2>&1 | grep -vE \"\$F\" | tail -26
echo '[step 5] in-model census at T=1024 (fixed no-ids path), utilization'; nice -n 19 \$PY \$BI --in-model --gpu --model $MODEL --arms cell16k --context-tokens 1024 --memory-limit-gib 60 --max-kv 4096 --in-model-steps 30 --utilization --out $OUT/in-model-1k.json 2>&1 | grep -vE \"\$F\" | tail -26
echo '[window 36 done]'
"
