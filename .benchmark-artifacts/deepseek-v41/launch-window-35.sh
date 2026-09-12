#!/bin/bash
# Window 35 (RUNNER FIRST; HEAD 646b55ee2): attribute the in-situ attention inflation — in-model census at T=1024 (same arm), at 16K with compile off, at 16K with win-memo off;
# then AR cell16k_ring_pool (W87 single pool) vs cell16k_ring on the cell; DSpark reruns (signature fix) LAST, depth 3 then 5.
WT=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41
MODEL=/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4
OUT=$WT/docs/deepseek-v41/receipts/gpu-windows/window-35
IDS=$WT/docs/deepseek-v41/receipts/gpu-windows/window-28b/ar-16k/prompt-ids-deepseek-v41.json
mkdir -p "$OUT"
cd "$WT"
for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8080/health | grep -q '"ok":true' && break; sleep 10; done
echo "[launcher] $(date -u +%FT%TZ) agent healthy; starting window 35"
export GPU_WINDOW_CHILD_RSS_CAP_BYTES=$((96*1024*1024*1024))
exec bash scripts/deepseek_v41/gpu_window.sh bash -c "
export PYTHONPATH=$WT DSV41_MODEL=$MODEL MTPLX_DSV41_PREFILL_CHUNK=1024
PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3
AB=scripts/deepseek_v41/ab_decode_env_levers.py
BI=scripts/deepseek_v41/metal_decode_attn_bisect.py
F='transformers\]|█|╗|╝|║|╭|╰|│'
echo '[step 1] in-model census at T=1024 (attribution: is in-situ attention 2 or 7 ms/layer at 1K?)'; nice -n 19 \$PY \$BI --in-model --gpu --model $MODEL --arms cell16k --context-tokens 1024 --memory-limit-gib 60 --max-kv 4096 --in-model-steps 30 --out $OUT/in-model-1k.json 2>&1 | grep -vE \"\$F\" | tail -24
echo '[step 2] in-model census at 16K, ATTN_COMPILE off'; nice -n 19 \$PY \$BI --in-model --gpu --model $MODEL --arms cell16k --context-tokens 16384 --memory-limit-gib 60 --max-kv 17408 --in-model-steps 30 --prompt-ids-file $IDS --prompt-seed 20260829 --compare-no-compile --out $OUT/in-model-16k-nocompile.json 2>&1 | grep -vE \"\$F\" | tail -24
echo '[step 3] in-model census at 16K, ATTN_WIN_MEMO off'; nice -n 19 \$PY \$BI --in-model --gpu --model $MODEL --arms cell16k --context-tokens 16384 --memory-limit-gib 60 --max-kv 17408 --in-model-steps 30 --prompt-ids-file $IDS --prompt-seed 20260829 --no-win-memo --out $OUT/in-model-16k-nomemo.json 2>&1 | grep -vE \"\$F\" | tail -24
echo '[step 4] W89 route-trace collection (prefill tail 2048 + decode 256, non-invasive) for the prefetch predictor'; nice -n 19 \$PY scripts/deepseek_v41/collect_route_traces.py --arm cell16k --memory-limit-gib 60 --max-kv 17408 --context-tokens 16384 --prefill-tail 2048 --decode-tokens 256 --prompt-ids-file $IDS --prompt-seed 20260829 --out $WT/.benchmark-artifacts/deepseek-v41/route-traces-w35 2>&1 | grep -vE \"\$F\" | tail -12
echo '[step 5] DSpark cell16k_ring_draft depth 3 @16384 (signature fix)'; MTPLX_ROUTE_STAGE_PROBE=1 \$PY \$AB --context-tokens 16384 --decode-tokens 256 --max-kv 17408 --decode-mode dspark --dspark-depth 3 --arms cell16k_ring_draft --stage-timing --memory-limit-gib 60 --out $OUT/dspark-d3-ring-draft.json 2>&1 | grep -vE \"\$F\" | tail -26
echo '[step 6] DSpark cell16k_ring_draft depth 5 @16384'; MTPLX_ROUTE_STAGE_PROBE=1 \$PY \$AB --context-tokens 16384 --decode-tokens 256 --max-kv 17408 --decode-mode dspark --dspark-depth 5 --arms cell16k_ring_draft --stage-timing --memory-limit-gib 60 --out $OUT/dspark-d5-ring-draft.json 2>&1 | grep -vE \"\$F\" | tail -26
echo '[step 7] served AR 16K x1 seed, profile default (60 GiB, cell levers), prefix reuse OFF (W83)'; DSV41_CONTEXTS=16384 DSV41_SEEDS=20260829 DSV41_RECEIPT_DIR=$OUT/served-ar-16k bash scripts/deepseek_v41/served_cell_bench.sh 2>&1 | grep -vE \"\$F\" | tail -8
echo '[window 35 done]'
"
