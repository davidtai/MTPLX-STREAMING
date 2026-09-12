#!/bin/bash
# Window 37 (RUNNER, drafted before reviews close — prune steps whose branch did not get MERGE):
# each reviewed lever alone on the 16K cell vs the window-36 cell16k_ring reference, --stage-timing --utilization, AR mode only:
# 1 cell16k_ring_fused (W91 K35; engagement + sha must match); 2 cell16k_ring_pool (W87; cold_start + steady hit rate); 3 cell16k_ring_switch (W92; allhit_fence_deferred == all_hit);
# 4 cell16k_ring_prefetch (W93 gate-oracle prefetch k=10; gate_prefetch counters); 5 GPU parity for the premix kernel + device-route parity (MTPLX_GPU_PARITY=1) — receipts only.
WT=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41
MODEL=/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4
OUT=$WT/docs/deepseek-v41/receipts/gpu-windows/window-37
mkdir -p "$OUT"
cd "$WT"
for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8080/health | grep -q '"ok":true' && break; sleep 10; done
echo "[launcher] $(date -u +%FT%TZ) agent healthy; starting window 37"
export GPU_WINDOW_CHILD_RSS_CAP_BYTES=$((96*1024*1024*1024))
exec bash scripts/deepseek_v41/gpu_window.sh bash -c "
export PYTHONPATH=$WT DSV41_MODEL=$MODEL MTPLX_DSV41_PREFILL_CHUNK=1024
PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3
AB=scripts/deepseek_v41/ab_decode_env_levers.py
C='--context-tokens 16384 --decode-tokens 256 --max-kv 17408 --stage-timing --memory-limit-gib 60 --utilization'
F='transformers\]|█|╗|╝|║|╭|╰|│'
echo '[step 0] AR cell16k_ring @16384 (paired reference for this window; run-to-run spread is ~7%, levers are ≤2.5% → compare within the window only)'; MTPLX_ROUTE_STAGE_PROBE=1 \$PY \$AB \$C --arms cell16k_ring --out $OUT/ar-ring-ref.json 2>&1 | grep -vE \"\$F\" | tail -20
echo '[step 1] AR cell16k_ring_fused @16384 (W91 K35 fused small stages; small_stages_engagement fused>0, sha 2d39390f5686)'; MTPLX_ROUTE_STAGE_PROBE=1 \$PY \$AB \$C --arms cell16k_ring_fused --out $OUT/ar-ring-fused.json 2>&1 | grep -vE \"\$F\" | tail -24
echo '[step 2] AR cell16k_ring_pool @16384 (W87 single pool; cold_start first-64 vs steady, expert_cache hit_rate; sha 2d39390f5686)'; MTPLX_ROUTE_STAGE_PROBE=1 \$PY \$AB \$C --arms cell16k_ring_pool --out $OUT/ar-ring-pool.json 2>&1 | grep -vE \"\$F\" | tail -24
echo '[step 3] AR cell16k_ring_switch @16384 (W92; allhit_fence_deferred == all_hit; sha 2d39390f5686)'; MTPLX_ROUTE_STAGE_PROBE=1 \$PY \$AB \$C --arms cell16k_ring_switch --out $OUT/ar-ring-switch.json 2>&1 | grep -vE \"\$F\" | tail -24
echo '[step 4] in-model census at T=1024 on stack_a (SELECTED_KEYS off = the window-16 arm; does attention drop to ~1.7 ms/layer?)'; nice -n 19 \$PY scripts/deepseek_v41/metal_decode_attn_bisect.py --in-model --gpu --model $MODEL --arms stack_a --context-tokens 1024 --memory-limit-gib 60 --max-kv 4096 --in-model-steps 30 --utilization --out $OUT/in-model-1k-stack_a.json 2>&1 | grep -vE \"\$F\" | tail -24
echo '[step 5] in-model census at T=1024 on cell16k_ring (selected keys on; paired control for step 4)'; nice -n 19 \$PY scripts/deepseek_v41/metal_decode_attn_bisect.py --in-model --gpu --model $MODEL --arms cell16k_ring --context-tokens 1024 --memory-limit-gib 60 --max-kv 4096 --in-model-steps 30 --utilization --out $OUT/in-model-1k-ring.json 2>&1 | grep -vE \"\$F\" | tail -24
echo '[step 6] GPU parity receipts (premix kernel; K3 convention)'; MTPLX_GPU_PARITY=1 MTPLX_PARITY_RECEIPT=$OUT/parity-premix.json nice -n 19 \$PY -m pytest tests/models/test_deepseek_v41_small_stages_fused.py -q -p no:cacheprovider -k parity_gpu 2>&1 | tail -3
echo '[window 37 done]'
"
