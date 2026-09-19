#!/bin/bash
# Window 13: stage-timing + warm-repeat ceiling, K3 parity receipt, K4/K3 arms @1K, K16 @16K, MTP serve re-check.
WT=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41
MODEL=/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4
OUT=$WT/docs/deepseek-v41/receipts/gpu-windows/window-13
mkdir -p "$OUT"
cd "$WT"
for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8080/health | grep -q '"ok":true' && break; sleep 10; done
echo "[launcher] $(date -u +%FT%TZ) agent healthy; starting window 13"
export GPU_WINDOW_CHILD_RSS_CAP_BYTES=$((90*1024*1024*1024))
exec bash scripts/deepseek_v41/gpu_window.sh bash -c "
export PYTHONPATH=$WT DSV41_MODEL=$MODEL MTPLX_DSV41_PREFILL_CHUNK=1024
PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3
F='transformers\]|█|╗|╝|║|╭|╰|│'
echo '[step 1] stage timing + warm repeat @1024'; \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 1024 --decode-tokens 64 --arms control --stage-timing --warm-repeat --memory-limit-gib 82 --out $OUT/stage-timing-1024.json 2>&1 | grep -vE \"\$F\" | tail -80
echo '[step 2] K3 parity (receipt)'; MTPLX_GPU_PARITY=1 MTPLX_PARITY_RECEIPT=$OUT/k3-parity.json \$PY -m pytest tests/models/test_deepseek_v41_sinkhorn_metal.py -k parity_gpu -q -s -p no:cacheprovider > $OUT/k3-parity.log 2>&1; tail -5 $OUT/k3-parity.log; cat $OUT/k3-parity.json 2>/dev/null | head -c 1500; echo
echo '[step 3] env arms @1024 (K3, K4, all)'; \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 1024 --decode-tokens 256 --arms control sinkhorn_metal hc_compile all_levers --syncs 16 --memory-limit-gib 82 --out $OUT/ab-1024-k3-k4.json 2>&1 | grep -vE \"\$F\" | tail -40
echo '[step 4] K16 A/B @16384'; \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 16384 --decode-tokens 64 --max-kv 16640 --arms control layer_major --memory-limit-gib 82 --out $OUT/ab-16384-k16.json 2>&1 | grep -vE \"\$F\" | tail -40
echo '[step 5] serve_health MTP'; DSV41_MAX_TOKENS=64 DSV41_SERVE_EXTRA_ARGS='--generation-mode mtp' bash scripts/deepseek_v41/serve_health.sh 2>&1 | grep -vE \"\$F\" | tail -30
echo '[window 13 done]'
"
