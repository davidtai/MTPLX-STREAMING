#!/bin/bash
# Window 12: re-run window 11 after the W34/W35/W36 fixes + K3 parity + K3 decode arm.
WT=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41
MODEL=/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4
OUT=$WT/docs/deepseek-v41/receipts/gpu-windows/window-12
mkdir -p "$OUT"
cd "$WT"
for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8080/health | grep -q '"ok":true' && break; sleep 10; done
echo "[launcher] $(date -u +%FT%TZ) agent healthy; starting window 12"
export GPU_WINDOW_CHILD_RSS_CAP_BYTES=$((90*1024*1024*1024))
exec bash scripts/deepseek_v41/gpu_window.sh bash -c "
export PYTHONPATH=$WT DSV41_MODEL=$MODEL MTPLX_DSV41_PREFILL_CHUNK=1024
PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3
F='transformers\]|█|╗|╝|║|╭|╰|│'
echo '[step 1] serve_health AR'; DSV41_MAX_TOKENS=64 bash scripts/deepseek_v41/serve_health.sh 2>&1 | grep -vE \"\$F\" | tail -40
echo '[step 2] serve_health MTP'; DSV41_MAX_TOKENS=64 DSV41_SERVE_EXTRA_ARGS='--generation-mode mtp' bash scripts/deepseek_v41/serve_health.sh 2>&1 | grep -vE \"\$F\" | tail -40
echo '[step 3] K3 sinkhorn parity'; MTPLX_GPU_PARITY=1 \$PY -m pytest tests/models/test_deepseek_v41_sinkhorn_metal.py::test_sinkhorn_kernel_parity_gpu -q -s -p no:cacheprovider 2>&1 | tail -8
echo '[step 4] env levers A/B @1024 (K1, K3)'; \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 1024 --decode-tokens 256 --arms control shared_overlap sinkhorn_metal --syncs 16 --memory-limit-gib 82 --out $OUT/ab-1024-env-levers.json 2>&1 | grep -vE \"\$F\" | tail -60
echo '[step 5] I/O levers A/B @1024 (W24 Gate D)'; \$PY scripts/deepseek_v41/ab_decode_levers.py --context-tokens 1024 --decode-tokens 256 --arms control fanout4 overlap overlap_fanout4 --memory-limit-gib 82 --out $OUT/ab-1024-io-levers.json 2>&1 | grep -vE \"\$F\" | tail -60
echo '[step 6] K16 A/B @16384'; \$PY scripts/deepseek_v41/ab_decode_env_levers.py --context-tokens 16384 --decode-tokens 64 --arms control layer_major --memory-limit-gib 82 --out $OUT/ab-16384-k16.json 2>&1 | grep -vE \"\$F\" | tail -60
echo '[window 12 done]'
"
