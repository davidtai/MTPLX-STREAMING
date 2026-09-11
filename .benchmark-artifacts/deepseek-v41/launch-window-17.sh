#!/bin/bash
# Window 17: served MTP smoke (256 tok) under the stack env + gather_qmm mxfp4 M=1 microbench (W43).
WT=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/deepseek-v41
MODEL=/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4
OUT=$WT/docs/deepseek-v41/receipts/gpu-windows/window-17
mkdir -p "$OUT"
cd "$WT"
for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8080/health | grep -q '"ok":true' && break; sleep 10; done
echo "[launcher] $(date -u +%FT%TZ) agent healthy; starting window 17"
export GPU_WINDOW_CHILD_RSS_CAP_BYTES=$((85*1024*1024*1024))
exec bash scripts/deepseek_v41/gpu_window.sh bash -c "
export PYTHONPATH=$WT DSV41_MODEL=$MODEL MTPLX_DSV41_PREFILL_CHUNK=1024
PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python3
F='transformers\]|█|╗|╝|║|╭|╰|│'
echo '[step 1] serve MTP 256 tok under stack env'; MTPLX_DSV41_HEAD_MODE=bf16 MTPLX_DSV41_SINKHORN_METAL=1 MTPLX_DSV41_ATTN_COMPILE=1 MTPLX_DSV41_ATTN_WIN_MEMO=1 DSV41_MAX_TOKENS=256 DSV41_SERVE_EXTRA_ARGS='--generation-mode mtp' bash scripts/deepseek_v41/serve_health.sh 2>&1 | grep -vE \"\$F\" | tail -30
echo '[step 2] gather_qmm microbench'; \$PY scripts/deepseek_v41/gather_qmm_microbench.py --arms mxfp4_switch mxfp4_convention affine_q4_gs64 affine_q8_gs64 bf16_dense --m-values 1 4 --top-k 6 --hidden 5120 --inter 2304 --bank-slots 92 --group-size-mxfp4 32 --iters 50 --warmup 10 --memory-limit-gib 8.0 --seed 0 --out $OUT/gather-qmm-microbench.json 2>&1 | grep -vE \"\$F\" | tail -30
echo '[window 17 done]'
"
