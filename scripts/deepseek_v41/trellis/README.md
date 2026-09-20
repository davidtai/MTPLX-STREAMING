# DSV4.1 F34 — trellis (TCQ) re-encoding into the eschamoe K=3 format (CPU-only)

Re-encodes DeepSeek-V4.1 mxfp4 experts into EschaLabs' `eschamoe` K=3 (3 bits/weight) trellis format, whose bit-exact
decoder already lives in `mtplx/eschamoe.py` on this branch. Answers the two F34 questions: does a serialized artifact
round-trip through the existing decoder, and how close does 3-bit TCQ get to the FP4 source.

## Modules

- `mxfp4_source.py` — read expert records from `~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4/experts.bin` via
  `expert-manifest.json`; dequantize to fp32 (`mx.dequantize(mode="mxfp4")`, numpy cross-checked bit-exact).
- `tcq_encode.py` — the codebook `DEC[65536]`; `cycle_tables()` derives the single 256-symbol **circular** trellis
  (state = 13 shared bits, 8 branches, conflict-free packing) from the decoder's own gather table; tail-biting
  **Viterbi** (`viterbi_encode`) and **beam** (`beam_encode`) + seam repair; packer; own numpy decoder; forward-chain
  scale/target math; `serialize()` → vendor-field `.npz`.
- `tcq_verify.py` — decode with the EXISTING `mtplx.eschamoe.decode_expert_weights`, assert bit-exact vs own decode,
  report cosine / rel-RMS / max|err| and the forward-chain check.
- `run_f34.py` — gate-aware sample runner; writes `reports/dsv41-f34-trellis/f34_results.json` incrementally.
- `tests/test_dsv41_f34_trellis.py` — 11 CPU tests (codebook == vendor + primitive; mxfp4 numpy == MLX; cycle-table
  validity; pack→decode bit-exact; Viterbi optimality vs brute force; tail-biting ring closure; forward-chain).

## Run

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd scripts/deepseek_v41/trellis
nice -n 19 <repo>/.venv/bin/python -m pytest tests/test_dsv41_f34_trellis.py -q     # 11 passed
nice -n 19 <repo>/.venv/bin/python run_f34.py all                                    # sample + full projection
```

Interpreter: the main worktree's `.venv` (MLX 0.32.2). The eschamoe decoder is loaded from this branch's
`mtplx/eschamoe.py` by file path (the editable-installed `mtplx` points at the main worktree, which lacks it).

## Result (see `reports/dsv41-f34-trellis-report.md`)

Bit-exact round-trip (0 mismatches). 3-bit TCQ vs FP4: cosine **0.9885** (exact Viterbi) / **0.9878** (beam-256),
rel-RMS ≈ 0.15 — beats 3-bit affine (0.9807), below 4-bit affine (0.9947). Serialized full projection = **3.010
bits/weight** (29 % smaller than mxfp4). Full-bank re-encode is a large offline job (~167 days beam, single CPU thread).
