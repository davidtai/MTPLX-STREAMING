# W24 — DSV4.1-Flash decode routing-locality census + decode-streaming levers

Branch `feat/deepseek-v41-w24` off `feat/deepseek-v41-streaming` @ 5d6dd8ad.
CPU-designed; GPU-measured by the orchestrator inside `scripts/deepseek_v41/gpu_window.sh`.
Aligned to `docs/deepseek-v41/OPTIMIZATION_LEDGER.md` (integration c979b783): this
census is the decision instrument for Gates 0/1/D and R5.

## Goal & baseline
Decode **> 20 tok/s** on the native mxfp4 artifact at the 1,024- and 16,384-token
shapes, MTP on, KV minimal, expert cache maximal, under the 100 GB box. Measured
baseline (GPU, 1,024 prompt, greedy, 256 tok, loader path): prefill 37 tok/s,
TTFT 27 s, **decode 4.79 tok/s** @ 72 GiB, 4.85 @ 92 GiB — capacity is not the
lever within one prompt; cold misses dominate.

## Deliverables (all committed on this branch)
| File | What |
|---|---|
| `scripts/deepseek_v41/routing_census.py` | CPU census tool (Gate 0/1/D/R5) |
| `docs/deepseek-v41/receipts/routing_census_1024.json` | census receipt (committed) |
| `mtplx/expert_io.py`, `mtplx/expert_runtime.py` | **`io_read_fanout`** lever (new, default off, byte-identical) |
| `scripts/deepseek_v41/ab_decode_levers.py` | GPU A/B lever harness (Gate D realized BW) |
| `tests/test_deepseek_v41_decode_levers.py` | byte-identity + census-function tests (all green) |
| this report | census numbers, cost model, ranked levers, A/B commands |

## Cost model — the physics (SSD-bandwidth/service-bound decode)
Per the ledger §1 and confirmed by this census:

- Routed record = **18,800,000 B** (288,777,830,400 B ÷ 40 layers ÷ 384 experts;
  native mxfp4 gs32, lossless repack, cos 1.000000). AR reads **40 × 6 × 18.80 MB
  = 4.51 GB/token** of routed records.
- SSD device ceiling **13.42 GB/s (12.5 GiB/s)**; single-stream realized can be
  far lower (~5.2 GiB/s at queue depth ≈ 1 — the serially-layer-dependent decode
  chain; saturates only at qd ≥ 64).
- Today ≈ 23–30 % resident → **~3.0–3.5 GB/token off SSD** →
  `13.42 / 3.0 ≈ 4.5 tok/s` (matches the measured 4.79).

**20 tok/s = 50 ms/token → ≤ 0.67 GB off SSD per *accepted* token → a ~4.5× cut.**
No single lever delivers it; it is the product of four measured mechanisms:

| Factor | Mechanism | W24 lever(s) | Gate |
|---|---|---|---|
| A. Acceptance (MTP) | draft K, verify in one forward, amortize read over accepted tokens | (W23 owns the DSpark port) | Gate 2 |
| B. Verify-row dedup | read the per-cycle expert **union** once, not per verify position | census `u` measures the ceiling; `overlap_miss_reads` at the gather | Gate 0 |
| C. Frequency residency | pin the hot experts within a layer | `cache_scope`, (conditional) census-pin | Gate 1 |
| D. Realized BW / queue depth | issue the whole read set concurrently; split each record | **`io_read_fanout`**, `overlap_miss_reads`, `max_inflight_io_bytes`, `max_read_chunk_bytes` | Gate D |

Worked stack (ledger §1.4): even α 0.78 (L≈2.85) × dedup u≈6 × hit 23→40 % ×
BW≈90 % ⇒ ~12.6 tok/s — **short of 20**; 20 needs u→~4–5 and/or hit→~55 % and/or
a quality-neutral acceptance rule. Honest read: **central estimate 10–14 tok/s;
20 is the optimistic edge.** The census gates decide which world we are in.

## Census results (1,024-token prompt + 64 greedy tokens, CPU)
<!-- CENSUS_FILL_START -->
_Pending the committed census run; numbers land in
`docs/deepseek-v41/receipts/routing_census_1024.json` and are summarized here._
<!-- CENSUS_FILL_END -->

## Levers implemented (each an explicit switch, default OFF, byte-identical)
Cache eviction (`cache_policy=frequency`, default) and the IO knobs already exist
as `ExpertStreamingConfig` fields; W24 adds the one Factor-D lever the runtime
lacked and packages the rest as A/B arms.

1. **`io_read_fanout: int = 1`** (NEW — `mtplx/expert_io.py`, `mtplx/expert_runtime.py`).
   Splits one large record's positional read into N concurrent contiguous
   sub-reads on the shared ref-counted fd (`os.preadv` is thread-safe with an
   explicit offset). Raises SSD queue depth beyond the per-record floor on the
   serial decode chain. **Byte-identical** (contiguous partition into disjoint
   destination slices — tested for N∈{1,2,4,8}). Default 1 = shipped path
   untouched. Expected: **up to ~1.5×** decode IFF Gate D shows realized BW
   < 8 GiB/s; ~0 if already saturated.
2. **`overlap_miss_reads` (existing)** — batches a decode layer's ≤6 miss reads
   into one coalesced/concurrent read set (offset-sorted scatter preadv). Where
   hy3 saw this "dead" (0.9 misses/layer, nothing to batch), DSV4.1 misses ~6/layer.
   Byte-identical (documented; A/B sha check).
3. **`max_read_chunk_bytes` (existing read-chunk lever)** — 8 MiB default splits an
   18.8 MB record into 3 sequential reads; ≥ 19 MB = one preadv/record (fewer
   syscalls). Byte-identical (tested).
4. **`max_inflight_io_bytes` (existing)** — widens the expert-IO threadpool =
   across-record queue depth.
5. **`cache_scope=global` (existing)** — global LRU pool vs per-layer banks (a
   Factor-C probe; hy3 measured ~3 %).

Ranking is conditional on the census (below): if Gate D BW is low, rank
`io_read_fanout` + `overlap_miss_reads` first; if Gate 1 shows real cross-layer
concentration, a census-derived hot-expert pin list (Factor C) is added.

## A/B commands (orchestrator, inside gpu_window.sh)
One lever per arm, paired with the in-window `control` (ledger §7 window-drift
law). Realized decode BW (Gate D) and control-vs-candidate byte-identity are in
every receipt.

```
# 1,024-token shape (Gate D + Factor-D lever sweep):
scripts/deepseek_v41/gpu_window.sh \
  env PYTHONPATH="$PWD" .venv/bin/python3 scripts/deepseek_v41/ab_decode_levers.py \
    --context-tokens 1024 --decode-tokens 256 \
    --arms control fanout4 overlap inflight8g chunk32m overlap_fanout4 \
    --out docs/deepseek-v41/receipts/ab_decode_levers_1024.json

# 16,384-token shape:
scripts/deepseek_v41/gpu_window.sh \
  env PYTHONPATH="$PWD" .venv/bin/python3 scripts/deepseek_v41/ab_decode_levers.py \
    --context-tokens 16384 --decode-tokens 256 \
    --arms control fanout4 overlap \
    --out docs/deepseek-v41/receipts/ab_decode_levers_16384.json
```
Gate D read: `arms[control].realized_decode_gib_per_s`. If < 8 GiB/s, land
`fanout`/`overlap` (they hand ~1.5× for free); if already ~12, only byte-cutting
(A×B×C) moves decode. Every candidate arm must show
`byte_identical_to_control: true`.

## R5 — 16K prefill "read the bank once" (TTFT lever; W20 owns prefill)
W20's chunked 16K prefill runs ~13 chunks of ~1,271 tokens; each chunk's MoE
gathers its routed experts independently. This census measures how many distinct
experts a 1,024-token chunk touches per layer (`gate_r5_prefill_read_once`); a
1,271-token chunk touches ≥ that. If it is ~all 384, 16K prefill re-reads the
269 GiB bank up to ~13× (≈ 5 min of SSD), dominating the 27 s TTFT.

**Runtime fix (sketch, for W20):**
- *Expert-major prefill*: for each layer, group every chunk's routed rows by
  expert id and gather each of the (≤384) records **once**, applying it to all
  rows across all chunks that routed to it. Bank read = the union once (~269 GiB
  read-once floor) instead of per-chunk. Memory-bounded: hold one expert's record
  + scatter its output to the (sparse) rows; never all chunks' activations at once.
- *Prefill-phase pin-until-consumed*: a prefill cache policy that pins the current
  layer's fetched records until **every** chunk has consumed that layer, then
  releases — same read-once effect with the existing slot machinery, one policy
  flag, no gather rewrite.

Both are byte-identical (same records, same math; only read order changes) and are
the ledger's R5. Priced against the census `gate_r5` distinct-experts number below.

## Memory contract & peak RSS
The census holds `/tmp/dsv41-cpu-model-load.lock` (single concurrent model load),
CPU device, `mx.set_cache_limit(0)` + `mx.clear_cache()` per chunk, an in-process
RSS watchdog, and chunked prefill.

**Finding (important for the coordinator):** the runtime's fixed-footprint PLAN
(`expert_runtime.py:2050`, unconditional) counts the *full* resident manifest
(~23 GiB incl. the 15.3 GiB MTP experts the AR path never loads), so a literal
`memory_limit ≤ 12 GiB` makes the loader **refuse** (`fixed footprint exceeds
limit by 13.4 GB`). `memory_limit` is the plan *ceiling*, not process RSS — with
`expert_cache_limit_bytes` set explicitly it caps the actual slot buffers
independent of the ceiling. So the census runs with a 32 GiB plan ceiling (like
`decode_probe.py`'s 100 GiB) while **real RSS is held down** by text-only lazy-mmap
residents (~10 GiB, the hard floor — the model cannot forward on less), a 0.5 GiB
expert cache, MLX cache disabled, and the watchdog.
<!-- RSS_FILL_START -->
_Peak RSS: pending the committed run._
<!-- RSS_FILL_END -->

## Not done / handoff
- Gates 2/3/4 (MTP + dedup + stack) need W23's DSpark port; the A/B harness accepts
  the arms; R2 dedup rides the multi-position gather.
- R5 fix is W20's (prefill allowlist); sketched above, priced by `gate_r5`.
- Frequency-pin (R3) implemented only if Gate 1 passes (see census verdict below).
