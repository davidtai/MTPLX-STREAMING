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

**Census verdict on the stack (this changes the ledger's central estimate down):**
- **Factor A/B (MTP) is dead-here on bytes.** Gate 0 measured u=19 at K=3 (dedup
  1.28×), so MTP reads 3.17× the AR records/cycle and needs L≥3.17 to break even —
  V4's ~2.85 loses. MTP is a *byte regression* on this bank unless a quality-neutral
  rule pushes α far up, and even then marginal. **Do not lead with MTP.**
- **Factor C (residency) is capped at the within-prompt cold ceiling 0.733 →
  ~11.2 tok/s.** No within-prompt policy beats it; the only feasible budget (115
  slots = 80.5 GiB) reaches a warm ~18 tok/s ceiling only under optimistic
  cross-prompt priming + full realized BW. Static pinning is worse than LRU
  (Gate 1); online frequency eviction (default) is already near the deployable best.
- **Factor D (realized BW) is the one live decode lever.** Measured 4.79 < the
  11.2 cold ceiling ⇒ the deficit is queue depth, recoverable without cutting
  bytes. This is where W24's implemented levers act.

**Honest read (revised by the census): the within-prompt decode ceiling is ~11
tok/s, not 20.** Realistic near-term win: Factor D lifts 4.79 → **~8–11 tok/s** (if
today's realized decode BW is ~3–5 GiB/s, per Gate D). Reaching 20 would require
cross-prompt warm residency at a budget the 82 GiB envelope cannot hold, so it is
**out of reach on the current bank/shape** — report this to David rather than
chase it.

## Census results (1,024-token prompt + 64 greedy tokens, CPU)
Receipt: `docs/deepseek-v41/receipts/routing_census_1024.json` (peak RSS 13.03 GiB;
40 MoE layers, top-6, record 18,800,000 B; decoded output coherent Python — model
healthy). Routing is gate-computed and identical under any cache config, so the
trace and every simulated number below are production-valid; the runtime IO/hit
telemetry in the receipt is from the memory-forced 0.3 GiB **global** cache and is
**not** production (labeled so in the JSON).

### Decode-window locality (the core result)
- A 64-token decode touches **~102 distinct experts/layer** (of 384). LRU hit rate
  is **flat at 0.733 for every budget ≥ 115 slots/layer, and Belady == LRU there**
  — i.e. more cache does **nothing** within one prompt once the ~102-expert
  working set fits. This *is* the measured "capacity is not the lever / cold misses
  dominate" (4.79 @ 72 GiB ≈ 4.85 @ 92 GiB).
- **Within-prompt decode ceiling = 0.733 hit** → miss `0.267 × 4.51 GB = 1.20
  GB/tok` → **~11.2 tok/s** at the 13.42 GB/s ceiling. **No within-prompt cache
  policy beats this** (it is the cold first-touch floor).

| slots/layer | resident if all 40 layers | cold LRU=Belady hit | warm (prefill-primed) LRU hit | warm miss GB/tok | warm SSD tok/s ceiling |
|---|---|---|---|---|---|
| 115 | **80.5 GiB (fits 82)** | 0.733 | 0.835 | 0.746 | **18.0** |
| 205 | 143.6 GiB (infeasible) | 0.733 | 0.941 | 0.265 | 50.5 |
| 384 | 268.9 GiB (infeasible) | 0.733 | 0.989 | 0.052 | 259 |

Warm = cross-prompt priming; the only budget that fits 82 GiB is **~115 slots/layer**,
whose warm ceiling is ~18 tok/s (optimistic — assumes the prefill-resident set
overlaps decode and realized BW = 12.5). The measured 4.79 sits **below the 11.2
cold ceiling**, so today's gap is **realized BW / queue depth (Factor D)**, not cache.

### Gate 0 — verify-row union `u` (MTP viability): **FAIL (MTP dead-here)**
Sliding K+1 windows over consecutive decode tokens, per-layer union of top-6:

| K | verify positions | u median | u p90 | u mean | dedup vs naive | verdict |
|---|---|---|---|---|---|---|
| 1 | 2 | 11 | 12 | 10.9 | 1.10× | — |
| 2 | 3 | 15 | 17 | 15.1 | 1.20× | u>14: regresses |
| 3 | 4 | **19** | 22 | 18.7 | 1.28× | **u>14: regresses** |

Pass needed median u ≤ 10 at K=3; measured **19**. Consecutive draft tokens route
to *largely different* experts (only 1.28× overlap), so an MTP verify reads ~19
records/layer/cycle vs AR's 6 = **3.17× more SSD bytes/cycle**. Break-even
acceptance is L ≥ 19/6 = **3.17 tokens/cycle**; V4 got ~2.85 at K3 → **MTP is a net
byte *loss* on this streaming bank** (the ledger §1.4 GLM-shaped trap, confirmed).
Prefill-adjacent (much larger sample) agrees: u median 11/15/18.

### Gate 1 — frequency residency (held-out 70/30): **FAIL for static pinning**
Top-N hottest-expert pin (from train freq) vs LRU vs Belady on the unseen eval:

| N slots | static-pin miss | LRU miss | Belady miss | pin vs LRU | Belady vs LRU | cross-layer coverage stdev |
|---|---|---|---|---|---|---|
| 32 | 0.516 | 0.338 | 0.191 | **−52.5 %** | +43.4 % | 0.064 |
| 64 | 0.380 | 0.194 | 0.102 | **−96.1 %** | +47.5 % | 0.051 |
| 96 | 0.283 | 0.130 | 0.067 | **−117.9 %** | +48.5 % | 0.037 |

Static pinning is **worse than plain LRU** on held-out traffic (the hot set churns
— the hy3 result, confirmed; cross-layer stdev 0.037–0.064 is only modestly above
hy3's dead 0.027 and a per-layer/pin allocation still captures nothing deployable).
**Do not ship a census pin list.** What *is* real: **Belady beats LRU 43–48 % at
tight budgets** — so a good *online eviction* policy has headroom when the budget
is < the ~102-expert working set. The shipped default is already
`cache_policy=frequency` (a decayed-LFU between LRU and Belady); that is the
residency lever, not pinning.

### Gate D — realized decode SSD BW
CPU census (non-production 0.3 GiB cache, ~all-miss): **0.81 GiB/s single-stream**
— far below the 12.5 ceiling, consistent with the queue-depth-1 problem, but not
the GPU number. **The GPU A/B (`ab_decode_levers.py`, control arm
`realized_decode_gib_per_s`) is authoritative for Gate D** and is what decides the
Factor-D levers.

### Gate R5 — 16K prefill read-once: **REAL, large TTFT lever**
A **1,024-token chunk touches 261 distinct experts/layer (68 % of 384)** (min 224,
max 313). So W20's 16K prefill (~13 chunks) naively re-reads the bank **~8.8×
= ~2,379 GiB ≈ ~3 min of SSD** vs the **269 GiB read-once floor** — this dominates
the 27 s TTFT. The expert-major / pin-until-consumed fix (below) saves ~8×.

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

**Census-decided ranking (final):**
1. **`io_read_fanout` + `overlap_miss_reads`** (Factor D) — the census makes these
   the *only* live decode levers: measured 4.79 < the 11.2 cold ceiling, so the gap
   is realized BW. Land both if the GPU Gate-D control arm shows < 8 GiB/s.
2. **`max_inflight_io_bytes`, `max_read_chunk_bytes`** (Factor D companions) — more
   queue depth / fewer syscalls; A/B them alongside.
3. **Residency**: keep `cache_policy=frequency` (default; already near Belady's
   deployable best). `cache_scope=global` is a small probe. **No pin list** —
   Gate 1 proved static pinning worse than LRU.
4. **MTP (R1/R2): dead-here on bytes** (Gate 0 u=19). Do not build the dedup lever
   for a byte win; if MTP ships for latency, R8 (quality-neutral acceptance, opt-in)
   is the only way it nets positive, and only above L≈3.2.
5. **R5 prefill read-once** (below) — the biggest *TTFT* lever (W20's allowlist).

A census-derived hot-expert pin lever was **considered and rejected** on the Gate 1
evidence (it would regress), so it is deliberately not implemented (avoids shipping
a measured-negative lever).

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
**Peak RSS = 13.03 GiB** (committed receipt). The census ran chunked-prefill
(chunk 8) + global slot scope + 0.3 GiB cache + MLX buffer cache disabled.

**The ≤ 12 GiB target is below this model's CPU forward floor.** The ~11 GiB
text-only residents are fully faulted on the first forward, and the sparse
attention/indexer needs a multi-token forward (single-token prefill crashes:
`compress_kv is None`), whose smallest working set pushes peak to ~12.6–13 GiB.
Global slot scope and cache-disable did not change this — the floor is residents +
one indexer-valid forward. I verified the box was light (used ~7.5 GB, resident
agent idle, no heavy workers) before running at a **13.5 GiB watchdog (under the
14 GiB kill)**; peak landed at 13.03. **Recommendation for the coordinator:** run
future DSV4.1 CPU census/probe work either (a) during a GPU window with the qwen
agent booted out (frees ~86 GB, so 13 GiB is trivially safe) or (b) accept a ~13
GiB worker floor for this model; ≤ 12 GiB is not physically reachable for a real
forward. `memory_limit` was the 32 GiB *plan ceiling* (the fixed-footprint plan
counts the full resident manifest incl. unloaded MTP experts), never the RSS.
<!-- RSS_FILL_END -->

## Not done / handoff
- **MTP (Gates 2/3): re-scope.** Gate 0 (u=19 at K=3) says MTP is a byte
  regression on this bank; before spending a GPU window on W23's DSpark port,
  confirm on GPU that MTP-on decode is not slower than AR (it likely is unless
  acceptance ≫ 3.2). The A/B harness accepts MTP arms once W23 lands.
- **R5 (16K TTFT) is W20's** (prefill allowlist); the fix is sketched above and now
  priced by the census (261/384 distinct/layer → ~8.8× re-read). This is the
  largest *TTFT* win available and independent of the decode story.
- **Frequency-pin (R3): rejected on Gate 1 evidence** (static pinning regresses vs
  LRU held-out) — deliberately not implemented.
- **Next GPU window (orchestrator):** run the 1,024 Gate-D A/B (command above) to
  get the real realized decode BW and confirm `io_read_fanout`/`overlap` lift it
  toward the ~11 tok/s cold ceiling. That is the highest-value remaining decode
  experiment; everything else is capped by the census's within-prompt physics.
