# W20 — Token-chunked prefill for DeepSeek-V4.1-Flash (streaming)

Branch: `feat/deepseek-v41-w20` off `feat/deepseek-v41-streaming` (integration @ 5d6dd8ad).
Status: DONE. CPU-only, tiny synthetic configs; no real-artifact load. Peak RSS 389 MB.

## Verdict

The 16,384-token standard-shape prefill dies with a **103,089,701,120-byte
`[metal::malloc]`** over the 86.5 GiB Metal buffer cap. That number is **not an
MoE tensor** — it is the **attention score** on a ratio-2 CSA layer. The MoE
traceback is a lazy-evaluation artifact. Fix: **token-chunk the forward** so every
per-query prefill transient is bounded, while the caches accumulate exactly. The
cache and `expert_mlx.py` needed no change; the entire fix is in
`deepseek_v41.py` plus a latent-bug guard. Chunked prefill is proven exact on CPU
(logits ≤ 1e-5, cache state equal) and the 16K transients are proven < 8 GB.

## 1. Naming the allocation (arithmetic)

The reported failure trace ends in the streamed switch:

```
deepseek_v41.py:723 layer(...) → :670 self.mlp(x)
  → deepseek_v41_moe.py:243 routed = self.switch_mlp(xf, indices)
  → expert_mlx.py HotExpertSwitchGLU.__call__ → RuntimeError:
    [metal::malloc] Attempting to allocate 103,089,701,120 bytes
    (> max buffer size 86,586,540,032 bytes)
```

But `103,089,701,120` factors onto the **attention score**, not the MoE:

```
Attention._sparse_attend (deepseek_v41.py:486):
  scores = einsum("bshd,btd->bsht", q[b,s,H,hd], KV[b,T,hd])  →  [b, s, H, T] fp32

standard shape:  b=1   s=16385 (16,384 + BOS)   H=64
ratio-2 CSA layer:  n_comp = s // 2 = 8192
                    T = window(all s tokens in phase-1 prefill) + n_comp
                      = 16385 + 8192 = 24577
  bytes = s * H * T * 4 = 16385 * 64 * 24577 * 4
        = 16385 * 6,291,712
        = 103,089,701,120         ← EXACT match
```

`16385 * 64 * 24576 * 4 = 103,079,215,104` for `s=16384`; the `+1` BOS token makes
it the reported `…701,120`. Either way the buffer clears the 86.5 GiB cap.

### Why it surfaces in the MoE

MLX is lazy. The attention of layer *L* builds a graph but is not evaluated until
something forces it. The first force in each layer is inside the streamed switch:
`HotExpertSwitchGLU._run` calls `mx.eval(indices)` (expert_mlx.py:1976), and
`indices` depends transitively on that layer's attention output. So `mx.eval`
materialises the layer's 103 GB score tensor, and the exception is raised there —
in the MoE frame — even though the tensor is the attention score. This is the same
class as the DeepSeek-V4 long-context OOM (head_dim 512 blocks MLX flash-SDPA, so
the score is materialised in full): see `deepseek-v4-longcontext-prefill`.

### The other layers and transients at 16K

| tensor | shape (fp32) | bytes | note |
|---|---|---|---|
| **ratio-2 CSA score** | `[1, 16385, 64, 24577]` | **103.09 GB** | > cap → the crash |
| SWA-only score | `[1, 16385, 64, 16385]` | 68.73 GB | < 86.5 GiB cap → *allocates* (layers 0/1), so the exact number is what discriminates the ratio-2 layer |
| indexer score | `[1, 16385, 32, 8192]` | 17.18 GB | `Indexer.select` einsum; also unbounded in `s` |
| head logits | `[1, 16385, 129280]` | 8.47 GB | `Model.head`; under the cap, present in one-shot and chunked alike (see §7) |
| MoE routed rows | `[16385*6, 5120]` per wave | ≤ 2 GB | already per-row in `_gather_component_bank`; not the crash |

The MoE switch itself was never the allocator — its per-wave gather is per routed
row and bounded (`_gather_component_bank`: `selected = x.reshape((rows,1,1,hidden))`,
outputs `[rows,1,1,N]`). The task's "something inside the MoE allocates ~103 GB" is
the lazy-eval artifact above; the arithmetic points squarely at attention.

## 2. Fix — token-chunked prefill (`deepseek_v41.py` only)

Query-chunking the *token* dimension bounds every per-query transient (attention
score `[chunk, H, T]`, indexer `[chunk, 32, n_comp]`, routed rows `chunk*top_k`)
because `T`/`n_comp` are fixed by the context while `chunk` is the query count.

`DeepseekV41Backbone.__call__` now splits a prompt into spans of ≤ `chunk` query
tokens. Each span runs `_forward_span` — the *unchanged* original forward body —
through all 40 layers, appending to the **same** accumulating cache and advancing
`cache.offset`; a fresh `cache.new_shared_runtime()` per span. After each span the
cache stores + span output are `mx.eval`'d so the span's score buffer frees before
the next span's graph is built (load-bearing under laziness). Span outputs are
concatenated. One-shot (chunk disabled) is `_forward_span` over the whole prompt —
byte-for-byte the old code.

### Why it is exact

The whole decoder block is **per-token** except attention:
- HC mixes / `hc_pre` / `hc_post` / final collapse — elementwise per `(b, s)`, no
  cross-token mixing; the identity `pre_mix` seed is the same for every token, and
  `pre_mix` threads across *layers*, not across the sequence.
- MoE — `xf.reshape(-1, dim)` is per-token routing.
- Attention reads the **accumulated** cache. For a span's query at absolute
  position `p`: the window uses a causal mask `(wp ≤ qp) & (wp > qp − W)` over
  absolute positions, so it sees exactly the ≤ `p` window rows (earlier spans
  included); the compressed reach `arange(n_comp) < (p+1)//ratio` selects exactly
  the groups completed by token `p`, all already appended; the indexer top-k is
  per-query. So each token's attention output is identical to one-shot.
- **Compressor frontier** — W13's `CompressorState.push` accumulates `raw_kv`/
  `raw_score` and pools each group from *its own* `ratio` rows: "the result is
  independent of how the rows were chunked" (deepseek_v41_cache.py). A group
  straddling a span boundary is parked and pooled when its last row arrives —
  identical latent.
- **Engram** — `NgramHashState.advance` concatenates into the full history buffer
  and hashes positions `[start, start+L)` with n-gram lookback into prior spans;
  per-span advance in order == one-shot advance. Each engram layer in a span reads
  `_current` (that span's row ids).

So the cache needs **no** change (it already supports incremental multi-token
appends and the compressor frontier), and `deepseek_v41_moe.py` needs no change
(backbone chunking hands it a `[chunk, hidden]` input directly).

### Latent bug fixed alongside

A span (or prompt) shorter than one compress group leaves `shared.compress_kv is
None`, and `_compressed` crashed on `compress_kv.shape[1]`. This would also crash a
one-shot prompt below `ratio` tokens. Fixed exactly: when there are no compressed
rows, every query's reach is 0, so the compressed branch is skipped (window-only),
matching one-shot's all-masked result. Guarded in `_compressed` (returns `None`)
and `Attention.__call__` (skips the concat).

## 3. Exactness proof (CPU, tiny synthetic config)

`tests/models/test_deepseek_v41_chunked_prefill.py`, an 8-layer config exercising
every CSA2 mode (swa, full-r2, reuse, full-r1 candidate, reindex) with window 8 and
ratio-2/1 groups. Chunk sizes `{1,2,3,5,7,8,9,13,24,25}` straddle the window (8) and
the ratio-2 groups (odd chunks split a 2-token group); chunk 1 exercises the
None-skip.

- **logits**: `max|chunked − one_shot| ≤ 1e-5` at every chunk (measured worst
  ~2e-6 at the test's scale-0.1 fixture; ~1e-5 at the parity suite's scale-0.3).
- **cache state**: `window`, `compress_kv`, `index_k`, `comp_state.raw_kv/raw_score`
  and `offset` all equal one-shot (≤ 1e-5; same key set).
- also proven for a pure SWA-only model (no CSA).

The residual is fp32 matmul-tiling noise — the `[b,s,H,T]·[b,T,d]` output einsum in
`_sparse_attend` accumulates over `T` in a different tile order for different
query-batch sizes (classic batch-vs-sequential non-associativity). It scales with
logit magnitude (rel ~1e-6 … 2.5e-5), propagates into the cache from attention, and
is **100× tighter than the port's own 1e-3 attention parity bar**. It is not a logic
difference: the algorithm is exact.

## 4. Transient bound at 16K (arithmetic + synthetic switch)

`test_real_geometry_transients_bounded_by_chunk` at the released 40-layer shapes
(no weights). Auto-derived chunk for `s=16384` at an 8 GB budget = **1271**:

| transient | formula @ chunk 1271 | bytes |
|---|---|---|
| attention score | `1271 * 64 * 24576 * 4` | 7.996 GB < 8 GB |
| indexer score | `1271 * 32 * 8192 * 4` | 1.33 GB |
| MoE routed wave | `1271 * 6 * max(2304,5120) * 4` | 156 MB |

The test also (a) allocates + `mx.eval`s the largest routed tensor for each chunk in
the real chunk grid via a synthetic recording switch and asserts the recorded max
equals the arithmetic and stays < 8 GB, and (b) proves via a recording switch on a
tiny model that the *forward* feeds the MoE ≤ `chunk` rows per call (one-shot feeds
`s`). The unchunked score is asserted to equal `103,089,701,120` as a regression
anchor.

**Is any 16K score materialised (head_dim 512 blocks flash-SDPA)?** Yes — two:
the attention score (the 103 GB tensor) and the indexer score (17.2 GB one-shot).
Both are unbounded in the query dimension `s`; query-chunking bounds both to
`chunk` rows. This is the V4 finding confirmed for V4.1.

## 5. Chunk-size policy

Precedence (in `_resolve_prefill_chunk`): explicit `prefill_chunk=` argument >
`MTPLX_DSV41_PREFILL_CHUNK` env > shape-aware auto-derivation.

- **Auto** (default): the largest chunk whose dominant transient (`H*T*4` per row)
  stays under `MTPLX_DSV41_PREFILL_CHUNK_TARGET_GB` (default 8.0). Shrinks as the
  context grows: 16K→1271, 64K→317, 128K→158 — so any context stays bounded.
- **Inert on the working path**: 1,024-cell → chunk ≥ s → one-shot (unchanged: the
  1,024 cell already ran at 37 tok/s); decode (s=1) → one-shot.
- `MTPLX_DSV41_PREFILL_CHUNK=<int>` forces a chunk; `0`/negative disables chunking;
  `auto` = derived.

The fix is transparent to callers (`model(ids, cache=cache)` unchanged) and composes
with the serve path's own prefill chunking (`generation.py` `_iter_prefill_chunks`,
default 2048): a 2048 serve-chunk is sub-chunked to ≤1271 internally, so even the
served path — whether it chunks at 2048 or feeds one-shot — is bounded. Applies to
serve, bench and gate through the one model entry point.

## 6. Recommended bench flags for the 16K cell

`scripts/deepseek_v41/bench_standard_shape.py` calls `model(ids, cache)` one-shot
(line 441) and reads only `logits[0,-1]`. With this fix it **runs unchanged** —
auto-chunking is on by default. Recommended:

- Nothing required — the default auto-derivation bounds the score to < 8 GB.
- For headroom + reproducible receipts, set **`MTPLX_DSV41_PREFILL_CHUNK=1024`**
  (attention score 6.44 GB; matches the proven-good 1,024 cell size) and record the
  value in the receipt. Or lower the budget with
  `MTPLX_DSV41_PREFILL_CHUNK_TARGET_GB=6` (→ chunk 953).
- Keep `--memory-limit-gib 100`. Budget note: the estimated 16K peak is the resident
  model + KV/compress caches (~1 GB at 16K) + one span's ≤ 8 GB attention transient
  (freed between spans) + the 8.47 GB head-logits buffer (§7), which do not all
  coexist — comfortably under 100 GB with chunk 1024.

## 7. Not fixed here (out of scope, documented)

- **Head logits `[1, 16385, 129280]` = 8.47 GB.** `Model.head` projects the full
  hidden; the bench uses only the last row. This buffer is under the 86.5 GiB cap
  and is identical in one-shot and chunked, so it is not the crash and not in the
  W20 mandate. If future memory pressure needs it, a last-token slice
  (`num_logits_to_keep`) before the head is a one-line change — left out to avoid
  changing `Model.__call__`'s return semantics for the gate/hidden-state consumers.
- **`expert_mlx.py`** — untouched. The 103 GB is the attention score, not a bug in
  the streamed switch; its per-wave gather is already row-bounded.
- **`deepseek_v41_moe.py`** — untouched. Backbone chunking bounds its input, so no
  chunk plumbing is needed inside the MoE.
- **`deepseek_v41_cache.py`** — untouched. `CompressorState.push` is already
  chunk-independent and the stores are append-only, so incremental appends are
  exact as-is (no PORT_CONTRACT change needed).

## 8. Coordination with W22

W22 is reshaping the cache container to the mlx_lm per-layer protocol on
`feat/deepseek-v41-w22` (currently at 5d6dd8ad, not yet diverged). My chunking rides
the frozen cache API (`cache.offset`, `cache.advance(n)`, `cache.new_shared_runtime()`,
`cache.layers[i]`, `cache.engram_state`) and `_eval_cache_state` reaches stored
arrays through `getattr(..., None)` so a renamed field is skipped, not a crash. When
W22 lands: merge its branch, then re-run
`tests/models/test_deepseek_v41_chunked_prefill.py` (the exactness + bound suite is
the merge gate).

## 9. Files changed

- `mtplx/models/deepseek_v41.py` — chunk-resolution helpers
  (`_resolve_prefill_chunk` / `_derive_prefill_chunk` /
  `_prefill_score_bytes_per_row`), `Backbone.__call__` chunked driver +
  `_forward_span` + `_eval_cache_state`, `prefill_chunk` threaded through
  `Model.__call__`, and the `compress_kv is None` guard in `Attention`.
- `tests/models/test_deepseek_v41_chunked_prefill.py` — 9 tests (arithmetic anchor,
  auto-derive/precedence, exactness CSA + SWA, MoE-row plumbing, real-geometry bound).
- `docs/deepseek-v41/W20_REPORT.md` — this report.

## 10. Verification (CPU, `nice -n 19`, `mx.set_default_device(mx.cpu)`)

- `test_deepseek_v41_chunked_prefill.py` — 9 passed.
- `test_deepseek_v41_parity.py` / `_cache.py` / `_moe.py` / `_config.py` — no
  regressions (26 passed / 6 skipped).
- Peak RSS of the chunked test file: **389 MB** (`/usr/bin/time -l`: 407,601,152 B).
- No GPU: no real-artifact load; the 16K/103 GB numbers are arithmetic + synthetic.
  A real GPU-window bench of the 16K cell is the remaining confirmation, out of this
  CPU task's scope.
