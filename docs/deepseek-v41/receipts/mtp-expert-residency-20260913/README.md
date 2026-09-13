# Native MTP expert residency feasibility

Source: `1f3b9bca7ae5aab6a4bf30eb969b1f6dd0368711`. Companion `census.json` records artifact/config/index/header identities and reviewed source hashes. Only metadata and safetensors headers were read; this census does not authenticate weight payload contents.

All 128 experts in each of three MTP stages are fully resident: **7,219,445,760 B** of **7,949,967,240 B** selected MTP weights (90.81%). Other MTP weights total **730,521,480 B**. Expert-only shards are `model-00047/48/49.safetensors` (768 keys and 2,406,481,920 B each); other MTP tensors occupy `model-00044/45/46.safetensors`. Three unused VL biases add 1,536 B on disk and are excluded from the selected total. See the JSON for exact per-shard key counts and bytes.

## Layout and arithmetic constraints

Each expert is 18,800,640 B, matching target expert geometry: MXFP4 gs32, three packed U32 matrices plus U8 E8M0 scales, no biases. Gate/up weight shapes are `[2304,640]`, scales `[2304,160]`; down weight `[5120,288]`, scales `[5120,72]`.

Preserve the native top-3 over 128 experts, score/routing order, five draft rows, unsorted gather at 15 assignments, asymmetric SwiGLU clamp 10 (up two-sided, gate upper-only), and fp32 routed sum plus shared expert. The resident switch invokes up, gate, activation, then down. Equivalent codecs alone do not prove identical kernels/reductions after reshaping or sorting. Sources: `mtplx/models/deepseek_v41_dspark.py:340,634,752`; `mtplx/models/deepseek_v41_moe.py:230,271,415`; installed `mlx_lm/models/switch_layers.py:75,176` (path/hash in JSON).

## Loader, reader and pricing seams

- `deepseek_v41_loader.py:255,763` selects residents before `load_text_only_resident_arrays` (`:611`). Exclude MTP expert keys there and replace their switches before strict load/evaluation. Avoid `_map_mtp_residents` (`deepseek_v41.py:5052,5089`) stacking all 128 experts. Merely filtering hot experts within a touched shard still eagerly loads the whole shard on macOS; `resident_io.py:93` correctly rejects over 64 MiB discarded payload. Skip expert-only shards in the resident loader and use bounded segment reads for all expert slots.
- Header-derived `TensorSegment`/`ExpertRecord` descriptions can reuse `ExpertReader.read_record_into(prefer_sidecar=False)` (`expert_io.py:1309,1387`) and component-bank storage (`models/expert_mlx.py:1240`). Map checkpoint w1/w3/w2 to gate/up/down; source file ordering need not equal target bank ordering. No artifact replacement is required. MTP residents do not already have the target bank's per-record digests; any integrity proof must explicitly bind these source tensors.
- The target binder (`models/expert_mlx.py:3844`) only covers the backbone's 40 layers, 384 experts/top-6. MTP stages use logical layer IDs 40–42, 128/top-3; the current target runtime/manifest cannot simply be rebound to them.
- Keep `mtp_included=True`. Extend pricing at `expert_runtime.py:1624` and the allocator/runtime plan construction together (`deepseek_v41_loader.py:490,513,523,530`). Discount only omitted MTP expert bytes, then charge actual MTP slot storage and I/O/graph overhead. Keep other MTP weights, SWA and wo_a reserves. Do not subtract the entire MTP reserve.

For scale only, 32 total expert slots per stage retain 1,804,861,440 B and release 5,414,584,320 B gross: seven additional target slots per layer before new overhead. This is not a proposed capacity or speed prediction. Dynamic misses must preserve GPU-consumer and writer completion ownership; sharing transient storage between stages would require its own lifetime proof.

## Separate diagnostic route capture proposal

Use a temporary diagnostic wrapper around only the three native `switch_mlp` callables after loading. Retain the actual `[5,3]` int32 index arrays, forward the original inputs unchanged, and count/serialize after the existing draft/generation synchronization (`deepseek_v41_dspark_decode.py:1127`). Do not call `.tolist()`/`mx.eval()` inside each stage, reroute experts, or add production counters. Existing census wrappers perform immediate host conversion and should not be copied unchanged.

Bound capture to the single 16K/1,023-step/depth-5 pass: at most 1,023 draft calls per stage, 3,069 arrays and 184,140 B of index payload, plus a separately bounded allowance for Python/array objects. Keep no activation references. Preserve ordered per-cycle routes, validate per-stage coverage against actual draft calls and `stats.cycles` (fixed positive depth drafts once per cycle), compare token digests, and analyze cache misses offline. Diagnostic throughput is not a promotion receipt.

**Route locality and streaming speed benefit are unmeasured.** Native drafting currently has no host synchronization per stage. Added SSD reads and routing barriers may outweigh gains from a larger target cache; obtain the bounded trace before selecting a capacity or designing the streaming lane.
