# DSpark prefill lifetime and MTP planning, 2026-09-13

This batch changes memory ownership and planning, not the full-workload TPS
receipt. The best measured 16K/1,024-output-cap result remains 6.535995651 TPS
on source d2582555402e0c727d14d69983f449edc44ab6c3. The 20 TPS goal is open.

The default direct DSpark prefill now requests one logits row. Previously its
16,384 x 129,280 fp32 logits occupied 8,472,494,080 bytes, although generation
used only the final row. Verification still evaluates every candidate row;
custom forward functions keep their existing two-argument API.

Both direct and served entrypoints seed the draft windows, gather the final
hidden row and retained window rows into independent storage, evaluate those
small buffers, and drop the full prompt locals before decoding. A plain slice
retains its parent allocation. MLX 0.32.2's `deepcopy` also uses the array copy
constructor ([upstream binding](https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/array.cpp#L518-L522));
the intermediate `green` receipt records that unsuccessful approach. Final code
uses `mx.take`, with no floating-point arithmetic in the copies.

The bounded memory regression uses under 64 MiB of generated prompt arrays,
no model weights, and all three direct/served/custom-forward routes. Original
retention was 50,733,068 / 33,955,852 / 50,733,068 bytes. Final synchronized
measurements retain 17,920 bytes in each route. Synchronization and garbage
collection are measurement-only; neither was added to the production decode
loop. One existing small-model greedy test still exercises acceptance and
rejection with exact output equality. Final guarded result: four checks passed.
Two pure-Python completion-lifetime cases also passed with MLX imports blocked.

Standalone loading now resolves the MTP choice before allocation and sets the
same spec and config for planning and model construction. The manifest-based
resident discount therefore keeps MTP weights charged. All loaded stages also
reserve fp32 sliding windows and, when enabled, fp32 wo_a caches. The canonical
serving route uses the same additional-resident helper. Separate construction
refuses an MTP head against an AR-only runtime plan before model construction.
The legacy benchmark reprice flags no longer subtract an approximate 7.4 GiB;
the loader owns residency within the original envelope. Five pure-Python
selection/admission checks passed after observed failures; five guarded existing
benchmark/default-reserve compatibility checks passed.

Static accounting reads the real artifact's config and manifest with MLX imports
blocked and the allocation backend replaced by a plan recorder. At the previous
86,369,798,112-byte engine budget, 17,664-token KV capacity and shared 48-record
transient pool, AR has 83 slots/layer and MTP 72. MTP adds 8,353,408,392 fixed
bytes: 7,949,968,776 manifest bytes plus 403,439,616 stage-cache bytes. The
manifest figure conservatively includes 1,536 bytes filtered out at loading.
Component-bank and runtime plans compare equal. This is **not** a bound on MTP
draft/verify temporary peak; establish that before a full-model MTP run.

Every GPU window restored the exact Qwen model and warmup, then released the
exclusive lock. Swapouts remained 4,399,765 pages. The first expected-failure
window also hit `live step state unreadable` during child exit and returned 8;
later windows preserved the child status. That exit-observation race needs a
bounded reproduction before changing the guard. There was no OOM in this batch.
