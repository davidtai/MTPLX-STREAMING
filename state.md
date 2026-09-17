# Current Goal

DeepSeek V4.1: correct memory reporting and runner bugs, then reach 20 decode
TPS on the exact 16,384-input / 1,024-output Python workload under 110 decimal GB.
**Best historical measured: 11.7203483 TPS. Latest: 11.6513006 TPS at cap93.
The 20 TPS goal remains open.**

# Decisions

- Separate allocator, process phys_footprint and whole-machine physical memory.
  Never add process usage to machine usage or substitute RSS. Missing peaks are
  null, not zero. Headline and detailed fields share the same observation.
- Reserve 2 GiB for Python/host caches and metadata. Current benchmark cache
  request is 1 GiB inside Metal. Its retention policy does not hard-bound
  instantaneous cached allocations; the prefill profile observed about 2.7 GB.
- Hold /tmp/mtplx-gpu-exclusive.lock before MLX import or stopping Qwen. Use
  scripts/deepseek_v41/gpu_window.sh directly. Never nest guards or disturb
  another GPU job. Automatically reclaim stopped Qwen and candidate file caches,
  restore the exact service, verify health/warmup, then release the lock.
- User permits tie breakers. Preserve full output digests and use index-matched
  candidate logits for tie classification. Do not accept arbitrary drift.
- Keep unvalidated bounded KV off. Preserve Claude's W126/W127/W128 worktrees.
  Minimal testing; add optimization tests only after a measured win. No agents.

# Current Changes and Evidence

Plan: docs/plans/2026-09-16-deepseek-v41-20tps-stage.md, Task4 remains open.
Prior promoted capture lifetime fix:9c4fac40f. Reporting fixes:c031119a1 records
requested/effective depth; b95f8d1a7 preserves missing peaks; ca207c3d7 binds the
three model flags before construction and records actual bound booleans.
Twenty-one focused no-MLX reporting checks passed before this continuation.

The new prefill change compiles only _hc_post_impl for the layer-major post-MoE
combine through the prebound _PREFILL_HC_POST. It preserves routing, captured
states, evaluation fences and decode, including the original diagnostic timing
hook. With that stateless inactive hook normalized, the method AST matches the
measured installation. The existing strict layer-major versus
chunk-major regression passed after the change; no new test module was added.

Receipts: docs/deepseek-v41/receipts/hc-post-prefill-20260917/README.md.
Measured source:ca207c3d7 plus the archived candidate installation. Both new
runs explicitly bound HC compile, attention compile and window memo false.

- Prefill-only cap93:140.618836208s; allocator peak95,208,123,264B.
  Historical cap94 control peak97,146,256,508B minus one752,025,600B slot band
  gives a capacity-normalized difference of1,186,107,644B. Historical actual
  globals were not recorded; window memo can affect prefill, so this is not a
  fresh identical-flags control or an exact isolated causal estimate.
- Full cap93,D5/M6:11.6513006265TPS/87.8013565s,206cycles,1024output tokens.
  Allocator peak95,208,121,956B. Separately measured MTP seed+decode peak
  88,755,252,592B. One counter reset before the decode timer isolates the phase;
  all full-run headline/detail values retain max(prefill,post-prefill).
- Full external sampled process95,512,926,728B/system105,458,794,496B.
  Internal process95,471,050,472B/system104,870,838,272B. Baseline9,586,573,312B;
  physical admission bound108,512,945,408B. No prefill saving was spent yet.
- Full MTP digest remains
  0d54d9b28a180c2c91ff5ef14f0dfb38320014bbed9d01827fb1b60c6e0417ac.
  AR reference remains
  2bd0ad017b9580c8fec340e297696a0bd81a7759b6c5dfe7c5d64de6d40c1090.
  Existing tie_flip297 has matching capture index, consistent rows and zero AR
  contested margin. Reference timing/memory is null; cached logits are hashed.
- Native-shaped post-only micro was bit-exact and reduced temporary+output
  storage83,886,080->41,943,040B. Whole-HC-chain compile is a different experiment
  with native numerical differences; it still needs a full output/tie gate.

# Lifecycle

All owned GPU jobs are terminal. Full-output session89807 returned0 and Qwen
was healthy/warm with its exact ID before lock release13:31:16 UTC. Focused
regression session56370 returned0 and restored/released13:39:41. Live API checks
confirmed restoration. Another benchd acquired the lane afterward; never signal
it. Last observed swap3165.44MiB, no increase in these windows.

# Next Work and Constraints

- Cache growth after prefill is now supported by a measured decode peak, but
  has NOT been implemented or admitted. Cap93->102 adds6,768,230,400B; steady
  MLX peak would project95,523,482,992B before new margins. Whole-machine and
  resize-copy headroom, caches and ownership still need a conservative bound.
- Preserve one physical component bank per layer. A separate tail bank breaks
  _run_component_bank_q4 and device-route LUT invariants. Study component-wise
  exact packed-byte copies within the same bank identity after draining all
  readers, pins and Metal consumers. Release every old exported memoryview.
- Keep allocator closure plan, physical slot maps, logical capacities, shifted
  transient indices, runtime reporting and device LUTs coherent. Updating the
  allocator's exposed plan alone leaves its captured plan stale. Subsequent
  requests must shrink physical backing before prefill; logical shrink is not
  enough. Partial resize failure must stop generation, with cleanup.
- Historical staged wrappers imported dsv41 before presets. arm_env alone does
  not prove active compile globals. New full performance arms need explicit
  actual flag provenance and an unchanged control. Never fabricate a reference
  diagnostic by replaying it through changed candidate arithmetic.
- Best prior cap94,D5 full run:11.7203483TPS,38,613reads,725.95GB,55.64s active
  I/O. D3 loses on the complete workload despite winning a short prefix.
  Twenty TPS requires<=51.15s decode, so I/O and verification both need work.
- Reject staged3+3, per-chunk combine fences, retirement-only tiny speed changes,
  serial rANS decoding, tuned full-M6 admission, and direct target-head rounding
  changes. Decode-only larger-cache replay is a projection, not GPU evidence.
- /tmp/dsv41-110-stage/CONTINUATION.md preserves detailed receipts, rejected
  candidates and ownership inspection. Old source-pinned wrappers become stale
  after promotion; regenerate their installation proof before another run.
