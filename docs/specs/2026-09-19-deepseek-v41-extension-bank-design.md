# Avoid the initial expert-bank copy

The retained complete packed-projection run spends3.6737s in bank growth. The
one-row extension avoids a second copy, but its first84→109 resize still
allocates and copies replacement backing arrays. Existing packed kernels
already group physical expert rows by backing bank; no new kernel is needed.

Measure an equal110-slot A/B/A component. Both arms start with84 packed rows,
the same saved cache state, all40 native packed output projections and the
same predictable BF16 expansion schedule. Control grows84→109, then appends
one separate row. Candidate keeps84 unchanged rows and adds one26-row bank.
Replay206 saved M6 routes for layer34 with synthetic projection inputs and
the covering router dependency. Attention is omitted. Charge allocation plus
the warmed replay; also retain complete replay timing. Compare every output.

The candidate can lose: frequent hits in both persistent banks create extra
gather groups and dispatches. A measured win beyond control variation is
required before full integration. Do not infer its result from the one-row
extension, or promote a memory/copy reduction as full throughput evidence.

The component retains the26GiB incremental bound:10GiB Metal/cache/compile,
4GiB host and12GiB source cache. Its explicit Metal payload/copy/scratch/cache/
compiler inventory is10,104,602,624B. Use the canonical parent-held GPU guard,
110 decimalGB physical ceiling,100GiB wired ceiling, and exact Qwen lifecycle.

Only installation changes bank ownership and cache capacity. All original row
objects and indices survive; the candidate also preserves original backing
arrays. New owners enter allocator cleanup before any subsequent operation can
fail. Runtime, policy, physical slots and memory plans update at the quiescent
boundary. The measured expert/projection path has no new branch or counter.
