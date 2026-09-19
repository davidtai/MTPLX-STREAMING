# Fund another Q4 expert-cache row with the screened smaller draft

The exact extension-bank result is13.8688TPS at111 target rows/layer. It still
reads558.68GB during decode. A previously completed native Q4 draft screen
uses80/40/24 experts, preserves198 teacher-scored target calls and1242 verify
rows, and has a real2,707,292,160-byte subset artifact. It removes733,224,960B
from the existing draft. Its prior full attempt refused on admission before
model load; it has no full result to reuse.

Compose that exact fixed alias map and physical subset with the measured
extension layout. Do not quantize or change target weights, arithmetic, KV16,
D5 plus causal lookup, or M<=8 verification. Keep84 original expert rows and
add at most29 rows/layer. Existing bank indices remain0..83; added indices
0..28 fit the same packed kernels and signed32-bit component offsets.

Subtract only proved draft payload from transition, native seed and steady
allocation bounds; retain the larger original prefill bound. All copy-free
extension, raw-scale temporary, three-buffer projection, KV, allocator cache,
compile and page-padding allowances remain. Source authentication reads the
existing artifact through F_NOCACHE in a16MiB CPU buffer envelope inside the
parent-held guard before MLX import. No model is downloaded or rewritten.

The previous machine peak exceeded its launch estimate by107,409,300B. Add an
explicit256MiB background-variation reserve inside the unchanged110GB ceiling.
The CLI's aggregate non-MLX allowance therefore includes both the original
1,438,773,248B host reserve and268,435,456B background reserve, itemized in the
receipt. Keep100GiB wired, strict256MiB decode cache and bounded Engram arenas.

Require fresh admission for at least112 final rows before loading the model;
otherwise refuse the intended comparison and restore Qwen. Run one full exact
16K/1K request only after CPU composition and budget checks. The prior head
screen is not a full-output guarantee: require fresh output IDs, tie/drift
classification, decoded-output sanity, wall/TPS and separate memory evidence.
No new regression suite follows unless this composition produces a real win.
