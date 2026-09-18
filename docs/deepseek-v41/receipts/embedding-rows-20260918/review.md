# Inline review of the optimization checkpoint

Reviewed source 575c3c8b3beb0420d16fc03c727f3a27c0f36edd through the staged
receipt/helper changes. The context snapshot is stale and was not used for
call-site coverage. Review was performed inline, respecting the no-agent scope.

No material issues found for this explicitly installed, single-request lane.
Critical and Important findings: none remaining. The initial Path argument bug
is fixed in full-v2; full-v1 remains archived as failed evidence.

Reviewed file_embedding.py: fixed host capacity, duplicate ordering, LRU slot
reuse, immutable returned output ownership, short-read propagation and close.
Reviewed full-v2/packed/embedding_install.py: cleanup registered before owner
replacement, prefill fence, actual source release, source-page reclamation,
completion-only reporting and measured-transition timing.
Reviewed full-v2/packed/packed_admission.py: extra host bytes reach the real CLI
resolver; no prefill credit; source retirement precedes all credited phases;
existing copy/seed/compiler/KV and wired reserves remain. The full receipt and
native outputs agree with this construction.
Reviewed the CPU clock installer and accumulators: fixed records, preserved
native tensor operations, no added fences, explicit diagnostic timing fields.

Coverage is intentionally narrow: one exact full candidate after the Path fix,
one bounded operator, and two post-win host resource regressions. Cold operator
rows do not prove cold OS pages. The successful full candidate reclaims source
pages explicitly. Background and capacity differ from the retained performance
control, so the result is not an isolated or repeatability claim. General
concurrent serving, repeated prefill, alternate artifacts and complete 256K
prefill are outside this construction. No API/security boundary changed.

Verdict: ready for the local optimization checkpoint. This is not a general
serving promotion, and the 20 TPS goal remains open.
