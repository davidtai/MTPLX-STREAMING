# Packed cache transition cost attribution

A single real layer-20 bank at the actual 84-to-102 geometry spends roughly
equal time loading/checking packed scales and growing its weight arrays.

| Existing transition operation | Seconds |
|---|---:|
| Release raw scale backings | 0.000064 |
| Load and hash packed scales | 0.039524 |
| Grow three weight components | 0.039921 |
| Final synchronization/cache clear | 0.000021 |
| Whole transition | 0.079530 |

Forty times the measured copy cost is about 1.60 seconds, not the entire
3.426-second transition charged in the retained full run. This scaling is an
estimate, not a measured full-model breakdown. The saved steady decode bound
already exceeds the resize bound by 514,906,000 bytes. Removing the copy peak
alone therefore does not admit more slots. An extension-bank design would need
to preserve decode cost before its small potential saving warrants promotion.
No new storage layout was implemented.

Old native weight rows 0, 41 and 83 retain exact hashes after growth. The
operator releases 92,897,280 raw-scale bytes and adds 318,504,960 weight bytes.
The 8 GiB bound covers host/compiler and allocator/cache/graph allocations.
Allocator peak is 2,483,786,252 bytes; cleanup leaves eight active bytes.
The guard samples process footprint at 327,664,480 bytes and whole-machine
physical use at 11,368,333,312 bytes. Those one-second samples miss shorter
allocation peaks; they do not replace the allocator measurement or static bound.

Guard 76769 is terminal exit 0. Exact Qwen identity, health and warmup returned
before lock release at 02:26:43 UTC on September 18. Independent verification
at 02:32:16 UTC found healthy/idle/warmed Qwen, a free lock and no owned child.
Source: `cc438324696301ce3b502227bcce9777781786df`.
`sha256.json` binds the diagnostic, source, byte checks and lifecycle.
