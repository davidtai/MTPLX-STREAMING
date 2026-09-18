# Prefix readiness diagnostic on the retained native route trace

No MLX execution or model read is used. The existing native policy replay
reproduces every retained per-layer/per-cycle miss count:35,164 demand misses
over8,240 layer calls, starting from73 captured residents plus29 empty slots.
These are policy misses, not the physical reads of the latest grown cache.

| Prefix rows | Prefix needs no reads | Share of all demand misses |
| --- | ---: | ---: |
| 1 | 50.59% | 17.21% |
| 2 | 29.70% | 33.29% |
| 3 | 18.25% | 49.50% |
| 6 | 5.23% | 100.00% |

This identifies potential work that could reach the next layer before all
current-layer reads finish. It does not establish an asynchronous decoder or
a latency gain. Row-by-row execution expands199,773 unique batched expert
requests into296,640 row requests, a48.49% increase before kernel and dispatch
costs. Actual read ordering, shared transient ownership and native cache
semantics are not simulated.

The next useful gate is a bounded measurement of native block cost and state
under partial-row execution. Establish the real attention, compressed/shared
KV, Hyper-Connection, expert reduction and slot-lease ownership contracts
before implementing inter-layer scheduling. A promising route histogram alone
does not justify a full-model run or a replacement hot path.

The first CPU attempt resolves the environment's editable checkout because an
absolute script path omits the intended worktree from `sys.path`; it refuses
at the first constructor. The corrected script puts the current worktree
first, reproduces all native counts, and completes in0.780CPU seconds.
Both attempts use the replay helper's explicit NoMLX import guard.
Source5ab1776598; SHA-256 provenance is retained. No runtime change or test
suite is added.
