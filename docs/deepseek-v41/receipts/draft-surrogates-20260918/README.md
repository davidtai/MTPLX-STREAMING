# Bounded draft expert aliases and admission refusal

No new full-model throughput result or default is promoted. The retained best
single result remains13.4141517619TPS;20TPS is unmet.

A144-expert draft subset preserves198 target calls in saved-state acceptance
replay and removes733,224,960B of serialized expert payload. Its full candidate
is refused before model loading because111 target slots would exceed110GB at
the live background baseline. A subsequent globally selected124-expert screen
needs200 target calls; it does not justify another full run. No regression tests
are added for these unpromoted candidates.

Measured source:`19ea3ac2f888edf5035e3a43bc314bea64f3cb6f`.

## Selection and acceptance screen

Selection uses only the first103 of206 previously captured native draft cycles.
The native route trace SHA-256 is
`6a90006c9c4829dac2a548298c1d237bfd1beb793c0bc6eaf99315fad8c9c3dd`.
Within the first screen's fixed stage capacities, experts rank by assignment
frequency, then ascending expert ID. Removed experts map to the retained
expert with the closest normalized router-weight vector; exact cosine ties use
ascending expert ID. Retained experts map to themselves. The maps are built
once, before replay. Native target verification and commit remain authoritative.

The screen loads the original93/58/32 physical bank and changes only its lookup
maps, so it measures draft quality without claiming physical retirement or TPS.
It uses the exact saved target hidden states and causal lookup extension from
the retained hybrid result. Future target IDs score proposals; they do not
choose the aliases or supply the causal lookup index. The positional split
between training and the second half is507 committed tokens.

| Head-only arm | Experts per stage | Target calls | Verify rows | Projected payload removal |
| --- | --- | ---: | ---: | ---: |
| Hybrid control | 93/58/32 | 198 | 1,242 | 0 B |
| Smaller cut | 80/40/24 | 198 | 1,242 | 733,224,960 B |
| Larger cut | 64/28/12 | 206 | 1,294 | 1,485,250,560 B |
| Global frequency124 | 52/46/26 | 200 | 1,254 | 1,109,237,760 B |

The fresh hybrid control reproduces all198 pinned commit boundaries. The
smaller cut has the same number of calls but different boundaries and proposals;
this is not a target parity result. Each candidate continues along its own
teacher-scored boundaries. The global124 arm is a separate single-arm follow-up
and explicitly reuses the completed same-source hybrid control. Its additional
verification work does not establish a useful speedup, so no physical124-expert
artifact or full run is made. Earlier cap95 pruning selected experts by cycle
presence and used slot-zero aliases; this screen evaluates a different draft
mapping and selection rule.

The49GiB incremental allocation proof includes41GiB active,4GiB cache and4GiB
host. Router similarity uses at most32MiB inside the host reserve. Both screens
peak at21,299,586,448 MLX bytes. Guard6139 samples17,760,330,416B peak process
footprint,27,621,670,912B physical machine usage and28,612,321,968B guard-accounted
usage across11 samples. Guard27488 samples14,995,279,736B process footprint and
26,463,338,496B physical/guard-accounted usage across7 samples. Different peak
measures are retained separately; neither screen grows the compressor.

## Physical subset and full-model bound

`make_subset.py` creates actual80/40/24 safetensors using bounded uncached CPU
I/O. All three original payload digests are verified during the copy. The
result contains864 tensors and2,707,292,160B payload, exactly733,224,960B below
the existing compact bank. It does not load the larger shards and then pretend
that a tensor allowlist released their memory. The source files are unchanged.
`draft-artifact-receipt.json` records output file/payload hashes and provenance;
the2.71GB payload itself is excluded from this archive.

The staged full runner installs the smaller physical expert axes and the exact
screened aliases. Target arithmetic,48 shared transients, native KV16, M<=8
verification, and native acceptance/commit stay unchanged. One additional
packed target slot in each of40 layers costs707,788,800B; the construction also
prices the larger resize copy. The previous prefill upper bound receives no
draft credit. Only resize/seed/steady subtract the known smaller resident payload.
No per-token validation, extra synchronization or proof counters are added.

The MLX-blocked preflight passes source hashes, syntax and CLI budget agreement.
At the reference10,240,868,352B baseline it admits111 slots, with97,480,665,308B
active bound,98,337,135,616B allocator limit and109,411,965,148B physical bound.
Host reserve remains1,421,996,032B; the strict256MiB decode cache, KV, compile,
page-padding, copy and wired allowances remain charged.

Guard85153 instead observes an11,137,220,608B baseline. The same111-slot envelope
would total110,308,317,404B, exceeding110,000,000,000B by308,317,404B. Admission
selects fewer slots and the candidate refuses model loading because the intended
additional band does not fit. No full model, prefill or decode executes; there
is no fresh output digest or TPS. The refusal is not an OOM and is not a reason
to lower the memory allowance or terminate unrelated jobs.

The candidate remains conditional, not an automatic next full run. Any later
use requires fresh source pins, the live physical/wired admission check and
one complete native output/throughput gate. The global124 variant is not promoted.

## Lifecycle and retained failed attempt

The initial head screen fails before MLX import on a duplicated `/private`
path. `v2/stage.py` fixes replacement order and validates both generated paths
with AST parsing before launch. The original helpers and failure are retained.

| Guard | Result | Lock released UTC | Independent health/idle/warm/free check UTC |
| --- | --- | --- | --- |
| 11847 | Head path error, exit1 before MLX | 11:28:36 | 11:28:55 |
| 6139 | Completed three-arm head screen, exit0 | 11:29:44 | 11:30:17 |
| 85153 | Full admission refusal, exit1 before load | 11:36:22 | 11:37:24 |
| 27488 | Completed global124 head screen, exit0 | 11:42:08 | 11:43:05 |

All dates are2026-09-18. Every guard owns the GPU lock before stopping idle Qwen,
reclaims the stopped service's clean model pages, restores the exact service
identity and background warmup, and releases the lock. Head subprocess source
caches end at0 before restoration. The final independent check confirms healthy,
idle, warmed Qwen and a free lock. No owned GPU child remains.

Scripts and commands retain historical absolute paths and source pins. Do not
rerun them against a later HEAD without auditing and refreshing those pins.
