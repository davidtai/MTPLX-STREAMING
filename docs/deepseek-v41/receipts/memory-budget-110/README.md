# 110 GB memory calibration

These receipts use decimal GB for the machine budget. They are memory and
correctness calibration, not evidence of the 20 TPS target.

## Findings measured on 2026-09-13

The machine is an Apple M5 Max with 128 GiB physical RAM and an unchanged
100 GiB wired limit. The unloaded baseline initially included about 38 GB of
file-backed pages. That cache belongs in physical memory reporting even though
the OS can reclaim it.

The OS-only purge probe made no change: unprivileged `purge` was refused, and
`sudo -n /usr/sbin/purge` required a password. A successful `sudo -l` permission
query did not establish password-free execution. The guarded attempt restored
the exact service with warmup complete. The memory budget continues to include
the measured file cache.

The benchmark omitted the expert profile's `bypass_page_cache` setting. Engram
row reads also used buffered descriptors, duplicating the bounded Python LRU
in the macOS file cache. Both now install `F_NOCACHE` at construction on macOS.
The resolved expert and Engram modes are included in new receipts.

| Measurement | Result |
| --- | ---: |
| Random Engram payload read per probe | 4,325,376 bytes |
| Buffered probe file-cache growth | 273,039,360 bytes |
| Uncached probe file-cache growth | 0 bytes |
| Corrected 16K calibration sampled physical peak | 99,188,310,016 bytes |
| Corrected calibration sampled process peak | 42,363,028,160 bytes |
| Corrected calibration prefill | 134.350 s |
| Corrected calibration decode | 3.047 TPS over 32 decode steps |

The row probes use different seeded rows, so their timings are not a controlled
speed comparison. The 250 ms OS sampler observed 564 samples during the full
calibration. Swapout counts did not increase. Sampling does not establish an
instantaneous peak bound.

The calibration used the historical window-54 16K prompt and
`cell16k_ring_v2_attn_pf0`, I/O fanout 4, a 100 GB allocation target, a 16 GiB
transient band, and a 2 GiB freed-allocator cache. Its 33 generated token IDs
exactly match the first 33 IDs from window 54. Its smaller expert cache makes
the 3.047 TPS result unsuitable as a regression comparison against that run's
larger, incompletely accounted memory plan.

Before these cache fixes, the guard aborted the same calibration when sampled
physical use reached about 110.8 GB despite a roughly 25.4 GB process footprint.
The guard restored the previous Qwen service and released its lock. No completed
generation receipt exists for that aborted attempt.

`load-phases.jsonl` records another remaining source of duplication: path-based
resident loading grew file-backed pages by about 9.737 GB. The load-only physical
sampled peak was 77,415,890,944 bytes (77,388,152,832 at the load-end boundary).
These files describe the earlier loader.

The `resident-uncached-load-*` files measure the corrected resident loader with
the same reduced allocation settings. Its sampled physical peak is
**68,360,978,432 bytes**. Used file-cache growth through load is 113,655,808 bytes,
versus 9,600,466,944 in the earlier load. File-backed pages themselves still grow:
the new pages are predominantly speculative, which the OS counts as free. Do
not use `file_backed_bytes` alone as physical used memory. The process sampled
peak is 21,129,175,200 bytes; the Engram sidecar is now materialized during load,
instead of waiting for the first forward. Swapouts did not increase. The exact
previous service returned healthy with background warmup done and the lock
released. `resident-load-comparison.json` contains the calculations.

## Receipt provenance and corrections

`calibration-fixed.jsonl` and its OS trace and guard log are unchanged raw
observations from base commit `a0d36575b` plus the three cache changes listed in
`calibration-summary.json`. Two reporting fixes landed after that process had
imported the runner:

- The transient pool is global: 48 slots × 18,800,640 bytes = **902,430,720
  bytes**. The raw receipt incorrectly multiplies it by 40 layers.
- The new typed `resident_load_report` field is absent. The guard log's
  `STORAGE_POLICY` line records expert and both Engram readers as `f-nocache`.

`calibration-summary.json` preserves these errata explicitly. Do not silently
rewrite the raw receipt or use its transient total for memory accounting.

## Python workload for the performance target

The historical prompt ends with a markdown-report instruction that says
"No code." The new pinned fixtures retain the repository's naturalistic Python
patch request, including its helper and pytest requirements, and omit that
contradictory suffix. They use the artifact's tokenizer and exact chat template
with thinking disabled. No generation has yet been measured on these fixtures.

| Input tokens | SHA-256 of JSON token IDs |
| --- | --- |
| 1,024 | `f915ac7f2373ef94fbc86d944da740e0772453f39d6efba0152bd47dfa43905d` |
| 16,384 | `38894d01011a0146f8621dfdaf4bc4e618092d772d8cc28a70e21163a16799a2` |

`python-prompt-ids.json` contains both inputs, the instruction's 1,024-token
output cap, tokenizer/template hashes, and context provenance. The plain and
rendered text files make the task reviewable. From the repository root,
`PYTHONPATH=. .venv/bin/python
docs/deepseek-v41/receipts/memory-budget-110/build_coding_prompts.py` regenerates
them using the already installed local artifact (use the shared repository venv
when running from a worktree). This generator does not import MLX.

Full-model measurements still require the exclusive GPU guard, a bounded load
and compile peak, service restoration, and an explicit prompt IDs file. Report
actual tokens through EOS and distinguish decode steps from the first token
produced by prefill.
