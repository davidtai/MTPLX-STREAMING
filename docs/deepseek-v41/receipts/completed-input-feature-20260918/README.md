# Completed-source feature screen

The completed preceding layer's mean hidden state improves top-six route
recall by only 1.0489 percentage points after RMS normalization. This does
not justify a full-model partial-expert capture at this stage.

At six issued experts and 27,648 reference assignments:

| Feature | Matches | Recall |
| --- | ---: | ---: |
| Native gate self-alignment | 27,631 | 99.9385% |
| Preceding post-attention router input | 20,051 | 72.5224% |
| Completed-source mean, raw | 20,288 | 73.3796% |
| Completed-source mean, target RMS norm | 20,341 | 73.5713% |

The CPU-only screen uses the last 128 of 256 historical W35 AR rows, target
layers 4 through 39. This is a different 16K prompt from the acceptance
workload. Saved files contain mean hidden and router input, not the full
four-channel Hyper-Connection state. A completed-source feature includes all
preceding expert output and has no early-read lead time. It is an optimistic
feature-quality check, not a mathematical bound on partial-HC prediction.
No miss coverage, read issuance, target execution or TPS is measured.

Execution takes 0.518840458 s, with MLX imports blocked and two BLAS threads.
Only selected rows and gate tensors are read through uncached bounded ranges.
The complete static incremental allowance is 512 MiB. The endpoint process
phys_footprint is 95,355,432 B and machine physical usage is 12,056,854,528 B.
The guard's one sample sees only 13,844,960 B process footprint and
11,978,702,848 B machine usage; it misses the short screen's peak. Its zero
compressor-growth observation is also sampled. None of these values is an
exact continuous peak.

The first launcher exits 2 before lock acquisition or service change because
the guard's child-cap minimum is 1 GiB. The corrected launcher uses that floor
while retaining the tighter 512 MiB construction bound. Both the refusal and
its independent health check are retained.

Measured source: `61b6899f27fee89f63bf7eed9f1c8aa77e73a53a`. Corrected guard 47710
exits 0 and restores exact Qwen identity/warmup before lock release at
12:06:04 UTC. Independent healthy/idle/warmed/free verification passes at
12:10:45 UTC. No new regression tests or production route follow this screen.
