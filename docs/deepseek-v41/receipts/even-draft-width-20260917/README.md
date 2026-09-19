# Even draft widths: bounded acceptance screen

The depth-5 control reproduces all 206 captured native commit boundaries.
Depths 4 and 6, and a depth-7 head with a constant six-proposal cut, expose
cycle/row tradeoffs without a compelling reduction in verification work.
No full target run or throughput promotion follows this screen.

| Head depth / verified proposals | Cycles | Verification rows |
|---|---:|---:|
| 5 / 5, unchanged control | 206 | 1,236 |
| 4 / 4 | 238 | 1,190 |
| 6 / 6 | 191 | 1,337 |
| 7 / 6 | 189 | 1,323 |

The fixed six-proposal cut reduces cycles by 8.25% but increases rows by 7.04%.
The earlier full D7/M8 run increased expert reads and lost throughput despite
its cycle reduction. Saved target states cannot establish a new verification
schedule's actual arithmetic, physical reads, or TPS. The head replay wall
times in the raw data are not decode throughput. The best full result remains
12.6731624 TPS; 20 TPS is unmet.

All four native compact heads share the same installed parameter arrays.
Target futures only score acceptance; they do not choose proposals or widths.
The raw BF16 teacher files and completion receipt are verified before use.
Native model, DSpark, loader and runner hashes match the earlier adaptive
screen. The controller additionally pins the current guard and reclaimer.

The bound is 41 GiB active, 4 GiB cache and 4 GiB Python/compiler space:
52,613,349,376 bytes plus the live baseline, below 110,000,000,000 bytes.
The child baseline is 10,994,647,040 bytes. The allocator peak is
21,299,586,448 bytes; the guard samples a process-tree footprint peak of
15,197,695,592 bytes and machine physical peak of 26,291,191,808 bytes.
These measures have separate scopes and are not added together.

After the draft subprocess exited 0, the stdlib controller invalidated
15,348,088,832 bytes of clean source-file cache to zero. The guard exited 0,
restored exact Qwen identity, health and warmup, and released the lock at
02:11:01 UTC on September 18. Independent verification at 02:11:48 UTC found
Qwen healthy, idle and warmed, the lock free, and no owned child remaining.
The source is `f384b2ce19df3f40d954e9cfc0c383f51cf122c1`.

`sha256.json` binds the complete proposals, controller, memory/lifecycle logs
and source installation. The proposal JSON is losslessly gzip compressed.
