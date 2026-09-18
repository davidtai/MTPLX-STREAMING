# Weight-only compression headroom

The current packed-plane lane reads weights only; it keeps scales resident.
Recomputing the twelve authenticated entropy samples for the three weight
components gives an ideal order0 byte-code saving of **5.9408%**, compared with
10.7525% for separately coded whole records. The earlier larger percentage
includes scale traffic that the current decode path already removes.

At the measured 13.08 GB/s useful read rate, even an ideal serial decoder must
exceed **220.17 GB/s** of reconstructed weights merely to break even. Framing,
directories, tables, scratch space and dispatch would raise that requirement.
This bound applies to the sampled independent-byte model, not all structured
compression methods or fully overlapped decoding. No codec is installed and
no new model or artifact read is performed. `summary.json` binds the inputs.

The next bounded candidate instead retires native packed target `wo_a` weights
after their existing first-use BF16 transpose is materialized. The native
manifest prices the 40 packed copies at1,384,120,320bytes. Its output arithmetic
will stay identical; a real one-layer ownership screen must establish release
before a full-model capacity change is considered.
