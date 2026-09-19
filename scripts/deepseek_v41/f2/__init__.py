"""F2b next-layer expert prefetch for the DeepSeek-V4.1 Q4 D5/M<=8 verify decode.

A LANE-PRIVATE HOST RING plus READER-LEVEL INTERCEPTION: the runtime keeps
``prefetch_slots == 0`` and its packed_phase / packed_admission / config stay byte-for-byte
retained (except the equal-capacity row cap). The runtime still sees an ordinary decode
MISS; the miss is fulfilled from RAM instead of the SSD by an intercepted reader derived
from the retained ``plane_lane.bind_reader``. Nothing in the 11 pinned runtime sources is
edited (the reader intercept is a derived-source rebind; the predictor is an outer wrap of
``switch._run``). Installed as the last step of ``observe_seed_prefill`` (after
``prime_model``) via one anchored, round-trip-checked staged edit of run_full.py.

Modules:
  * ``host_ring``       -- R records x 3 planes of anonymous host RAM; state machine +
                           try_serve/recycle; F2bCounters. MLX-free.
  * ``reader_intercept``-- derive the intercepted reader from the retained bind_reader
                           (anchored round-trip) + rebind read_record_into/…components.
  * ``predictor``       -- parameter-free next-layer predictor (device biased-gate max) +
                           host ranking (top-3 not-resident, not-in-ring) + source geometry.
  * ``speculative``     -- private N-worker pool (default 3) filling the ring, window-stop.
  * ``install``         -- wire ring + pool + intercept + per-layer run wrappers.
  * ``stage_f2_runner`` -- anchored staged edits (admission cap + run_full install/traceback).
  * ``window_preflight``-- CPU preflight: source pin + seam resolution + staged compile.

See docs/deepseek-v41/receipts/f2-prefetch-build-20260919/README.md for the design, the
install-point analysis (file:line), the memory arithmetic and the counter schema.
"""
