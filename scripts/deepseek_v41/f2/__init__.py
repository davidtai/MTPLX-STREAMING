"""F2 next-layer expert prefetch lane for the DeepSeek-V4.1 Q4 D5/M<=8 verify decode.

Composed onto the exact configuration of the retained 13.87 TPS packed decode run
(``sources/packed/plane_lane.py`` ``PackedDecode``). This package is the CANDIDATE
lane; the retained ``plane_lane.install`` is the CONTROL. A construction-time switch
(the run_full stage edit) selects one lane once; the stock path is byte-for-byte
untouched when the candidate is off.

Modules:
  * ``priority_reads``       -- N-worker demand-priority isolated reader (fix #1).
  * ``issue``                -- live next-layer predictor: device biased-gate max
                                over rows, host top-k not-READY, one issue (fix #2/#3).
  * ``plane_lane_prefetch``  -- ``PrefetchDecode`` runner + ``install`` (the 3 fixes),
                                derived from ridge-prefetch v2.
  * ``full_config``          -- ``FullPrefetchConfig``: transition-window + ring R.
  * ``gpu_smoke``            -- WRITE-not-run bounded Metal smoke for the one
                                interaction (plane-split mxfp4 reader + Metal gather)
                                that cannot run on CPU.

Every runtime interaction the lane relies on is a real, shipped ``mtplx`` API. See
``docs/deepseek-v41/receipts/f2-prefetch-build-20260919/README.md`` for the seam
table (every attribute -> defining file:line -> covering test).
"""
