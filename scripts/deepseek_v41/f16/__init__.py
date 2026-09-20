"""F16 verify-row-group pipeline for the DeepSeek-V4.1 Q4 D5/M<=8 verify decode.

Interleave two causal MTP-verify row groups (A = first 4 rows, B = the rest, B
attending to A's KV) so one group's SSD miss-read wait overlaps the other group's
routing barrier + submit, under a STRICT BATON with one hand-off point inside the
expert switch.  Bit-identical to the retained sequential (4,4) ``verify_chunks``
arithmetic (digest 172830a9...); the whole lever is (a) new modules here and (b)
anchored, round-trip-checked staged edits of the retained runner helpers.

Modules:
  * ``pipeline``          -- the clone of ``_forward_span`` + Model head tail, the
                             strict baton, the yield-run derivation, and the source pins.
  * ``install``           -- arm/passthrough wiring from ``observe_seed_prefill``
                             (rebind yield run, wrap issue_next, refuse non-scheduled
                             lanes / global-bank locks / device route), + provenance.
  * ``stage_f16_runner``  -- the three anchored staged edits (run_full hook,
                             hybrid_install replacement + injection, projection 4-buffer).
  * ``preflight``         -- CPU-only attribute/anchor/derivation existence check.
"""
