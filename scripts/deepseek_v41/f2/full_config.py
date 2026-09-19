"""Full-model prefetch config: the retained transition-window geometry + ring R.

The public runtime deliberately forbids the transition-window cache policy combined
with speculative prefetch (mtplx/expert_runtime.py:457-461, inside
``ExpertStreamingConfig.__post_init__``). Codex's 3-layer screen admitted the two
together with ``PairedPrefetchConfig`` (ridge-prefetch v2), which validated every
native field with ``prefetch_slots`` temporarily zeroed, then restored the ring.

``FullPrefetchConfig`` is the full 40-layer analog of that type: it admits ONLY the
exact retained DeepSeek-V4.1 packed decode geometry (transition-window, layer scope,
component-banks, transient_slots=48, decode_miss_records_per_part=3, deferred split
release, overlap_miss_reads) with a fixed prefetch ring R in {16, 32}. Every OTHER
field -- memory limit, expert-cache limit, codec, islands, KV, io fraction -- is
validated UNCHANGED by the native ``__post_init__`` (called once with the ring
zeroed so the mutual-exclusion guard does not fire, then the ring is restored). No
public configuration or production eligibility changes here.

The ring is a RESIDENT reserve of R expert records. It must be charged into the
retained run's admission bound through the SAME admission code
(sources/packed/packed_admission.py), NOT a parallel formula -- see
``ring_reserve_bytes`` and the receipt's memory arithmetic.
"""
from __future__ import annotations

from mtplx.expert_runtime import ExpertStreamingConfig

# Admitted ring sizes (records). R=32 is the build default; R=16 is also supported.
ADMITTED_RING_SLOTS = (16, 32)

# One packed mxfp4 weight-only expert record on the retained profile
# (sources/packed/packed_admission.py ``WEIGHTS``; == f2_predictor.EXPERT_RECORD_BYTES).
EXPERT_WEIGHT_RECORD_BYTES = 17_694_720


def ring_reserve_bytes(ring_slots: int) -> int:
    """Resident bytes an R-record prefetch ring reserves (one weight record/slot).

    This is the reserve the admission loop must add on top of the row weights; the
    row-count that fits is derived by ``packed_admission.resolve_admission`` (its
    capacity search at sources/packed/packed_admission.py:108-131), never here.
    """
    ring_slots = int(ring_slots)
    if ring_slots < 0:
        raise ValueError("ring_slots must be >= 0")
    return ring_slots * EXPERT_WEIGHT_RECORD_BYTES


class FullPrefetchConfig(ExpertStreamingConfig):
    def __post_init__(self) -> None:
        if (
            not str(self.model_key).startswith("deepseek-v41")
            or self.cache_policy != "transition-window"
            or self.cache_scope != "layer"
            or self.slot_layout != "component-banks"
            or type(self.prefetch_slots) is not int
            or self.prefetch_slots not in ADMITTED_RING_SLOTS
            or self.transient_slots != 48
            or self.decode_miss_records_per_part != 3
            or self.split_route_release != "deferred"
            or not self.overlap_miss_reads
        ):
            raise ValueError(
                "FullPrefetchConfig admits only the retained DeepSeek-V4.1 packed "
                "decode geometry (deepseek-v41 model, transition-window cache policy, "
                "layer scope, component-banks, transient_slots=48, "
                "decode_miss_records_per_part=3, deferred split release, "
                "overlap_miss_reads) with prefetch_slots in "
                f"{ADMITTED_RING_SLOTS}"
            )
        ring = self.prefetch_slots
        # Defeat the native transition-window + prefetch mutual exclusion so the ring
        # is admitted ALONGSIDE the transition window. Every other native field is
        # validated unchanged with the ring zeroed, then the ring is restored.
        object.__setattr__(self, "prefetch_slots", 0)
        super().__post_init__()
        object.__setattr__(self, "prefetch_slots", ring)
