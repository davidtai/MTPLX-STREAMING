"""Explicit configuration type for this bounded two-layer experiment only.

The public runtime deliberately excludes transition-window plus prefetch.
This type admits only the fixed packed experiment; the caller must validate
its two-layer spec, shared16-slot ring and source-pinned bank implementation
before allocation. Native LayerExpertSlotBank.plan already resolves published
ring hits before transition-window admission and excludes them from pool loads.
No public configuration or production eligibility changes here.
"""
from mtplx.expert_runtime import ExpertStreamingConfig


class PairedPrefetchConfig(ExpertStreamingConfig):
    def __post_init__(self):
        if (self.model_key!='deepseek-v41-flash-expert-mxfp4'
                or self.cache_policy!='transition-window'
                or self.cache_scope!='layer' or self.slot_layout!='component-banks'
                or type(self.prefetch_slots) is not int or self.prefetch_slots!=16
                or self.transient_slots!=48 or self.decode_miss_records_per_part!=3
                or self.split_route_release!='deferred' or not self.overlap_miss_reads
                or self.memory_limit_bytes!=7*1024**3
                or self.expert_cache_limit_bytes!=210*18800640):
            raise ValueError('only the fixed paired105/48/16 packed experiment is admitted')
        # Validate every unchanged native field, including all transition-window
        # constraints. The additional capability is the fixed shared ring above.
        object.__setattr__(self,'prefetch_slots',0)
        super().__post_init__()
        object.__setattr__(self,'prefetch_slots',16)
