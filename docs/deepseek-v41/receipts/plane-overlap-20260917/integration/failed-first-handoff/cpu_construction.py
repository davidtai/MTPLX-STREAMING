import dataclasses
import hashlib
import importlib.abc
import json
import sys
from pathlib import Path

class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='mlx' or fullname.startswith('mlx.'):
            raise RuntimeError('MLX forbidden in CPU construction accounting')

sys.meta_path.insert(0,NoMLX())
from mtplx.expert_streaming_models import get_model_spec,ExpertMemoryPlan
from mtplx.expert_runtime import ExpertStreamingConfig

root=Path(__file__).resolve().parent
s=get_model_spec('deepseek-v41-flash-expert-mxfp4')
record=s.expert_record_bytes
s=dataclasses.replace(s,routed_layer_start=20,routed_layer_count=1,total_tensor_bytes=384*record+4096,
 router_bytes=0,kv_bytes_per_token=0,mtp_layer_index=None,mtp_included=False,full_indexer_layers=(),island_pin_order=())
p=ExpertMemoryPlan(model_key=s.key,total_limit_bytes=2*1024**3,runtime_reserve_bytes=0,io_staging_bytes=0,
 execution_workspace_bytes=0,context_tokens=0,resident_bytes=4096,kv_bytes=0,transient_slots=48,transient_bytes=48*record,
 expert_cache_limit_bytes=8*record,persistent_budget_bytes=8*record,cache_scope='layer',persistent_slots=8,slots_per_layer=8,
 persistent_cache_bytes=8*record,unallocated_bytes=2*1024**3-56*record-4096,fits_fixed=True,batch_admission_slots=48)
c=ExpertStreamingConfig(model_key=s.key,memory_limit_bytes=2*1024**3,max_live_kv_tokens=17664,runtime_reserve_bytes=0,
 expert_cache_limit_bytes=8*record,transient_slots=48,slot_layout='component-banks',cache_scope='layer',
 cache_policy='transition-window',bypass_page_cache=True,resource_telemetry=False,decode_miss_records_per_part=3,
 overlap_miss_reads=True,verify_shared_overlap=True,split_route_release='deferred',deferred_pin_release=True,
 verify_record_hashes=False,io_read_fanout=4)
proof=json.loads((root/'construction.json').read_text())
proof.update(bound_scope='One real layer: 8 persistent plus 48 shared transient slots; fixed packed scale owners; BF16 M1/M6 operator inputs and intermediate roots; Metal capped at 2 GiB within an 8 GiB total incremental envelope.', bank_capacity=56, comparison='unchanged packed runtime versus specialized full-ready and early-GU runtime; exact operator outputs, not full-model throughput', spec=dataclasses.asdict(s),plan=dataclasses.asdict(p),config=c.to_dict(),
 source_scope='Exact layer20 subset only; no backbone, attention or other layer weights. 4096 descriptor-resident bytes conservatively priced.',
 native_slot_bytes=56*record,packed_slot_bytes=56*17694720,
 lane_sha256=hashlib.sha256((root/'plane_lane.py').read_bytes()).hexdigest(),
 purpose='Real ExpertStreamingRuntime/SlotPool/HotExpertSwitchGLU integration; complete records remain authoritative.')
(root/'integration-construction.json').write_text(json.dumps(proof,indent=2)+'\n')
compile((root/'plane_lane.py').read_text(),str(root/'plane_lane.py'),'exec')
print(json.dumps({'mlx_imported':any(k=='mlx' or k.startswith('mlx.') for k in sys.modules),
 'routed_layers':s.routed_layer_indices,'native_slot_bytes':proof['native_slot_bytes'],
 'packed_slot_bytes':proof['packed_slot_bytes'],'static_incremental_bound_bytes':proof['static_incremental_bound_bytes']},indent=2))
