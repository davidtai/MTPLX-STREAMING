"""Construction and completion boundaries for one native packed decode request."""
from mtplx.models import deepseek_v41 as dv
from owned_projection import BF16Output, install_attention


def install_model(model, *, backbone_type=dv.DeepseekV41Backbone):
    if (type(model) is not dv.Model or type(model.model) is not backbone_type
        or len(model.model.layers) != 40 or model.args.hidden_size != 5120
        or model.mtp is None or not all(dv._fused_proj_use(rows) for rows in (1, 6, 8))):
        raise RuntimeError('single-request native target fused decode geometry required')
    identities = [(id(layer.attn.wo_a.weight), id(layer.attn.wo_a.scales))
                  for layer in model.model.layers]
    if len(set(identities)) != 40:
        raise RuntimeError('native target projection owners must be independent')
    reports = [install_attention(layer.attn) for layer in model.model.layers]
    total = sum(r['raw_bytes_released_on_first_use'] for r in reports)
    if total != 1384120320:
        raise RuntimeError('target projection source inventory differs')
    return {'target_layers': 40, 'source_packed_projection_bytes': total,
            'cold_source_overlap_allowance_bytes': 34603008,
            'prefill_and_growth_credit_bytes': 0,
            'resident_plan_retains_source_reserve': True,
            'scope': 'One native KV16 D5/M6 request; native lazy BF16 materialization followed by packed-owner retirement. Module ownership and measured process/allocator bytes are separate.'}


def verify_retirement(model):
    # Completion boundary, after the native measured generation returns.
    retained = 0
    for layer in model.model.layers:
        attn = layer.attn
        lane = attn._out_prep_fused_impl
        if (type(lane) is not BF16Output or 'wo_a' in attn
            or attn._wo_a_bf16T_cache is not None or attn._wo_a_dense_cache is not None
            or lane.wo_b is not attn.wo_b):
            raise RuntimeError('target projection retirement did not complete')
        retained += int(lane.weight.nbytes)
    if retained != 2684354560:
        raise RuntimeError('retained native BF16 projection bytes differ')
    return {'module_ownership_verified': True, 'retired_module_source_bytes': 1384120320,
            'retained_module_bf16_bytes': retained}
