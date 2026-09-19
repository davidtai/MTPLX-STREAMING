"""Construction-bound, single-request DSpark tail capture and native seeding."""
import ast
from pathlib import Path


def build_forward(keep_rows, *, prompt_tokens=None):
    from mtplx.models import deepseek_v41 as dv
    if not isinstance(keep_rows, int) or keep_rows <= 0:
        raise ValueError('positive retained row count required at construction')
    path = Path(__file__).with_name('tail_prefill_body.py')
    tree = ast.parse(path.read_text())
    if prompt_tokens is not None:
        if not isinstance(prompt_tokens, int) or prompt_tokens <= keep_rows:
            raise ValueError('explicit prompt size must exceed the retained rows')
        body = tree.body[0].body
        at = next(i for i, node in enumerate(body)
                  if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple)
                  and [n.id for n in node.targets[0].elts] == ['b', 's'])
        guard = ast.parse(f"if (b, s) != (1, {prompt_tokens}):\n    raise RuntimeError('tail prefill prompt differs from its admitted shape')").body[0]
        body.insert(at + 1, guard)

    class BindRows(ast.NodeTransformer):
        def visit_Name(self, node):
            return ast.copy_location(ast.Constant(keep_rows), node) if node.id == '_tail_keep_rows' else node

    tree = ast.fix_missing_locations(BindRows().visit(tree))
    namespace = {}
    # Use the native globals so construction-time prefill kernel choices stay
    # identical. No function or model is stored into the native module globals.
    exec(compile(tree, str(path), 'exec'), dv.__dict__, namespace)
    return namespace['_forward_layer_major']


def install(model):
    from mtplx.models import deepseek_v41 as dv
    from mtplx.models import deepseek_v41_dspark as ds
    from mtplx.models import deepseek_v41_dspark_decode as decode
    args = model.args
    if (not isinstance(model, dv.Model) or type(model.model) is not dv.DeepseekV41Backbone
        or model.mtp is None or args.hidden_size != 5120 or args.sliding_window != 128
        or tuple(model.model._mtp_target_layer_ids) != (37, 38, 39)
        or args.num_hidden_layers != 40):
        raise RuntimeError('tail prefill requires the native16K DSpark geometry')
    prototypes = model.make_mtp_cache()
    if len(prototypes) != 3 or any(type(c) is not ds.DSparkStageCache for c in prototypes):
        raise RuntimeError('native draft caches required; Q8 partial seeding is not installed')
    del prototypes
    forward = build_forward(2048, prompt_tokens=16384)
    # A type method avoids a model -> stored bound method -> model ownership
    # cycle. The native verification and generic forward methods are inherited.
    cls = type('Tail2048DSparkBackbone', (dv.DeepseekV41Backbone,), {'_forward_layer_major': forward})
    object.__setattr__(model.model, '__class__', cls)
    original_seed = decode._seed_prefill_state
    model_identity = id(model)
    seeded = False

    def seed(target, hidden, caches):
        nonlocal seeded
        # This boundary executes once, outside the token/layer/cycle hot path.
        if seeded or id(target) != model_identity or hidden.shape != (1, 2048, 15360):
            raise RuntimeError('single-request16K tail seed shape or ownership differs')
        if len(caches) != 3 or any(type(c) is not ds.DSparkStageCache or c.offset or c.window is not None for c in caches):
            raise RuntimeError('tail seed requires fresh native caches')
        for cache in caches:
            cache.offset = 16384 - 2048
        seeded = True
        return original_seed(target, hidden, caches)

    decode._seed_prefill_state = seed
    return {'prompt_tokens': 16384, 'retained_main_hidden_rows': 2048,
            'absolute_seed_start': 14336, 'draft_window_rows': 128,
            'retained_main_hidden_bytes': 62914560,
            'scope': 'Single-request native KV16; target verification unchanged; original memory allowances required until fresh full evidence'}
