"""CPU test that the COMMITTED validator seam edit in this worktree's mtplx/models/expert_mlx.py round-trips: the
real ``make_mlx_component_bank_allocator`` accepts a tcq3 record AND still accepts an mxfp4 record (and still rejects
a genuine geometry mismatch).  Plus the eval driver's editable-install-shadowing guard (mtplx.__file__ under the
worktree) and the serve PYTHONPATH order.  No serve, no model, no GPU.
"""
import sys
import types
from pathlib import Path

import pytest

_TRELLIS = Path(__file__).resolve().parents[1]
_DSV41 = Path(__file__).resolve().parents[2]
_ROOT = Path(__file__).resolve().parents[4]
for p in (str(_TRELLIS), str(_DSV41), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _seg(c, d, sh, ln):
    return types.SimpleNamespace(component=c, dtype=d, shape=sh, length=ln)


def _manifest(segs):
    return types.SimpleNamespace(records=[types.SimpleNamespace(layer=0, expert=0, segments=segs)])


def _spec(codec):
    return types.SimpleNamespace(expert_codec=codec, routed_layer_indices=(0,), hidden_size=5120,
                                 expert_hidden_size=2304, quant_bits=(3 if codec == "tcq3" else 4),
                                 quant_group_size=32, expert_count=384)


_PLAN = types.SimpleNamespace(cache_scope="layer")   # skips the global-keys check; validation is the same
_TCQ3 = [_seg("gate_proj.code", "I16", (320, 144, 48), 4423680), _seg("gate_proj.rout", "F16", (2304,), 4608),
         _seg("up_proj.code", "I16", (320, 144, 48), 4423680), _seg("up_proj.rout", "F16", (2304,), 4608),
         _seg("down_proj.code", "I16", (144, 320, 48), 4423680), _seg("down_proj.rout", "F16", (5120,), 10240)]
_MXFP4 = [_seg("gate_proj.weight", "U32", (2304, 640), 5898240), _seg("gate_proj.scales", "U8", (2304, 160), 368640),
          _seg("up_proj.weight", "U32", (2304, 640), 5898240), _seg("up_proj.scales", "U8", (2304, 160), 368640),
          _seg("down_proj.weight", "U32", (5120, 288), 5898240), _seg("down_proj.scales", "U8", (5120, 72), 368640)]


def test_seam_edit_accepts_tcq3_and_still_accepts_mxfp4():
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from mtplx.models.expert_mlx import make_mlx_component_bank_allocator as mk
    # the COMMITTED seam edit must accept a tcq3 record (returns the allocator, no raise)...
    assert callable(mk(_PLAN, _spec("tcq3"), _manifest(_TCQ3)))
    # ...and still accept an mxfp4 record (no regression)...
    assert callable(mk(_PLAN, _spec("mxfp4"), _manifest(_MXFP4)))
    # ...and still reject a genuine geometry mismatch (tcq3 spec vs an mxfp4-signature record).
    with pytest.raises(ValueError):
        mk(_PLAN, _spec("tcq3"), _manifest(_MXFP4))
    with pytest.raises(ValueError):
        mk(_PLAN, _spec("mxfp4"), _manifest(_TCQ3))


def test_seam_edit_present_in_worktree_source():
    src = (_ROOT / "mtplx" / "models" / "expert_mlx.py").read_text()
    assert 'if expert_codec == "tcq3":' in src and '.code", "I16"' in src and '.rout", "F16"' in src


# ---------------------------------------------------------------- editable-install-shadowing guard

def test_mtplx_under_worktree_guard():
    from tcq import eval_driver as E
    wt = E.MTPLX_WORKTREE
    assert E.mtplx_under_worktree(f"{wt}/mtplx/__init__.py", wt) is True
    assert E.mtplx_under_worktree(f"{wt}/mtplx/models/expert_mlx.py", wt) is True
    # a shadowing editable/site-packages mtplx is rejected
    assert E.mtplx_under_worktree("/opt/venv/lib/python3.12/site-packages/mtplx/__init__.py", wt) is False
    assert E.mtplx_under_worktree("/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/mtplx/__init__.py", wt) is False
    assert E.mtplx_under_worktree("", wt) is False


def test_serve_pythonpath_puts_worktree_first():
    from tcq import eval_driver as E
    # tcq3 serve env carries the tcq package + site hook; the worktree root must be FIRST
    _, env = E.build_serve_command(bank="tcq3", model_dir=E.MODEL_DIRS["tcq3"], host="127.0.0.1", port=18183,
                                   lane="ar", depth=3)
    pp = E.serve_pythonpath(E.MTPLX_WORKTREE, env)
    assert pp[0] == E.MTPLX_WORKTREE                              # my mtplx wins over the editable install
    assert E.TCQPKG in pp and any(p.endswith("tcq_serve_site") for p in pp)
    # mxfp4 (control): still worktree-first, no tcq paths -> identical mtplx, only the env flag differs
    _, env_m = E.build_serve_command(bank="mxfp4", model_dir=E.MODEL_DIRS["mxfp4"], host="127.0.0.1", port=18183,
                                     lane="ar", depth=3)
    assert E.serve_pythonpath(E.MTPLX_WORKTREE, env_m) == [E.MTPLX_WORKTREE]


def test_served_mtplx_file_resolves_this_worktree():
    # a real (light) CPU preflight: import mtplx under [worktree] and confirm it's this worktree's mtplx
    from tcq import eval_driver as E
    mtplx_file = E._served_mtplx_file(sys.executable, [E.MTPLX_WORKTREE], E.MTPLX_WORKTREE)
    assert E.mtplx_under_worktree(mtplx_file, E.MTPLX_WORKTREE), mtplx_file
