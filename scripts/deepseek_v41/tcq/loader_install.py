"""F39: make the pinned runner's model loader tcq3-aware, as an installable monkeypatch (a staged edit calls it).

Like the other levers, this does NOT edit mtplx source — it patches the narrow, identified seams at construction:

  * ``mtplx.expert_manifest``: accept ``quantization.mode == "tcq3"`` and its six components
    (`{gate,up,down}_proj.{code,rout}`), so a tcq3 manifest parses.
  * ``mtplx.expert_streaming_models.ExpertStreamingModelSpec``: accept the ``tcq3`` codec in ``__post_init__`` and return
    ``expert_record_bytes == 13,290,496`` for it (the class property is otherwise mxfp4/affine/shadow only).
  * ``stamp_spec_tcq3(spec)``: force a loaded spec's codec to tcq3 via ``object.__setattr__`` (the spec is a frozen
    dataclass; this is the same escape the retained packed_phase uses for switches), so ``rt.spec.expert_codec ==
    "tcq3"`` and ``rt.spec.expert_record_bytes == 13,290,496`` downstream.

All patches are idempotent and applied only inside ``install_tcq_loader()`` / ``stamp_spec_tcq3()`` (importing this
module does not touch mtplx).  The tcq3 pure facts are module constants and are unit-tested.

BOUNDARY (documented; needs the real F38 bank + GPU): the whole-record int16 bank component per slot and the
decode route are handled by the F39 reader (`tcq.install.bind_tcq_reader`) and the plane-lane route
(`tcq.install.route_plane_lane`); the spec REGISTRY entry (get_model_spec producing a tcq3 spec directly) and the
switch-binding decode branch are the remaining runtime integration exercised only when the bank lands.  This module
covers the manifest-parse + spec-codec/record-bytes seam so the loaded mxfp4 spec can be stamped to tcq3.
"""
from __future__ import annotations

TCQ3_MODE = "tcq3"
TCQ3_RECORD_BYTES = 13_290_496
WHOLE_RECORD_COMPONENT = "expert_record"          # the single int16 component the tcq3 slot bank presents
_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
TCQ3_MANIFEST_COMPONENTS = tuple(f"{p}.{leaf}" for p in _PROJECTIONS for leaf in ("code", "rout"))


def tcq3_manifest_components() -> tuple:
    """The six per-record component names an F38 tcq3 manifest carries (pure; matches transcode_bank)."""
    return TCQ3_MANIFEST_COMPONENTS


def install_manifest_support() -> bool:
    """Patch ``mtplx.expert_manifest`` to accept the tcq3 mode + components.  Idempotent; returns True if applied."""
    from mtplx import expert_manifest as em
    if getattr(em, "_tcq3_installed", False):
        return False
    original = em.expert_components_for_mode

    def patched(quant_mode: str):
        if quant_mode == TCQ3_MODE:
            return TCQ3_MANIFEST_COMPONENTS
        return original(quant_mode)

    em.expert_components_for_mode = patched
    try:
        em._KNOWN_COMPONENTS = em._KNOWN_COMPONENTS | frozenset(TCQ3_MANIFEST_COMPONENTS)
    except Exception:  # noqa: BLE001 - _KNOWN_COMPONENTS shape may evolve; the components patch is the load-bearing one
        pass
    em._tcq3_installed = True
    em._tcq3_original_components_for_mode = original
    return True


def install_spec_support() -> bool:
    """Patch ``ExpertStreamingSpec`` to accept the tcq3 codec and report its record bytes.  Idempotent."""
    from mtplx.expert_streaming_models import ExpertStreamingModelSpec as Spec
    if getattr(Spec, "_tcq3_installed", False):
        return False
    original_post_init = Spec.__post_init__
    original_record_bytes = Spec.expert_record_bytes.fget

    def patched_post_init(self):
        if getattr(self, "expert_codec", None) == TCQ3_MODE:
            # tcq3 is validated at the F39 manifest reader / geometry gate; skip the mxfp4-only codec reject but keep
            # every other structural check by temporarily presenting an accepted codec to the original validator.
            object.__setattr__(self, "_tcq3", True)
            object.__setattr__(self, "expert_codec", "affine")
            try:
                original_post_init(self)
            finally:
                object.__setattr__(self, "expert_codec", TCQ3_MODE)
            return
        original_post_init(self)

    def patched_record_bytes(self):
        if getattr(self, "expert_codec", None) == TCQ3_MODE:
            return TCQ3_RECORD_BYTES
        return original_record_bytes(self)

    Spec.__post_init__ = patched_post_init
    Spec.expert_record_bytes = property(patched_record_bytes)
    Spec._tcq3_installed = True
    return True


def stamp_spec_tcq3(spec) -> None:
    """Force a loaded (frozen) spec to the tcq3 codec so downstream reads see the right codec + record bytes.

    Uses ``object.__setattr__`` (frozen dataclass escape, as the retained packed_phase does for switches).  Call
    after ``install_spec_support()`` so ``expert_record_bytes`` resolves to 13,290,496.
    """
    object.__setattr__(spec, "expert_codec", TCQ3_MODE)


def install_tcq_loader() -> dict:
    """Install every tcq3 loader seam (manifest + spec).  Idempotent; safe to call once at construction."""
    return {
        "manifest_support_installed": install_manifest_support(),
        "spec_support_installed": install_spec_support(),
        "tcq3_record_bytes": TCQ3_RECORD_BYTES,
        "tcq3_manifest_components": TCQ3_MANIFEST_COMPONENTS,
        "whole_record_component": WHOLE_RECORD_COMPONENT,
    }
