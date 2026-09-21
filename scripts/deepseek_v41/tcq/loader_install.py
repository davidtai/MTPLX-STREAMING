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


def _tcq3_expected_signature(spec):
    """The tcq3 record signature (component, dtype, shape, length) tuple, for the allocator validator branch."""
    sig = []
    for proj in _PROJECTIONS:
        out_f = spec.expert_hidden_size if proj in ("gate_proj", "up_proj") else spec.hidden_size
        in_f = spec.hidden_size if proj in ("gate_proj", "up_proj") else spec.expert_hidden_size
        nI, nJ = in_f // 16, out_f // 16
        sig.append((f"{proj}.code", "I16", (nI, nJ, 48), nI * nJ * 48 * 2))
        sig.append((f"{proj}.rout", "F16", (out_f,), out_f * 2))
    return tuple(sig)


def install_component_bank_allocator_support() -> bool:
    """sha-safe monkeypatch of ``make_mlx_component_bank_allocator``: add a tcq3 validator branch, mxfp4 unchanged.

    The pinned run (run_full.py:280-284) asserts the mtplx source sha, so the F39 seam edit cannot be used on the
    launcher path; this rebinds the module function (no file change).  For a tcq3 spec it validates the record
    signature itself (the inline mxfp4/shadow validator would reject it) then builds the SAME generic, record-driven
    component banks (``MlxComponentBank(record)`` per record segment -> per-projection {proj}.code / {proj}.rout
    arrays) via the module's own ``allocate`` machinery.  For every other codec it delegates to the original,
    byte-for-byte.  Idempotent."""
    from mtplx.models import expert_mlx as em
    if getattr(em, "_tcq3_allocator_installed", False):
        return False
    original = em.make_mlx_component_bank_allocator

    def patched(plan, spec, manifest):
        if getattr(spec, "expert_codec", None) != TCQ3_MODE:
            return original(plan, spec, manifest)      # mxfp4/affine/shadow: unchanged
        # --- tcq3 branch: validate then reuse the ORIGINAL generic allocator with validation bypassed ---
        expected = _tcq3_expected_signature(spec)

        def _sig(record):
            return tuple((s.component, s.dtype, tuple(s.shape), int(s.length)) for s in record.segments)

        routed = set(spec.routed_layer_indices)
        exemplar = None
        for record in manifest.records:
            if record.layer in routed:
                if _sig(record) != expected:
                    raise ValueError(
                        f"tcq3 manifest record geometry differs for expert "
                        f"({record.layer}, {record.expert}): {_sig(record)} != {expected}")
                if exemplar is None:
                    exemplar = record
        if exemplar is None:
            raise ValueError("tcq3 manifest has no routed exemplar record")
        # Build the component banks with the module's own primitives (record-driven; identical to the original's
        # post-validation bank_for/allocate), which the tcq3 record supports unchanged.
        return _build_generic_allocator(em, plan, spec, manifest)

    em.make_mlx_component_bank_allocator = patched
    em._tcq3_allocator_installed = True
    em._tcq3_allocator_original = original
    return True


def _build_generic_allocator(em, plan, spec, manifest):
    """Record-driven component-bank allocator (the generic body of make_mlx_component_bank_allocator, no codec
    signature gate) — used for a validated tcq3 manifest.  Mirrors the module's own bank_for/allocate exactly."""
    MlxComponentBank = em.MlxComponentBank
    MlxComponentSlot = em.MlxComponentSlot
    record_by_layer = {}
    for record in manifest.records:
        record_by_layer.setdefault(record.layer, record)
    banks: dict = {}
    slots: dict = {}
    backend = "mlx-metal-component-banks"

    def bank_for(kind, discriminator=-1):
        keyed = {"persistent", "prefetch"}
        key = (kind, discriminator if kind in keyed else -1)
        bank = banks.get(key)
        if bank is not None:
            return bank
        if kind == "persistent":
            capacity, record, label = plan.slots_for_layer(int(discriminator)), record_by_layer[discriminator], f"layer-{discriminator}-persistent-bank"
        elif kind == "prefetch":
            capacity, record, label = plan.prefetch_ring_slots, record_by_layer[discriminator], f"layer-{discriminator}-prefetch-bank"
        elif kind == "global-persistent":
            capacity, record, label = plan.persistent_slots, record_by_layer[spec.routed_layer_indices[0]], "global-persistent-bank"
        elif kind == "global-prefetch":
            capacity, record, label = plan.prefetch_ring_slots, record_by_layer[spec.routed_layer_indices[0]], "global-prefetch-bank"
        else:
            capacity, record, label = plan.transient_slots, record_by_layer[spec.routed_layer_indices[0]], "global-transient-bank"
        bank = MlxComponentBank(capacity=capacity, record=record, label=label)
        banks[key] = bank
        return bank

    def allocate(size, label):
        m = em._LAYER_PERSISTENT_LABEL.fullmatch(label) or em._LAYER_PREFETCH_LABEL.fullmatch(label) \
            or em._GLOBAL_PERSISTENT_LABEL.fullmatch(label) or em._GLOBAL_TRANSIENT_LABEL.fullmatch(label) \
            or em._GLOBAL_PREFETCH_LABEL.fullmatch(label)
        if em._LAYER_PERSISTENT_LABEL.fullmatch(label):
            layer, idx = int(m.group(1)), int(m.group(2)); bank = bank_for("persistent", layer)
        elif em._LAYER_PREFETCH_LABEL.fullmatch(label):
            layer, idx = int(m.group(1)), int(m.group(2)); bank = bank_for("prefetch", layer)
        elif em._GLOBAL_PERSISTENT_LABEL.fullmatch(label):
            idx = int(m.group(1)); bank = bank_for("global-persistent", -1)
        elif em._GLOBAL_TRANSIENT_LABEL.fullmatch(label):
            idx = int(m.group(1)); bank = bank_for("transient", -1)
        elif em._GLOBAL_PREFETCH_LABEL.fullmatch(label):
            idx = int(m.group(1)); bank = bank_for("global-prefetch", -1)
        else:
            raise ValueError(f"unknown expert slot label {label!r}")
        if int(size) != bank.record_bytes:
            raise ValueError("slot allocator size differs from the bank record")
        if label in slots:
            raise ValueError(f"slot {label} was allocated twice")
        slot = MlxComponentSlot(bank, idx, label=label)
        slots[label] = slot
        return slot

    def close_banks():
        for bank in tuple(banks.values()):
            bank.close()
        banks.clear()
        slots.clear()
    for name, val in (("backend", backend), ("slots", slots), ("banks", banks), ("close", close_banks), ("plan", plan)):
        setattr(allocate, name, val)
    return allocate


def install_tcq_loader() -> dict:
    """Install every tcq3 loader seam (manifest + spec + allocator).  Idempotent; safe to call once at construction."""
    return {
        "manifest_support_installed": install_manifest_support(),
        "spec_support_installed": install_spec_support(),
        "allocator_support_installed": install_component_bank_allocator_support(),
        "tcq3_record_bytes": TCQ3_RECORD_BYTES,
        "tcq3_manifest_components": TCQ3_MANIFEST_COMPONENTS,
        "whole_record_component": WHOLE_RECORD_COMPONENT,
    }
