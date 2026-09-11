"""Serve-path loader for the DeepSeek-V4.1-Flash SSD-streamed MoE artifact.

Everything between ``mtplx serve --model <local dir>`` and a constructed model
whose routed experts stream from the Q2 bank (``experts.bin`` via
``expert-manifest.json``).  The Hy3/GLM lane is the template: the generic
machinery lives in :mod:`mtplx.resident_loader`,
:mod:`mtplx.expert_runtime`, and :func:`mtplx.models.expert_mlx.bind_streamed_switches`.
This module adds only the three things that are specific to DeepSeek-V4.1:

1. a text-only resident filter (drop ``vision.*`` / ``aligner.*`` / ``image_*``
   and, for phase-1 autoregressive serving, ``mtp.*`` tensors -- the converter
   kept q8 MTP dense + q8 MTP experts + the vision/aligner residents in the
   artifact, none of which a text-only AR forward needs);
2. the SWA sliding-window reserve priced as a fixed additional-resident buffer;
3. the engram bank path handed to the model via a constructor argument (this
   module does NOT implement engram -- worker W2 owns ``mtplx/engram*``).

The native model module (``mtplx.models.deepseek_v41``, worker W1) is imported
lazily and guarded, so until it lands the resolver raises a clear error rather
than an ``ImportError`` deep in the loader.  Once it lands, integration is the
single guarded import in :func:`deepseek_v41_model_classes`.

Convention choice (W3): the loader body lives here, in a new
``mtplx/models/deepseek_v41_loader.py``.  ``mtplx/resident_loader.py`` -- the
equivalent place hy3's loader lives -- carries only the two-line dispatch that
routes ``model_type == "deepseek_v41"`` into this module
(``get_streaming_model_classes`` and ``construct_resident_model``), so the
production serve path (``runtime.py`` -> ``construct_resident_model``) reaches
this loader with NO ``runtime.py`` edit, exactly like the hy3 lane.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..expert_manifest import (
    ExpertManifest,
    ResidentTensor,
    load_expert_manifest,
    resolve_artifact_member,
)
from ..expert_runtime import ExpertStreamingConfig, ExpertStreamingRuntime
from ..expert_streaming_models import ExpertStreamingModelSpec, get_model_spec
from ..resident_loader import (
    ResidentLoadError,
    ResidentLoadReport,
    ResidentModel,
    _dtype_name,
)

# Top-level config model_type for the merged (text + vision) checkpoint; the
# text sub-config is "deepseek_v41_text".  Every serve-path dispatch gate reads
# the TOP-LEVEL model_type, so this is the string those gates register.
MODEL_TYPE = "deepseek_v41"
TEXT_MODEL_TYPE = "deepseek_v41_text"

# Streamed-bank model key (== spec.key == manifest.model_key).
MODEL_KEY = "deepseek-v41-flash-expert-q2"

# Phase-1 SWA sliding window, priced as a fixed wired buffer (NOT per token):
# num_layers * sliding_window * (latent_dim * bf16_bytes) = 40 * 128 * 512 * 2.
# Matches PORT_PLAN 3b (bf16 window) and the kv_bytes_per_token spec comment.
NUM_TEXT_LAYERS = 40
SLIDING_WINDOW = 128
KV_LATENT_DIM = 512
SWA_WINDOW_BYTES = NUM_TEXT_LAYERS * SLIDING_WINDOW * (KV_LATENT_DIM * 2)

# Default runtime reserve: the promoted streaming profiles reserve exactly
# 7 GiB (docs/advanced/ssd-streamed-moe.md); the ExpertStreamingConfig default
# is 16 GiB, so pass this explicitly.
DEFAULT_RUNTIME_RESERVE_BYTES = 7 * 1024**3

# Engram resident-row LRU budget. The engram banks (layers 1 and 14) stream
# their affine-q8 rows from the 2x101 GB SSD row banks through a byte-budgeted
# LRU (mtplx.ngram_row_cache); 2 GiB is the serve default here, raising the
# module's bare 1 GiB env fallback. MTPLX_ENGRAM_CACHE_LIMIT still overrides it.
DEFAULT_ENGRAM_CACHE_BYTES = 2 * 1024**3


def resolve_engram_cache_bytes() -> int:
    """Resident-row LRU byte budget for the engram banks.

    Reads ``MTPLX_ENGRAM_CACHE_LIMIT`` (Pydantic ByteSize: "2GiB", raw bytes,
    ...) when set, otherwise the 2 GiB serve default. The loader passes the
    resolved value to :meth:`Model.attach_engram` so the engram row cache is a
    resolved config value rather than the module's implicit 1 GiB fallback.
    """

    from ..ngram_row_cache import cache_bytes_from_env

    return cache_bytes_from_env(default=DEFAULT_ENGRAM_CACHE_BYTES)

# Resident tensors skipped for text-only autoregressive serving.  A tensor is
# skipped when its dotted name starts with any of these prefixes.  Everything
# else -- ``embed.*``, ``head.*``, ``layers.*`` (attention, router gate, shared
# experts, indexer, hyper-connections, norms), ``norm.*`` -- is a text resident.
TEXT_ONLY_SKIP_PREFIXES = ("mtp.", "vision.", "aligner.", "image_")


def is_text_resident(name: str) -> bool:
    """Whether a resident tensor belongs to the text-only AR forward path."""

    return not name.startswith(TEXT_ONLY_SKIP_PREFIXES)


def _skip_reason(name: str) -> str:
    if name.startswith("mtp."):
        return "mtp"
    if name.startswith(("vision.", "aligner.", "image_")):
        return "vision"
    return "text"


def manifest_has_mtp_residents(manifest: ExpertManifest) -> bool:
    """Whether the artifact ships DSpark MTP residents (``mtp.*``)."""

    return any(t.tensor.startswith("mtp.") for t in manifest.resident_tensors)


def _config_declares_mtp(config: dict[str, Any]) -> bool:
    def _stages(d):
        d = d or {}
        return int(d.get("n_mtp_layers") or d.get("num_nextn_predict_layers") or 0)

    return max(_stages(config), _stages((config or {}).get("text_config"))) > 0


def resolve_with_mtp(
    config: dict[str, Any], manifest: ExpertManifest, with_mtp: bool | None
) -> bool:
    """Decide whether to build + load the DSpark head (worker W23).

    Opt-in, so the default stays phase-1 text-only AR (the MTP head's 3x128
    resident mxfp4 experts are ~6.7 GiB, not wanted for AR serving): explicit
    ``with_mtp`` (True/False) wins, then the ``MTPLX_DSV41_MTP`` env flag
    (``1``/``true``/... builds the head), else False.  Building it additionally
    requires the config to declare MTP stages and the artifact to ship ``mtp.*``
    residents -- opting in against an artifact that ships none is a load error the
    operator should see, not a silent AR fallback.

    The served ``--generation-mode mtp`` reaches this via ``MTPLX_DSV41_MTP=1``
    (the serve-path glue that maps the flag to the env lives in cli.py /
    resident_loader.py, outside W23's allowlist -- see W23_REPORT / PORT_CONTRACT).
    The head is then published by ``inject_deepseek_v41_mtp_support`` at the
    runtime's MTP dispatch."""

    if with_mtp is not None:
        want = bool(with_mtp)
    else:
        env = os.environ.get("MTPLX_DSV41_MTP")
        want = (
            env.strip().lower() in {"1", "true", "yes", "on"}
            if env is not None and env.strip() != ""
            else False
        )
    if not want:
        return False
    if not _config_declares_mtp(config):
        raise ResidentLoadError(
            "DSpark MTP requested (with_mtp) but the config declares no MTP stages"
        )
    if not manifest_has_mtp_residents(manifest):
        raise ResidentLoadError(
            "DSpark MTP requested (with_mtp) but the artifact ships no mtp.* residents"
        )
    return True


@dataclass(frozen=True)
class TextResidentPartition:
    """Text-only vs skipped split of a manifest's resident tensors."""

    kept: tuple[ResidentTensor, ...]
    skipped: tuple[ResidentTensor, ...]
    kept_bytes: int
    kept_count: int
    skipped_bytes: int
    skipped_count: int
    skipped_mtp_bytes: int
    skipped_mtp_count: int
    skipped_vision_bytes: int
    skipped_vision_count: int


def partition_text_residents(
    manifest: ExpertManifest, *, with_mtp: bool = False
) -> TextResidentPartition:
    """Split ``manifest.resident_tensors`` into kept vs skipped.

    Always skips ``vision.*``/``aligner.*``/``image_*``.  ``mtp.*`` residents are
    skipped on the text-only AR path (``with_mtp=False``) and KEPT on the opt-in
    DSpark MTP path (``with_mtp=True``, worker W23) so the DSpark head's resident
    mxfp8/mxfp4 tensors load.  The returned byte/count fields are the exact
    figures the W3 report and the text-only-filter test assert against the real
    artifact (unchanged for ``with_mtp=False``).
    """

    kept: list[ResidentTensor] = []
    skipped: list[ResidentTensor] = []
    skipped_mtp_bytes = skipped_mtp_count = 0
    skipped_vision_bytes = skipped_vision_count = 0
    for tensor in manifest.resident_tensors:
        if is_text_resident(tensor.tensor) or (
            with_mtp and tensor.tensor.startswith("mtp.")
        ):
            kept.append(tensor)
            continue
        skipped.append(tensor)
        if _skip_reason(tensor.tensor) == "mtp":
            skipped_mtp_bytes += tensor.length
            skipped_mtp_count += 1
        else:
            skipped_vision_bytes += tensor.length
            skipped_vision_count += 1
    kept_bytes = sum(tensor.length for tensor in kept)
    skipped_bytes = skipped_mtp_bytes + skipped_vision_bytes
    return TextResidentPartition(
        kept=tuple(kept),
        skipped=tuple(skipped),
        kept_bytes=kept_bytes,
        kept_count=len(kept),
        skipped_bytes=skipped_bytes,
        skipped_count=len(skipped),
        skipped_mtp_bytes=skipped_mtp_bytes,
        skipped_mtp_count=skipped_mtp_count,
        skipped_vision_bytes=skipped_vision_bytes,
        skipped_vision_count=skipped_vision_count,
    )


def deepseek_v41_model_classes() -> tuple[type, type]:
    """Resolve ``(Model, ModelArgs)`` for the DeepSeek-V4.1 text model.

    Guarded until worker W1 lands ``mtplx/models/deepseek_v41.py``.  This is the
    single "one-line import" the whole W3 loader is built around.
    """

    try:
        from .deepseek_v41 import Model, ModelArgs  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised until W1 lands
        raise ResidentLoadError(
            "the DeepSeek-V4.1 text model module (mtplx.models.deepseek_v41, "
            "worker W1) is not available yet; the streaming loader is wired and "
            "will construct the model as soon as that module provides "
            "`Model(ModelArgs, *, engram_bank_path=None)` and `ModelArgs`. "
            f"Underlying import error: {exc}"
        ) from exc
    return Model, ModelArgs


def engram_bank_path_for(root: Path | str) -> Path | None:
    """The engram bank directory inside the artifact, or None if absent.

    The engram table is a separate on-disk artifact (worker W2 owns its
    runtime).  This loader only exposes its path to the model constructor; it
    does not read or requantize it.
    """

    engram = Path(root).resolve() / "engram"
    return engram if engram.is_dir() else None


def build_streaming_config(
    spec: ExpertStreamingModelSpec,
    *,
    memory_limit_bytes: int,
    max_live_kv_tokens: int,
    runtime_reserve_bytes: int = DEFAULT_RUNTIME_RESERVE_BYTES,
    expert_cache_limit_bytes: int | None = None,
    **overrides: Any,
) -> ExpertStreamingConfig:
    """An ExpertStreamingConfig for this affine Q2 bank.

    Phase-1 defaults: layer cache scope, no islands, verify_record_hashes off
    at open (the admission receipt covers integrity), affine records through the
    default direct-slot layout.  ``overrides`` pass through to
    :class:`ExpertStreamingConfig` for callers that need to tune it.
    """

    return ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=memory_limit_bytes,
        max_live_kv_tokens=max_live_kv_tokens,
        runtime_reserve_bytes=runtime_reserve_bytes,
        expert_cache_limit_bytes=expert_cache_limit_bytes,
        **overrides,
    )


def open_deepseek_v41_runtime(
    root: Path | str,
    *,
    memory_limit_bytes: int,
    max_live_kv_tokens: int,
    runtime_reserve_bytes: int = DEFAULT_RUNTIME_RESERVE_BYTES,
    manifest_path: Path | str | None = None,
    spec: ExpertStreamingModelSpec | None = None,
    expert_cache_limit_bytes: int | None = None,
    admission_receipt: Mapping[str, Any] | None = None,
    apply_memory_cap: bool = True,
    mx_module: Any | None = None,
    config: ExpertStreamingConfig | None = None,
    **config_overrides: Any,
) -> ExpertStreamingRuntime:
    """Construct the ExpertStreamingRuntime from ``expert-manifest.json``.

    The SWA window is priced as a fixed ``additional_resident_bytes`` reserve.
    Bank I/O is lazy: opening the runtime reads the manifest JSON and (with the
    default header verification) size-checks ``experts.bin`` only -- it never
    reads or hashes the 169 GiB bank.  ``spec`` defaults to the pinned
    descriptor for the manifest's model key.
    """

    artifact_root = Path(root).resolve()
    resolved_manifest = (
        Path(manifest_path)
        if manifest_path is not None
        else resolve_artifact_member(artifact_root, "expert-manifest.json")
    )
    loaded_manifest: ExpertManifest | None = None
    if spec is None:
        loaded_manifest = load_expert_manifest(resolved_manifest)
        spec = get_model_spec(loaded_manifest.model_key)
    if config is None:
        config = build_streaming_config(
            spec,
            memory_limit_bytes=memory_limit_bytes,
            max_live_kv_tokens=max_live_kv_tokens,
            runtime_reserve_bytes=runtime_reserve_bytes,
            expert_cache_limit_bytes=expert_cache_limit_bytes,
            **config_overrides,
        )
    buffer_allocator = _component_bank_allocator_for(
        config, spec, artifact_root, resolved_manifest, loaded_manifest
    )
    return ExpertStreamingRuntime.open(
        artifact_root,
        resolved_manifest,
        config,
        spec=spec,
        buffer_allocator=buffer_allocator,
        additional_resident_bytes=SWA_WINDOW_BYTES,
        apply_memory_cap=apply_memory_cap,
        mx_module=mx_module,
        expert_admission_receipt=admission_receipt,
    )


def _component_bank_allocator_for(
    config: ExpertStreamingConfig,
    spec: ExpertStreamingModelSpec,
    artifact_root: Path,
    manifest_path: Path,
    manifest: ExpertManifest | None = None,
) -> Callable[[int, str], Any] | None:
    """The component-major slot allocator for a ``component-banks`` layout.

    ``ExpertStreamingRuntime.open`` allocates a routed slot's storage through
    the ``buffer_allocator`` it is handed; when none is supplied it falls back
    to a raw ``bytearray`` (``mtplx/expert_slots.py`` -- ``buffer_allocator or
    (lambda size, _label: bytearray(size))``).  The *direct-slot* dispatch reads
    that byte buffer fine (``_component_array``/``_run_q4_expert`` in
    ``mtplx/models/expert_mlx.py`` treat a non-``mx.array`` buffer as raw
    record bytes), but the *component-banks* dispatch requires each binding's
    buffer to be an ``MlxComponentSlot`` backed by a gather-qmm-ready
    ``MlxComponentBank``: it reaches for ``binding.buffer.bank``
    (``mtplx/models/expert_mlx.py`` ``evaluate_component_bindings``), which a
    bytearray does not have -- the exact ``'bytearray' object has no attribute
    'bank'`` failure this loader's separate open entry hit under the P1.7 gate.

    The production serve path builds this allocator in ``mtplx/runtime.py`` for
    the same layout; ``open_deepseek_v41_runtime`` is a *second* runtime-open
    entry (the P1.7 streamed==resident gate and the CPU end-to-end proof), so it
    must wire the same allocator.  For any other layout (the default
    ``direct-slots``) ``None`` is returned and the bytearray fallback stands.

    The plan handed to the allocator is built the same way
    ``ExpertStreamingRuntime.open`` builds its own (island placement resolved,
    the SWA window priced as ``additional_resident_bytes``, the resident-quant
    discounts applied, mixed-official per-layer record sizes when applicable),
    so the allocator's per-bank capacities match the slot pool's plan exactly.
    """

    if config.slot_layout != "component-banks":
        return None

    from ..expert_runtime import (
        proj_quant_plan_discount,
        proj_requant_plan_discount,
        resolve_island_placement,
    )
    from .expert_mlx import make_mlx_component_bank_allocator

    if manifest is None:
        manifest = load_expert_manifest(manifest_path)
    resolved_config = resolve_island_placement(config, artifact_root, spec=spec)
    plan = resolved_config.memory_plan(
        spec,
        additional_resident_bytes=SWA_WINDOW_BYTES,
        resident_discount_bytes=(
            proj_quant_plan_discount(manifest, resolved_config.proj_quant)
            + proj_requant_plan_discount(manifest, resolved_config.proj_requant)
        ),
        layer_record_bytes=(
            manifest.record_bytes_by_layer() if spec.is_mixed_official else None
        ),
    )
    return make_mlx_component_bank_allocator(plan, spec, manifest)


def load_text_only_resident_arrays(
    root: Path | str,
    manifest: ExpertManifest,
    *,
    mx_module: Any | None = None,
    partition: TextResidentPartition | None = None,
) -> dict[str, Any]:
    """Lazily load ONLY the text-only resident tensors from the q8 shards.

    Mirrors :func:`mtplx.resident_loader.load_resident_arrays` but iterates the
    text-only subset (vision/aligner/image and ``mtp.*`` skipped) and asserts
    coverage of that subset, not of the full manifest.  Routed expert arrays
    returned by ``mx.load`` stay lazy and are dropped unmaterialized.
    """

    if mx_module is None:
        try:
            import mlx.core as mx
        except Exception as exc:  # pragma: no cover - environment guard
            raise ResidentLoadError(
                f"MLX is required for resident loading: {exc}"
            ) from exc
    else:
        mx = mx_module
    if partition is None:
        partition = partition_text_residents(manifest)
    artifact_root = Path(root).resolve()
    by_shard: dict[str, list[ResidentTensor]] = {}
    for tensor in partition.kept:
        by_shard.setdefault(tensor.shard, []).append(tensor)
    selected: dict[str, Any] = {}
    for shard_name, expected_tensors in sorted(by_shard.items()):
        shard_path = resolve_artifact_member(artifact_root, shard_name)
        try:
            loaded = mx.load(str(shard_path), format="safetensors")
        except Exception as exc:
            raise ResidentLoadError(
                f"could not lazily load {shard_name}: {exc}"
            ) from exc
        if not isinstance(loaded, dict):
            raise ResidentLoadError(f"MLX returned a non-dictionary for {shard_name}")
        for expected in expected_tensors:
            try:
                value = loaded[expected.tensor]
            except KeyError as exc:
                raise ResidentLoadError(
                    f"resident tensor {expected.tensor} is missing from {shard_name}"
                ) from exc
            shape = tuple(int(dimension) for dimension in value.shape)
            if shape != expected.shape:
                raise ResidentLoadError(
                    f"resident tensor {expected.tensor} shape {shape} != {expected.shape}"
                )
            dtype = _dtype_name(value)
            if dtype != expected.dtype:
                raise ResidentLoadError(
                    f"resident tensor {expected.tensor} dtype {dtype} != {expected.dtype}"
                )
            if int(value.nbytes) != expected.length:
                raise ResidentLoadError(
                    f"resident tensor {expected.tensor} bytes {value.nbytes} != {expected.length}"
                )
            if expected.tensor in selected:
                raise ResidentLoadError(f"duplicate resident tensor {expected.tensor}")
            selected[expected.tensor] = value
        del loaded
    if len(selected) != partition.kept_count:
        raise ResidentLoadError("text-only resident allowlist was not loaded completely")
    return selected


def construct_deepseek_v41_resident_model(
    root: Path | str,
    runtime: ExpertStreamingRuntime,
    *,
    config: dict[str, Any] | None = None,
    engram_bank_path: Path | str | None = None,
    mx_module: Any | None = None,
    model_class_resolver: Callable[[], tuple[type, type]] | None = None,
    switch_binder: Callable[[Any, Any], int] | None = None,
    strict: bool = True,
    with_mtp: bool | None = None,
) -> ResidentModel:
    """Instantiate, bind, and load the DeepSeek-V4.1 model.

    The DeepSeek-V4.1 analogue of
    :func:`mtplx.resident_loader.construct_resident_model`, with two additions:
    the model constructor receives the engram bank path, and only the kept
    residents are materialized.  ``mtplx/resident_loader.py`` delegates here for
    ``model_type == "deepseek_v41"``.

    ``with_mtp`` (worker W23) selects the opt-in DSpark head build: ``None`` auto-
    detects (:func:`resolve_with_mtp` -- config declares MTP stages and the
    manifest ships ``mtp.*`` residents, ``MTPLX_DSV41_MTP`` overriding), so the MTP
    artifact loads its head (mtp.* residents kept + mapped, backbone experts still
    streamed) ready for ``--generation-mode mtp``, while phase-1 AR artifacts are
    unchanged.  The backbone switch binder (``bind_streamed_switches``) walks only
    ``model.model.layers``, so the DSpark head's 128 resident mxfp4 experts stay
    resident."""

    artifact_root = Path(root).resolve()
    if config is None:
        try:
            from mlx_lm.utils import load_config

            config = load_config(artifact_root)
        except Exception as exc:
            raise ResidentLoadError(f"could not load model config: {exc}") from exc
    config = dict(config)
    model_type = str(config.get("model_type") or "")
    if model_type != MODEL_TYPE:
        raise ResidentLoadError(
            f"deepseek_v41 loader received model_type={model_type!r}; "
            f"expected {MODEL_TYPE!r}"
        )
    resolver = model_class_resolver or deepseek_v41_model_classes
    model_class, args_class = resolver()
    if engram_bank_path is None:
        engram_bank_path = engram_bank_path_for(artifact_root)
    resolved_with_mtp = resolve_with_mtp(config, runtime.manifest, with_mtp)
    try:
        model_args = args_class.from_dict(config)
        # The artifact's config ``quantization`` block selects the resident codec
        # (affine q8 gs64 for the original artifact; a native float mode --
        # mxfp8/mxfp4/nvfp4 -- for the exact-repack artifact). Passed through so
        # the model's nn.quantize matches the residents on disk for a strict load.
        model = model_class(
            model_args,
            engram_bank_path=engram_bank_path,
            quantization=config.get("quantization"),
            mtp=resolved_with_mtp,
        )
    except Exception as exc:
        raise ResidentLoadError(f"could not construct deepseek_v41 model: {exc}") from exc

    from .expert_mlx import bind_streamed_switches

    try:
        bound = (switch_binder or bind_streamed_switches)(model, runtime)
    except Exception as exc:
        raise ResidentLoadError(
            f"could not bind streamed expert layers: {exc}"
        ) from exc
    if bound != runtime.spec.routed_layer_count:
        raise ResidentLoadError(
            f"bound {bound} sparse layers; expected {runtime.spec.routed_layer_count}"
        )

    partition = partition_text_residents(runtime.manifest, with_mtp=resolved_with_mtp)
    weights = load_text_only_resident_arrays(
        artifact_root, runtime.manifest, mx_module=mx_module, partition=partition
    )
    try:
        if hasattr(model, "sanitize"):
            weights = model.sanitize(weights)
        model.eval()
        model.load_weights(list(weights.items()), strict=strict)
    except Exception as exc:
        raise ResidentLoadError(f"resident parameter validation failed: {exc}") from exc

    if mx_module is None:
        import mlx.core as mx
    else:
        mx = mx_module
    try:
        parameters = model.parameters()
        mx.eval(parameters)
    except Exception as exc:
        raise ResidentLoadError(f"resident parameter evaluation failed: {exc}") from exc

    report = ResidentLoadReport(
        shard_count=len({tensor.shard for tensor in partition.kept}),
        tensor_count=partition.kept_count,
        raw_tensor_bytes=partition.kept_bytes,
        evaluated_parameter_count=sum(1 for _ in _flatten(parameters)),
        bound_sparse_layers=bound,
        strict=strict,
    )

    # Attach the real Engram hooks (layers 1 and 14) end to end when the artifact
    # ships both the engram bank/manifest and the resident projection sidecar (W4).
    # The model's ``attach_engram`` owns which layers are engram layers and how
    # ``make_cache`` hands each sequence its own history; the loader only supplies
    # the on-disk path.  Done AFTER the resident report so its parameter count
    # stays the 1,616 text residents.
    engram_layer_ids: tuple[int, ...] = ()
    if engram_bank_path is not None:
        engram_dir = Path(engram_bank_path)
        has_manifest = (engram_dir / "engram-manifest.json").is_file()
        has_sidecar = (engram_dir / "engram-residents.safetensors").is_file()
        if has_manifest and has_sidecar:
            try:
                engram_layer_ids = tuple(
                    model.attach_engram(
                        engram_dir, cache_bytes=resolve_engram_cache_bytes()
                    )
                )
            except Exception as exc:
                raise ResidentLoadError(
                    f"could not attach engram from {engram_dir}: {exc}"
                ) from exc

    setattr(model, "_mtplx_expert_runtime", runtime)
    setattr(model, "_mtplx_resident_load_report", report.as_dict())
    setattr(model, "_mtplx_engram_bank_path", str(engram_bank_path) if engram_bank_path else None)
    setattr(model, "_mtplx_engram_layer_ids", engram_layer_ids)
    return ResidentModel(model=model, config=config, report=report)


def _flatten(value: Any):
    if isinstance(value, dict):
        for child in value.values():
            yield from _flatten(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _flatten(child)
    else:
        yield value


def load_deepseek_v41_streaming(
    root: Path | str,
    *,
    memory_limit_bytes: int,
    max_live_kv_tokens: int,
    runtime_reserve_bytes: int = DEFAULT_RUNTIME_RESERVE_BYTES,
    receipt_root: Path | str | None = None,
    admit: bool = True,
    admission_receipt: Mapping[str, Any] | None = None,
    spec: ExpertStreamingModelSpec | None = None,
    expert_cache_limit_bytes: int | None = None,
    apply_memory_cap: bool = True,
    mx_module: Any | None = None,
    strict: bool = True,
    with_mtp: bool | None = None,
    **config_overrides: Any,
) -> ResidentModel:
    """End-to-end serve entry: admit, open the runtime, construct the model.

    Mirrors the hy3 serve flow (admission -> ``ExpertStreamingRuntime.open`` ->
    resident construction).  When ``admit`` is set and no receipt is supplied,
    the real one-time admission runs via
    :func:`mtplx.expert_admission.ensure_expert_admitted`, which validates the
    manifest against the pinned descriptor and writes a revision/digest-bound
    receipt.  NOTE: that identity check compares the manifest's source_repo/
    source_revision against the spec's quant_model/quant_revision; the shipped
    artifact carries pre-publish ``local/...`` identity, so production admission
    fails until the manifest is rebuilt with the HF identity (see
    docs/deepseek-v41/W3_REPORT.md).  Callers/tests may inject a pre-built
    ``admission_receipt`` (and/or a source-rebased ``spec``) to run the rest of
    the pipeline in the meantime.
    """

    artifact_root = Path(root).resolve()
    receipt: Mapping[str, Any] | None = admission_receipt
    if admit and receipt is None:
        from ..expert_admission import ensure_expert_admitted

        receipt = ensure_expert_admitted(artifact_root, receipt_root=receipt_root)
    runtime = open_deepseek_v41_runtime(
        artifact_root,
        memory_limit_bytes=memory_limit_bytes,
        max_live_kv_tokens=max_live_kv_tokens,
        runtime_reserve_bytes=runtime_reserve_bytes,
        spec=spec,
        expert_cache_limit_bytes=expert_cache_limit_bytes,
        admission_receipt=receipt,
        apply_memory_cap=apply_memory_cap,
        mx_module=mx_module,
        **config_overrides,
    )
    try:
        return construct_deepseek_v41_resident_model(
            artifact_root,
            runtime,
            mx_module=mx_module,
            strict=strict,
            with_mtp=with_mtp,
        )
    except Exception:
        runtime.close()
        raise
