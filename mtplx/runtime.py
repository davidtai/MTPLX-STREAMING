"""High-level MTPLX runtime loading primitives."""

from __future__ import annotations

import hashlib
import inspect as py_inspect
import json
import logging
import os
import re
import subprocess
import sys
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from .artifacts import (
    inspect_model,
    load_config,
    mtp_weights_present_on_disk,
    text_config,
)
from .backends.registry import ARCHITECTURE_DECLARED_MODULES
from .mtp_adapters import (
    install_saved_mtp_lora_adapter,
    merge_installed_mtp_lora_adapters,
    mtp_adapter_depth,
)
from .mtp_patch import MTPContract, inject_mtp_support, validate_mtp_support

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .a3b_compiled_target_prefix import A3BCompiledTargetPrefixFactory


def _detect_total_system_memory_bytes() -> int | None:
    try:
        import psutil

        total = int(psutil.virtual_memory().total)
        if total > 0:
            return total
    except Exception:
        pass
    if sys.platform == "darwin":
        try:
            total = int(
                subprocess.check_output(
                    ["/usr/sbin/sysctl", "-n", "hw.memsize"],
                    text=True,
                ).strip()
            )
            if total > 0:
                return total
        except Exception:
            pass
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        total = page_size * pages
        return total if total > 0 else None
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _preflight_laguna_system_memory(config: dict[str, Any]) -> None:
    if not _is_laguna_s_2_1_mlx_4bit_config(config):
        return
    from .models.laguna_config import LAGUNA_S_2_1_MIN_RESIDENT_BYTES

    system_reserve = 16 * 1024**3
    total = _detect_total_system_memory_bytes()
    if total is None or total >= LAGUNA_S_2_1_MIN_RESIDENT_BYTES + system_reserve:
        return
    required = LAGUNA_S_2_1_MIN_RESIDENT_BYTES + system_reserve
    raise RuntimeError(
        "Laguna-S-2.1 requires at least "
        f"{required / 1024**3:.1f} GiB unified memory "
        "for weights, runtime headroom, and the system reserve"
    )


def _streamed_mtp_backend(model_key: str, precision: str) -> str:
    """Resolve the strict external MTP adapter before model allocation."""

    support = {
        "hy3-q4": ("hy3", {"bf16", "q4"}),
        "hy3-expert-only-q4": ("hy3", {"bf16"}),
        "hy3-expert-q2": ("hy3", {"bf16"}),
        "hy3-expert-oq2e": ("hy3", {"bf16"}),
        # The Q4 head sibling (issue #100) is lane-agnostic: it is selectable
        # for every GLM streamed lane: q2 and q1t have the same resident trunk,
        # router, and layer-78 contract; only routed record storage differs.
        "glm52-expert-q2": ("glm52", {"bf16", "q4"}),
        "glm52-expert-q1t": ("glm52", {"bf16", "q4"}),
        "glm52-q4": ("glm52", {"bf16", "q4"}),
    }
    selected = support.get(str(model_key))
    if selected is None:
        raise RuntimeError(f"streamed MTP is not supported for model key {model_key!r}")
    backend, precisions = selected
    if precision not in precisions:
        if precisions == {"bf16"}:
            raise RuntimeError(
                f"streamed MTP for {model_key!r} requires the validated BF16 head"
            )
        raise RuntimeError(
            f"streamed MTP precision {precision!r} is not supported for {model_key!r}"
        )
    return backend


@dataclass
class MTPLXRuntime:
    model: Any
    tokenizer: Any
    model_path: Path
    mtp_enabled: bool
    contract: MTPContract
    mtp_adapter_path: Path | None = None
    mtp_adapter_metadata: dict[str, Any] | None = None
    mtp_adapter_merge_report: dict[str, Any] | None = None
    deepseek_v4_o_lora_report: dict[str, Any] | None = None
    deepseek_v4_attn_proj_wide_m3_report: dict[str, Any] | None = None
    deepseek_v4_attention_island_report: dict[str, Any] | None = None
    a3b_compiled_target_prefix_factory: A3BCompiledTargetPrefixFactory | None = None
    a3b_whole_moe_installed: bool = False
    qwen4_relaxed_draft_ties: bool = False
    qwen_row_owned_router_report: dict[str, Any] = field(default_factory=dict)
    # Expert-streaming (SSD-MoE) wiring. ``expert_streaming`` holds the owning
    # ExpertStreamingRuntime for a streamed checkpoint and is None for every
    # fully-resident (upstream) load, which keeps the streaming methods below
    # inert and preserves upstream behavior.
    expert_streaming: Any | None = None
    resident_load_report: dict[str, Any] | None = None
    _a3b_whole_moe_request_preflights: dict[str, dict[str, Any]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _a3b_whole_moe_request_geometry_keys: dict[
        tuple[int, str, str], str
    ] = field(default_factory=dict, init=False, repr=False)
    diagnostic_counters: dict[str, int] = field(default_factory=dict)
    _forward_ar_supports_emit_logits: bool | None = field(default=None, init=False, repr=False)
    _forward_ar_supports_logits_keep: bool | None = field(default=None, init=False, repr=False)
    _plain_ar_decode: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # The promoted streamed lane has no construction-proven compiled
        # profile. Install its eager target route once so decode never enters
        # compiled eligibility or an eligible-or-eager fallback in the hot
        # path. Fully resident runtimes retain the upstream opt-in route.
        if self.expert_streaming is None:
            self._plain_ar_decode = self._resident_plain_ar_decode
        else:
            self._plain_ar_decode = self._streamed_plain_ar_decode

    def _count(self, key: str, amount: int = 1) -> None:
        self.diagnostic_counters[key] = int(self.diagnostic_counters.get(key, 0)) + int(amount)

    @staticmethod
    def _sequence_len(input_ids: Any) -> int:
        shape = getattr(input_ids, "shape", ())
        if len(shape) >= 2:
            return int(shape[1])
        if shape:
            return int(shape[0])
        return 1

    def _forward_ar_capabilities(self) -> tuple[bool, bool]:
        if (
            self._forward_ar_supports_emit_logits is None
            or self._forward_ar_supports_logits_keep is None
        ):
            try:
                params = py_inspect.signature(self.model.__call__).parameters
            except Exception:
                params = {}
            accepts_kwargs = any(
                param.kind == py_inspect.Parameter.VAR_KEYWORD
                for param in params.values()
            )
            patched_kwargs = bool(self.mtp_enabled and accepts_kwargs)
            self._forward_ar_supports_emit_logits = (
                "emit_logits" in params or patched_kwargs
            )
            self._forward_ar_supports_logits_keep = (
                "logits_keep" in params or patched_kwargs
            )
        return (
            bool(self._forward_ar_supports_emit_logits),
            bool(self._forward_ar_supports_logits_keep),
        )

    def embed_tokens(self, input_ids):
        """Embed token ids with the text model's embedding table."""

        text_model = getattr(self.model, "language_model", self.model)
        return text_model.model.embed_tokens(input_ids)

    def _streamed_plain_ar_decode(self, input_ids, cache, _kwargs):
        return self.model(input_ids, cache=cache)

    def _resident_plain_ar_decode(self, input_ids, cache, kwargs):
        compiled = self._compiled_ar_forward(cache)
        if compiled is not None:
            # Preserve the upstream fully resident diagnostic. Streamed
            # runtimes never install this callable.
            self._count("compiled_forward_calls")
            return compiled(input_ids, cache)
        if not kwargs:
            return self.model(input_ids, cache=cache)
        return self.model(
            input_ids,
            cache=cache,
            return_hidden=False,
            **kwargs,
        )

    def _expert_routing_context(self, input_ids: Any):
        if self.expert_streaming is None:
            return nullcontext()
        from .attention_context import current_attention_phase
        from .expert_streaming import RoutingPhase
        from .models.expert_mlx import expert_routing_phase

        attention = current_attention_phase()
        if attention == "prefill":
            # A one-token prefill tail chunk is still prefill traffic: the
            # width heuristic below would classify it as decode and pollute
            # the persistent decode hot set.
            return expert_routing_phase(RoutingPhase.PREFILL)
        if attention in {"ar_decode", "decode_verify", "postcommit"}:
            # MTP verify batches are decode traffic regardless of width.
            return expert_routing_phase(RoutingPhase.DECODE)

        decode_width = 1
        if self.mtp_enabled:
            # MTP verify batches are decode traffic: routing them as prefill
            # would stop the persistent decode hot set from ever training
            # once speculation is on.  With MTP off this stays exactly the
            # historical single-token decode classification.
            decode_width = max(
                decode_width,
                int(getattr(self.model, "mtp_verify_width", 1)),
            )
        phase = (
            RoutingPhase.PREFILL
            if self._sequence_len(input_ids) > decode_width
            else RoutingPhase.DECODE
        )
        return expert_routing_phase(phase)

    def forward_ar(
        self,
        input_ids,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        emit_logits: bool = True,
        logits_keep: int | None = None,
        input_embeddings=None,
    ):
        self._count("forward_ar_hidden_calls" if return_hidden else "forward_ar_plain_calls")
        if not self.mtp_enabled and return_hidden:
            raise RuntimeError("return_hidden requires an MTP-patched runtime")
        if input_embeddings is not None and not self.mtp_enabled:
            raise RuntimeError("vision splice requires the MTP-patched runtime")
        kwargs = {}
        if hidden_variant is not None:
            kwargs["hidden_variant"] = hidden_variant
        if input_embeddings is not None:
            # Vision splice path: the patched text model takes the rows
            # directly; ids still travel for mask construction.
            kwargs["input_embeddings"] = input_embeddings
        supports_emit_logits, supports_logits_keep = self._forward_ar_capabilities()
        if supports_emit_logits:
            kwargs["emit_logits"] = bool(emit_logits)
        elif not emit_logits:
            self._count("forward_ar_emit_logits_unsupported")
        if logits_keep is not None and supports_logits_keep:
            kwargs["logits_keep"] = int(logits_keep)
        elif logits_keep is not None:
            self._count("forward_ar_logits_keep_unsupported")
        sequence_len = self._sequence_len(input_ids)
        if bool(emit_logits) or not supports_emit_logits:
            if logits_keep is not None and supports_logits_keep:
                emitted = min(sequence_len, max(1, int(logits_keep)))
            else:
                emitted = sequence_len
            self._count("logits_tokens_emitted", emitted)
            if emitted == 1:
                self._count("final_logits_tokens_emitted", 1)
            else:
                self._count("full_logits_tokens_emitted", emitted)
        with self._expert_routing_context(input_ids):
            # kwargs == {"emit_logits": True} is semantically the plain call —
            # MTP-patched wrappers advertise emit_logits via **kwargs, so on MTP
            # runtimes the bare-kwargs case never occurs and the compiled hook
            # must accept the default-emit form too.
            plain_call = not kwargs or (
                set(kwargs) == {"emit_logits"} and kwargs["emit_logits"] is True
            )
            if not return_hidden and hidden_variant is None and plain_call:
                # Decode-only (seq_len == 1). Prefill is multi-token over an
                # unprimed cache: seeding the compiled graph from its None KV
                # leaves throws, and its shape differs from a single-token
                # decode step, forcing a retrace. Prefill stays eager. The
                # prebound ``_plain_ar_decode`` route keeps a streamed runtime
                # off the compiled eligibility check entirely (arm A/B proof:
                # ``compiled_forward_calls`` counts only on the resident lane).
                if sequence_len == 1:
                    return self._plain_ar_decode(input_ids, cache, kwargs)
                if not kwargs:
                    return self.model(input_ids, cache=cache)
            return self.model(
                input_ids,
                cache=cache,
                return_hidden=return_hidden,
                **kwargs,
            )

    def _compiled_ar_forward(self, cache):
        """Compiled target forward (MTPLX_COMPILE_AR_FORWARD).

        Kills the per-token Python graph rebuild by tracing the full trunk
        forward once (CompiledARForward, KV state threaded). Applies to
        fully-resident loads with a standard per-layer KV cache; a host-sync
        buried in the model forward surfaces as an error on the first traced
        call rather than silently degrading. Rebuilds per cache identity so a
        new generation gets fresh threaded state. Returns None (the eager
        path) otherwise.
        """
        from .compiled_forward import CompiledARForward, compile_forward_enabled

        if not compile_forward_enabled() or not cache:
            return None
        # An unprimed cache (empty context / first token) has None KV leaves
        # that would crash the compiled graph. Only compile once the cache
        # holds real keys, and only for the plain growable KVCache shape the
        # fixed-buffer conversion understands.
        first = cache[0]
        if getattr(first, "keys", None) is None:
            return None
        if any(
            not hasattr(entry, "keys")
            or not hasattr(entry, "values")
            or not hasattr(entry, "offset")
            for entry in cache
        ):
            return None
        cache_key = id(first)
        if (
            getattr(self, "_compiled_ar", None) is None
            or getattr(self, "_compiled_ar_key", None) != cache_key
        ):
            import os as _os

            reserve = int(_os.environ.get("MTPLX_COMPILE_AR_RESERVE_TOKENS", "4096"))
            self._compiled_ar = CompiledARForward(self.model, reserve_tokens=reserve)
            self._compiled_ar_key = cache_key
        return self._compiled_ar

    def forward_ar_capture(
        self,
        input_ids,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        capture_backend: str | None = None,
    ):
        text_model = getattr(self.model, "language_model", self.model)
        inner = getattr(text_model, "model", None)
        if not (hasattr(inner, "fa_idx") and hasattr(inner, "ssm_idx")):
            # Uniform full-attention model (e.g. hy_v3): every layer is plain
            # causal attention, so there is no GDN/recurrent state to capture
            # and forward_with_gdn_capture's hybrid layout (fa_idx/ssm_idx,
            # layer.is_linear) does not exist. The verify forward is just the
            # plain AR forward; commit_captured_prefix with empty captures
            # commits by trimming the standard (trimmable) KV caches, which is
            # the correct prefix commit for pure-attention layers.
            self._count("forward_ar_capture_plain_attention_calls")
            if return_hidden:
                logits, hidden = self.forward_ar(
                    input_ids,
                    cache=cache,
                    return_hidden=True,
                    hidden_variant=hidden_variant,
                )
                return logits, hidden, {}
            logits = self.forward_ar(input_ids, cache=cache)
            return logits, {}

        from .gdn_capture import forward_with_gdn_capture

        with self._expert_routing_context(input_ids):
            return forward_with_gdn_capture(
                self.model,
                input_ids,
                cache=cache,
                return_hidden=return_hidden,
                hidden_variant=hidden_variant,
                capture_backend=capture_backend,
            )

    def _forward_ar_capture_a3b_postconv(
        self,
        input_ids,
        *,
        cache,
        hidden_variant: str | None,
        postconv_implementations: tuple[Callable[..., Any], ...],
    ):
        from .gdn_capture import forward_with_a3b_gdn_postconv_capture

        return forward_with_a3b_gdn_postconv_capture(
            self.model,
            input_ids,
            cache=cache,
            hidden_variant=hidden_variant,
            postconv_implementations=postconv_implementations,
        )

    def draft_mtp(
        self,
        hidden_states,
        next_token_ids,
        mtp_cache=None,
        concat_order: str | None = None,
        return_hidden: bool = False,
        mtp_hidden_variant: str | None = None,
        mtp_depth: int | None = None,
        position_offset: int | None = None,
    ):
        if not self.mtp_enabled:
            raise RuntimeError("MTP is not enabled for this runtime")
        self._count("draft_mtp_calls")
        resolved_hidden_variant = (
            self.contract.hidden_variant
            if mtp_hidden_variant in {None, "auto", "contract"}
            else str(mtp_hidden_variant)
        )
        resolved_concat_order = (
            self.contract.concat_order if concat_order in {None, "auto", "contract"} else concat_order
        )
        with mtp_adapter_depth(self.model, mtp_depth):
            kwargs = {
                "mtp_cache": mtp_cache,
                "concat_order": resolved_concat_order,
                "return_hidden": return_hidden,
                "mtp_hidden_variant": resolved_hidden_variant,
                "position_offset": position_offset,
            }
            try:
                params = py_inspect.signature(self.model.mtp_forward).parameters
            except Exception:
                params = {}
            if "mtp_depth" in params:
                kwargs["mtp_depth"] = mtp_depth
            return self.model.mtp_forward(hidden_states, next_token_ids, **kwargs)

    def update_mtp_cache(
        self,
        hidden_states,
        next_token_ids,
        mtp_cache=None,
        concat_order: str | None = None,
        mtp_hidden_variant: str | None = None,
        position_offset: int | None = None,
        input_embeddings=None,
    ):
        if not self.mtp_enabled:
            raise RuntimeError("MTP is not enabled for this runtime")
        self._count("update_mtp_cache_calls")
        resolved_hidden_variant = (
            self.contract.hidden_variant
            if mtp_hidden_variant in {None, "auto", "contract"}
            else str(mtp_hidden_variant)
        )
        resolved_concat_order = (
            self.contract.concat_order if concat_order in {None, "auto", "contract"} else concat_order
        )
        update = getattr(self.model, "mtp_update_cache", None)
        if update is not None:
            try:
                params = py_inspect.signature(update).parameters
            except Exception:
                params = {}
            accepts_kwargs = any(
                param.kind == py_inspect.Parameter.VAR_KEYWORD
                for param in params.values()
            )
            candidates = {
                "mtp_cache": mtp_cache,
                "concat_order": resolved_concat_order,
                "mtp_hidden_variant": resolved_hidden_variant,
                "position_offset": position_offset,
                "input_embeddings": input_embeddings,
            }
            kwargs = {
                key: value
                for key, value in candidates.items()
                if accepts_kwargs or key in params
            }
            if input_embeddings is not None and "input_embeddings" not in kwargs:
                # Silently dropping the spliced vision rows would rebuild the
                # exact draft-history corruption this parameter fixes (#103).
                raise RuntimeError(
                    "this MTP backend does not accept input_embeddings; "
                    "vision history append is unsupported for it"
                )
            if "mtp_depth" in params:
                kwargs["mtp_depth"] = None
            return update(hidden_states, next_token_ids, **kwargs)
        if input_embeddings is not None:
            raise RuntimeError(
                "mtp_forward fallback does not accept input_embeddings; "
                "vision history append is unsupported for it"
            )
        _logits, hidden = self.model.mtp_forward(
            hidden_states,
            next_token_ids,
            mtp_cache=mtp_cache,
            concat_order=resolved_concat_order,
            return_hidden=True,
            mtp_hidden_variant=resolved_hidden_variant,
            position_offset=position_offset,
        )
        return hidden

    def make_cache(self):
        inner = getattr(self.model, "language_model", self.model)
        if hasattr(inner, "make_cache"):
            cache = inner.make_cache()
        else:
            # Plain mlx-lm models (the generic AR fallback lane) declare no
            # custom cache; mlx-lm's own factory builds their default
            # KVCache list, exactly as mlx_lm.generate would.
            from mlx_lm.models.cache import make_prompt_cache

            cache = make_prompt_cache(inner)
        from .cache_state import (
            configure_owned_recurrent_state_cache,
            configure_tail_owned_attention_kv_cache,
        )

        configure_owned_recurrent_state_cache(cache)
        configure_tail_owned_attention_kv_cache(cache)
        return cache

    def repage_target_prefill_cache(self, cache: Any) -> bool:
        """Install the runtime's decode cache layout after contiguous prefill."""

        from .cache_state import configure_tail_owned_attention_kv_cache

        configure_tail_owned_attention_kv_cache(cache)
        return True

    def make_mtp_cache(self):
        if not self.mtp_enabled:
            raise RuntimeError("MTP is not enabled for this runtime")
        self._count("make_mtp_cache_calls")
        cache = self.model.make_mtp_cache()
        from .cache_state import configure_mtp_attention_kv_cache

        configure_mtp_attention_kv_cache(cache)
        return cache

    def admit_kv_tokens(self, tokens: int):
        """Reserve request KV capacity under the streamed memory plan."""

        if self.expert_streaming is None:
            return nullcontext()
        return self.expert_streaming.admit_kv_tokens(tokens)

    def expert_streaming_snapshot(self) -> dict[str, Any] | None:
        if self.expert_streaming is None:
            return None
        return self.expert_streaming.snapshot()

    def expert_resource_telemetry_snapshot(self) -> dict[str, Any] | None:
        if self.expert_streaming is None:
            return None
        return self.expert_streaming.resource_telemetry_snapshot()

    # Name compatibility alias for the streamed resource telemetry snapshot.
    def resource_telemetry_snapshot(self) -> dict[str, Any] | None:
        return self.expert_resource_telemetry_snapshot()

    def close(self, *, timeout: float | None = None) -> None:
        if self.expert_streaming is not None:
            self.expert_streaming.close(timeout=timeout)


class LagunaARRuntime(MTPLXRuntime):
    """Target-only runtime that preserves Laguna's native cache ownership."""

    def forward_ar(
        self,
        input_ids,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        emit_logits: bool = True,
        logits_keep: int | None = None,
        input_embeddings=None,
    ):
        del return_hidden, hidden_variant
        return self.model(
            input_ids,
            cache=cache,
            input_embeddings=input_embeddings,
            emit_logits=emit_logits,
            logits_keep=logits_keep,
        )

    def make_cache(self):
        inner = getattr(self.model, "language_model", self.model)
        return inner.make_cache()

    def repage_target_prefill_cache(self, cache: Any) -> bool:
        del cache
        return False


# HF class name (as declared in config ``architectures``) -> mlx-lm module
# implementing it. The table lives in backends.registry (single source of
# truth shared with the compatibility verdicts); extend it only with
# verified schema-compatible pairs — an architecture absent there keeps the
# fail-loud unknown-model_type behavior.
_ARCHITECTURE_DECLARED_MODULES = ARCHITECTURE_DECLARED_MODULES


def _install_architectures_declared_module_alias(config: dict[str, Any]) -> bool:
    """Alias ``mlx_lm.models.<model_type>`` to the module implementing the
    checkpoint's declared ``architectures`` class, when mlx-lm has no module
    for the model_type itself.

    ``mlx_lm.utils.load`` resolves the model class from ``model_type`` alone,
    so a schema-compatible checkpoint under a fresh model_type string (the
    Qwen3.6 -> "qwen3_5" precedent, expected again for Qwen3.8) would
    hard-fail even though the checkpoint itself names the implementing class.
    This honors that declaration — transformers' own class resolution works
    the same way — and logs loudly so an alias load is never silent.
    Returns True when an alias was installed.
    """
    import importlib
    import importlib.util

    tcfg = text_config(config)
    model_type = str(config.get("model_type") or tcfg.get("model_type") or "").strip()
    if not model_type:
        return False
    alias_name = f"mlx_lm.models.{model_type}"
    if alias_name in sys.modules:
        return False
    try:
        if importlib.util.find_spec(alias_name) is not None:
            return False  # mlx-lm knows this model_type natively
    except (ImportError, ValueError):
        return False
    architectures: list[str] = []
    for source in (config, tcfg):
        raw = source.get("architectures")
        if isinstance(raw, list):
            architectures.extend(str(item) for item in raw)
    for arch in architectures:
        target = _ARCHITECTURE_DECLARED_MODULES.get(arch)
        if target is None:
            continue
        try:
            module = importlib.import_module(f"mlx_lm.models.{target}")
        except ImportError:
            continue
        sys.modules[alias_name] = module
        logger.warning(
            "[model-alias] model_type %r has no mlx-lm module; loading via the "
            "checkpoint's declared architecture %s (mlx_lm.models.%s)",
            model_type,
            arch,
            target,
        )
        return True
    return False


def _load_impl(
    model_path: Path | str,
    *,
    mtp: bool = True,
    contract: MTPContract | None = None,
    mtp_adapter: Path | str | None = None,
    merge_mtp_adapter: bool = False,
    gemma4_draft_block_size: int | None = None,
    gemma4_target_distribution_mode: str | None = None,
    proj_quant: str | None = None,
    proj_requant: str | None = None,
    expert_streaming_config: Any | None = None,
    expert_manifest: Path | str | None = None,
    mtp_artifacts: Path | str | None = None,
    mtp_precision: str = "bf16",
    expert_admission_receipt: Mapping[str, Any] | None = None,
    _expert_runtime_owner: list[Any] | None = None,
) -> MTPLXRuntime:
    """Load an MLX model and optionally inject native MTP support.

    ``proj_quant`` / ``proj_requant`` (or the ``MTPLX_PROJ_QUANT`` /
    ``MTPLX_PROJ_REQUANT`` environment variables) quantize the trunk
    ``*_proj`` Linears at load time — see :mod:`mtplx.proj_quant`. Applied
    to the trunk only, before MTP injection, so a draft head's precision is
    never reduced. Streamed loads encode resident projection precision in
    ``expert_streaming_config`` so the memory plan and installed layout agree
    before allocation.

    Supplying ``expert_streaming_config`` and ``expert_manifest`` (together)
    routes construction through the SSD-resident expert-streaming path.
    ``mtp_artifacts`` / ``mtp_precision`` select the streamed external draft
    head; ``expert_admission_receipt`` pins the admitted routed-expert banks.
    ``_expert_runtime_owner`` is a private out-parameter used by :func:`load`
    to transfer streamed-runtime ownership only on success; it defaults to a
    fresh list so every direct caller keeps working unchanged.
    """
    if _expert_runtime_owner is None:
        _expert_runtime_owner = []
    path = Path(model_path)
    from .hy3_mtp_patch import HY3_MTP_PRECISIONS

    if mtp_precision not in HY3_MTP_PRECISIONS:
        raise ValueError(
            f"mtp_precision must be one of {HY3_MTP_PRECISIONS}; got {mtp_precision!r}"
        )
    streaming_requested = (
        expert_streaming_config is not None or expert_manifest is not None
    )
    if (expert_streaming_config is None) != (expert_manifest is None):
        raise ValueError(
            "expert_streaming_config and expert_manifest must be supplied together"
        )
    if expert_admission_receipt is not None and not streaming_requested:
        raise ValueError(
            "expert_admission_receipt applies to streamed checkpoints only"
        )
    if streaming_requested and (proj_quant is not None or proj_requant is not None):
        raise ValueError(
            "streamed loads configure proj_quant/proj_requant through "
            "expert_streaming_config so the construction-time memory plan "
            "matches the installed resident layout"
        )
    if mtp_artifacts is not None and not streaming_requested:
        raise ValueError(
            "mtp_artifacts applies to streamed checkpoints only; non-streamed "
            "models carry their own MTP weights"
        )
    from .gemma4_pair import resolve_gemma4_pair_paths

    gemma4_pair = resolve_gemma4_pair_paths(path)
    if gemma4_pair is not None:
        if streaming_requested:
            raise ValueError("expert streaming does not support Gemma assistant pairs")
        if mtp:
            from .backends.gemma4_assistant import (
                DEFAULT_DRAFT_BLOCK_SIZE,
                Gemma4AssistantRuntimeConfig,
                load_gemma4_assistant_pair,
            )

            metadata = gemma4_pair["metadata"]
            benchmark = (
                metadata.get("benchmark") if isinstance(metadata, dict) else {}
            )
            draft_block_size = DEFAULT_DRAFT_BLOCK_SIZE
            if isinstance(benchmark, dict):
                try:
                    draft_block_size = int(
                        benchmark.get("best_block_size") or draft_block_size
                    )
                except (TypeError, ValueError):
                    draft_block_size = DEFAULT_DRAFT_BLOCK_SIZE
            if gemma4_draft_block_size is not None:
                draft_block_size = int(gemma4_draft_block_size)
            runtime = load_gemma4_assistant_pair(
                Gemma4AssistantRuntimeConfig.from_paths(
                    target_model_path=gemma4_pair["target_model"],
                    assistant_model_path=gemma4_pair["assistant_model"],
                    draft_block_size=draft_block_size,
                    target_distribution_mode=gemma4_target_distribution_mode,
                )
            )
            runtime.model_path = path
            runtime.path = path
            runtime.bundle_path = path
            return runtime
        path = Path(gemma4_pair["target_model"])
    config = load_config(path)
    from .a3b_whole_moe import validate_a3b_whole_moe_load_options

    validate_a3b_whole_moe_load_options(
        mtp_adapter=mtp_adapter,
        merge_mtp_adapter=merge_mtp_adapter,
    )
    if mtp and _is_laguna_s_2_1_mlx_4bit_config(config):
        raise ValueError(
            "Laguna-S-2.1 has no native MTP head; "
            "load it with mtp=False (CLI: --no-mtp)."
        )
    _preflight_laguna_system_memory(config)
    from .step3p5_mtp_patch import is_step3p5_mtp_config
    from .qwen3_5_mtp_patch import (
        install_qwen3_5_mtp_trunk_shim,
        is_qwen3_5_mtp_config,
    )

    # Qwen3.5-MoE MTP exports carry model_type ``qwen3_5_mtp`` (no mlx-lm module);
    # the trunk is a vanilla ``qwen3_5_moe``. Alias it so the trunk loads.
    if is_qwen3_5_mtp_config(config):
        install_qwen3_5_mtp_trunk_shim()

    # hy_v3 has no model class in any released mlx-lm; register the vendored
    # one (kept MTP head) before mlx_lm.utils.load resolves the model type.
    from .hy_v3_mtp_patch import install_hy_v3_model_shim, is_hy_v3_config

    if is_hy_v3_config(config):
        install_hy_v3_model_shim()

    # A checkpoint whose model_type has no mlx-lm module may still declare the
    # implementing class in ``architectures`` — new Qwen generations reuse the
    # qwen3_5 schema under fresh model_type strings (Qwen3.6 shipped as
    # qwen3_5; vLLM loads Qwen3.8-Max FP8 through the same classes). Honor the
    # checkpoint's own declaration instead of hard-failing the load.
    _install_architectures_declared_module_alias(config)

    if streaming_requested:
        runtime_metadata = _load_runtime_metadata(path)
        contract = (
            (contract or MTPContract())
            .with_runtime_metadata(runtime_metadata, preserve_explicit=True)
            .with_config_defaults(config)
        )
        expert_runtime = None
        resident_load_report = None
        streamed_mtp_backend = None
        streamed_mtp_resident_bytes = 0
        hy3_router_incremental_bytes = 0
        mtp_enabled = False
        from .expert_runtime import (
            ExpertStreamingConfig,
            ExpertStreamingConfigurationError,
            ExpertStreamingRuntime,
            apply_mlx_memory_cap,
            validate_deepseek_v41_runtime_env,
        )
        if not isinstance(expert_streaming_config, ExpertStreamingConfig):
            raise TypeError("expert_streaming_config must be an ExpertStreamingConfig")
        # Native MTP serving can set the allocator cap before runtime.open.
        # Validate here as well as at the direct runtime/benchmark boundary.
        validate_deepseek_v41_runtime_env(expert_streaming_config.model_key)
        from .expert_streaming_models import get_model_spec
        from .models.expert_mlx import (
            make_mlx_component_bank_allocator,
            make_mlx_slot_buffer_allocator,
        )
        from .resident_loader import construct_resident_model

        import mlx.core as mx

        streaming_spec = get_model_spec(expert_streaming_config.model_key)
        # DeepSeek-V4.1 DSpark MTP is a NATIVE in-artifact draft head (worker W23),
        # not an external hy3/glm MTP adapter. When --generation-mode mtp selects
        # it (mtp=True) the loader keeps the mtp.* residents and builds the head,
        # and the head-injection dispatch below publishes it; the external-MTP path
        # (_streamed_mtp_backend, which requires mtp_artifacts) must be skipped.
        # The plan must also PRICE the wired mtp.* residents: mtp_included=True so
        # text_only_resident_discount stops discounting them (+7.95 GB / 7.404 GiB
        # for the mxfp4 artifact -- the DSpark head's ~6.7 GiB active experts plus
        # the remaining mtp.* residents partition_text_residents(with_mtp=True)
        # keeps). The swap flows to the pre-flight plan and open()'s pool plan.
        from .models.deepseek_v41 import is_deepseek_v41_mtp_config

        native_streamed_mtp = bool(mtp) and is_deepseek_v41_mtp_config(config)
        if native_streamed_mtp:
            streaming_spec = replace(streaming_spec, mtp_included=True)
        verified_artifact_context = nullcontext(None)
        if mtp and not native_streamed_mtp:
            streamed_mtp_backend = _streamed_mtp_backend(
                expert_streaming_config.model_key,
                mtp_precision,
            )
            if mtp_artifacts is None:
                raise RuntimeError(
                    "this streamed checkpoint omits its trained MTP layer; pass "
                    "mtp_artifacts=<validated external artifact directory> or "
                    "load with mtp=False"
                )
            if streamed_mtp_backend == "glm52":
                from .glm52_mtp_patch import _validate_glm52_mtp_contract

                _validate_glm52_mtp_contract(contract)
                if mtp_precision == "q4":
                    from .glm52_mtp_artifact import (
                        open_verified_glm52_mtp_layer78_q4,
                    )

                    verified_artifact_context = open_verified_glm52_mtp_layer78_q4(
                        Path(mtp_artifacts), deep=True
                    )
                else:
                    from .glm52_mtp_artifact import (
                        open_verified_glm52_mtp_layer78,
                    )

                    verified_artifact_context = open_verified_glm52_mtp_layer78(
                        Path(mtp_artifacts), deep=True
                    )
            elif streamed_mtp_backend == "hy3":
                from .hy3_mtp_patch import open_verified_hy3_mtp_artifacts

                verified_artifact_context = open_verified_hy3_mtp_artifacts(
                    Path(mtp_artifacts),
                    precision=mtp_precision,
                    expected_revision=streaming_spec.source_revision,
                )
        with verified_artifact_context as verified_streamed_artifact:
            if streamed_mtp_backend == "glm52":
                receipt = verified_streamed_artifact.manifest
                inventory = receipt.get("inventory")
                if not isinstance(inventory, dict):
                    raise RuntimeError("GLM-5.2 MTP manifest inventory is missing")
                payload_bytes = inventory.get("payload_bytes")
                if (
                    isinstance(payload_bytes, bool)
                    or not isinstance(payload_bytes, int)
                    or payload_bytes <= 0
                ):
                    raise RuntimeError(
                        "GLM-5.2 MTP manifest payload byte count is invalid"
                    )
                streamed_mtp_resident_bytes = payload_bytes
            elif streamed_mtp_backend == "hy3":
                streamed_mtp_resident_bytes = verified_streamed_artifact.payload_bytes
                if (
                    isinstance(streamed_mtp_resident_bytes, bool)
                    or not isinstance(streamed_mtp_resident_bytes, int)
                    or streamed_mtp_resident_bytes <= 0
                ):
                    raise RuntimeError("Hy3 MTP artifact payload byte count is invalid")

            if (
                str(config.get("model_type") or "") == "hy_v3"
                and expert_streaming_config.hy3_router_kernel != "stock"
            ):
                from .models.hy3_mlx import (
                    estimate_hy3_router_kernel_incremental_bytes,
                )

                hy3_router_incremental_bytes = (
                    estimate_hy3_router_kernel_incremental_bytes(
                        config,
                        expert_streaming_config.hy3_router_kernel,
                        include_mtp=streamed_mtp_backend == "hy3" and bool(mtp),
                    )
                )
            additional_resident_bytes = (
                streamed_mtp_resident_bytes + hy3_router_incremental_bytes
            )
            if streaming_spec.key.startswith("deepseek-v41-"):
                from .models.deepseek_v41_loader import (
                    deepseek_v41_additional_resident_bytes,
                    deepseek_v41_mtp_layers,
                )

                additional_resident_bytes += deepseek_v41_additional_resident_bytes(
                    mtp_layers=deepseek_v41_mtp_layers(config) if native_streamed_mtp else 0
                )
            plan_kwargs = (
                {"additional_resident_bytes": additional_resident_bytes}
                if additional_resident_bytes
                else {}
            )
            # ExpertStreamingRuntime.open computes the same discount from the
            # manifest itself, so plan_kwargs stays free of it.
            # Resolve a pending island_layer_count BEFORE the pre-flight plan
            # (census-first precedence; open() re-resolves idempotently) —
            # census-only specs previously hit the unresolved-count guard here.
            if expert_streaming_config.island_layer_count is not None:
                from .expert_runtime import resolve_island_placement

                expert_streaming_config = resolve_island_placement(
                    expert_streaming_config, Path(expert_manifest).parent
                )
            preflight_plan_kwargs = dict(plan_kwargs)
            # The pre-flight plan sizes the component-bank slot allocator below,
            # so its resident discount must equal the one ExpertStreamingRuntime.open
            # applies to its own pool plan (proj_quant + proj_requant + the
            # text-only skip); otherwise the allocator's per-layer bank capacity
            # and the runtime slot pool disagree. The text-only term is 0 for any
            # manifest with no MTP/vision residents (hy3/glm), so their plans are
            # unchanged.
            from .expert_manifest import load_expert_manifest
            from .expert_runtime import (
                proj_quant_plan_discount,
                proj_requant_plan_discount,
                text_only_resident_discount,
            )

            _preflight_manifest = load_expert_manifest(expert_manifest)
            _resident_discount = (
                proj_quant_plan_discount(
                    _preflight_manifest,
                    expert_streaming_config.proj_quant,
                )
                + proj_requant_plan_discount(
                    _preflight_manifest,
                    getattr(expert_streaming_config, "proj_requant", None),
                )
                + text_only_resident_discount(_preflight_manifest, streaming_spec)
            )
            if _resident_discount:
                preflight_plan_kwargs["resident_discount_bytes"] = _resident_discount
            if bool(getattr(streaming_spec, "is_mixed_official", False)):
                # Mixed-official has no uniform record size; the preflight gate
                # must see the same manifest-derived per-layer sizes as open()
                # (issue #51 M2, D2).
                from .expert_manifest import load_expert_manifest

                preflight_plan_kwargs["layer_record_bytes"] = load_expert_manifest(
                    expert_manifest
                ).record_bytes_by_layer()
            streaming_plan = expert_streaming_config.memory_plan(
                streaming_spec,
                **preflight_plan_kwargs,
            )
            if not streaming_plan.fits_fixed:
                raise ExpertStreamingConfigurationError(
                    "fixed expert-streaming footprint exceeds limit by "
                    f"{-streaming_plan.unallocated_bytes} bytes"
                )
            prebuilt_glm_mtp = None
            prebuilt_hy3_mtp = None
            if mtp:
                # Materialize external MTP heads before allocating expert-cache
                # banks. Stacking their routed experts has a large transient
                # footprint that can breach an otherwise-valid steady-state plan.
                apply_mlx_memory_cap(streaming_plan, mx_module=mx)
            if streamed_mtp_backend == "glm52":
                from .glm52_mtp_patch import build_glm52_mtp_module
                from .models.glm52_mlx import ModelArgs as Glm52ModelArgs

                prebuilt_glm_mtp = build_glm52_mtp_module(
                    mtp_artifacts,
                    Glm52ModelArgs.from_dict(config),
                    expected_revision=streaming_spec.source_revision,
                    precision=mtp_precision,
                    verified_artifact=verified_streamed_artifact,
                )
            elif streamed_mtp_backend == "hy3":
                from .hy3_mtp_patch import build_hy3_mtp_module
                from .models.hy3_mlx import ModelArgs as Hy3ModelArgs

                prebuilt_hy3_mtp = build_hy3_mtp_module(
                    mtp_artifacts,
                    Hy3ModelArgs.from_dict(config),
                    expected_revision=streaming_spec.source_revision,
                    precision=mtp_precision,
                    shared_kernel=expert_streaming_config.hy3_mtp_shared_kernel,
                    shared_kernel_depth=(
                        expert_streaming_config.hy3_mtp_shared_kernel_depth
                    ),
                    verified_artifacts=verified_streamed_artifact,
                )
            if expert_streaming_config.slot_layout == "component-banks":
                from .expert_manifest import load_expert_manifest

                streaming_manifest = load_expert_manifest(expert_manifest)
                slot_allocator = make_mlx_component_bank_allocator(
                    streaming_plan,
                    streaming_spec,
                    streaming_manifest,
                )
            else:
                slot_allocator = make_mlx_slot_buffer_allocator(
                    streaming_plan, streaming_spec
                )

            if mtp_adapter is not None or merge_mtp_adapter:
                raise RuntimeError("MTP adapters are unavailable for streamed loading")
            expert_runtime = ExpertStreamingRuntime.open(
                path,
                expert_manifest,
                expert_streaming_config,
                spec=streaming_spec,
                buffer_allocator=slot_allocator,
                device_synchronize=mx.synchronize,
                apply_memory_cap=True,
                mx_module=mx,
                expert_admission_receipt=expert_admission_receipt,
                **plan_kwargs,
            )
            _expert_runtime_owner[:] = [expert_runtime]
            try:
                resident = construct_resident_model(
                    path, expert_runtime, config=config, with_mtp=native_streamed_mtp
                )
                model = resident.model
                resident_load_report = dict(
                    getattr(model, "_mtplx_resident_load_report", resident.report.as_dict())
                )
                tokenizer = _load_tokenizer_resilient(path, config)
                if mtp:
                    if native_streamed_mtp:
                        # DeepSeek-V4.1 DSpark native draft head (worker W23): the
                        # head is already built into the model by the mtp=True
                        # resident construct above; there is no external MTP
                        # artifact, so publish the in-model head here (the general
                        # is_deepseek_v41_mtp_config dispatch lives past the
                        # streaming block's `return runtime`, so the streamed lane
                        # must publish it itself).
                        from .models.deepseek_v41 import (
                            inject_deepseek_v41_mtp_support,
                        )

                        mtp_enabled = inject_deepseek_v41_mtp_support(
                            model, path, config, contract
                        )
                    elif streamed_mtp_backend == "hy3":
                        from .hy3_mtp_patch import inject_hy3_streamed_mtp_support

                        mtp_enabled = inject_hy3_streamed_mtp_support(
                            model,
                            mtp_artifacts,
                            config,
                            contract,
                            expected_revision=streaming_spec.source_revision,
                            mtp_precision=mtp_precision,
                            shared_kernel=(
                                expert_streaming_config.hy3_mtp_shared_kernel
                            ),
                            shared_kernel_depth=(
                                expert_streaming_config.hy3_mtp_shared_kernel_depth
                            ),
                            mtp_module=prebuilt_hy3_mtp,
                        )
                    elif streamed_mtp_backend == "glm52":
                        from .glm52_mtp_patch import (
                            inject_glm52_streamed_mtp_support,
                        )

                        mtp_enabled = inject_glm52_streamed_mtp_support(
                            model,
                            mtp_artifacts,
                            config,
                            contract,
                            expected_revision=streaming_spec.source_revision,
                            verified_artifact=verified_streamed_artifact,
                            mtp_module=prebuilt_glm_mtp,
                        )
                    else:
                        raise RuntimeError(
                            f"unresolved streamed MTP backend {streamed_mtp_backend!r}"
                        )
                    if not mtp_enabled or not validate_mtp_support(model):
                        raise RuntimeError(f"streamed MTP injection failed for {path}")
                if (
                    str(config.get("model_type") or "") == "hy_v3"
                    and expert_streaming_config.hy3_router_kernel != "stock"
                ):
                    from .models.hy3_mlx import configure_hy3_router_kernels

                    import os as _os

                    # EXPERIMENT gate (2026-07-21): "all" extends the split-K
                    # kernel to trunk routers at rows==1 (AR decode), which
                    # historically fell back to the stock host path. Default
                    # "mtp" preserves the measured ladder behavior exactly.
                    router_kernel_report = configure_hy3_router_kernels(
                        model,
                        expert_streaming_config.hy3_router_kernel,
                        sigmoid_mode=expert_streaming_config.hy3_router_sigmoid,
                        splitk_m1_scope=_os.environ.get(
                            "MTPLX_HY3_ROUTER_SPLITK_M1", "mtp"
                        ),
                    )
                    actual_incremental = int(
                        router_kernel_report.get("incremental_bytes", -1)
                    )
                    if actual_incremental != hy3_router_incremental_bytes:
                        raise RuntimeError(
                            "Hy3 router prepared-layout bytes do not match "
                            f"admission plan: {actual_incremental} != "
                            f"{hy3_router_incremental_bytes}"
                        )
                    setattr(
                        model,
                        "_mtplx_hy3_router_kernel_report",
                        router_kernel_report,
                    )
                    if isinstance(resident_load_report, dict):
                        resident_load_report["hy3_router_kernel"] = router_kernel_report
            except BaseException:
                expert_runtime.close()
                _expert_runtime_owner.clear()
                raise
        # Common streamed post-construction wiring (the resident-lane tail
        # below is skipped for streamed loads: proj-quant/MTP-injection run
        # inside the branch above, and adapters are refused before open).
        from .attention_split import configure_split_full_attention
        from .native_mlp import configure_native_mlp

        configure_split_full_attention(model)
        configure_native_mlp(model)
        from .nax_verify import install_nax_qlinear_patch, nax_env_enabled

        if nax_env_enabled():
            nax_report = install_nax_qlinear_patch()
            logger.info("[nax-verify] %s", nax_report)
        from .kernel_selfcheck import maybe_run_model_selfcheck

        # Expert-streaming loads pass their spec so the routed expert bank's
        # gather_qmm lane is validated at its own (possibly different) quant
        # format, matching the resident-lane self-check.
        maybe_run_model_selfcheck(
            model,
            expert_spec=getattr(expert_runtime, "spec", None),
        )
        runtime = MTPLXRuntime(
            model,
            tokenizer,
            path,
            mtp_enabled,
            contract,
            expert_streaming=expert_runtime,
            resident_load_report=resident_load_report,
        )
        return runtime

    if is_step3p5_mtp_config(config):
        from mlx_lm.utils import load_model

        tokenizer = _load_tokenizer_resilient(path, config)
        model, _loaded_config = load_model(path)
    else:
        model, tokenizer = _load_base_model(path, config)
    import os as _os

    proj_quant = proj_quant or _os.environ.get("MTPLX_PROJ_QUANT") or None
    proj_requant = proj_requant or _os.environ.get("MTPLX_PROJ_REQUANT") or None
    if proj_quant or proj_requant:
        from .proj_quant import quantize_projections, requantize_projections

        if proj_quant:
            touched = quantize_projections(model, proj_quant)
            logger.info(
                "[proj-quant] quantized %d trunk *_proj modules to %s",
                len(touched), proj_quant,
            )
        if proj_requant:
            touched = requantize_projections(model, proj_requant)
            logger.info(
                "[proj-quant] requantized %d trunk *_proj modules to %s",
                len(touched), proj_requant,
            )
    deepseek_v4_attn_proj_wide_m3_report = None
    if str((config or {}).get("model_type") or "").lower() == "deepseek_v4":
        from .models.deepseek_v4 import configure_deepseek_v4_moe_tail

        configure_deepseek_v4_moe_tail(model, config)
        from .deepseek_v4_attn_proj_wide_m3 import (
            deepseek_v4_attn_proj_wide_m3_enabled,
        )

        if deepseek_v4_attn_proj_wide_m3_enabled():
            from .deepseek_v4_attn_proj_wide_m3 import (
                install_deepseek_v4_attn_proj_wide_m3,
            )

            deepseek_v4_attn_proj_wide_m3_report = (
                install_deepseek_v4_attn_proj_wide_m3(model, config)
            )
            logger.info(
                "[deepseek-v4-attn-proj-wide-m3] %s",
                deepseek_v4_attn_proj_wide_m3_report,
            )
    runtime_metadata = _load_runtime_metadata(path)
    contract = (
        (contract or MTPContract())
        .with_runtime_metadata(runtime_metadata, preserve_explicit=True)
        .with_config_defaults(config)
    )
    mtp_enabled = False
    if mtp:
        from .deepseek_mtp_patch import inject_deepseek_mtp_support, is_deepseek_mtp_config
        from .glm_mtp_patch import inject_glm_mtp_support, is_glm_mtp_config
        from .mimo_mtp_patch import inject_mimo_mtp_support, is_mimo_mtp_config
        from .nemotron_h_mtp_patch import inject_nemotron_h_mtp_support, is_nemotron_h_mtp_config
        from .step3p5_mtp_patch import inject_step3p5_mtp_support
        from .hy_v3_mtp_patch import inject_hy_v3_mtp_support, is_hy_v3_mtp_config
        from .models.deepseek_v4 import (
            inject_deepseek_v4_mtp_support,
            is_deepseek_v4_mtp_config,
        )
        from .models.deepseek_v41 import (
            inject_deepseek_v41_mtp_support,
            is_deepseek_v41_mtp_config,
        )
        from .models.qwen4_exp import (
            inject_qwen4_exp_mtp_support,
            is_qwen4_exp_mtp_config,
        )
        from .qwen3_5_mtp_patch import inject_qwen3_5_mtp_support

        if is_deepseek_v4_mtp_config(config):
            # Native draft head: the block binds through the ordinary load path
            # and the model already carries the runtime surface, so this only
            # publishes it. Placed ahead of is_deepseek_mtp_config defensively --
            # that predicate keys on model_type in {deepseek_v3, deepseek_v32,
            # glm_moe_dsa}, so it cannot match a deepseek_v4 config today, but it
            # is the arm that would build a V3 head if the sets ever overlap.
            mtp_enabled = inject_deepseek_v4_mtp_support(model, path, config, contract)
        elif is_deepseek_v41_mtp_config(config):
            # DeepSeek-V4.1 DSpark native draft head: the head binds through the
            # opt-in mtp=True load path and the model carries the runtime surface,
            # so this only publishes it.  A dedicated arm is required because
            # is_deepseek_v4_mtp_config never matches a V4.1 config and the generic
            # inject_mtp_support builds a qwen3_5 graft, not this native head (W23).
            mtp_enabled = inject_deepseek_v41_mtp_support(model, path, config, contract)
        elif is_nemotron_h_mtp_config(config):
            mtp_enabled = inject_nemotron_h_mtp_support(model, path, config, contract)
        elif is_mimo_mtp_config(config):
            mtp_enabled = inject_mimo_mtp_support(model, path, config, contract)
        elif is_glm_mtp_config(config):
            mtp_enabled = inject_glm_mtp_support(model, path, config, contract)
        elif is_step3p5_mtp_config(config):
            mtp_enabled = inject_step3p5_mtp_support(model, path, config, contract)
        elif is_hy_v3_mtp_config(config):
            mtp_enabled = inject_hy_v3_mtp_support(model, path, config, contract)
        elif is_qwen4_exp_mtp_config(config):
            # Flash-Next native draft head: attach_mtp builds it from the
            # pack's self-describing mtp.safetensors sidecar and publishes
            # the runtime surface on language_model.
            mtp_enabled = inject_qwen4_exp_mtp_support(model, path, config, contract)
        elif is_qwen3_5_mtp_config(config):
            mtp_enabled = inject_qwen3_5_mtp_support(model, path, config, contract)
        elif is_deepseek_mtp_config(config):
            mtp_enabled = inject_deepseek_mtp_support(model, path, config, contract)
        else:
            mtp_enabled = inject_mtp_support(model, path, config, contract)
        if mtp_enabled:
            if not validate_mtp_support(model):
                raise RuntimeError(f"MTP injection failed for {path}")
        elif mtp_weights_present_on_disk(path, config):
            # MTP weights ship with the model but injection could not use
            # them: a genuine failure the operator should see.
            raise RuntimeError(f"MTP injection failed for {path}")
        else:
            # The config declares MTP layers but no MTP weights are present on
            # disk (e.g. a quant conversion that dropped the draft head).
            # Degrade to autoregressive rather than failing the load.
            logger.warning(
                "[MTP] %s declares MTP layer(s) but ships no MTP weights; "
                "serving autoregressive (no speculative draft head).",
                path,
            )
    compiled_target_factory = None
    whole_moe_plan = None
    selfcheck_report = None
    router_report: dict[str, Any] = {}
    # Laguna skips the qwen3-next kernel stack entirely; its own env-gated
    # fused lanes install right before runtime construction below.
    if not _is_laguna_s_2_1_mlx_4bit_config(config):
        from .attention_split import configure_split_full_attention
        from .moe_packed_projections import (
            configure_moe_packed_projections,
            moe_pack_gate_up_enabled,
        )
        from .native_mlp import configure_native_mlp

        configure_split_full_attention(model)
        configure_native_mlp(model)
        from .lfm2_fast import is_lfm2_config, install_lfm2_fast

        # LFM2 (LiquidAI) dense hybrid: bit-exact decode fast-path that fuses
        # the ShortConv sliding window (see mtplx/lfm2_fast.py). Decode-only;
        # prefill and masked steps fall back to stock. Toggle: MTPLX_LFM2_FAST=0.
        if is_lfm2_config(config) and os.environ.get("MTPLX_LFM2_FAST", "1") != "0":
            lfm2_report = install_lfm2_fast(model)
            logger.info("[lfm2-fast] %s", lfm2_report)
        # Construction-time only: replaces the MoE gate/up projections with one
        # packed matmul each. Must run after MTP injection so the draft block's
        # MoE layer is packed too, and after load-coverage validation so the
        # packed parameter tree is never compared against checkpoint keys.
        if moe_pack_gate_up_enabled():
            pack_report = configure_moe_packed_projections(model)
            logger.info("[moe-pack] %s", pack_report)
        # Must run after MTP injection and after load-coverage validation.
        from .proj_fusion import (
            configure_fused_projections,
            fuse_projections_enabled,
        )

        if fuse_projections_enabled():
            fuse_report = configure_fused_projections(model)
            logger.info("[proj-fusion] %s", fuse_report)
        from .nax_verify import install_nax_qlinear_patch, nax_env_enabled

        if nax_env_enabled():
            nax_report = install_nax_qlinear_patch()
            logger.info("[nax-verify] %s", nax_report)
        from .kernels.gdn_blocked_prefill import (
            blocked_prefill_env_enabled,
            install_gdn_blocked_prefill_patch,
        )

        if blocked_prefill_env_enabled():
            gdn_prefill_report = install_gdn_blocked_prefill_patch()
            logger.info("[gdn-blocked-prefill] %s", gdn_prefill_report)
        from .qwen_row_owned_router import (
            install_qwen_row_owned_routers,
            prepare_qwen_row_owned_routers,
        )
        from .a3b_whole_moe import (
            install_a3b_whole_moe,
            prepare_a3b_whole_moe,
            run_a3b_whole_moe_selfcheck,
        )

        from .gdn_capture import (
            install_a3b_gdn_postconv,
            prepare_a3b_gdn_postconv,
        )
        from .a3b_compiled_target_prefix import (
            preflight_a3b_k1_target_prefix_load_graph,
            prepare_a3b_compiled_target_prefix,
        )

        whole_moe_plan = prepare_a3b_whole_moe(model, config=config)
        router_plan = prepare_qwen_row_owned_routers(model, config=config)
        postconv_plan = prepare_a3b_gdn_postconv(model, config=config)
        postconv_factory = None
        from .kernel_selfcheck import maybe_run_model_selfcheck

        selfcheck_report = maybe_run_model_selfcheck(model)
        if whole_moe_plan is not None and router_plan is None:
            from .a3b_whole_moe import A3BWholeMoeConfigError

            raise A3BWholeMoeConfigError(
                "whole-MoE target M2 requires the accepted row-owned router/combine route"
            )
        if router_plan is not None:
            router_report = install_qwen_row_owned_routers(router_plan, selfcheck_report)
            logger.info("[qwen-row-owned-router] %s", router_report)
        if whole_moe_plan is not None:
            selfcheck_report = run_a3b_whole_moe_selfcheck(
                whole_moe_plan,
                selfcheck_report,
            )
        if postconv_plan is not None:
            postconv_factory = install_a3b_gdn_postconv(
                postconv_plan, selfcheck_report
            )
            from .gdn_capture import gdn_postconv_stats

            logger.info("[a3b-gdn-postconv] %s", gdn_postconv_stats())
        compiled_target_factory = prepare_a3b_compiled_target_prefix(
            model,
            config=config,
            gdn_postconv_factory=postconv_factory,
        )
    adapter_path = Path(mtp_adapter) if mtp_adapter is not None else None
    adapter_metadata = None
    adapter_merge_report = None
    if adapter_path is not None:
        if not mtp_enabled:
            raise RuntimeError("MTP adapter requires mtp=True")
        adapter_metadata = install_saved_mtp_lora_adapter(model, adapter_path)
        if merge_mtp_adapter:
            adapter_merge_report = merge_installed_mtp_lora_adapters(model)
    elif merge_mtp_adapter:
        raise RuntimeError("merge_mtp_adapter requires mtp_adapter")
    deepseek_v4_o_lora_report = None
    deepseek_v4_attention_island_report = None
    if str(config.get("model_type") or "").lower() == "deepseek_v4":
        from .models.deepseek_v4 import (
            _o_lora_mode_from_env,
            install_deepseek_v4_o_lora_routes,
        )

        selected_o_lora_mode = _o_lora_mode_from_env()
        # The canonical mixed route hard-validates the exact DeepSeek-V4-Flash
        # topology (43 body layers, rank-1024 Q4/g64 wo_a/wo_b, one dense-BF16
        # MTP block) and refuses anything else. That strictness is correct for
        # the explicit gather_qmm opt-in, but the default "cached" mode must
        # keep loading every DSV4 MTP artifact (8-bit/bf16 user conversions,
        # other group sizes) exactly as v2.4.2 did via the per-module dense
        # route — which is bit-identical on the canonical artifact anyway
        # (test_cached_dequant_is_bit_identical).
        canonical_mixed_route = bool(
            mtp_enabled and selected_o_lora_mode == "gather_qmm"
        )
        if not mtp_enabled:
            # An artifact that declared but did not ship MTP weights already
            # degraded to AR above. It has no dense MTP module to validate or
            # route, so bind the trunk's explicit stock/cached construction.
            selected_o_lora_mode = "cached"
        deepseek_v4_o_lora_report = install_deepseek_v4_o_lora_routes(
            model,
            mode=selected_o_lora_mode,
            canonical_mixed_route=canonical_mixed_route,
        )
        logger.info("[deepseek-v4-o-lora] %s", deepseek_v4_o_lora_report)
        from .deepseek_v4_attention_island import (
            deepseek_v4_attention_island_enabled,
            install_deepseek_v4_attention_island,
        )

        if deepseek_v4_attention_island_enabled():
            deepseek_v4_attention_island_report = (
                install_deepseek_v4_attention_island(model, config)
            )
            logger.info(
                "[deepseek-v4-attention-island] %s",
                deepseek_v4_attention_island_report,
            )
    fused_report: list[dict[str, Any]] = []
    if _is_laguna_s_2_1_mlx_4bit_config(config):
        # Env-gated fused decode paths (MTPLX_LAGUNA_*): with no switches set
        # this returns an empty report and changes nothing, so default serving
        # behavior is untouched; a serving wrapper that exports the measured
        # stack gets it engaged at load.
        from .models.laguna_fused import install_from_env as _laguna_install_fused

        fused_report = _laguna_install_fused(model)
        if fused_report:
            logger.info("[laguna-fused] %s", fused_report)
    runtime_class = (
        LagunaARRuntime
        if _is_laguna_s_2_1_mlx_4bit_config(config)
        else MTPLXRuntime
    )
    runtime = runtime_class(
        model,
        tokenizer,
        path,
        mtp_enabled,
        contract,
        mtp_adapter_path=adapter_path,
        mtp_adapter_metadata=adapter_metadata,
        mtp_adapter_merge_report=adapter_merge_report,
        deepseek_v4_o_lora_report=deepseek_v4_o_lora_report,
        deepseek_v4_attn_proj_wide_m3_report=deepseek_v4_attn_proj_wide_m3_report,
        deepseek_v4_attention_island_report=deepseek_v4_attention_island_report,
        a3b_compiled_target_prefix_factory=compiled_target_factory,
        a3b_whole_moe_installed=False,
        qwen_row_owned_router_report=router_report,
    )
    if str((config or {}).get("model_type") or "").lower() in {
        "qwen4_exp",
        "qwen4_exp_text",
    }:
        relaxed_draft_ties = (
            os.environ.get("MTPLX_QWEN4_RELAXED_DRAFT_TIES", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if relaxed_draft_ties:
            if not mtp_enabled:
                raise RuntimeError("relaxed Qwen4 draft ties require native MTP")
            if adapter_path is not None:
                raise RuntimeError("relaxed Qwen4 draft ties do not accept adapters")
            runtime.qwen4_relaxed_draft_ties = True
            # Engagement receipt: the flag only swaps the cycle draft reader
            # inside generate_mtpk, so the load log is where an A/B proves
            # the relaxed-tie arm is the one serving.
            logger.info(
                "[qwen4-relaxed-draft-ties] installed: cycle draft reader will "
                "use sparse_distribution_from_mlx_logits_relaxed_ties"
            )
        compiled_mtp_prepare = (
            os.environ.get("MTPLX_QWEN4_COMPILED_MTP_PREPARE", "")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
        )
        if compiled_mtp_prepare:
            if adapter_path is not None:
                raise RuntimeError(
                    "compiled Qwen4 MTP preparation requires the native draft head"
                )
            text_model = getattr(model, "language_model", model)
            mtp_module = getattr(text_model, "mtp", None)
            install_prepare = getattr(mtp_module, "install_compiled_prepare", None)
            if install_prepare is None:
                raise RuntimeError(
                    "compiled Qwen4 MTP preparation is unavailable on this model"
                )
            runtime.qwen4_compiled_mtp_prepare_report = install_prepare()
            logger.info(
                "[qwen4-compiled-MTP-prepare] %s",
                runtime.qwen4_compiled_mtp_prepare_report,
            )
        from .qwen4_fixed_verify import (
            install_qwen4_fixed_verify_route,
            qwen4_fixed_verify_enabled,
        )

        if qwen4_fixed_verify_enabled():
            qwen4_verify_report = install_qwen4_fixed_verify_route(runtime)
            runtime.qwen4_fixed_verify_report = qwen4_verify_report
            logger.info("[qwen4-fixed-M4-verify] %s", qwen4_verify_report)
            # ---- stacked auxiliary lane: cached async PLE (PR #475) ---------
            # Wraps the fixed-M4 compiled-verify aux builder installed just
            # above so the auxiliary PLE plane is produced outside the compiled
            # verifier via mx.async_eval. Exact by construction (the stock
            # owner-side row cache is preserved). Needs the native
            # ple_cpu_rows extension; declines with a printed reason and serves
            # stock when it is not built. Any other failure (a contract miss)
            # escapes and fails the load, keeping the exactness contract
            # unhealthy-on-failure.
            from .qwen4_aux_lanes import (
                ple_cached_aux_enabled,
                qsa_pooled_rowsel_enabled,
            )

            if ple_cached_aux_enabled():
                from .native import (
                    load_ple_cpu_rows_extension,
                    ple_cpu_rows_unavailable_reason,
                )

                decline = ple_cpu_rows_unavailable_reason()
                if decline is not None:
                    ple_cached_aux_report = {
                        "lane": "ple_cached_aux",
                        "status": "declined",
                        "reason": decline,
                    }
                else:
                    from .ple_cached_aux import (
                        PENDING_LIMIT,
                        install_fixed_m4_cached_aux_builder,
                    )

                    native_module = load_ple_cpu_rows_extension()
                    installation = install_fixed_m4_cached_aux_builder(
                        runtime, native_module=native_module
                    )
                    runtime._ple_cached_aux_installation = installation
                    ple_cached_aux_report = {
                        "lane": "ple_cached_aux",
                        "status": "installed",
                        "variant": "async_aux",
                        "pending_limit": PENDING_LIMIT,
                        "native_ext": getattr(native_module, "__file__", None),
                    }
                runtime.ple_cached_aux_report = ple_cached_aux_report
                logger.info("[qwen4-ple-cached-aux] %s", ple_cached_aux_report)
                # A stderr install line (print, not logger.info, which the serve
                # log drops at the default level) so a benchmark window can tell
                # an engaged lane from a decline-to-stock, matching the
                # qsa_sparse_decode convention.
                if ple_cached_aux_report["status"] == "declined":
                    print(
                        "[mtplx] ple_cached_aux declined to stock: "
                        f"{ple_cached_aux_report['reason']}",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    print(
                        "[mtplx] ple_cached_aux armed: "
                        f"{ple_cached_aux_report['variant']} "
                        f"pending_limit={ple_cached_aux_report['pending_limit']} "
                        f"native_ext={ple_cached_aux_report['native_ext']}",
                        file=sys.stderr,
                        flush=True,
                    )
            # ---- stacked auxiliary lane: fixed-M4 pooled-key rowsel ---------
            # Rebinds the twelve QSA indexers' pooled-key preparation to the
            # construction-bound rowsel method, sharing one inv_freq object.
            # Exact by construction; contract failures escape (unhealthy).
            if qsa_pooled_rowsel_enabled():
                from .qsa_pooled_rowsel import install_fixed_m4_pool

                pool_report = install_fixed_m4_pool(runtime)
                qsa_pooled_rowsel_report = {
                    "lane": "qsa_pooled_rowsel",
                    "status": "installed",
                    **pool_report,
                }
                runtime.qsa_pooled_rowsel_report = qsa_pooled_rowsel_report
                logger.info(
                    "[qwen4-qsa-pooled-rowsel] %s", qsa_pooled_rowsel_report
                )
                print(
                    "[mtplx] qsa_pooled_rowsel armed: "
                    f"bank_mode={pool_report['bank_mode']} "
                    f"indexers={pool_report['kernel_binding_count']} "
                    f"shared_inv_freq_objects={pool_report['shared_inv_freq_object_count']}",
                    file=sys.stderr,
                    flush=True,
                )
        from .qwen4_m4_stage3 import (
            install_qwen4_m4_stage3,
            qwen4_m4_stage3_flags,
        )

        (
            m4_stage3_enabled,
            routed_reduce_enabled,
            residual_tail_enabled,
            routed_glu_enabled,
        ) = qwen4_m4_stage3_flags()
        if m4_stage3_enabled:
            qwen4_m4_stage3_report = install_qwen4_m4_stage3(
                runtime,
                routed_down_reduce_enabled=routed_reduce_enabled,
                routed_down_residual_tail_enabled=residual_tail_enabled,
                routed_glu_enabled=routed_glu_enabled,
            )
            logger.info("[qwen4-M4-stage3] %s", qwen4_m4_stage3_report)
        # MTPLX_QWEN4_HC_M4's PACK contract, checked here because the weights
        # exist by now. Everything it validates is process-invariant, so a
        # miss is a deployment error: it must stop the server coming up with a
        # precise reason rather than turn the first request that reaches verify
        # width into an HTTP 500. No-op (armed=False) when the flag is off.
        from .models.qwen4_exp import install_hc_m4_pack_validation

        hc_m4_report = install_hc_m4_pack_validation(runtime.model)
        runtime.qwen4_hc_m4_report = hc_m4_report
        if hc_m4_report.get("armed"):
            logger.info("[qwen4-hc-m4] %s", hc_m4_report)
    if whole_moe_plan is not None:
        if compiled_target_factory is None:
            from .a3b_whole_moe import A3BWholeMoeConfigError

            raise A3BWholeMoeConfigError(
                "whole-MoE requires exact compiled target-prefix construction"
            )
        whole_moe_report = install_a3b_whole_moe(
            whole_moe_plan,
            selfcheck_report,
            compiled_preflight=lambda: preflight_a3b_k1_target_prefix_load_graph(
                runtime, compiled_target_factory
            ),
        )
        runtime.a3b_whole_moe_installed = True
        logger.info("[a3b-whole-moe] %s", whole_moe_report)
    # The server prints this as its startup engagement receipt; logger.info
    # alone is invisible under `python -m mtplx.server.openai` (no handler).
    runtime.laguna_fused_report = fused_report
    # Gate on the object that actually runs the layer loop, not the config
    # string: dense Qwen3.8 loads as plain `qwen3_5`, and mtp_patch shadows
    # the TextModel class with its own loop — but the layers stay stock
    # `qwen3_5.DecoderLayer`, which is what the rung wrapper patches.
    try:
        from mlx_lm.models import qwen3_5 as _qwen3_5_module

        _inner_text = getattr(
            getattr(model, "language_model", model), "model", None
        )
        if isinstance(_inner_text, _qwen3_5_module.Qwen3_5TextModel):
            from .packed_concats import install_qwen3_next_packed_concats
            from .prefill_rungs import install_qwen3_5_prefill_rungs

            # Env-gated (MTPLX_PREFILL_ASYNC_RUNGS); no-op without a stride.
            install_qwen3_5_prefill_rungs()
            # Env-gated (MTPLX_PACKED_PROJ_CONCATS); no-op unless enabled.
            install_qwen3_next_packed_concats(model)
    except ImportError:
        pass
    return runtime


def load(
    model_path: Path | str,
    *,
    mtp: bool = True,
    contract: MTPContract | None = None,
    mtp_adapter: Path | str | None = None,
    merge_mtp_adapter: bool = False,
    gemma4_draft_block_size: int | None = None,
    gemma4_target_distribution_mode: str | None = None,
    proj_quant: str | None = None,
    proj_requant: str | None = None,
    expert_streaming_config: Any | None = None,
    expert_manifest: Path | str | None = None,
    mtp_artifacts: Path | str | None = None,
    mtp_precision: str = "bf16",
    expert_admission_receipt: Mapping[str, Any] | None = None,
) -> MTPLXRuntime:
    """Load a model and transfer streamed-runtime ownership only on success.

    Thin public wrapper over :func:`_load_impl`. It owns the streamed-runtime
    lifecycle: a partially-opened ``ExpertStreamingRuntime`` is closed if any
    later construction step raises, so a failed load never leaks pinned
    routed-expert file descriptors. Non-streamed loads leave the owner list
    empty and this is a pass-through.
    """

    expert_runtime_owner: list[Any] = []
    try:
        runtime = _load_impl(
            model_path,
            mtp=mtp,
            contract=contract,
            mtp_adapter=mtp_adapter,
            merge_mtp_adapter=merge_mtp_adapter,
            gemma4_draft_block_size=gemma4_draft_block_size,
            gemma4_target_distribution_mode=gemma4_target_distribution_mode,
            proj_quant=proj_quant,
            proj_requant=proj_requant,
            expert_streaming_config=expert_streaming_config,
            expert_manifest=expert_manifest,
            mtp_artifacts=mtp_artifacts,
            mtp_precision=mtp_precision,
            expert_admission_receipt=expert_admission_receipt,
            _expert_runtime_owner=expert_runtime_owner,
        )
    except BaseException:
        if expert_runtime_owner:
            expert_runtime_owner[0].close()
        raise
    expert_runtime_owner.clear()
    return runtime


def inspect(path: Path | str):
    return inspect_model(path)


def _is_laguna_s_2_1_mlx_4bit_config(config: dict[str, Any]) -> bool:
    from .models.laguna_config import is_laguna_s_2_1_mlx_4bit_config

    return is_laguna_s_2_1_mlx_4bit_config(config)


def _deepseek_v4_model_classes() -> tuple[type, type]:
    from .models.deepseek_v4 import Model, ModelArgs

    return Model, ModelArgs


def _qwen4_exp_model_classes() -> tuple[type, type]:
    from .models.qwen4_exp import Model, ModelArgs

    return Model, ModelArgs


# model_type -> loader of MTPLX-owned (Model, ModelArgs) classes for
# architectures the pinned mlx-lm does not implement. A new in-tree
# architecture (e.g. the Qwen3.8-Flash-Next backend) registers its loader
# here AND its model_type in backends.registry._INTREE_MODEL_TYPES, keeping
# the compatibility verdict and the loader in lockstep. Laguna stays a
# geometry-gated special case below because its match is not model_type-keyed.
_INTREE_MODEL_CLASS_LOADERS: dict[str, Callable[[], tuple[type, type]]] = {
    "deepseek_v4": _deepseek_v4_model_classes,
    "qwen4_exp": _qwen4_exp_model_classes,
    # Text-config spelling of the same family: inspection prefers the nested
    # text_config.model_type for multimodal checkpoints, and text-only
    # re-exports carry it at top level. Same trunk, same classes.
    "qwen4_exp_text": _qwen4_exp_model_classes,
}


def _model_classes_for_config(config: dict[str, Any]) -> tuple[type, type] | None:
    """Return MTPLX-owned model classes for architectures missing in mlx-lm."""

    loader = _INTREE_MODEL_CLASS_LOADERS.get(
        str(config.get("model_type") or "").lower()
    )
    if loader is not None:
        return loader()
    if not _is_laguna_s_2_1_mlx_4bit_config(config):
        return None
    from .models.laguna import Model, ModelArgs

    return Model, ModelArgs


def _load_base_model(path: Path, config: dict[str, Any]) -> tuple[Any, Any]:
    if (
        config.get("architectures") == ["LagunaForCausalLM"]
        and str(config.get("model_type") or "").lower() == "laguna"
        and "model_file" in config
    ):
        raise ValueError("Laguna model_file execution is not permitted")
    model_classes = _model_classes_for_config(config)
    if model_classes is not None:
        from mlx_lm.utils import load_model

        from .models.laguna_config import laguna_module_quantization

        tokenizer = _load_tokenizer_resilient(path, config)
        load_kwargs: dict[str, Any] = {
            "get_model_classes": lambda config: model_classes,
        }
        module_quantization = laguna_module_quantization(config)
        if module_quantization is not None:
            # The pinned oQ4e checkpoint keys its mixed-precision quantization
            # dict by the ``language_model.``-prefixed export path. Strip the
            # prefix so mlx-lm's config-driven quantizer addresses each module
            # by its tree path (the BF16 routers carry no entry and stay
            # unquantized). mlx-lm reads this from config["quantization"], not
            # from any model-level predicate.
            load_kwargs["model_config"] = {
                "quantization": module_quantization,
                "quantization_config": module_quantization,
            }
        model, _loaded_config = load_model(path, **load_kwargs)
        # In-tree models with SSD-resident sidecars (e.g. the qwen4_exp
        # n-gram table) finish wiring here — after weights, before serving.
        post_load = getattr(model, "post_weight_load", None)
        if callable(post_load):
            post_load(path)
        return model, tokenizer

    from mlx_lm.utils import load as mlx_lm_load

    model, tokenizer = mlx_lm_load(str(_mtp_alias_load_path(path, config)))
    _refuse_double_shifted_trunk_norms(model, config)
    return model, tokenizer


def _refuse_double_shifted_trunk_norms(model: Any, config: dict[str, Any]) -> None:
    """Fail loud on a +1.0 double-shifted trunk (#306), never serve it slow.

    mlx-lm's qwen3.5-family sanitize keys the +1.0 delta restoration on the
    bare PRESENCE of mtp.* keys in the shards, so an artifact that embeds an
    already-absolute MTP head gets every trunk RMSNorm shifted a second time
    — the model loads and generates, just badly (the #306 reports measured
    0.9-6.7% acceptance and blamed the engine). Healthy absolute q-norm
    means sit in the 1.74-1.83 fleet band; a double shift lands ~2.79. One
    tensor mean decides it. Refusal with the cause beats silently serving a
    corrupted trunk — the hide-nothing law.
    """
    family = str(config.get("model_type") or "").lower()
    if family not in {"qwen3_5", "qwen3_next", "qwen3next"}:
        return
    weight = None
    try:
        layers = getattr(getattr(model, "model", model), "layers", None) or []
        for layer in layers:
            candidate = getattr(
                getattr(layer, "self_attn", None), "q_norm", None
            )
            weight = getattr(candidate, "weight", None)
            if weight is not None:
                break
        if weight is None or getattr(weight, "ndim", None) != 1:
            return
        mean = float(weight.mean().item())
    except Exception:
        return
    if mean > 2.4:
        raise ValueError(
            "trunk RMSNorm weights read double-shifted "
            f"(q_norm mean {mean:.2f}; healthy packs sit near 1.79): this "
            "artifact embeds mtp.* keys in its shards with absolute gains, "
            "and mlx-lm's presence-keyed sanitize added +1.0 to an "
            "already-absolute trunk (issue #306). Rebuild the pack with the "
            "MTP head as a standalone mtp.safetensors (mtplx forge does "
            "this), or strip the embedded mtp.* tensors from the shards."
        )


# A chat_template that is nothing but a Jinja ``{% include %}`` redirect to a
# sidecar file. The pinned Laguna-S-2.1 oQ4e checkpoint ships the 35-char stub
# ``{% include 'chat_template.jinja' %}`` in tokenizer_config.json. transformers
# compiles embedded chat templates in a loader-less Jinja Environment, so any
# apply_chat_template on such a stub raises
# ``TypeError('no loader for this environment specified')`` — the failure the
# 2026-07-22 laguna serving window hit on both the one-shot and server paths.
_JINJA_INCLUDE_CHAT_TEMPLATE_RE = re.compile(r"\{%-?\s*include\b")


def _is_jinja_include_chat_template(chat_template: Any) -> bool:
    """True when ``chat_template`` is a string carrying a Jinja include."""

    return isinstance(chat_template, str) and bool(
        _JINJA_INCLUDE_CHAT_TEMPLATE_RE.search(chat_template)
    )


def _pinned_chat_template_text(model_path: Path) -> str | None:
    """Contents of the sidecar chat_template.jinja pinned next to the model.

    Returns None when the file is absent or empty. The file is only read, never
    mutated — its sha256 is load-bearing for artifact-integrity checks.
    """

    jinja = model_path / "chat_template.jinja"
    if not jinja.exists():
        return None
    text = jinja.read_text(encoding="utf-8")
    return text if text.strip() else None


def _repair_included_chat_template(tokenizer: Any, model_path: Path) -> None:
    """Swap an include-stub chat_template for the pinned sidecar contents.

    The oQ4e tokenizer_config.json redirects its chat_template to a sidecar via
    ``{% include 'chat_template.jinja' %}``. transformers cannot resolve the
    include (no Jinja loader), so apply_chat_template raises the moment it runs.
    The real template — self-contained, no include/import/extends — lives in
    chat_template.jinja beside the weights; substitute its contents in memory.
    Setting ``tokenizer.chat_template`` on the mlx-lm TokenizerWrapper forwards
    to the underlying HF tokenizer, which is what apply_chat_template renders.
    """

    current = getattr(tokenizer, "chat_template", None)
    if not _is_jinja_include_chat_template(current):
        return
    replacement = _pinned_chat_template_text(model_path)
    if replacement is None:
        return
    tokenizer.chat_template = replacement


def _load_tokenizer_resilient(model_path: Path, config: dict[str, Any]) -> Any:
    from mlx_lm.utils import load_tokenizer

    try:
        tokenizer = load_tokenizer(model_path)
    except Exception as exc:  # noqa: BLE001 - transformers raises several strict-config errors
        logger.warning(
            "[tokenizer] AutoTokenizer parse failed (%s); using tokenizer.json fallback",
            exc,
        )
    else:
        _repair_included_chat_template(tokenizer, model_path)
        return tokenizer

    from mlx_lm.tokenizer_utils import TokenizerWrapper
    from transformers import PreTrainedTokenizerFast

    tcfg_path = model_path / "tokenizer_config.json"
    tcfg = json.loads(tcfg_path.read_text(encoding="utf-8")) if tcfg_path.exists() else {}
    passthrough = {
        key: tcfg[key]
        for key in ("bos_token", "eos_token", "pad_token", "unk_token", "additional_special_tokens")
        if key in tcfg
    }
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(model_path / "tokenizer.json"),
        **passthrough,
    )
    chat_template = tcfg.get("chat_template")
    # An include-stub is not a usable template (transformers has no loader for
    # it); treat it as absent so the pinned sidecar below supplies the real one.
    if _is_jinja_include_chat_template(chat_template):
        chat_template = None
    if not chat_template:
        replacement = _pinned_chat_template_text(model_path)
        if replacement is not None:
            chat_template = replacement
    if chat_template:
        hf_tokenizer.chat_template = chat_template
    eos = config.get("eos_token_id")
    if eos is None:
        eos = (config.get("text_config") or {}).get("eos_token_id")
    if isinstance(eos, int):
        eos_ids = [eos]
    elif isinstance(eos, (list, tuple)):
        eos_ids = list(eos)
    else:
        eos_ids = None
    return TokenizerWrapper(
        hf_tokenizer,
        eos_token_ids=eos_ids,
        chat_template=None,
    )


def _mtp_alias_load_path(path: Path, config: dict[str, Any] | None) -> Path:
    """Loadable path for `*_mtp`-typed checkpoints (issue #147).

    vLLM-convention MTP checkpoints ship config.json with model_type like
    ``qwen3_5_mtp``: the trunk is the plain base architecture plus an
    embedded MTP head. mlx_lm's class table has no ``*_mtp`` modules, so
    handing it the raw dir fails with "Model type ... not supported" even
    though the forge probe correctly reports the family as supported. When
    the stripped base module exists in mlx_lm and the full name does not,
    build a symlink wrapper with a patched config.json (model_type=base)
    and load through it; MTP injection later picks the head up from the
    original weights. Everything else returns the path untouched.
    """

    model_type = str((config or {}).get("model_type") or "")
    if not model_type.endswith("_mtp"):
        return path
    base_type = model_type[: -len("_mtp")]
    import importlib.util

    def _mlx_lm_has(model_type_name: str) -> bool:
        return (
            importlib.util.find_spec(f"mlx_lm.models.{model_type_name}")
            is not None
        )

    if _mlx_lm_has(model_type) or not _mlx_lm_has(base_type):
        return path
    try:
        wrapper_root = Path.home() / ".mtplx" / "build-cache" / "mtp-alias-load"
        digest = hashlib.sha256(
            f"{path.resolve()}::{base_type}".encode("utf-8")
        ).hexdigest()[:16]
        wrapper = wrapper_root / f"{path.name}-{base_type}-{digest}"
        patched_config = dict(config or {})
        patched_config["model_type"] = base_type
        marker = wrapper / ".mtplx-alias-source"
        if not marker.exists() or marker.read_text(encoding="utf-8") != str(
            path.resolve()
        ):
            wrapper.mkdir(parents=True, exist_ok=True)
            for item in path.iterdir():
                if item.name in {"config.json", ".mtplx-alias-source"}:
                    continue
                link = wrapper / item.name
                if link.is_symlink() or link.exists():
                    continue
                link.symlink_to(item)
            marker.write_text(str(path.resolve()), encoding="utf-8")
        # Rewrite the config every time: the source config may have changed.
        (wrapper / "config.json").write_text(
            json.dumps(patched_config, indent=2), encoding="utf-8"
        )
        return wrapper
    except Exception:
        # Wrapper construction is best-effort; the raw path preserves the
        # original (informative) mlx_lm error.
        return path


def _load_runtime_metadata(path: Path) -> dict[str, Any] | None:
    runtime_path = path / "mtplx_runtime.json"
    if not runtime_path.exists():
        return None
    try:
        data = json.loads(runtime_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None
