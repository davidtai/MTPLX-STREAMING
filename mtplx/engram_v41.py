"""DeepSeek-V4.1-Flash Engram conditional-memory runtime for MTPLX.

Three pieces, ported from the reference ``inference/engram.py`` + ``model.py`` and wired
onto the shared resident-row cache:

  * :func:`build_compressed_token_map` -- collapse token ids that normalize alike onto a
    smaller id space (the space every hash multiplier is derived from).  Mirrors the
    reference ``build_compressed_token_map``; yields 99092 entries on the real tokenizer.

  * :class:`NgramHashState` -- maps each position to the 24 engram row ids (per layer) of the
    n-grams ending there, exactly per the manifest ``hashing`` recipe (rolling XOR over the
    ``max_ngram_size-1`` lookbacks, mod prime buckets, plus per-bucket flat offsets).  It is
    streaming-safe: it keeps a per-sequence compressed-id history so lookbacks cross the
    prefill/decode split, and supports :meth:`trim` for speculative-decode rollback.

  * :class:`EngramV41` -- an ``nn.Module`` hook for layers 1 and 14: gather the 24 rows/token
    from the (dequantized) engram bank, run ``engram.wkv`` + the q/k gate, and write the
    result into the hc-expanded residual stream.  Callable as
    ``(hidden_states, token_ids, cache_state) -> mx.array`` so the model worker can attach it
    to a layer's ``engram_hook`` attribute (default ``None``).

The engram bank rows come through :class:`mtplx.ngram_row_cache.NGramRowCache` (byte-budgeted
resident LRU, positional misses, MLX affine dequant).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

import mlx.core as mx
import mlx.nn as nn

from mtplx.models import deepseek_v41_stage_timing as _stime
from mtplx.ngram_row_cache import FileRowReader, NGramRowCache, RowGeometry

__all__ = [
    "build_compressed_token_map",
    "load_engram_tokenizer",
    "NgramHashState",
    "EngramV41",
    "EngramResidents",
    "load_engram_residents",
    "load_engram_resident_tensors",
    "open_engram_row_cache",
    "n_hash_cols",
]

DEAD = -1  # a masked / image-span token: no n-gram may span it (matches reference)


def n_hash_cols(max_ngram_size: int, n_heads: int) -> int:
    """Rows a single token gathers per engram layer (== 24 for max_ngram=4, n_heads=8)."""
    return (max_ngram_size - 1) * n_heads


# --------------------------------------------------------------------------
# compressed token map (mirrors reference engram.build_compressed_token_map)
# --------------------------------------------------------------------------
def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Map every token id onto a smaller id space where tokens that normalize alike collapse.

    ``tokenizer`` is a HuggingFace fast tokenizer (``.backend_tokenizer`` + ``len()``).
    Decode each id via the Rust backend (``skip_special_tokens=False``), normalize with the
    manifest's normalizer sequence, and collapse ids whose normalized form matches.
    Partial-UTF8 tokens (``\\ufffd``) are keyed by their raw ``id_to_token`` form.

    Returns ``(lookup, compressed_vocab_size)``; the size drives every hash multiplier.
    """
    from tokenizers import Regex, normalizers

    # a private-use char, so a token that is exactly one space survives Strip() instead of
    # collapsing to the empty string and merging with unrelated tokens
    sentinel = ""
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text

        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id

    return lookup, len(key_to_new)


def load_engram_tokenizer(directory: str | Path):
    """Load the artifact's HuggingFace fast tokenizer (for :func:`build_compressed_token_map`)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(directory))


# --------------------------------------------------------------------------
# mode-aware row-cache constructor (reads the manifest's quant mode)
# --------------------------------------------------------------------------
def open_engram_row_cache(
    engram_dir: str | Path,
    layer: int,
    *,
    cache_rows: int = 32768,
    cache_bytes: int | None = None,
) -> NGramRowCache:
    """Open one engram layer's on-disk bank as a mode-aware :class:`NGramRowCache`.

    Reads ``engram-manifest.json``, picks the layer's ``quant`` block, and builds the resident
    LRU with a :class:`RowGeometry` matching the declared ``mode`` (``affine`` -> 272 B/row bf16
    scales+biases; ``mxfp8`` -> 264 B/row E4M3 codes + E8M0 scales, dequantized via
    ``mx.dequantize(mode="mxfp8")``).  The cache's :meth:`NGramRowCache.dequantize` therefore
    returns the correctly dequantized ``[.., head_dim]`` rows for whichever codec the bank uses,
    so :class:`EngramV41` needs no codec awareness of its own.
    """
    directory = Path(engram_dir)
    manifest = json.loads((directory / "engram-manifest.json").read_text())
    entry = next((e for e in manifest["layers"] if e["layer_id"] == layer), None)
    if entry is None:
        raise KeyError(f"layer {layer} not in manifest ({directory})")
    q = entry["quant"]
    mode = q.get("mode", "affine")
    geom = RowGeometry(
        values_per_row=int(q["head_dim"]), bits=int(q["bits"]),
        group_size=int(q["group_size"]), mode=mode,
    )
    if geom.row_bytes != int(entry["record_bytes"]):
        raise ValueError(
            f"layer {layer}: geometry row_bytes {geom.row_bytes} != manifest record_bytes "
            f"{entry['record_bytes']} (mode {mode})"
        )
    reader = FileRowReader(
        directory / entry["file"], row_bytes=int(entry["record_bytes"]), num_rows=int(entry["rows"]),
    )
    return NGramRowCache(
        reader, geom, num_rows=int(entry["rows"]),
        cache_bytes=cache_bytes, cache_rows=cache_rows,
    )


# --------------------------------------------------------------------------
# streaming n-gram hash state
# --------------------------------------------------------------------------
class NgramHashState:
    """Streaming per-sequence engram row-id source (exact manifest hash recipe).

    Constructed from the manifest ``hashing`` block plus a compressed token map.  Feed tokens
    with :meth:`advance` (once per model step); read a layer's ``[B, L, 24]`` row ids with
    :meth:`current_row_ids`.  The compressed-id history persists across steps, so n-gram
    lookbacks cross the prefill/decode boundary; :meth:`trim` drops the last ``n`` positions so
    a rejected speculative draft can be re-fed to identical ids.
    """

    def __init__(
        self,
        *,
        token_map: Sequence[int],
        multipliers: np.ndarray,   # [n_layers, max_ngram_size] int64
        primes: np.ndarray,        # [n_layers, max_ngram_size-1, n_heads] int64
        flat_offsets: np.ndarray,  # [n_layers, n_hash_cols] int64
        pad_compressed: int,
        max_ngram_size: int,
        n_heads: int,
        layer_ids: Sequence[int],
        num_embeddings: Sequence[int] | None = None,
    ) -> None:
        self.token_map = np.asarray(token_map, dtype=np.int64)
        self.multipliers = np.asarray(multipliers, dtype=np.int64)
        self.primes = np.asarray(primes, dtype=np.int64)
        self.flat_offsets = np.asarray(flat_offsets, dtype=np.int64)
        self.pad_compressed = int(pad_compressed)
        self.max_ngram_size = int(max_ngram_size)
        self.n_heads = int(n_heads)
        self.layer_ids = tuple(int(x) for x in layer_ids)
        self.num_embeddings = tuple(num_embeddings) if num_embeddings is not None else None
        self.n_layers = len(self.layer_ids)
        self.n_hash_cols = n_hash_cols(self.max_ngram_size, self.n_heads)

        if self.multipliers.shape != (self.n_layers, self.max_ngram_size):
            raise ValueError("multipliers shape mismatch")
        if self.primes.shape != (self.n_layers, self.max_ngram_size - 1, self.n_heads):
            raise ValueError("primes shape mismatch")
        if self.flat_offsets.shape != (self.n_layers, self.n_hash_cols):
            raise ValueError("flat_offsets shape mismatch")

        self._buf: np.ndarray | None = None   # [B, T] compressed ids (incl DEAD)
        self._len = 0
        self._current: np.ndarray | None = None  # [B, L, n_layers, n_hash_cols]

    # -- construction from the engram manifest ------------------------------
    @classmethod
    def from_manifest(cls, manifest: dict, tokenizer) -> "NgramHashState":
        h = manifest["hashing"]
        token_map, size = build_compressed_token_map(tokenizer)
        expected = int(h.get("compressed_vocab_size", size))
        if size != expected:
            raise ValueError(f"compressed vocab size {size} != manifest {expected}")
        multipliers = np.asarray(h["hash_multipliers"], dtype=np.int64)
        per = h["per_layer"]
        primes = np.asarray([layer["primes"] for layer in per], dtype=np.int64)
        flat_offsets = np.asarray([layer["flat_offsets"] for layer in per], dtype=np.int64)
        pad_id = int(h["pad_id"])
        return cls(
            token_map=token_map,
            multipliers=multipliers,
            primes=primes,
            flat_offsets=flat_offsets,
            pad_compressed=int(token_map[pad_id]),
            max_ngram_size=int(h["max_ngram_size"]),
            n_heads=int(h["n_heads"]),
            layer_ids=list(h["layer_ids"]),
            num_embeddings=[layer["total_rows"] for layer in per],
        )

    @classmethod
    def from_manifest_path(cls, directory: str | Path, tokenizer) -> "NgramHashState":
        manifest = json.loads((Path(directory) / "engram-manifest.json").read_text())
        return cls.from_manifest(manifest, tokenizer)

    # -- streaming ----------------------------------------------------------
    def fresh(self) -> "NgramHashState":
        """A new streaming state that shares this one's (immutable) hash config.

        Building the config (the compressed token map + primes/offsets) is the
        expensive part; :meth:`advance` never mutates it (it only indexes the
        token map and concatenates into a *new* history buffer), so many
        per-sequence states can safely share one config.  Used by the model to
        hand each KV cache its own engram history without rebuilding the map.
        """
        return NgramHashState(
            token_map=self.token_map,
            multipliers=self.multipliers,
            primes=self.primes,
            flat_offsets=self.flat_offsets,
            pad_compressed=self.pad_compressed,
            max_ngram_size=self.max_ngram_size,
            n_heads=self.n_heads,
            layer_ids=self.layer_ids,
            num_embeddings=self.num_embeddings,
        )

    def reset(self) -> None:
        self._buf = None
        self._len = 0
        self._current = None

    def trim(self, n: int) -> None:
        """Drop the last ``n`` fed positions (speculative-decode rollback)."""
        if n < 0:
            raise ValueError("trim count must be >= 0")
        if n == 0:
            return
        if self._buf is None or n > self._len:
            raise ValueError(f"cannot trim {n} of {self._len} positions")
        self._len -= n
        self._buf = self._buf[:, : self._len]
        self._current = None

    @property
    def length(self) -> int:
        return self._len

    def advance(self, input_ids, token_mask=None) -> np.ndarray:
        """Feed ``[B, L]`` token ids; return their engram row ids ``[B, L, n_layers, n_hash_cols]``.

        ``token_mask`` ``[B, L]`` (bool, ``False`` == image-span / no n-gram) marks positions
        as DEAD so no n-gram spans them, matching the reference.
        """
        ids = np.asarray(input_ids, dtype=np.int64)
        if ids.ndim != 2:
            raise ValueError("input_ids must be [B, L]")
        B, L = ids.shape
        compressed = self.token_map[ids]
        if token_mask is not None:
            mask = np.asarray(token_mask, dtype=bool)
            if mask.shape != (B, L):
                raise ValueError("token_mask must be [B, L]")
            compressed = np.where(mask, compressed, DEAD)

        if self._buf is None:
            self._buf = compressed.copy()
        else:
            if self._buf.shape[0] != B:
                raise ValueError(f"batch size changed {self._buf.shape[0]} -> {B}; call reset()")
            self._buf = np.concatenate([self._buf, compressed], axis=1)
        start = self._len
        self._len = start + L

        row_ids = self._hash_positions(start, L)
        self._current = row_ids
        return row_ids

    def _hash_positions(self, start: int, L: int) -> np.ndarray:
        """Row ids for positions ``[start, start+L)`` given the full compressed history."""
        assert self._buf is not None
        full = self._buf                                   # [B, T]
        B = full.shape[0]
        pos = np.broadcast_to(np.arange(start, start + L)[None, :], (B, L))
        blocked = np.zeros((B, L), dtype=bool)
        toks = []
        for shift in range(self.max_ngram_size):
            idx = np.clip(pos - shift, 0, None)
            src = np.take_along_axis(full, idx, axis=1)
            blocked = blocked | (pos < shift) | (src == DEAD)
            toks.append(np.where(blocked, self.pad_compressed, src))
        toks = np.stack(toks, axis=-1)                     # [B, L, max_ngram]

        # products[..., k] = token k-back * multiplier[layer, k]
        products = toks[:, :, None, :] * self.multipliers[None, None, :, :]  # [B,L,n_layers,ng]
        rolling = products[..., 0]
        hashes = []
        for i in range(1, self.max_ngram_size):
            rolling = np.bitwise_xor(rolling, products[..., i])
            # rolling [B,L,n_layers] % primes[:, i-1] [n_layers, n_heads]
            hashes.append(rolling[..., None] % self.primes[:, i - 1][None, None])
        return np.concatenate(hashes, axis=-1) + self.flat_offsets[None, None]  # [B,L,n_layers,cols]

    def current_row_ids(self, layer_hash_index: int) -> np.ndarray:
        """Row ids ``[B, L, n_hash_cols]`` for one engram layer from the last :meth:`advance`."""
        if self._current is None:
            raise RuntimeError("no positions advanced yet")
        return self._current[:, :, layer_hash_index, :]

    # -- serialisation (mlx_lm session-state contract) ----------------------
    @property
    def state(self) -> mx.array:
        """The streaming n-gram history as one ``mx.array`` for save/restore.

        Returns the ``[B, T]`` compressed-id history buffer (``int32``; ``DEAD``
        ``== -1``) that :meth:`advance` / :meth:`trim` maintain -- exactly the
        per-sequence state a warm-turn (session-bank / SSD) restore must
        reinstate so n-gram lookbacks cross the restore seam.  A never-advanced
        state serialises to the canonical empty ``[1, 0]`` buffer.

        Only the *history* travels: the immutable hash config (compressed token
        map, multipliers, primes, flat offsets) is shared across sequences and is
        rebuilt by :meth:`fresh`, so it is deliberately NOT serialised.  Being a
        single ``mx.array`` leaf (no numpy, no ``None``), this round-trips through
        both ``mtplx.cache_state.snapshot_cache`` / ``restore_cache`` and
        ``mlx_lm.save_prompt_cache`` / ``load_prompt_cache`` (whose ``tree_flatten``
        + ``mx.save_safetensors`` reject the raw numpy ``_buf`` with
        ``std::bad_cast``).
        """
        if self._buf is None:
            return mx.zeros((1, 0), dtype=mx.int32)
        return mx.array(self._buf.astype(np.int32))

    @state.setter
    def state(self, value) -> None:
        self.replace_state(value)

    def replace_state(self, value) -> None:
        """Reinstate the streaming history from :attr:`state` (mlx_lm contract).

        Accepts the ``mx.array`` (or any ``[B, T]`` array-like) produced by
        :attr:`state`; ``None`` resets to a fresh history.  :meth:`fresh` /
        :meth:`advance` / :meth:`trim` semantics are preserved -- the internal
        ``_buf`` is restored to its exact contents (as ``int64``, the working
        dtype) and ``_len`` is derived from it, while the transient last-advance
        cache ``_current`` is cleared (the next :meth:`advance` recomputes it
        before any :meth:`current_row_ids` read, exactly as :meth:`trim` does).
        """
        if value is None:
            self.reset()
            return
        arr = np.asarray(value)
        if arr.ndim != 2:
            raise ValueError(f"engram state must be [B, T]; got shape {tuple(arr.shape)}")
        if arr.shape[1] == 0:
            # empty history (never advanced, or fully trimmed): a fresh buffer is
            # equivalent -- the next advance rebuilds from scratch for whatever
            # batch it is fed.
            self._buf = None
            self._len = 0
        else:
            self._buf = np.ascontiguousarray(arr.astype(np.int64))
            self._len = int(arr.shape[1])
        self._current = None


# --------------------------------------------------------------------------
# engram layer module
# --------------------------------------------------------------------------
@dataclass
class _StepState:
    """Convenience carrier for the hook's ``cache_state`` when not passing the hash state.

    Holds precomputed ``[B, L, n_layers, n_hash_cols]`` row ids and an optional token mask.
    """

    row_ids: np.ndarray
    token_mask: mx.array | None = None

    def current_row_ids(self, layer_hash_index: int) -> np.ndarray:
        return self.row_ids[:, :, layer_hash_index, :]


class EngramV41(nn.Module):
    """Engram residual write for one layer (1 or 14): gather 24 rows/token, ``wkv``, q/k gate.

    ``__call__(hidden_states, token_ids, cache_state) -> mx.array`` returns the **updated**
    hc-expanded residual stream (reference ``Engram.forward`` output ``h + gate * value``); the
    pure additive contribution is ``result - hidden_states``.  The model worker attaches an
    instance to a layer's ``engram_hook`` attribute (default ``None``) and calls
    ``h = layer.engram_hook(h, token_ids, cache_state)``.

    ``cache_state`` supplies this layer's row ids via ``cache_state.current_row_ids(layer_hash_index)``
    -- pass the shared :class:`NgramHashState` (advanced once per step) or a :class:`_StepState`.
    ``wkv`` is a callable ``[.., n_hash_cols*head_dim] -> [.., dim*(hc_mult+1)]`` (build one for a
    dense resident weight with :meth:`dense_wkv`; a resident q8 wkv is wrapped as a
    ``mx.quantized_matmul`` callable).  ``q_weight``/``k_weight`` are ``[hc_mult, dim]``.
    """

    def __init__(
        self,
        *,
        layer_id: int,
        layer_hash_index: int,
        row_cache: NGramRowCache,
        wkv: Callable[[mx.array], mx.array],
        q_weight: mx.array,
        k_weight: mx.array,
        dim: int,
        hc_mult: int,
        norm_eps: float,
        clamp_value: float = 1e-6,
    ) -> None:
        super().__init__()
        self.layer_id = int(layer_id)
        self.layer_hash_index = int(layer_hash_index)
        self.row_cache = row_cache
        self.wkv = wkv
        self.q_weight = q_weight
        self.k_weight = k_weight
        self.dim = int(dim)
        self.hc_mult = int(hc_mult)
        self.head_dim = int(row_cache.geometry.values_per_row)
        self.norm_eps = float(norm_eps)
        self.clamp_value = float(clamp_value)

    @staticmethod
    def dense_wkv(weight: mx.array) -> Callable[[mx.array], mx.array]:
        """Wrap a dense ``[out, in]`` weight as a linear callable ``x -> x @ weight.T``."""
        def apply(x: mx.array) -> mx.array:
            return x @ weight.T
        return apply

    def __call__(self, hidden_states: mx.array, token_ids, cache_state) -> mx.array:
        # W37 engram.hash: the per-layer row-id read (the rolling n-gram hash was
        # computed once for the step in ``NgramHashState.advance`` -> engram.advance;
        # this slices out THIS layer's [B, L, cols] ids).
        with _stime.stage("engram.hash"):
            row_ids = cache_state.current_row_ids(self.layer_hash_index)   # np [B, L, cols]
        B, L = int(hidden_states.shape[0]), int(hidden_states.shape[1])
        if tuple(row_ids.shape[:2]) != (B, L):
            raise ValueError(f"row ids {row_ids.shape[:2]} do not match hidden states {(B, L)}")
        if token_ids is not None and tuple(np.asarray(token_ids).shape) != (B, L):
            raise ValueError("token_ids shape does not match hidden_states / cache_state")

        # W37 engram.row_fetch: the byte-budgeted row-cache lookup (the SSD/LRU
        # gather that dequantizes the 24 rows/token) -- the I/O-bound half.
        with _stime.stage("engram.row_fetch") as _st:
            embed = self.row_cache.dequantize(row_ids)                 # mx [B, L, cols, head_dim]
            _st.add(embed)

        # W37 engram.apply: wkv projection + q/k gate + the additive residual write.
        with _stime.stage("engram.apply") as _st:
            kv = self.wkv(embed.reshape(B, L, -1))                     # [B, L, dim*(hc_mult+1)]
            split = self.hc_mult * self.dim
            key = kv[..., :split].astype(mx.float32).reshape(B, L, self.hc_mult, self.dim)
            value = kv[..., split:].astype(mx.float32)                 # [B, L, dim]

            h = hidden_states.astype(mx.float32)                       # [B, L, hc_mult, dim]
            weight = (self.q_weight * self.k_weight).astype(mx.float32)  # [hc_mult, dim]
            eps = self.norm_eps
            rstd = mx.rsqrt(mx.mean(h * h, axis=-1) + eps) * mx.rsqrt(mx.mean(key * key, axis=-1) + eps)
            dot = mx.sum(h * weight * key, axis=-1) * rstd * (self.dim ** -0.5)   # [B, L, hc_mult]
            # signed sqrt before the sigmoid (copysign(sqrt(clamp|dot|), dot)); +0 -> + branch
            mag = mx.sqrt(mx.maximum(mx.abs(dot), self.clamp_value))
            signed = mx.where(dot < 0, -mag, mag)
            gate = mx.sigmoid(signed)

            mask = getattr(cache_state, "token_mask", None)
            if mask is not None:
                gate = gate * mask.astype(mx.float32)[..., None]   # [B, L] -> [B, L, 1] over hc copies

            contribution = gate[..., None] * value[:, :, None, :]      # [B, L, hc_mult, dim]
            out = (h + contribution).astype(hidden_states.dtype)
            _st.add(out)
        return out


# --------------------------------------------------------------------------
# resident Engram projections (W4 sidecar loader)
# --------------------------------------------------------------------------
@dataclass
class EngramResidents:
    """One engram layer's resident projection tensors, loaded from the W4 sidecar.

    ``wkv`` is the ``[.., n_hash_cols*head_dim] -> [.., dim*(hc_mult+1)]`` callable that
    :class:`EngramV41` expects, backed by ``mx.quantized_matmul`` over the wkv weight -- affine
    q8/gs64 (``mode="affine"``, with ``wkv_biases``) or native mxfp8/gs32 (``mode="mxfp8"``,
    ``wkv_biases`` is ``None``).  ``q_weight``/``k_weight`` are the exact ``[hc_mult, dim]`` F32
    gates.  Build the module with :meth:`build_module` (adds the streamed row cache + geometry).
    """

    layer_id: int
    wkv: Callable[[mx.array], mx.array]
    q_weight: mx.array
    k_weight: mx.array
    wkv_packed: mx.array
    wkv_scales: mx.array
    dim: int
    hc_mult: int
    group_size: int
    bits: int
    wkv_biases: mx.array | None = None
    mode: str = "affine"

    def build_module(self, *, row_cache: NGramRowCache, layer_hash_index: int,
                     norm_eps: float, clamp_value: float = 1e-6) -> "EngramV41":
        """Construct the :class:`EngramV41` hook from these residents + a row cache."""
        return EngramV41(
            layer_id=self.layer_id, layer_hash_index=layer_hash_index,
            row_cache=row_cache, wkv=self.wkv,
            q_weight=self.q_weight, k_weight=self.k_weight,
            dim=self.dim, hc_mult=self.hc_mult,
            norm_eps=norm_eps, clamp_value=clamp_value,
        )


def load_engram_resident_tensors(
    sidecar: str | Path, *, layer_ids: Sequence[int], mode: str = "affine"
) -> dict[str, mx.array]:
    """Load the complete projection sidecar once for all attached Engram layers."""
    from .resident_io import ResidentShardReader

    # Snapshot sidecars may be symlinks to a content-addressed blob. Resolve
    # once, then keep the no-follow descriptor open for admission and loading.
    path = Path(sidecar).resolve(strict=True)
    suffixes = ("wkv.weight", "wkv.scales", "q_weight", "k_weight")
    if mode == "affine":
        suffixes += ("wkv.biases",)
    retained = {f"layers.{layer}.engram.{suffix}" for layer in layer_ids for suffix in suffixes}
    with ResidentShardReader({path.name: path}, retained_names={path.name: retained}) as reader:
        tensors = reader.load(path.name, mx)
    return {name: value for name, value in tensors.items() if name in retained}


def load_engram_residents(
    artifact_dir: str | Path,
    layer_id: int,
    *,
    sidecar_name: str = "engram-residents.safetensors",
    preloaded: dict[str, mx.array] | None = None,
) -> EngramResidents:
    """Load one engram layer's resident projections from ``engram-residents.safetensors``.

    ``artifact_dir`` is the artifact's ``engram/`` directory (holding the sidecar and
    ``engram-manifest.json``).  The wkv codec is read from the manifest ``residents.quant.wkv``
    (``mode`` ``affine`` or ``mxfp8``, default affine for old sidecars) and wrapped as the matching
    ``mx.quantized_matmul(..., mode=...)`` callable -- affine uses ``wkv.{weight,scales,biases}``,
    mxfp8 uses ``wkv.{weight,scales}`` (no bias).  ``q_weight``/``k_weight`` are returned as the
    exact F32 arrays.  ``dim`` and ``hc_mult`` are read off ``q_weight``'s ``[hc_mult, dim]`` shape
    and cross-checked against the wkv output width (``dim*(hc_mult+1)``).

    Production attachment supplies one admitted, uncached ``preloaded`` dictionary
    shared by all Engram layers. Standalone one-layer callers retain lazy path
    loading so they do not eagerly allocate the sidecar's unused sibling layers.
    """
    directory = Path(artifact_dir)
    manifest_path = directory / "engram-manifest.json"
    res_meta = None
    if manifest_path.is_file():
        res_meta = json.loads(manifest_path.read_text()).get("residents")
    sidecar = directory / (res_meta["file"] if res_meta and "file" in res_meta else sidecar_name)
    if not sidecar.is_file():
        raise FileNotFoundError(f"engram residents sidecar not found: {sidecar}")

    # wkv codec from the manifest residents entry (default affine for old sidecars)
    bits, group, mode = 8, 64, "affine"
    if res_meta and isinstance(res_meta.get("quant"), dict):
        wkv_q = res_meta["quant"].get("wkv", {})
        if isinstance(wkv_q, dict):
            bits = int(wkv_q.get("bits", bits))
            group = int(wkv_q.get("group_size", group))
            mode = str(wkv_q.get("mode", mode))

    tensors = preloaded if preloaded is not None else mx.load(str(sidecar))
    base = f"layers.{layer_id}.engram"
    try:
        packed = tensors[f"{base}.wkv.weight"]
        scales = tensors[f"{base}.wkv.scales"]
        # mxfp8 wkv has no bias; affine does.  Fall back to sidecar contents if the manifest
        # codec is absent (old artifact) but a bias tensor is present.
        has_bias = f"{base}.wkv.biases" in tensors
        biases = tensors[f"{base}.wkv.biases"] if (mode == "affine" and has_bias) else None
        if mode == "affine" and biases is None:
            raise KeyError(f"{base}.wkv.biases")
        q_weight = tensors[f"{base}.q_weight"]
        k_weight = tensors[f"{base}.k_weight"]
    except KeyError as exc:
        raise KeyError(f"layer {layer_id} residents missing from {sidecar}: {exc}") from exc

    hc_mult = int(q_weight.shape[0])
    dim = int(q_weight.shape[1])
    out_width = int(packed.shape[0])
    if out_width != dim * (hc_mult + 1):
        raise ValueError(
            f"wkv out width {out_width} != dim*(hc_mult+1) {dim * (hc_mult + 1)} "
            f"(q_weight shape {tuple(q_weight.shape)})"
        )

    if mode == "mxfp8":
        def wkv(x: mx.array, _p=packed, _s=scales, _g=group, _bits=bits) -> mx.array:
            return mx.quantized_matmul(x, _p, _s, transpose=True,
                                       group_size=_g, bits=_bits, mode="mxfp8")
    else:
        def wkv(x: mx.array, _p=packed, _s=scales, _b=biases, _g=group, _bits=bits) -> mx.array:
            return mx.quantized_matmul(x, _p, _s, _b, transpose=True,
                                       group_size=_g, bits=_bits, mode="affine")

    return EngramResidents(
        layer_id=int(layer_id), wkv=wkv, q_weight=q_weight, k_weight=k_weight,
        wkv_packed=packed, wkv_scales=scales, wkv_biases=biases,
        dim=dim, hc_mult=hc_mult, group_size=group, bits=bits, mode=mode,
    )
