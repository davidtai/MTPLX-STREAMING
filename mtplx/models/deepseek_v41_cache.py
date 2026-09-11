"""DeepSeek-V4.1-Flash per-sequence attention STATE (W13).

Everything ``inference/model.py`` (the DeepSeek MIT reference) keeps as
per-sequence attention state, and how it is updated per forward call, lifted out
of the weight-bearing modules into one cache object so the MTPLX serve path can
own it (many sequences share one set of weights; the state trims/rolls back for
speculative verify and the resident==stream gate).  The arithmetic that lives on
learned weights -- ``wkv``/``wgate``/``wk``/``weights_proj``/``wq_b``, RMSNorm,
RoPE, quantisation -- stays in ``deepseek_v41.py`` (W10) and the MoE module
(W11); this module owns only the *stored, mutated* state and the pure index/roll
mechanics that update it, so every method here is weight-free and unit-testable
on random arrays.

Reference state -> this module's field
--------------------------------------
The reference keeps state as buffers on the weight modules.  ``max_batch_size`` /
``max_seq_len`` prealloc is dropped: the serve path grows history append-only and
the reference's fixed ring / group buffers are realised as pure *views* of that
history, so the trim/rollback seam can undo any depth (phase 1 keeps full history
in bf16/float; phase 2 bounds it -- see W13_REPORT.md).

* ``Attention.window_kv_cache``  ([B, window_size, head_dim] ring, model.py
  L663-668, written L708-719, read L716/719) -> ``LayerAttentionCache.window``
  (append-only post-RoPE rows) + :func:`ring_view` / :meth:`LayerAttentionCache.ring`,
  which reproduces the reference ring's slot layout *exactly* (slot ``s`` holds
  the newest fed position ``p`` with ``p % window_size == s``).  The window
  attend indices are :func:`window_topk_idxs` (reference ``get_window_topk_idxs``
  L409-426).
* ``Attention.compress_kv_cache``  ([B, max_seq_len//ratio, head_dim], model.py
  L669-679, written L761, published L748) -> ``LayerAttentionCache.compress_kv``
  (append-only, one row per completed group).
* ``Indexer.k_cache``  ([B, max_seq_len//ratio, index_head_dim], model.py
  L520-525, written L547, published L548) -> ``LayerAttentionCache.index_k``.
* ``Compressor.kv_state`` / ``Compressor.score_state``  ([B, ratio, head_dim],
  ``score_state`` -inf-filled, model.py L449-456, updated L466-485) ->
  :class:`CompressorState` (``raw_kv`` / ``raw_score`` retain every fed row; the
  reference's current partial group is the trailing ``n_fed % ratio`` of them).
  The reference pads unfilled ``score_state`` slots with ``-inf``, but it only
  ever pools a *complete* group (all ``ratio`` slots filled, model.py L481-482 /
  L467-475), so those pad slots never enter a softmax and an unpadded tail is
  exactly equivalent.
* ``shared_attn`` (module-global ``SharedAttentionRuntime``, model.py L1166-1180)
  -> :class:`SharedAttentionRuntime` (one per cache, not a process global).
* ``start_pos`` bookkeeping threaded through every forward (model.py forward
  signatures; ``generate.py`` increments it) -> ``DeepseekV41Cache.offset``.
* ``Transformer.engram_hash`` streaming n-gram history
  (``NgramHashState.cache``, engram.py L155-157, advanced L167, look-back
  L169-175) -> ``DeepseekV41Cache.engram_state`` -- a
  :class:`mtplx.engram_v41.NgramHashState` clone whose ``advance`` / ``trim`` the
  W10 backbone and this cache's :meth:`~DeepseekV41Cache.trim` drive in step.

Call surface for W10 / the serve path is frozen in docs/deepseek-v41/PORT_CONTRACT.md
(W13 heading).
"""

from __future__ import annotations

from typing import Optional, Sequence

import mlx.core as mx

#: Reference ``ModelArgs.window_size`` default and released config value.
WINDOW_SIZE_DEFAULT = 128

#: Version tag for a per-layer entry's ``meta_state`` (mlx_lm session contract).
_LAYER_META_VERSION = "mtplx-deepseek-v41-layer-cache-v1"


# ---------------------------------------------------------------------------
# append-only history primitives (the phase-1 backing store)
# ---------------------------------------------------------------------------
def _grow(rows: Optional[mx.array], new: Optional[mx.array]) -> Optional[mx.array]:
    """Append ``new`` rows along the sequence axis (axis 1) of an append-only
    store.  ``None`` seeds it; a zero-length ``new`` is a no-op."""
    if new is None or (new.ndim >= 2 and new.shape[1] == 0):
        return rows
    if rows is None:
        return new
    return mx.concatenate([rows, new], axis=1)


def _truncate(rows: Optional[mx.array], n: int) -> Optional[mx.array]:
    """Keep the first ``n`` rows along axis 1; ``n == 0`` drops the store."""
    if rows is None:
        return None
    if n <= 0:
        return None
    if n >= rows.shape[1]:
        return rows
    return rows[:, :n]


def _rows(rows: Optional[mx.array]) -> int:
    return 0 if rows is None else rows.shape[1]


# ---------------------------------------------------------------------------
# Sliding-window ring (reference get_window_topk_idxs L409-426, _window_kv L700-720)
# ---------------------------------------------------------------------------
def window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int) -> mx.array:
    """Which sliding-window slots each query attends to; ``-1`` marks an empty
    slot.  Byte-for-byte transliteration of ``get_window_topk_idxs``
    (model.py L409-426): ``[bsz, m, topk]`` int32 with ``m == seqlen`` for a
    ``start_pos == 0`` prefill (one causal window per query, indices into the
    chunk) and ``m == 1`` for a decode step (the whole ring, oldest first).
    """
    win = window_size
    if start_pos == 0:
        end = mx.arange(seqlen).reshape(seqlen, 1)                      # L418
        idxs = mx.maximum(end - win + 1, 0) + mx.arange(min(seqlen, win))  # L419
        idxs = mx.where(idxs > end, -1, idxs)                          # L420
    else:
        oldest = start_pos % win + 1                                   # L422
        idxs = mx.concatenate([mx.arange(oldest, win), mx.arange(oldest)])  # L423
        idxs = mx.where(idxs > start_pos, -1, idxs)                    # L424
        idxs = idxs.reshape(1, -1)
    idxs = idxs.astype(mx.int32)                                       # L426
    return mx.broadcast_to(idxs[None], (bsz, idxs.shape[0], idxs.shape[1]))


def ring_view(window: Optional[mx.array], window_size: int, length: int) -> Optional[mx.array]:
    """The reference ``window_kv_cache[:bsz]`` contents after ``length`` tokens.

    Given the append-only ``window`` history ([B, T>=length, head_dim] of
    post-RoPE rows) this reproduces the reference ring exactly (model.py
    L708-719): a full ``[B, window_size, head_dim]`` buffer where slot ``s`` holds
    the newest fed position ``p < length`` with ``p % window_size == s``, and any
    slot with no such position (only while ``length < window_size``) reads zero,
    matching the zero-initialised ``window_kv_cache`` (L664-668).  Returning the
    full ring -- not the filled prefix -- is what the reference decode reads
    (``window_kv = self.window_kv_cache[:bsz]``, L719) and is what pairs with the
    slot indices of :func:`window_topk_idxs`, whose ``-1`` entries mask the empty
    slots.
    """
    if window is None or length <= 0:
        return None
    win = window_size
    slots = mx.arange(win)
    # newest position congruent to each slot, mod win, at or below length-1
    pos = (length - 1) - ((length - 1 - slots) % win)                 # L712-716 layout
    valid = slots < min(length, win)                                  # empty slots while filling
    rows = window[:, mx.clip(pos, 0, length - 1)]
    return mx.where(valid[None, :, None], rows, 0)


# ---------------------------------------------------------------------------
# Compressor incremental frontier (reference Compressor.kv_state/score_state
# + pooling, model.py L449-485)
# ---------------------------------------------------------------------------
class CompressorState:
    """Carries the compressor's partial pooling group across forward calls and
    pools each completed group (reference ``Compressor``, model.py L449-485).

    Instantiated only for ``compress_ratio > 1`` layers -- the reference keeps
    ``kv_state`` / ``score_state`` only there (L447-456); ``ratio == 1`` is a
    plain per-token projection with no gate and no state (L461-462), handled by
    W10 with ``comp_state is None``.

    ``push`` takes the *already projected* ``kv = wkv(x)`` and ``score =
    wgate(x)`` (both fp32, W10's weights) and returns the pre-RoPE, pre-norm
    pooled latents for whatever groups this push completed -- the softmax-gated
    sum of model.py L475 / L482.  W10 applies the compressor's RMSNorm to the
    result (reference ``self.norm(...)`` L462/L485) and then RoPE.  Retaining the
    raw rows (rather than the reference's fixed ``ratio``-slot buffer) is what
    lets :meth:`trim` restore the exact frontier after a rollback that crosses a
    group boundary.
    """

    def __init__(self, ratio: int):
        if ratio <= 1:
            raise ValueError("CompressorState is only for compress_ratio > 1")
        self.ratio = int(ratio)
        self.raw_kv: Optional[mx.array] = None      # [B, n_fed, head_dim] fp32
        self.raw_score: Optional[mx.array] = None    # [B, n_fed, head_dim] fp32

    @property
    def n_fed(self) -> int:
        """Rows fed so far == tokens this layer has processed."""
        return _rows(self.raw_kv)

    @property
    def n_groups(self) -> int:
        """Completed pooled groups so far (== rows in this layer's compress_kv)."""
        return self.n_fed // self.ratio

    def push(self, kv: mx.array, score: mx.array) -> mx.array:
        """Append ``kv``/``score`` ([B, L, head_dim]) and return the pooled
        latents ``[B, g, head_dim]`` for the groups completed by this push
        (``g >= 0``; empty when the group is still filling).

        One code path covers prefill and decode: the reference prefill pools
        ``floor(seqlen / ratio)`` groups and parks the remainder (L466-475), and
        the reference decode pools one group when it just completed (L476-485);
        both are "pool every group whose ``ratio`` rows are now present", which
        is exactly what a running row count gives.  Pooling reads only a group's
        own rows, so the result is independent of how the rows were chunked.
        """
        n_before = self.n_fed
        self.raw_kv = _grow(self.raw_kv, kv)
        self.raw_score = _grow(self.raw_score, score)
        n_after = self.n_fed
        g_before = n_before // self.ratio
        g_after = n_after // self.ratio
        if g_after == g_before:
            return kv[:, :0]  # nothing completed; keep dtype/shape for concat
        lo = g_before * self.ratio
        hi = g_after * self.ratio
        b = self.raw_kv.shape[0]
        d = self.raw_kv.shape[-1]
        grp_kv = self.raw_kv[:, lo:hi].reshape(b, g_after - g_before, self.ratio, d)
        grp_sc = self.raw_score[:, lo:hi].reshape(b, g_after - g_before, self.ratio, d)
        # softmax over the ratio axis (reference softmax(dim=2)/(dim=1), L475/L482)
        pooled = mx.sum(grp_kv * mx.softmax(grp_sc, axis=2), axis=2)
        return pooled

    def trim(self, n: int) -> None:
        """Drop the last ``n`` fed rows, restoring the frontier n tokens back."""
        if n < 0:
            raise ValueError("trim count must be >= 0")
        if n == 0:
            return
        keep = self.n_fed - n
        if keep < 0:
            raise ValueError(f"cannot trim {n} of {self.n_fed} compressor rows")
        self.raw_kv = _truncate(self.raw_kv, keep)
        self.raw_score = _truncate(self.raw_score, keep)

    def mark(self) -> int:
        """Snapshot for :meth:`rollback` -- the row count fully determines the
        state because the retained arrays are immutable."""
        return self.n_fed

    def rollback(self, mark: int) -> None:
        self.raw_kv = _truncate(self.raw_kv, mark)
        self.raw_score = _truncate(self.raw_score, mark)


# ---------------------------------------------------------------------------
# Shared cross-layer runtime (reference SharedAttentionRuntime, model.py L1166-1180)
# ---------------------------------------------------------------------------
class SharedAttentionRuntime:
    """One slot each for what a source layer hands down the stack this forward:
    the group's compressed KV and index keys, the index-source's selected rows,
    and the candidate-source's candidate-block mask.  Layers run in order and
    every source writes before its consumers read, so one slot is enough and
    nothing needs resetting between forwards (reference ``SharedAttentionRuntime``
    L1166-1180).  Unlike the reference module-global, one of these belongs to
    each cache, so concurrent sequences never collide.

    Sources: ``compress_kv`` and ``index_k`` from ``kv_source_layers``,
    ``topk_idxs`` from ``index_source_layers``, ``candidates`` from
    ``candidate_source_layer``.
    """

    def __init__(self):
        self.compress_kv: Optional[mx.array] = None   # [B, n_comp, head_dim]  (L1173)
        self.index_k: Optional[mx.array] = None        # [B, n_comp, index_head_dim] (L1174)
        self.topk_idxs: Optional[mx.array] = None      # index source's published selection (L1175)
        self.candidates: Optional[mx.array] = None     # candidate-block mask (L1176)

    # W10's Attention publishes the selection as a boolean row mask rather than
    # the reference's integer indices; ``topk_mask`` is that view's name for the
    # same ``topk_idxs`` slot (see PORT_CONTRACT.md W13).
    @property
    def topk_mask(self) -> Optional[mx.array]:
        return self.topk_idxs

    @topk_mask.setter
    def topk_mask(self, value: Optional[mx.array]) -> None:
        self.topk_idxs = value


# ---------------------------------------------------------------------------
# Per-layer attention state
# ---------------------------------------------------------------------------
class LayerAttentionCache:
    """The append-only window / compressed-KV / index-key history for one layer,
    plus its compressor frontier and the trim/rollback seam.

    ``compress_ratio`` and ``is_kv_source`` come from the model config so
    :meth:`trim` can restore the compressed stores exactly (they hold one row per
    completed group of ``compress_ratio`` tokens, so the reachable count is a
    function of the token length).  A pure sliding-window layer
    (``compress_ratio == 0``) keeps only ``window``.
    """

    def __init__(self, window_size: int = WINDOW_SIZE_DEFAULT,
                 compress_ratio: int = 0, is_kv_source: bool = False,
                 engram_state=None):
        self.window_size = int(window_size)
        self.compress_ratio = int(compress_ratio)
        self.is_kv_source = bool(is_kv_source)
        #: this entry's own position counter == the mlx_lm ``cache.offset`` the
        #: serve/generate path reads and trims per entry (all entries of one
        #: sequence advance/trim in lockstep, so ``cache[0].offset`` is the
        #: sequence position -- reference ``start_pos``).
        self.offset: int = 0
        #: streaming engram n-gram history, when this entry owns it (only the
        #: first entry of a sequence does -- see :class:`DeepseekV41Cache`); its
        #: rewind rides this entry's :meth:`trim`/:meth:`rollback` so a per-entry
        #: trim moves the shared history exactly once.  ``None`` otherwise.
        self.engram_state = engram_state
        #: post-RoPE window KV rows, one per token (reference window_kv_cache seed)
        self.window: Optional[mx.array] = None
        #: pooled+RoPE'd compressed KV, one per completed group (compress_kv_cache)
        self.compress_kv: Optional[mx.array] = None
        #: index keys, one per completed group (Indexer.k_cache)
        self.index_k: Optional[mx.array] = None
        #: compressor frontier (CompressorState for ratio>1, else None)
        self.comp_state: Optional[CompressorState] = (
            CompressorState(self.compress_ratio)
            if self.is_kv_source and self.compress_ratio > 1
            else None
        )

    # -- window (reference _window_kv L700-720) -----------------------------
    def append_window(self, kv_new: mx.array) -> None:
        """Seed the ring with this call's post-RoPE window KV (reference
        L708-719).  History is kept append-only; the reference's fixed ring is
        :meth:`ring`."""
        self.window = _grow(self.window, kv_new)

    def ring(self, length: int) -> Optional[mx.array]:
        """The reference ``window_kv_cache`` view after ``length`` tokens
        (reference read L716/L719)."""
        return ring_view(self.window, self.window_size, length)

    def window_len(self) -> int:
        return _rows(self.window)

    # -- compressed KV (reference compress_kv_cache write L761) --------------
    def append_compress(self, compress_new: mx.array) -> None:
        """Append pooled, RoPE'd, (phase-2) quantised compressed latents; one row
        per completed group (reference L761)."""
        self.compress_kv = _grow(self.compress_kv, compress_new)

    # -- index keys (reference Indexer.k_cache write L547) ------------------
    def append_index_k(self, index_new: mx.array) -> None:
        """Append this group's index keys (reference L547)."""
        self.index_k = _grow(self.index_k, index_new)

    # -- length / mlx_lm per-entry contract --------------------------------
    def advance(self, n: int) -> None:
        """Record ``n`` more processed tokens on this entry's own position
        counter (reference ``start_pos += seqlen``, kept per entry so the serve
        path reads ``cache[i].offset``).  The per-layer stores are grown by the
        layer forward itself; this only moves the entry offset."""
        self.offset += int(n)

    def is_trimmable(self) -> bool:
        """Every lane here rewinds exactly -- window/compressed/index truncate to
        the shorter length, the compressor frontier and the engram history
        truncate their append-only journals -- so the engine's snapshot-free
        verify repair (``mtplx.cache_state.trim_verified_window_without_snapshot``)
        can drive this entry with a plain :meth:`trim`."""
        return True

    def size(self) -> int:
        return int(self.offset)

    def empty(self) -> bool:
        return self.offset == 0

    # -- trim / rollback seam ----------------------------------------------
    def trim(self, n: int) -> int:
        """Restore this entry to ``n`` tokens earlier and return the number of
        tokens trimmed (mlx_lm cache ``trim`` contract).  The window drops its
        last ``n`` rows; the compressed stores drop back to the number of groups
        the shortened length still completes (``new_len // compress_ratio``); the
        compressor frontier re-exposes the shortened partial group; the engram
        history (only the entry that owns it) drops its last ``n`` fed positions.
        ``offset`` decreases by exactly ``n`` -- the per-entry property the
        verify repair (``trim_verified_window_to_prefix``) checks."""
        n = int(n)
        if n < 0:
            raise ValueError("trim count must be >= 0")
        if n == 0:
            return 0
        cur = int(self.offset)
        new_len = cur - n
        if new_len < 0:
            raise ValueError(f"cannot trim {n} of {cur} tokens")
        self.window = _truncate(self.window, new_len)
        if self.is_kv_source and self.compress_ratio >= 1:
            groups = new_len // self.compress_ratio
            self.compress_kv = _truncate(self.compress_kv, groups)
            self.index_k = _truncate(self.index_k, groups)
            if self.comp_state is not None:
                self.comp_state.trim(n)
        if self.engram_state is not None:
            self.engram_state.trim(n)
        self.offset = new_len
        return n

    def mark(self):
        return (
            int(self.offset),
            self.window_len(),
            _rows(self.compress_kv),
            _rows(self.index_k),
            None if self.comp_state is None else self.comp_state.mark(),
            None if self.engram_state is None else int(self.engram_state.length),
        )

    def rollback(self, mark) -> None:
        offset, nw, nc, ni, comp_mark, engram_len = mark
        if self.engram_state is not None and engram_len is not None:
            cur = int(self.engram_state.length)
            if cur > engram_len:
                self.engram_state.trim(cur - engram_len)
        self.window = _truncate(self.window, nw)
        self.compress_kv = _truncate(self.compress_kv, nc)
        self.index_k = _truncate(self.index_k, ni)
        if self.comp_state is not None and comp_mark is not None:
            self.comp_state.rollback(comp_mark)
        self.offset = int(offset)

    # -- mlx_lm session state contract -------------------------------------
    @property
    def state(self):
        """The append-only KV lanes as a tuple of arrays (mlx_lm ``cache.state``)
        for :func:`mtplx.cache_state.snapshot_cache` / :func:`restore_cache`.  The
        compressor frontier rows travel with it so a snapshot restore rebuilds the
        exact frontier.  The engram history is deliberately NOT in ``state`` -- it
        rewinds through :meth:`trim`, the only rollback the served trunk cache
        drives (session/SSD save-restore is disabled for this model; see
        docs/deepseek-v41/W22_REPORT.md)."""
        comp_kv = None if self.comp_state is None else self.comp_state.raw_kv
        comp_sc = None if self.comp_state is None else self.comp_state.raw_score
        return (self.window, self.compress_kv, self.index_k, comp_kv, comp_sc)

    @state.setter
    def state(self, value) -> None:
        if value is None:
            self.window = None
            self.compress_kv = None
            self.index_k = None
            if self.comp_state is not None:
                self.comp_state.raw_kv = None
                self.comp_state.raw_score = None
            return
        window, compress_kv, index_k, comp_kv, comp_sc = value
        self.window = window
        self.compress_kv = compress_kv
        self.index_k = index_k
        if self.comp_state is not None:
            self.comp_state.raw_kv = comp_kv
            self.comp_state.raw_score = comp_sc

    def replace_state(self, value) -> None:
        self.state = value

    @property
    def meta_state(self):
        return (
            _LAYER_META_VERSION,
            str(int(self.offset)),
            str(int(self.window_size)),
            str(int(self.compress_ratio)),
            "1" if self.is_kv_source else "0",
        )

    @meta_state.setter
    def meta_state(self, value) -> None:
        if value is None:
            return
        if (
            not isinstance(value, (tuple, list))
            or len(value) != 5
            or value[0] != _LAYER_META_VERSION
        ):
            raise ValueError(
                f"unsupported DeepSeek-V4.1 layer cache meta state: {value!r}"
            )
        self.offset = int(value[1])


# Inline-name aliases (the names W10's in-progress code and the serve path use).
_LayerCache = LayerAttentionCache
_SharedRuntime = SharedAttentionRuntime


# ---------------------------------------------------------------------------
# Top-level per-sequence cache
# ---------------------------------------------------------------------------
class DeepseekV41Cache:
    """Per-sequence attention state, presented as the mlx_lm *list of per-layer
    caches* the MTPLX serve/generate path consumes.

    Iterating or indexing this object yields the per-layer
    :class:`LayerAttentionCache` entries -- ``mlx_lm.models.cache.make_prompt_cache``
    returns it from :meth:`Model.make_cache`, and ``mtplx.generation`` /
    ``mtplx.runtime`` then treat it as the standard ``list[cache]`` (iterate for
    ``_cache_has_recurrent_entries`` / ``snapshot_cache``, index ``cache[0].offset``
    for the position, call ``cache[i].trim`` per entry to un-decode a rejected
    speculative tail).  The cross-layer state the reference keeps process-global
    -- a fresh :class:`SharedAttentionRuntime` per forward and the streaming
    engram n-gram history -- stays reachable as :meth:`new_shared_runtime` and
    :attr:`engram_state` for the W10 backbone, and the running position is
    ``cache[i].offset`` (every entry advances/trims in lockstep).

    This mirrors ``mtplx.models.deepseek_v4.Model.make_cache`` (a list of
    per-layer :class:`~mtplx.models.deepseek_v4.DeepseekV4Cache`), with the one
    difference V4.1 forces: the engram history is a single per-sequence object,
    so the first entry owns its rewind (its :meth:`~LayerAttentionCache.trim` /
    rollback moves it exactly once even though the serve path trims every entry)
    and it is also published here for the backbone to advance.

    :meth:`trim` and :meth:`mark`/:meth:`rollback` are the W13 whole-sequence
    seam (kept for the unit tests and the bare-forward path); the served verify
    path drives the per-entry :meth:`~LayerAttentionCache.trim` instead.  Both
    restore the state -- KV, compressor frontier and engram history together --
    to exactly what it was ``n`` tokens earlier, so a rejected draft re-feeds to
    identical logits.
    """

    def __init__(
        self,
        n_layers: int,
        *,
        window_size: int = WINDOW_SIZE_DEFAULT,
        compress_ratios: Optional[Sequence[int]] = None,
        kv_source_layer_ids: Sequence[int] = (),
        engram_state=None,
    ):
        kv_sources = set(int(i) for i in kv_source_layer_ids)
        ratios = list(compress_ratios) if compress_ratios is not None else [0] * n_layers
        if len(ratios) < n_layers:
            ratios = ratios + [0] * (n_layers - len(ratios))
        self.window_size = int(window_size)
        self.layers = [
            LayerAttentionCache(
                window_size=window_size,
                compress_ratio=ratios[i],
                is_kv_source=i in kv_sources,
            )
            for i in range(n_layers)
        ]
        #: the engram history is one object per sequence; the first entry owns
        #: its rewind so a per-entry ``trim`` (the rollback the serve path calls)
        #: moves it exactly once, and the backbone reaches it via
        #: :attr:`engram_state`.
        if self.layers:
            self.layers[0].engram_state = engram_state
        #: an :class:`mtplx.engram_v41.NgramHashState` clone, or None when engram
        #: is not wired; advanced by the W10 backbone (once per forward, before
        #: the layers), rewound through the owning entry in step.
        self.engram_state = engram_state

    # -- mlx_lm sequence protocol (a list of per-layer caches) -------------
    def __iter__(self):
        return iter(self.layers)

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, idx):
        return self.layers[idx]

    def __setitem__(self, idx, value) -> None:
        # The serve path's cache-layout installers (``install_tail_owned_...``)
        # rebind entries by index; ours are skipped (no ``keys``/``values``), but
        # honour the contract so a future layout swap stays consistent.
        self.layers[idx] = value

    def __bool__(self) -> bool:
        return bool(self.layers)

    @property
    def offset(self) -> int:
        """The sequence position (reference ``start_pos``); every entry tracks
        its own and they advance/trim in lockstep, so the first entry's is the
        sequence's."""
        return int(self.layers[0].offset) if self.layers else 0

    # a fresh shared runtime per forward (reference re-uses one module global;
    # the backbone calls this once at the top of every forward and threads it
    # through the layers)
    def new_shared_runtime(self) -> SharedAttentionRuntime:
        return SharedAttentionRuntime()

    # -- length bookkeeping -------------------------------------------------
    def advance(self, n: int) -> None:
        """Record that ``n`` more tokens were processed (reference ``start_pos +=
        seqlen``) on every entry.  The per-layer stores are grown by the layers
        themselves and the engram is advanced by the backbone; this only moves
        each entry's position counter."""
        for layer in self.layers:
            layer.advance(n)

    # -- trim / rollback seam ----------------------------------------------
    def trim(self, n: int) -> int:
        """Drop the last ``n`` tokens from every entry (KV, compressor frontier
        and -- through the owning entry -- the engram history), restoring the
        state to ``n`` tokens earlier.  Returns the number of tokens trimmed
        (``mlx_lm`` cache convention)."""
        if n < 0:
            raise ValueError("trim count must be >= 0")
        n = min(int(n), self.offset)
        if n == 0:
            return 0
        for layer in self.layers:
            layer.trim(n)
        return n

    def is_trimmable(self) -> bool:
        return True

    def mark(self):
        return tuple(layer.mark() for layer in self.layers)

    def rollback(self, mark) -> None:
        for layer, m in zip(self.layers, mark):
            layer.rollback(m)


def make_cache(model_or_n_layers, *, window_size: Optional[int] = None,
               engram_state=None) -> DeepseekV41Cache:
    """Build a fresh :class:`DeepseekV41Cache`.

    Accepts either a ModelArgs-like object (``num_hidden_layers`` /
    ``window_size`` / ``compress_ratios`` / ``kv_source_layer_ids``) or a plain
    ``n_layers`` int.  W10's ``Model.make_cache(self)`` calls this with the model
    args and then attaches the per-sequence engram clone; the serve-path loader
    and the gate reach it through ``mlx_lm.make_prompt_cache(model)`` ->
    ``model.make_cache()`` (see PORT_CONTRACT.md W13).
    """
    if isinstance(model_or_n_layers, int):
        n_layers = model_or_n_layers
        ws = window_size if window_size is not None else WINDOW_SIZE_DEFAULT
        return DeepseekV41Cache(n_layers, window_size=ws, engram_state=engram_state)
    args = model_or_n_layers
    n_layers = int(args.num_hidden_layers)
    ws = window_size if window_size is not None else int(getattr(args, "window_size", WINDOW_SIZE_DEFAULT))
    return DeepseekV41Cache(
        n_layers,
        window_size=ws,
        compress_ratios=getattr(args, "compress_ratios", None),
        kv_source_layer_ids=getattr(args, "kv_source_layer_ids", ()),
        engram_state=engram_state,
    )
