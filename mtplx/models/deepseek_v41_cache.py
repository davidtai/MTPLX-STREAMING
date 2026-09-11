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

import os
from typing import Optional, Sequence

import mlx.core as mx

#: Reference ``ModelArgs.window_size`` default and released config value.
WINDOW_SIZE_DEFAULT = 128

#: W73 / K32: chunk-grown append backing for the window / compressed-KV / index-key
#: stores.  The phase-1 backing (:func:`_grow`) re-``concatenate``s the WHOLE store
#: on every token, an O(current-length) copy per layer per token -- at T=16384 the
#: window append alone is ~2 ms/layer on the CPU double (measured), ~40x its 1K cost
#: and, across 40 layers, the dominant per-token O(T) work once K30 selected keys
#: has already bounded the attention score (docs/deepseek-v41/W73_DECODE_16K_AUDIT.md).
#: Under this flag the stores grow via a geometric-capacity buffer with a logical
#: length and a donated ``mx.slice_update`` in-place write (amortized O(new-rows) per
#: token -- the resize concatenate fires only on the O(log T) doublings).  The logical
#: view ``buf[:, :length]`` is BYTE-IDENTICAL to the concatenated store, so every
#: downstream read (attention score/gather, indexer, trim/rollback, mlx_lm state) is
#: unchanged.  Read at construction (per request, after the serve harness stamps the
#: key; NOT frozen at import -- [[env-flags-read-at-use-not-import]]).  Default OFF ->
#: the plain :func:`_grow` path, byte-for-byte the shipped cache.
_KV_CHUNK_GROW_ENV = "MTPLX_DSV41_KV_CHUNK_GROW"


def _kv_chunk_grow_enabled() -> bool:
    """Whether ``MTPLX_DSV41_KV_CHUNK_GROW`` arms the chunk-grown append backing.

    Read at call time (never frozen at import): the serving harness stamps the key
    after importing this module, and each request builds a fresh cache."""
    return (os.environ.get(_KV_CHUNK_GROW_ENV) or "").strip().lower() in (
        "1", "true", "yes", "on",
    )

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


#: W73 engagement + O(T) telemetry.  ``rows_copied`` is the LOGICAL rows an append
#: copies under the chosen strategy: a chunk-grown donated ``slice_update`` write
#: touches only ``new`` rows (a geometric resize copies the live prefix once);
#: ``_grow`` (the plain path) copies the whole live prefix every time.  ``layers_*``
#: prove engagement (how many layer caches picked each backing); ``buffers`` counts
#: the geometric buffers allocated (>0 iff chunk-grow actually built one).  Process-
#: global and cumulative (like the Sinkhorn counters); the ab harness resets after
#: model load and snapshots after the run.  NOTE: ``rows_copied`` is the STRATEGY's
#: logical cost, not what MLX's Metal backend physically moves -- compare it against
#: the measured ``cache_append`` census stage to detect a slice_update that did not
#: donate in-place (logical ~O(1) but the stage still O(T) == no donation on Metal).
_KV_STATS = {
    "layers_chunk_grown": 0,
    "layers_plain": 0,
    "buffers": 0,
    "appends": 0,
    "rows_copied": 0,
}


def reset_kv_chunk_grow_stats() -> None:
    """Zero the W73 telemetry (call after model load to scope it to one run)."""
    for k in _KV_STATS:
        _KV_STATS[k] = 0


def kv_chunk_grow_stats() -> dict:
    """Snapshot the W73 telemetry: ``enabled`` (any layer cache chose the chunk-grown
    backing), per-backing layer counts, geometric buffers allocated, total appends
    and the logical rows copied across them."""
    s = dict(_KV_STATS)
    s["enabled"] = bool(_KV_STATS["layers_chunk_grown"] > 0)
    return s


def _note_rows_copied(n: int) -> None:
    _KV_STATS["appends"] += 1
    _KV_STATS["rows_copied"] += int(n)


# Back-compat alias for the W73 unit test's direct read of the cumulative counter.
def _rows_copied_total() -> int:
    return _KV_STATS["rows_copied"]


class _GrowBuffer:
    """Geometric-capacity append backing for one store lane (W73 / K32).

    Holds a ``[b, cap, *tail]`` buffer and a logical ``length``; :meth:`append`
    writes the new rows in place with a donated ``mx.slice_update`` (amortized
    O(new-rows) when the previous buffer is unreferenced at the write -- confirmed
    on the CPU double: ~constant per step across cap 4k/16k/64k, vs ``concatenate``'s
    O(cap)), growing the capacity geometrically so the copy-everything resize fires
    only O(log T) times.  :meth:`view` returns ``buf[:, :length]`` -- BYTE-IDENTICAL
    to the equivalent :func:`_grow` (concatenate) store, so no downstream reader
    changes.  :meth:`set` rebuilds from a full array (trim/rollback/state restore).

    Correctness never depends on the donation: if a live view keeps the buffer
    referenced at write time MLX copies instead of donating -- slower (same class as
    the plain path) but the same bytes.
    """

    __slots__ = ("_buf", "_len", "_init_cap")

    def __init__(self, init_cap: int = 256):
        self._buf: Optional[mx.array] = None
        self._len: int = 0
        self._init_cap = int(init_cap)

    @staticmethod
    def _starts(ndim: int, row: int) -> mx.array:
        return mx.array([0, int(row)] + [0] * (ndim - 2), dtype=mx.int32)

    def _write(self, buf: mx.array, new: mx.array, row: int) -> mx.array:
        axes = tuple(range(new.ndim))
        return mx.slice_update(buf, new, self._starts(new.ndim, row), axes=axes)

    def append(self, new: Optional[mx.array]) -> None:
        if new is None or (new.ndim >= 2 and new.shape[1] == 0):
            return
        n = int(new.shape[1])
        if self._buf is None:
            cap = max(self._init_cap, n)
            tail = tuple(new.shape[2:])
            buf = mx.zeros((new.shape[0], cap) + tail, dtype=new.dtype)
            self._buf = self._write(buf, new, 0)
            self._len = n
            _KV_STATS["buffers"] += 1
            _note_rows_copied(n)
            return
        cap = int(self._buf.shape[1])
        if self._len + n <= cap:
            # in-place donated write of just the new rows
            self._buf = self._write(self._buf, new, self._len)
            self._len += n
            _note_rows_copied(n)
            return
        # geometric resize: copy the live prefix once into a larger buffer
        new_cap = max(cap * 2, self._len + n)
        head = self._buf[:, : self._len]
        tail = tuple(new.shape[2:])
        buf = mx.zeros((new.shape[0], new_cap) + tail, dtype=new.dtype)
        buf = self._write(buf, head, 0)
        buf = self._write(buf, new, self._len)
        self._buf = buf
        _KV_STATS["buffers"] += 1  # geometric resize allocation
        _note_rows_copied(self._len + n)
        self._len += n

    def view(self) -> Optional[mx.array]:
        if self._buf is None or self._len == 0:
            return None
        if self._len == self._buf.shape[1]:
            return self._buf
        return self._buf[:, : self._len]

    def set(self, arr: Optional[mx.array]) -> None:
        """Replace the whole store from a full array (or clear on ``None``)."""
        self._buf = None
        self._len = 0
        self.append(arr)

    def truncate_to(self, n: int) -> None:
        """Drop back to the first ``n`` logical rows (length-only; keeps capacity)."""
        n = max(0, int(n))
        if n <= 0:
            self._buf = None
            self._len = 0
        else:
            self._len = min(n, self._len)

    def rows(self) -> int:
        return 0 if self._buf is None else self._len

    def raw_backing(self) -> Optional[mx.array]:
        """The whole preallocated buffer (NOT the logical ``view()`` slice).

        :meth:`_eval_cache_state` forces this instead of ``view()`` so a settle
        fence does not materialise a fresh ``[b, length]`` slice per token (W80
        item 4): the buffer already holds the in-place-written bytes, so forcing it
        realises only this step's ``slice_update`` writes, not an O(length) copy."""
        return self._buf


# ---------------------------------------------------------------------------
# W80 / K34: bounded sliding-window RING + preallocated compress/index stores
# ---------------------------------------------------------------------------
#: W80: the window store is a genuine sliding window of ``window_size`` (128); the
#: decode gather / SWA mask only ever read the last ``window_size`` rows.  The
#: phase-1 store keeps FULL history append-only (row j == token j) so at T=16384 the
#: resident window is ~40x16384x512x2 = ~0.7 GB, and W76/W78 pinned that resident
#: churn as the memory-pressure amplifier that inflates the whole 16K decode step
#: (attention-proper 1.7->6.6 ms/layer, uniform across CSA modes).  Under
#: ``MTPLX_DSV41_WINDOW_RING`` the window store is a BOUNDED ring of
#: ``window_size + max_verify + slack`` rows (~136) in a fixed pair of ping-pong
#: buffers allocated once at construction (~5.5 MB total across 40 layers, a ~128x
#: cut), and the compress_kv / index_k stores (which the indexer needs in FULL, so
#: they cannot be bounded) are preallocated to ``max_kv`` and written in place --
#: removing the per-token full-store realloc for all three lanes.  A logical
#: ``drop_offset`` (== the number of dropped front rows) is threaded through
#: ``_window_selected_idx`` / ``_window_attend`` / the SWA mask so they address the
#: SAME absolute positions as the full store; dropped rows are always >= window_size
#: behind the newest query, hence always masked to -inf / never gathered, so every
#: reachable read is BYTE-IDENTICAL to the full store by construction.  Default OFF.
#: Read at construction (per request, after the harness stamps the key; NOT frozen
#: at import -- [[env-flags-read-at-use-not-import]]).
_WINDOW_RING_ENV = "MTPLX_DSV41_WINDOW_RING"
#: Tuning knobs (env, read at construction).  ``max_verify`` sizes the ring for the
#: widest speculative-verify block appended in ONE decode forward (DSpark depth 3 ->
#: K+1 = 4 rows); ``slack`` a safety margin; ``headroom`` the number of in-place
#: appends between compactions (bigger => fewer compaction copies, more resident
#: rows); ``maxkv`` the preallocated capacity of the compress/index lanes (0/unset
#: => fall back to the geometric _GrowBuffer, still byte-identical).
_WINDOW_RING_MAX_VERIFY_ENV = "MTPLX_DSV41_WINDOW_RING_MAX_VERIFY"
_WINDOW_RING_SLACK_ENV = "MTPLX_DSV41_WINDOW_RING_SLACK"
_WINDOW_RING_HEADROOM_ENV = "MTPLX_DSV41_WINDOW_RING_HEADROOM"
_WINDOW_RING_MAXKV_ENV = "MTPLX_DSV41_WINDOW_RING_MAXKV"


def _window_ring_enabled() -> bool:
    """Whether ``MTPLX_DSV41_WINDOW_RING`` arms the bounded-ring window store and
    the preallocated compress/index stores.  Read at call time (never frozen at
    import): the serving harness stamps the key after importing this module, and
    each request builds a fresh cache."""
    return (os.environ.get(_WINDOW_RING_ENV) or "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _ring_env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}")
    if v < 0:
        raise ValueError(f"{name} must be >= 0, got {v}")
    return v


def _window_ring_config() -> tuple:
    """(max_verify, slack, headroom, maxkv) from the env, with defaults."""
    return (
        _ring_env_int(_WINDOW_RING_MAX_VERIFY_ENV, 8),
        _ring_env_int(_WINDOW_RING_SLACK_ENV, 8),
        _ring_env_int(_WINDOW_RING_HEADROOM_ENV, 64),
        _ring_env_int(_WINDOW_RING_MAXKV_ENV, 0) or None,
    )


#: W80 engagement + drop/copy telemetry.  ``layers_ring`` proves engagement (how
#: many layer caches chose the ring); ``capacity`` is the steady-state logical keep
#: (window_size + max_verify + slack), ``phys_capacity`` the physical ping-pong
#: buffer rows; ``drops`` counts compactions that advanced the drop_offset,
#: ``rows_dropped`` the total logical rows dropped, ``rows_copied`` the rows
#: physically written (appends + compaction carries).  Process-global + cumulative
#: (like the Sinkhorn / chunk-grow counters); the ab harness resets after model load
#: and snapshots after the run.
_RING_STATS = {
    "layers_ring": 0,
    "capacity": 0,
    "phys_capacity": 0,
    "drops": 0,
    "rows_dropped": 0,
    "rows_copied": 0,
    "appends": 0,
    "reallocs": 0,
}


def reset_window_ring_stats() -> None:
    """Zero the W80 window-ring telemetry (call after model load to scope a run)."""
    for k in _RING_STATS:
        _RING_STATS[k] = 0


def window_ring_stats() -> dict:
    """Snapshot the W80 telemetry: ``enabled`` (any layer cache chose the ring),
    ring capacities, drop/copy counts."""
    s = dict(_RING_STATS)
    s["enabled"] = bool(_RING_STATS["layers_ring"] > 0)
    return s


class _WindowRing:
    """Bounded sliding-window store for one layer (W80 / K34).

    Keeps at most a contiguous SUFFIX of the append-only window history in a fixed
    pair of ping-pong buffers (``phys_cap`` rows each), allocated once at
    construction (grown transiently only for a prefill chunk wider than
    ``phys_cap``).  ``_drop`` is the absolute position of physical slot 0 (rows
    ``[_drop, _drop + _len)`` are resident); the logical length is
    ``_drop + _len``.  :meth:`view` returns the resident rows as a CONTIGUOUS array
    whose row j is absolute position ``_drop + j`` -- so a reader translates an
    absolute index by subtracting ``_drop`` (:attr:`drop_offset`), addressing the
    same positions the full store would.

    Append writes the new rows in place with a donated ``mx.slice_update`` while the
    buffer has room (``_len + n <= phys_cap``), advancing nothing; when it would
    overflow, one COMPACTION copies the last ``keep`` rows into the OTHER ping-pong
    buffer (never the same buffer -- no aliasing) and advances ``_drop``, dropping
    the older rows.  ``keep = max(cap_keep, n + window_size - 1)`` so the current
    forward's oldest query (at the append's first position) always retains its full
    causal window; dropped rows are strictly older than that window, so they are
    never read again and dropping them is exact.  No per-token allocation on the
    decode path (ping-pong reuse); no T-sized array is ever built during decode.

    Correctness never depends on ``slice_update`` donating: if a live view keeps a
    buffer referenced MLX copies instead of donating -- slower, same bytes.
    """

    __slots__ = (
        "window_size", "cap_keep", "phys_cap",
        "_bufs", "_cur", "_len", "_drop", "_b", "_dtype", "_tail",
    )

    def __init__(self, window_size: int, max_verify: int, slack: int, headroom: int):
        self.window_size = int(window_size)
        self.cap_keep = int(window_size) + int(max_verify) + int(slack)
        self.phys_cap = self.cap_keep + int(headroom)
        self._bufs = [None, None]
        self._cur = 0
        self._len = 0
        self._drop = 0
        self._b: Optional[int] = None
        self._dtype = None
        self._tail: tuple = ()

    # -- absolute-position accessors ---------------------------------------
    @property
    def drop_offset(self) -> int:
        return self._drop

    def logical_len(self) -> int:
        return self._drop + self._len

    def rows(self) -> int:
        """Resident physical rows (what ``view().shape[1]`` reports)."""
        return self._len

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _write(buf: mx.array, new: mx.array, row: int) -> mx.array:
        n = new.ndim
        starts = mx.array([0, int(row)] + [0] * (n - 2), dtype=mx.int32)
        return mx.slice_update(buf, new, starts, axes=tuple(range(n)))

    def _alloc(self, cap: int) -> mx.array:
        return mx.zeros((self._b, cap) + self._tail, dtype=self._dtype)

    def _init_from(self, new: mx.array) -> None:
        self._b = int(new.shape[0])
        self._dtype = new.dtype
        self._tail = tuple(new.shape[2:])
        n = int(new.shape[1])
        cap = max(self.phys_cap, n)
        self.phys_cap = cap
        self._bufs[0] = self._alloc(cap)
        self._bufs[1] = self._alloc(cap)
        self._bufs[self._cur] = self._write(self._bufs[self._cur], new, 0)
        self._len = n
        self._drop = 0
        _RING_STATS["rows_copied"] += n
        _RING_STATS["capacity"] = self.cap_keep
        _RING_STATS["phys_capacity"] = self.phys_cap

    def append(self, new: Optional[mx.array]) -> None:
        if new is None or (new.ndim >= 2 and new.shape[1] == 0):
            return
        n = int(new.shape[1])
        _RING_STATS["appends"] += 1
        if self._bufs[self._cur] is None:
            self._init_from(new)
            return
        if self._len + n <= self.phys_cap:
            # in-place donated write of just the new rows; drop_offset unchanged
            self._bufs[self._cur] = self._write(self._bufs[self._cur], new, self._len)
            self._len += n
            _RING_STATS["rows_copied"] += n
            return
        # Compaction: keep the last ``keep`` rows, drop the older ones.  ``keep``
        # covers the current forward's oldest query's full causal window
        # (``n + window_size - 1``, the append's first position looking back
        # ``window_size - 1``), never less than the steady ``cap_keep``.
        L = self._drop + self._len
        L_new = L + n
        keep = min(L_new, max(self.cap_keep, n + (self.window_size - 1)))
        old_drop = self._drop
        new_drop = L_new - keep
        retained = L - new_drop            # old rows carried over (>= 0)
        src = self._bufs[self._cur]        # read the retained suffix from OLD buffer
        target_cap = self.phys_cap
        if keep > target_cap:
            # transient grow for a prefill chunk wider than the ping-pong buffers
            # (allowed off the decode path; decode/verify appends never trip this)
            target_cap = keep
            _RING_STATS["reallocs"] += 1
        dst_idx = 1 - self._cur
        dst = self._bufs[dst_idx]
        if dst is None or int(dst.shape[1]) != target_cap:
            dst = self._alloc(target_cap)  # only when growing (else reuse ping-pong)
        if retained > 0:
            head = src[:, self._len - retained: self._len]   # OLD buffer != dst
            dst = self._write(dst, head, 0)
        dst = self._write(dst, new, retained)
        if target_cap != self.phys_cap:
            self.phys_cap = target_cap
            self._bufs = [None, None]      # other slot re-allocated at next compaction
            _RING_STATS["phys_capacity"] = self.phys_cap
        self._bufs[dst_idx] = dst
        self._cur = dst_idx
        self._len = retained + n
        self._drop = new_drop
        _RING_STATS["drops"] += 1
        _RING_STATS["rows_dropped"] += max(0, new_drop - old_drop)
        _RING_STATS["rows_copied"] += retained + n

    def view(self) -> Optional[mx.array]:
        buf = self._bufs[self._cur]
        if buf is None or self._len == 0:
            return None
        if self._len == buf.shape[1]:
            return buf
        return buf[:, : self._len]

    def raw_backing(self) -> Optional[mx.array]:
        """The whole current ping-pong buffer (see :meth:`_GrowBuffer.raw_backing`)."""
        return self._bufs[self._cur]

    def set(self, arr: Optional[mx.array]) -> None:
        """Replace the resident rows from a full array (state restore).  ``_drop``
        is reset to 0 here and re-seated by :meth:`reseat` once the logical length
        (the entry offset) is known."""
        self._bufs = [None, None]
        self._cur = 0
        self._len = 0
        self._drop = 0
        if arr is not None:
            self.append(arr)

    def reseat(self, logical_len: int) -> None:
        """Set ``_drop`` so the resident rows end at ``logical_len`` -- restoring
        the absolute frame after a state/offset restore (the ring always holds a
        contiguous suffix, so ``drop == logical_len - resident``)."""
        self._drop = max(0, int(logical_len) - self._len)

    def truncate_to_length(self, logical_len: int) -> None:
        """Drop back to logical length ``logical_len`` (trim/rollback).  ``_drop``
        stays where it is (advanced by any compaction since the mark -- those rows
        are unrecoverable but always beyond the window, so the reachable state is
        exact); only the resident count shrinks."""
        logical_len = max(0, int(logical_len))
        if logical_len <= self._drop:
            # trimming at/under the drop frontier: nothing resident remains reachable
            self._len = 0
            self._drop = logical_len
            return
        self._len = min(self._len, logical_len - self._drop)


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
        #: W59 / K30: the index source's selection as integer row indices into
        #: ``compress_kv`` ([B, s, k] int32, -1 = unreachable pad), the gather form
        #: of ``topk_idxs``.  Published once per index source and reused by its
        #: downstream Reuse layers, mirroring ``topk_idxs``.  Only populated under
        #: ``MTPLX_DSV41_SELECTED_KEYS`` (prefill); ``None`` otherwise.
        self.selected_idx: Optional[mx.array] = None

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
        #: W80 / K34: the bounded window ring + preallocated compress/index stores
        #: (master switch; takes precedence over W73 chunk-grow for all three lanes).
        #: Picked once at construction (per request, after the harness stamps the
        #: env key).  See :class:`_WindowRing` and :data:`_WINDOW_RING_ENV`.
        self._window_ring = _window_ring_enabled()
        #: W73 / K32: pick the append backing once, at construction (per request,
        #: after the harness stamps the env key).  OFF -> the three store lanes are
        #: plain ``mx.array`` attributes grown by :func:`_grow` (byte-for-byte the
        #: shipped cache); ON -> :class:`_GrowBuffer` lanes (chunk-grown append).
        self._chunk_grow = _kv_chunk_grow_enabled()
        if self._window_ring:
            _mv, _slk, _hr, _maxkv = _window_ring_config()
            _RING_STATS["layers_ring"] += 1
            #: bounded ring for the window lane (window_size + max_verify + slack);
            #: compress/index are preallocated to ``maxkv`` (indexer needs them in
            #: full -> cannot be bounded), or geometric when ``maxkv`` is unset.
            self._window = _WindowRing(self.window_size, _mv, _slk, _hr)
            _icap = int(_maxkv) if _maxkv else 256
            self._compress_kv = _GrowBuffer(init_cap=_icap)
            self._index_k = _GrowBuffer(init_cap=_icap)
        else:
            _KV_STATS["layers_chunk_grown" if self._chunk_grow else "layers_plain"] += 1
            #: post-RoPE window KV rows, one per token (reference window_kv_cache
            #: seed); pooled+RoPE'd compressed KV / index keys, one per completed
            #: group.  Held in ``_window`` / ``_compress_kv`` / ``_index_k`` (plain
            #: array, :class:`_GrowBuffer`, or :class:`_WindowRing`) and read/written
            #: through the same-named properties, so every existing reader/writer of
            #: ``.window`` etc. is unchanged.
            self._window = _GrowBuffer() if self._chunk_grow else None
            self._compress_kv = _GrowBuffer() if self._chunk_grow else None
            self._index_k = _GrowBuffer() if self._chunk_grow else None
        #: compressor frontier (CompressorState for ratio>1, else None)
        self.comp_state: Optional[CompressorState] = (
            CompressorState(self.compress_ratio)
            if self.is_kv_source and self.compress_ratio > 1
            else None
        )

    # -- store lanes: plain array (default) or _GrowBuffer (W73 chunk-grow) ----
    # The properties keep every reader/writer of ``.window`` / ``.compress_kv`` /
    # ``.index_k`` unchanged (attention, indexer, trim/rollback, mlx_lm state);
    # only the append + backing changes under MTPLX_DSV41_KV_CHUNK_GROW.  A
    # _GrowBuffer view is byte-identical to the concatenated store.
    @staticmethod
    def _lane_get(lane):
        return lane.view() if isinstance(lane, (_GrowBuffer, _WindowRing)) else lane

    @staticmethod
    def _lane_set(lane, value):
        if isinstance(lane, (_GrowBuffer, _WindowRing)):
            lane.set(value)
            return lane
        return value

    @property
    def window(self) -> Optional[mx.array]:
        return self._lane_get(self._window)

    @window.setter
    def window(self, value: Optional[mx.array]) -> None:
        self._window = self._lane_set(self._window, value)

    @property
    def window_drop_offset(self) -> int:
        """W80: the number of logically-dropped front rows of the window store
        (== absolute position of physical row 0 of :attr:`window`).  0 for the
        plain / chunk-grow backing (full history retained); the ring's ``_drop``
        otherwise.  A reader translates an absolute window index to a physical one
        by subtracting this."""
        return self._window.drop_offset if isinstance(self._window, _WindowRing) else 0

    @property
    def compress_kv(self) -> Optional[mx.array]:
        return self._lane_get(self._compress_kv)

    @compress_kv.setter
    def compress_kv(self, value: Optional[mx.array]) -> None:
        self._compress_kv = self._lane_set(self._compress_kv, value)

    @property
    def index_k(self) -> Optional[mx.array]:
        return self._lane_get(self._index_k)

    @index_k.setter
    def index_k(self, value: Optional[mx.array]) -> None:
        self._index_k = self._lane_set(self._index_k, value)

    # -- window (reference _window_kv L700-720) -----------------------------
    def append_window(self, kv_new: mx.array) -> None:
        """Seed the ring with this call's post-RoPE window KV (reference
        L708-719).  History is kept append-only; the reference's fixed ring is
        :meth:`ring`."""
        if isinstance(self._window, (_GrowBuffer, _WindowRing)):
            self._window.append(kv_new)
        else:
            self._window = _grow(self._window, kv_new)
            _note_rows_copied(_rows(self._window))

    def eval_backing(self):
        """W80 (item 4) / W73: the arrays a settle fence should force to free a
        span's transients.  For a ring / :class:`_GrowBuffer` lane this is the RAW
        preallocated backing buffer (:meth:`_WindowRing.raw_backing`), NOT the
        logical ``view()`` slice: forcing the buffer realises this step's in-place
        ``slice_update`` writes without materialising a fresh ``[b, length]`` slice
        (an O(length)/O(n_comp) copy) per token.  For the plain backing it is the
        stored array (unchanged).  Plus the compressor frontier."""
        out = []
        for lane in (self._window, self._compress_kv, self._index_k):
            a = lane.raw_backing() if isinstance(lane, (_GrowBuffer, _WindowRing)) else lane
            if a is not None:
                out.append(a)
        if self.comp_state is not None:
            for a in (self.comp_state.raw_kv, self.comp_state.raw_score):
                if a is not None:
                    out.append(a)
        return out

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
        if isinstance(self._compress_kv, _GrowBuffer):
            self._compress_kv.append(compress_new)
        else:
            self._compress_kv = _grow(self._compress_kv, compress_new)
            _note_rows_copied(_rows(self._compress_kv))

    # -- index keys (reference Indexer.k_cache write L547) ------------------
    def append_index_k(self, index_new: mx.array) -> None:
        """Append this group's index keys (reference L547)."""
        if isinstance(self._index_k, _GrowBuffer):
            self._index_k.append(index_new)
        else:
            self._index_k = _grow(self._index_k, index_new)
            _note_rows_copied(_rows(self._index_k))

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
        if isinstance(self._window, _WindowRing):
            # W80 drop-aware: the ring restores its LOGICAL length; _drop stays
            # advanced (dropped rows are always beyond the window -> exact).
            self._window.truncate_to_length(new_len)
        else:
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
        )

    def rollback(self, mark) -> None:
        offset, nw, nc, ni, comp_mark = mark
        # The engram history advances one position per token, in lockstep with
        # ``offset``, so the rollback depth is exactly the offset delta -- trim by
        # it (no dependence on the engram exposing a length, so a hook stand-in
        # that only records ``trim`` still rewinds correctly).
        if self.engram_state is not None:
            back = int(self.offset) - int(offset)
            if back > 0:
                self.engram_state.trim(back)
        if isinstance(self._window, _WindowRing):
            # W80 drop-aware: restore to the marked LOGICAL length (== offset, the
            # window advances one row per token); _drop stays advanced.
            self._window.truncate_to_length(int(offset))
        else:
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
        for :func:`mtplx.cache_state.snapshot_cache` / :func:`restore_cache` and
        ``mlx_lm.save_prompt_cache``.  The compressor frontier rows travel with it
        so a snapshot restore rebuilds the exact frontier.

        **W26:** the entry that owns the per-sequence engram history (only the
        first entry of a sequence -- see :class:`DeepseekV41Cache`) also carries
        it here, as a trailing ``mx.array`` leaf
        (:attr:`~mtplx.engram_v41.NgramHashState.state`).  Without it a KV-only
        snapshot restore desyncs the engram n-gram hashing on the engram layers
        (1 and 14) across a warm-turn near-prefix restore -- the exact reason the
        session bank's near-prefix restore / store-on-prefill were held off for
        this backend (docs/deepseek-v41/W22_REPORT.md).  The engram's immutable
        hash config is shared and does NOT travel; only its streaming history
        does.  Non-owning entries (``engram_state is None``, i.e. every entry but
        the first, and every entry of a no-engram model) return the plain 5-tuple
        unchanged."""
        comp_kv = None if self.comp_state is None else self.comp_state.raw_kv
        comp_sc = None if self.comp_state is None else self.comp_state.raw_score
        kv = (self.window, self.compress_kv, self.index_k, comp_kv, comp_sc)
        if self.engram_state is not None:
            return kv + (self.engram_state.state,)
        return kv

    @state.setter
    def state(self, value) -> None:
        if value is None:
            self.window = None
            self.compress_kv = None
            self.index_k = None
            if self.comp_state is not None:
                self.comp_state.raw_kv = None
                self.comp_state.raw_score = None
            # A whole-state clear does not touch the engram history: the engram
            # rewinds via trim only, and no restore flow that owns an engram ever
            # passes ``None`` (snapshot_untrimmable stores ``None`` for trimmable
            # entries and restore_cache then skips them).  Leaving it keeps the
            # no-engram path byte-identical to before W26.
            return
        values = tuple(value)
        engram_blob = None
        if len(values) == 6:
            window, compress_kv, index_k, comp_kv, comp_sc, engram_blob = values
        elif len(values) == 5:
            # a 5-tuple (no engram, or an older KV-only snapshot) leaves the
            # engram history untouched -- backward compatible with pre-W26 state.
            window, compress_kv, index_k, comp_kv, comp_sc = values
        else:
            raise ValueError(
                f"unexpected DeepSeek-V4.1 layer state arity {len(values)} (want 5 or 6)"
            )
        self.window = window
        self.compress_kv = compress_kv
        self.index_k = index_k
        if self.comp_state is not None:
            self.comp_state.raw_kv = comp_kv
            self.comp_state.raw_score = comp_sc
        # restore the engram history into the owning entry's live state object
        # (kept by make_cache with the shared hash config); if this entry carries
        # no engram, the blob has nowhere to go and the KV restore still stands.
        if engram_blob is not None and self.engram_state is not None:
            self.engram_state.replace_state(engram_blob)

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
        # W80: re-seat the ring's drop_offset now the logical length (offset) is
        # known -- the saved window ``state`` is a contiguous suffix, so
        # ``_drop == offset - resident_rows`` (no-op for the plain / chunk-grow
        # backing, which keeps full history with drop_offset 0).
        if isinstance(self._window, _WindowRing):
            self._window.reseat(self.offset)

    #: set on entries reconstructed by :meth:`from_state` when the saved ``state``
    #: carried an engram history (the 6th leaf).  The reconstruction cannot rebuild
    #: the shared, unserialised hash config, so the raw ``mx.array`` history buffer
    #: is parked here for the caller to rehydrate into a live
    #: :class:`~mtplx.engram_v41.NgramHashState` (``fresh().replace_state(...)``).
    loaded_engram_state = None

    @classmethod
    def from_state(cls, state, meta_state):
        """Reconstruct an entry from a saved ``(state, meta_state)`` --
        ``mlx_lm.load_prompt_cache``'s contract (``globals()[cls].from_state``).

        Rebuilds the window / compressed-KV / index-key lanes, the compressor
        frontier and the offset.  When ``state`` carries the engram history as
        its 6th leaf (the owning entry), the raw buffer is parked on
        :attr:`loaded_engram_state`; the immutable hash config is shared and
        unserialised, so the caller rehydrates it into a fresh
        :class:`~mtplx.engram_v41.NgramHashState` (see W26_REPORT)."""
        version, offset, window_size, compress_ratio, is_kv_source = meta_state
        if version != _LAYER_META_VERSION:
            raise ValueError(f"unsupported DeepSeek-V4.1 layer meta version: {version!r}")
        entry = cls(
            window_size=int(window_size),
            compress_ratio=int(compress_ratio),
            is_kv_source=(str(is_kv_source) == "1"),
            engram_state=None,
        )
        values = tuple(state)
        entry.loaded_engram_state = values[5] if len(values) == 6 else None
        entry.state = values[:5]          # KV lanes only (this entry owns no engram)
        entry.offset = int(offset)
        if isinstance(entry._window, _WindowRing):
            entry._window.reseat(entry.offset)  # W80: re-seat drop_offset from length
        return entry


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
        #: fallback slot for the (unreachable) zero-layer case; the real owner of
        #: the engram history is the first entry -- see :attr:`engram_state`.
        self._engram_state = None
        #: an :class:`mtplx.engram_v41.NgramHashState` clone, or None when engram
        #: is not wired; advanced by the W10 backbone (once per forward, before
        #: the layers) and rewound through the owning (first) entry in step.  The
        #: property setter keeps that entry's reference in sync, so re-pointing
        #: ``cache.engram_state`` re-owns the history on the entry that trims it.
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

    @property
    def engram_state(self):
        """The per-sequence engram n-gram history, owned by the first entry (so
        its trim/rollback moves it) and published here for the backbone to
        advance.  Setting it re-owns the history on that entry."""
        return self.layers[0].engram_state if self.layers else self._engram_state

    @engram_state.setter
    def engram_state(self, value) -> None:
        if self.layers:
            self.layers[0].engram_state = value
        else:
            self._engram_state = value

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
