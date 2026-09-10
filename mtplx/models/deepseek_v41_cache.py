"""DeepSeek-V4.1 text KV cache (per-model + per-layer) with the trim/rollback seam.

STUB owned by worker W13 (``feat/deepseek-v41-w13``) per
``docs/deepseek-v41/PORT_CONTRACT.md``.  Until W13's branch lands this is the
faithful append-only cache the W10 attention reads/writes; on integration W13's
module (ring-buffered, mirroring the reference ``window_kv_cache`` /
``compress_kv_cache`` / ``k_cache`` + ``kv_state`` / ``score_state`` buffers)
replaces it.  The public surface below is the contract and must not change:

- :class:`_LayerCache`: attributes ``window`` / ``compress_kv`` / ``index_k`` /
  ``comp_state`` (all read/write by the attention), ``mark()`` / ``rollback()``.
- :class:`DeepseekV41Cache`: ``layers`` (list, len n_layers), ``offset``,
  ``engram_state``, ``mark()`` / ``rollback()`` / ``trim(n)``.
"""

from __future__ import annotations


def _truncate(rows, n):
    if rows is None or n == 0:
        return None if n == 0 else rows
    return rows[:, :n]


class _LayerCache:
    """Append-only window / compressed-KV / index-key rows for one layer, plus a
    mark/rollback seam.  Full history is kept and the sliding window is realised
    by the attention mask (equivalent to the reference ring buffer for the
    positions any query can still reach); W13 swaps it for a bounded ring.

    ``ratio`` is recorded the first time the attention appends a compressed row,
    so :meth:`DeepseekV41Cache.trim` can re-derive the compressed row count.
    """

    def __init__(self):
        self.window = None
        self.compress_kv = None
        self.index_k = None
        self.comp_state = None  # (kv_acc, score_acc) partial compressor group, or None
        self.ratio = None       # this layer's compress_ratio, set on first append

    def mark(self):
        def n(a):
            return 0 if a is None else a.shape[1]
        # comp_state holds immutable arrays, so the tuple itself is the snapshot
        return (n(self.window), n(self.compress_kv), n(self.index_k), self.comp_state)

    def rollback(self, mark):
        nw, nc, ni, comp_state = mark
        self.window = _truncate(self.window, nw)
        self.compress_kv = _truncate(self.compress_kv, nc)
        self.index_k = _truncate(self.index_k, ni)
        self.comp_state = comp_state


class DeepseekV41Cache:
    """Per-model KV cache: one :class:`_LayerCache` per layer plus the running
    token offset.  ``mark``/``rollback`` undo a decoded tail (the trim/rollback
    seam of the V4 ``DeepseekV4Cache``), restoring identical logits."""

    def __init__(self, n_layers: int):
        self.layers = [_LayerCache() for _ in range(n_layers)]
        self.offset = 0
        #: Engram row-id history (an ``NgramHashState``-like object owned by the
        #: engram worker); trimmed in step with the KV rollback below.  None when
        #: engram is not wired.
        self.engram_state = None

    def mark(self):
        return (self.offset, [lc.mark() for lc in self.layers])

    def rollback(self, mark):
        target_offset, layer_marks = mark
        # trim the engram token history by the same number of decoded tokens
        if self.engram_state is not None and self.offset > target_offset:
            self.engram_state.trim(self.offset - target_offset)
        self.offset = target_offset
        for lc, m in zip(self.layers, layer_marks):
            lc.rollback(m)

    def trim(self, n: int) -> None:
        """Drop the last ``n`` decoded tokens from the offset and every layer.

        Window rows are 1:1 with tokens (append-only), so they truncate by ``n``;
        compressed / index rows are re-derived from the new offset and the layer's
        ratio (``new_offset // ratio`` completed groups), and the partial-group
        compressor state is reset so the next step re-pools from scratch.  Engram
        history is trimmed by the same ``n``.
        """
        if n <= 0:
            return
        n = min(n, self.offset)
        new_offset = self.offset - n
        if self.engram_state is not None:
            self.engram_state.trim(n)
        for lc in self.layers:
            if lc.window is not None:
                keep = max(lc.window.shape[1] - n, 0)
                lc.window = _truncate(lc.window, keep)
            ratio = lc.ratio
            if ratio and lc.compress_kv is not None:
                comp_keep = new_offset // ratio
                lc.compress_kv = _truncate(lc.compress_kv, comp_keep)
                lc.index_k = _truncate(lc.index_k, comp_keep)
                lc.comp_state = None
        self.offset = new_offset
