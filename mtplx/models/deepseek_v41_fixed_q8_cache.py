"""Fixed-capacity Q8 attention storage for DeepSeek V4.1.

The cache factories are installed explicitly on a loaded model. Storage uses
affine Q8 groups of 64 with FP32 scales and biases. Compressor working rows
remain FP32 and retain only the sliding-window rollback history. All backing
arrays are allocated at cache construction; append and trim never grow them.
"""
from __future__ import annotations

from dataclasses import dataclass
import json

import mlx.core as mx

from .deepseek_v41_cache import DeepseekV41Cache, ring_view


@dataclass(frozen=True)
class FixedQ8CacheConfig:
    max_kv: int
    max_append: int
    window_size: int
    head_dim: int
    index_head_dim: int
    ratios: tuple[int, ...]
    sources: tuple[int, ...]
    draft_layers: int = 3
    max_rollback: int = 8

    def __post_init__(self):
        if (self.max_kv < 1 or not 1 <= self.max_append <= self.max_kv
                or self.window_size < 1 or self.max_rollback < 1
                or self.head_dim % 64 or self.index_head_dim % 64
                or min(self.head_dim, self.index_head_dim) < 64
                or self.draft_layers < 0 or not self.ratios):
            raise ValueError("invalid fixed Q8 cache geometry")
        if len(set(self.sources)) != len(self.sources) or any(
            not 0 <= i < len(self.ratios) or self.ratios[i] < 1
            or self.ratios[i] > self.window_size + self.max_rollback
            for i in self.sources
        ):
            raise ValueError("invalid fixed Q8 cache source layers")

    @property
    def ring_capacity(self):
        return self.window_size + self.max_rollback + self.max_append

    def storage_bytes(self):
        # Q8 values plus two FP32 numbers per group; window/frontier ping-pong.
        def q8(rows, width):
            return rows * (width + 8 * (width // 64))

        window = 2 * len(self.ratios) * q8(self.ring_capacity, self.head_dim)
        compressed = sum(q8((self.max_kv + r - 1) // r, self.head_dim)
                         for i, r in enumerate(self.ratios) if i in self.sources)
        index = sum(q8((self.max_kv + r - 1) // r, self.index_head_dim)
                    for i, r in enumerate(self.ratios) if i in self.sources)
        frontier = sum(4 * self.ring_capacity * self.head_dim * 4
                       for i, r in enumerate(self.ratios) if i in self.sources and r > 1)
        draft = 2 * self.draft_layers * q8(self.ring_capacity, self.head_dim)
        return dict(window=window, compressed=compressed, index=index,
                    frontier=frontier, draft=draft,
                    total=window + compressed + index + frontier + draft)

    def reserve_bytes(self):
        """Price backing copies, decoded views and allocator/compiler slack.

        This reserve is added before expert-slot planning. Existing native KV
        allowances remain available; no measured native peak is discounted.
        """
        views = (len(self.ratios) + self.draft_layers) * self.ring_capacity * self.head_dim * 4
        views += sum(((self.max_kv + self.ratios[i] - 1) // self.ratios[i])
                     * (self.head_dim + self.index_head_dim) * 4 for i in self.sources)
        total = 2 * self.storage_bytes()['total'] + views + 64 * 1024**2
        page_band = 16 * 1024**2
        return ((total + page_band - 1) // page_band) * page_band

    @classmethod
    def from_args(cls, args, *, max_kv, max_append, draft_layers, max_rollback=8):
        return cls(max_kv=int(max_kv), max_append=int(max_append),
                   window_size=int(args.window_size), head_dim=int(args.head_dim),
                   index_head_dim=int(args.index_head_dim),
                   ratios=tuple(args.compress_ratios[:int(args.num_hidden_layers)]),
                   sources=tuple(args.kv_source_layer_ids), draft_layers=int(draft_layers),
                   max_rollback=int(max_rollback))


class FixedRows:
    """Packed rows with a fixed append capacity and optional sliding suffix.

    Functional slice writes preserve consumers holding older array views. A
    second fixed bank handles compaction without overlapping source/destination.
    """

    def __init__(self, capacity, width, *, max_append, keep=0, q8=True):
        self.capacity, self.width = int(capacity), int(width)
        self.max_append, self.keep = int(max_append), int(keep)
        self.q8 = bool(q8)
        self.length = self.drop = self.current = 0
        self.dtype = mx.float32
        shapes = ((width // 4, mx.uint32), (width // 64, mx.float32),
                  (width // 64, mx.float32)) if q8 else ((width, mx.float32),)
        self.banks = [[mx.zeros((1, capacity, cols), dtype=dtype)
                       for cols, dtype in shapes] for _ in range(2 if keep else 1)]
        self._encode = self._quantize if q8 else lambda x: (x.astype(mx.float32),)
        self._decode = FixedRows._dequantize if q8 else FixedRows._float_view
        # Bind the input dtype once when the first prefill reaches this store.
        self._append_impl = FixedRows._first_append

    @staticmethod
    def _quantize(x):
        return mx.quantize(x.astype(mx.float32), group_size=64, bits=8)

    def _dequantize(self, arrays):
        return mx.dequantize(*arrays, group_size=64, bits=8).astype(self.dtype)

    def _float_view(self, arrays):
        return arrays[0].astype(self.dtype)

    def append(self, rows):
        self._append_impl(self, rows)

    @staticmethod
    def _write(buf, rows, offset):
        return mx.slice_update(buf, rows, mx.array([0, offset, 0], mx.int32), axes=(0, 1, 2))

    def _first_append(self, rows):
        if rows.ndim != 3 or rows.shape[0] != 1 or rows.shape[2] != self.width:
            raise ValueError("fixed Q8 stores require the constructed batch/width")
        self.dtype = rows.dtype
        self._append_impl = FixedRows._append
        self._append(rows)

    def _append(self, rows):
        n = int(rows.shape[1])
        if n == 0:
            return
        if n > self.max_append or (not self.keep and self.length + n > self.capacity):
            raise ValueError("fixed cache append exceeds its constructed capacity")
        encoded = self._encode(rows)
        if self.length + n > self.capacity:
            retained = min(self.length, self.keep)
            next_bank = 1 - self.current
            for i, src in enumerate(self.banks[self.current]):
                tail = src[:, self.length - retained:self.length]
                self.banks[next_bank][i] = self._write(self.banks[next_bank][i], tail, 0)
            self.drop += self.length - retained
            self.length = retained
            self.current = next_bank
        bank = self.banks[self.current]
        for i, value in enumerate(encoded):
            bank[i] = self._write(bank[i], value, self.length)
        self.length += n

    @property
    def end(self):
        return self.drop + self.length

    def view(self, start=None, end=None):
        if self.length == 0:
            return None
        lo = self.drop if start is None else int(start)
        hi = self.end if end is None else int(end)
        return self._decode(self, tuple(x[:, lo - self.drop:hi - self.drop]
                                        for x in self.banks[self.current]))

    def truncate(self, end):
        end = int(end)
        if not self.drop <= end <= self.end:
            raise ValueError("cache rollback is outside retained history")
        self.length = end - self.drop

    def backings(self):
        return [x for bank in self.banks for x in bank]

    def payload(self):
        return tuple(x[:, :self.length] for x in self.banks[self.current])

    def restore(self, values, *, drop, length, dtype):
        if not 0 <= length <= self.capacity:
            raise ValueError("saved cache exceeds constructed capacity")
        self.drop, self.length, self.current = int(drop), int(length), 0
        self.dtype = getattr(mx, str(dtype))
        self._append_impl = FixedRows._append
        if length:
            for i, value in enumerate(values):
                self.banks[0][i] = self._write(self.banks[0][i], value, 0)

    def metadata(self):
        return (self.drop, self.length, str(self.dtype).removeprefix("mlx.core."))


class FixedCompressorState:
    """Native FP32 pooling with a fixed, recoverable frontier journal."""

    def __init__(self, ratio, config):
        self.ratio = ratio
        keep = config.window_size + config.max_rollback
        self.kv = FixedRows(config.ring_capacity, config.head_dim,
                            max_append=config.max_append, keep=keep, q8=False)
        self.score = FixedRows(config.ring_capacity, config.head_dim,
                               max_append=config.max_append, keep=keep, q8=False)

    @property
    def n_fed(self):
        return self.kv.end

    def push(self, kv, score):
        before = self.n_fed // self.ratio
        self.kv.append(kv)
        self.score.append(score)
        after = self.n_fed // self.ratio
        if before == after:
            return kv[:, :0]
        lo, hi = before * self.ratio, after * self.ratio
        shape = (1, after - before, self.ratio, kv.shape[-1])
        k = self.kv.view(lo, hi).reshape(shape)
        s = self.score.view(lo, hi).reshape(shape)
        return mx.sum(k * mx.softmax(s, axis=2), axis=2)

    def rollback(self, mark):
        self.kv.truncate(mark)
        self.score.truncate(mark)

    def raw_backings(self):
        return self.kv.backings() + self.score.backings()


class FixedQ8LayerCache:
    def __init__(self, config, layer):
        if not 0 <= layer < len(config.ratios):
            raise ValueError("fixed Q8 cache layer is outside the model")
        self.config, self.layer = config, layer
        self.window_size, self.compress_ratio = config.window_size, config.ratios[layer]
        self.is_kv_source = layer in config.sources
        self.offset = 0
        self.engram_state = None
        keep = config.window_size + config.max_rollback
        self._window = FixedRows(config.ring_capacity, config.head_dim,
                                  max_append=config.max_append, keep=keep)
        self._compress = self._index = self.comp_state = None
        if self.is_kv_source:
            rows = (config.max_kv + self.compress_ratio - 1) // self.compress_ratio
            self._compress = FixedRows(rows, config.head_dim, max_append=config.max_append)
            self._index = FixedRows(rows, config.index_head_dim, max_append=config.max_append)
            if self.compress_ratio > 1:
                self.comp_state = FixedCompressorState(self.compress_ratio, config)

    @property
    def window(self):
        return self._window.view()

    @property
    def window_drop_offset(self):
        return self._window.drop

    @property
    def compress_kv(self):
        return self._compress.view() if self._compress is not None else None

    @property
    def index_k(self):
        return self._index.view() if self._index is not None else None

    def append_window(self, rows):
        self._window.append(rows)

    def append_compress(self, rows):
        self._compress.append(rows)

    def append_index_k(self, rows):
        self._index.append(rows)

    def ring(self, length):
        return ring_view(self.window, self.window_size, length)

    def window_len(self):
        return self._window.length

    def advance(self, n):
        self.offset += int(n)

    def assert_can_admit(self, n):
        # Layer-major prefill admits the entire prompt before processing chunks.
        if self.offset + n > self.config.max_kv:
            raise ValueError("forward exceeds the fixed Q8 context capacity")

    def size(self):
        return self.offset

    def empty(self):
        return self.offset == 0

    def is_trimmable(self):
        return True

    def can_restore(self, end):
        need = max(0, end - self.window_size + 1)
        if not 0 <= end <= self.offset or need < self._window.drop:
            return False
        if self.comp_state is not None:
            need = end - end % self.compress_ratio
            if need < self.comp_state.kv.drop:
                return False
        return True

    def trim(self, n):
        if n < 0 or n > self.offset:
            raise ValueError("invalid fixed-cache trim")
        end = self.offset - int(n)
        if not self.can_restore(end):
            return 0
        self._window.truncate(end)
        if self.is_kv_source:
            groups = end // self.compress_ratio
            self._compress.truncate(groups)
            self._index.truncate(groups)
            if self.comp_state is not None:
                self.comp_state.rollback(end)
        if self.engram_state is not None:
            self.engram_state.trim(n)
        self.offset = end
        return int(n)

    def mark(self):
        return self.offset

    def rollback(self, mark):
        if not self.can_restore(mark):
            raise ValueError("fixed-cache rollback is outside retained history")
        self.trim(self.offset - mark)

    def _stores(self):
        stores = [self._window]
        if self.is_kv_source:
            stores.extend((self._compress, self._index))
        if self.comp_state is not None:
            stores.extend((self.comp_state.kv, self.comp_state.score))
        return stores

    def eval_backing(self):
        return [x for store in self._stores() for x in store.backings()]

    @property
    def state(self):
        dtypes = {mx.float16: 0, mx.bfloat16: 1, mx.float32: 2}
        stores = tuple((mx.array([s.drop, s.length, dtypes[s.dtype]], mx.int64), s.payload())
                       for s in self._stores())
        engram = self.engram_state.state if self.engram_state is not None else None
        return mx.array([self.offset], mx.int64), stores, engram

    @state.setter
    def state(self, value):
        offset, packed_stores, engram = value
        offset = int(offset.item())
        if not 0 <= offset <= self.config.max_kv or len(packed_stores) != len(self._stores()):
            raise ValueError("saved fixed Q8 state differs from constructed capacity")
        dtypes = ("float16", "bfloat16", "float32")
        decoded = []
        for store, (metadata, arrays) in zip(self._stores(), packed_stores):
            drop, length, dtype = map(int, metadata.tolist())
            expected_end = (offset // self.compress_ratio
                            if store is self._compress or store is self._index else offset)
            if (drop < 0 or not 0 <= length <= store.capacity or not 0 <= dtype < len(dtypes)
                    or drop + length != expected_end or (not store.keep and drop != 0)
                    or len(arrays) != len(store.banks[0])
                    or any(tuple(a.shape) != (1, length, b.shape[-1]) or a.dtype != b.dtype
                           for a, b in zip(arrays, store.banks[0]))):
                raise ValueError("invalid fixed Q8 snapshot planes")
            decoded.append((store, arrays, drop, length, dtypes[dtype]))
        for store, arrays, drop, length, dtype in decoded:
            store.restore(arrays, drop=drop, length=length, dtype=dtype)
        self.offset = offset
        if self.engram_state is not None and engram is not None:
            self.engram_state.replace_state(engram)

    def replace_state(self, value):
        self.state = value

    @property
    def meta_state(self):
        return ("deepseek-v41-fixed-q8-v1", str(self.layer), json.dumps(self.config.__dict__))

    @meta_state.setter
    def meta_state(self, value):
        version, layer, config = value
        if (version != "deepseek-v41-fixed-q8-v1" or int(layer) != self.layer
                or json.loads(config) != json.loads(json.dumps(self.config.__dict__))):
            raise ValueError("saved fixed Q8 configuration differs from constructed cache")

    @classmethod
    def from_state(cls, state, meta_state):
        version, layer, config = meta_state
        if version != "deepseek-v41-fixed-q8-v1":
            raise ValueError("unsupported fixed Q8 cache state version")
        entry = cls(FixedQ8CacheConfig(**json.loads(config)), int(layer))
        entry.state = state
        entry.loaded_engram_state = state[2]
        return entry


class FixedQ8Cache(DeepseekV41Cache):
    def trim(self, n):
        if n < 0 or n > self.offset:
            raise ValueError("invalid fixed-cache trim")
        end = self.offset - n
        if not all(layer.can_restore(end) for layer in self.layers):
            return 0
        for layer in self.layers:
            layer.trim(n)
        return n

    def rollback(self, mark):
        if len(mark) != len(self.layers) or not all(
                layer.can_restore(end) for layer, end in zip(self.layers, mark)):
            raise ValueError("fixed-cache rollback is outside retained history")
        for layer, end in zip(self.layers, mark):
            layer.rollback(end)


def make_fixed_q8_cache(config, *, engram_state=None):
    cache = FixedQ8Cache(0, window_size=config.window_size)
    cache.layers = [FixedQ8LayerCache(config, i) for i in range(len(config.ratios))]
    cache.engram_state = engram_state
    cache.fixed_q8_config = config
    mx.eval([layer.eval_backing() for layer in cache.layers])
    return cache


class FixedQ8DraftCache:
    def __init__(self, config):
        self.window_size = config.window_size
        self.offset = 0
        self._window = FixedRows(config.ring_capacity, config.head_dim,
                                  max_append=config.max_append,
                                  keep=config.window_size + config.max_rollback)
        self._append_impl = FixedQ8DraftCache._seed

    def append_main(self, rows):
        self._append_impl(self, rows)

    @property
    def window(self):
        return self._window.view(max(self._window.drop, self.offset - self.window_size))

    def _seed(self, rows):
        n = int(rows.shape[1])
        # The first seed can be the complete prefill. Only the final window is
        # reachable by draft attention; do not quantize or retain earlier rows.
        retained = min(n, self.window_size)
        self._window.drop = n - retained
        tail = rows[:, -retained:]
        for start in range(0, retained, self._window.max_append):
            self._window.append(tail[:, start:start + self._window.max_append])
        self.offset = n
        self._append_impl = FixedQ8DraftCache._append_main

    def _append_main(self, rows):
        self._window.append(rows)
        self.offset += int(rows.shape[1])

    def detach_prefill_backings(self):
        # Packed fixed buffers already own their data; avoid a dequant/requant
        # cycle when the native prefill helper detaches its floating window.
        return self._window.backings()

    def trim(self, n):
        if n < 0 or n > self.offset:
            raise ValueError("invalid fixed draft-cache trim")
        end = self.offset - n
        if max(0, end - self.window_size) < self._window.drop:
            return 0
        self._window.truncate(end)
        self.offset = end
        return n

    def mark(self):
        return self.offset

    def rollback(self, mark):
        n = self.offset - mark
        if self.trim(n) != n:
            raise ValueError("draft rollback is outside retained history")

    def is_trimmable(self):
        return True


def install_fixed_q8_cache(model, *, max_kv, max_append, max_rollback=8):
    """Install explicit cache factories; allocate only when a request starts."""
    args = model.args
    config = FixedQ8CacheConfig.from_args(args, max_kv=max_kv, max_append=max_append,
                                         draft_layers=len(model.mtp_blocks), max_rollback=max_rollback)
    if len(config.ratios) != int(args.num_hidden_layers):
        raise ValueError("fixed Q8 cache layer inventory differs from the model")

    engram = model.model.engram_hash

    def make_cache():
        return make_fixed_q8_cache(config, engram_state=engram.fresh() if engram is not None else None)

    def make_mtp_cache():
        caches = [FixedQ8DraftCache(config) for _ in range(config.draft_layers)]
        mx.eval([c._window.backings() for c in caches])
        return caches

    object.__setattr__(model, "make_cache", make_cache)
    object.__setattr__(model, "make_mtp_cache", make_mtp_cache)
    report = dict(bits=8, group_size=64, metadata_dtype="float32",
                  max_kv=config.max_kv, max_append=config.max_append,
                  max_rollback=config.max_rollback,
                  storage_bytes=config.storage_bytes(),
                  additional_reserve_bytes=config.reserve_bytes(),
                  scope="target window/compressed/index and draft KV; native fixed compressor working rows")
    object.__setattr__(model, "_mtplx_fixed_q8_cache", report)
    return report
