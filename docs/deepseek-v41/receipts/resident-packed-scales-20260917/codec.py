"""Exact row-adaptive E8M0 storage; CPU construction only, no MLX imports."""
import numpy as np


def pack_rows(scales):
    if scales.ndim != 2 or scales.dtype != np.uint8:
        raise ValueError('native two-dimensional E8M0 bytes required')
    rows, columns = scales.shape
    base = scales.min(axis=1)
    spread = scales.max(axis=1).astype(np.uint16) - base
    widths = np.select([spread == 0, spread < 2, spread < 4, spread < 16],
                       [0, 1, 2, 4], default=8).astype(np.uint32)
    counts = (columns * widths + 31) // 32
    offsets = np.zeros(rows, np.uint32)
    offsets[1:] = np.cumsum(counts[:-1], dtype=np.uint32)
    total = int(counts.sum())
    if total >= 1 << 20:
        raise ValueError('component payload exceeds descriptor offset field')
    descriptor = (offsets << 12) | (widths << 8) | base.astype(np.uint32)
    payload = np.zeros(total, dtype=np.uint32)
    for width in (1, 2, 4, 8):
        which = np.flatnonzero(widths == width)
        if not len(which):
            continue
        per_word = 32 // width
        row_words = (columns + per_word - 1) // per_word
        deltas = np.zeros((len(which), row_words * per_word), dtype=np.uint32)
        deltas[:, :columns] = scales[which].astype(np.uint32) - base[which, None]
        packed = np.zeros((len(which), row_words), dtype=np.uint32)
        for lane in range(per_word):
            packed |= deltas[:, lane::per_word] << (lane * width)
        payload[offsets[which, None] + np.arange(row_words)] = packed
    return descriptor, payload


def unpack_rows(descriptor, payload, columns):
    widths = (descriptor >> 8) & 15
    base = descriptor & 255
    offsets = descriptor >> 12
    out = np.broadcast_to(base[:, None], (len(base), columns)).astype(np.uint8).copy()
    for width in (1, 2, 4, 8):
        which = np.flatnonzero(widths == width)
        if not len(which):
            continue
        indices = np.arange(columns, dtype=np.uint32)
        values = (payload[offsets[which, None] + indices * width // 32]
                  >> ((indices * width) % 32)) & ((1 << width) - 1)
        out[which] = values + base[which, None]
    return out
