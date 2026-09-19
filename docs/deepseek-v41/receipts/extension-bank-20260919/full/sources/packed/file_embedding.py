"""Exact native BF16 input rows with a fixed host arena, installed after prefill."""
from collections import OrderedDict
import fcntl
import json
import os
from pathlib import Path
import struct

import numpy as np
import mlx.core as mx
import mlx.nn as nn

ARENA_BYTES = 16 * 1024**2
HOST_BOUND_BYTES = 32 * 1024**2


class FileRowEmbedding(nn.Module):
    def __init__(self, source):
        super().__init__()
        self._fd = -1
        self._arena = None
        self._rows = None
        self._lru = OrderedDict()
        self._dims = source['shape'][1]
        self._vocab = source['shape'][0]
        self._row_bytes = self._dims * 2
        self._capacity = ARENA_BYTES // self._row_bytes
        path = Path(source['path'])
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        try:
            st = os.fstat(fd)
            actual = {key: getattr(st, key) for key in source['identity']}
            if actual != source['identity']:
                raise RuntimeError('native embedding source identity changed')
            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
            header_size = struct.unpack('<Q', os.pread(fd, 8, 0))[0]
            header = json.loads(os.pread(fd, header_size, 8))
            tensor = header[source['tensor']]
            if tensor != source['tensor_header']:
                raise RuntimeError('native embedding layout changed')
            if tensor['dtype'] != 'BF16' or tensor['shape'] != [129280, 5120]:
                raise RuntimeError('file rows require the native BF16 input embedding')
            self._offset = 8 + header_size + tensor['data_offsets'][0]
            if self._offset != source['offset']:
                raise RuntimeError('native embedding file offset changed')
            if tensor['data_offsets'][1] - tensor['data_offsets'][0] != self._vocab * self._row_bytes:
                raise RuntimeError('native embedding byte count changed')
            self._arena = bytearray(ARENA_BYTES)
            self._rows = np.frombuffer(self._arena, dtype=np.uint16,
                                      count=self._capacity * self._dims).reshape(self._capacity, self._dims)
            self._free = list(range(self._capacity - 1, -1, -1))
            self._fd = fd
        except BaseException:
            os.close(fd)
            raise

    def __call__(self, ids):
        # IDs and shape vary at runtime; model topology/layout were fixed once.
        tokens = ids.reshape(-1).tolist()
        output = np.empty((len(tokens), self._dims), dtype=np.uint16)
        for row, token in enumerate(tokens):
            slot = self._lru.get(token)
            if slot is None:
                if self._free:
                    slot = self._free.pop()
                else:
                    _, slot = self._lru.popitem(last=False)
                view = memoryview(self._arena)[slot * self._row_bytes:(slot + 1) * self._row_bytes]
                copied = 0
                while copied < self._row_bytes:
                    payload = os.pread(self._fd, self._row_bytes - copied,
                                       self._offset + token * self._row_bytes + copied)
                    if not payload:
                        raise RuntimeError('short native embedding row')
                    view[copied:copied + len(payload)] = payload
                    copied += len(payload)
                self._lru[token] = slot
            else:
                self._lru.move_to_end(token)
            # A pending GPU graph must never alias the mutable cache arena.
            output[row] = self._rows[slot]
        return mx.array(output).view(mx.bfloat16).reshape((*ids.shape, self._dims))

    def close(self):
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1
        self._lru.clear()
        self._free.clear()
        self._rows = None
        self._arena = None
