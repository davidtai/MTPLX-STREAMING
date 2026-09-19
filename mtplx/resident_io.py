"""Construction-only admission for eager, uncached safetensors shard reads."""
from __future__ import annotations

import io
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence

from .expert_manifest import (
    ResidentTensor,
    ShardInfo,
    _pread_exact,
    _read_safetensors_header,
    _readonly_flags,
)

MAX_RESIDENT_SHARD_BYTES = 3 * 1024**3
MAX_RESIDENT_HEADER_BYTES = 1024**2
# FileIO.readinto makes one OS read; keep every tensor below the signed read limit.
MAX_RESIDENT_TENSOR_BYTES = 2**31 - 1
MAX_UNSELECTED_RESIDENT_BYTES = 64 * 1024**2


class ResidentShardReader:
    """Validate every touched shard before the first array is evaluated.

    macOS file-object mx.load is eager and reads directly into MLX-owned buffers.
    Admission prices all discarded tensors, verifies the complete header, and
    keeps the same uncached descriptors alive through evaluation. On other
    platforms load() retains the existing lazy path-string behavior.
    """

    def __init__(
        self,
        paths: Mapping[str, Path],
        *,
        selected: Mapping[str, Sequence[ResidentTensor]] | None = None,
        shards: Sequence[ShardInfo] | None = None,
        retained_names: Mapping[str, Collection[str]] | None = None,
    ) -> None:
        self.paths = paths
        self.io_cache_mode = "f-nocache" if sys.platform == "darwin" else "buffered"
        self.unselected_bytes = 0
        self.max_shard_bytes = 0
        self._files: dict[str, io.FileIO] = {}
        self._stack = ExitStack()
        if self.io_cache_mode == "buffered":
            return
        expected_shards = {shard.name: shard for shard in (shards or ())}
        try:
            for name, path in sorted(paths.items()):
                handle = self._stack.enter_context(io.FileIO(
                    path, "rb", opener=lambda filename, flags: os.open(filename, _readonly_flags())
                ))
                self._files[name] = handle
                try:
                    import fcntl

                    fcntl.fcntl(handle.fileno(), fcntl.F_NOCACHE, 1)
                except (ImportError, AttributeError, OSError) as exc:
                    raise RuntimeError(f"{name}: required F_NOCACHE could not be applied") from exc
                size = os.fstat(handle.fileno()).st_size
                self.max_shard_bytes = max(self.max_shard_bytes, size)
                if size > MAX_RESIDENT_SHARD_BYTES:
                    raise RuntimeError(f"{name}: eager resident shard exceeds 3 GiB admission limit")
                header_length = int.from_bytes(_pread_exact(handle.fileno(), 0, 8), "little")
                if header_length > MAX_RESIDENT_HEADER_BYTES:
                    raise RuntimeError(f"{name}: resident header exceeds 1 MiB admission limit")
                current, tensors = _read_safetensors_header(path, relative_name=name, fd=handle.fileno())
                if any(tensor.length > MAX_RESIDENT_TENSOR_BYTES for tensor in tensors):
                    raise RuntimeError(f"{name}: tensor exceeds FileIO read-size admission limit")
                if shards is not None:
                    expected = expected_shards.get(name)
                    if expected is None or any(getattr(current, field) != getattr(expected, field)
                                               for field in ("size", "header_bytes", "header_sha256")):
                        raise RuntimeError(f"{name}: resident shard provenance mismatch")
                if selected is not None:
                    actual = {tensor.name: tensor for tensor in tensors}
                    expected = {tensor.tensor: tensor for tensor in selected[name]}
                    for key, tensor in expected.items():
                        value = actual.get(key)
                        if value is None or any(getattr(value, field) != getattr(tensor, field)
                                                for field in ("offset", "length", "dtype", "shape")):
                            raise RuntimeError(f"{name}: resident tensor metadata mismatch for {key}")
                    kept_names = set(expected)
                else:
                    kept_names = (set(retained_names[name]) if retained_names is not None
                                  else {tensor.name for tensor in tensors})
                self.unselected_bytes += sum(tensor.length for tensor in tensors
                                             if tensor.name not in kept_names)
            if self.unselected_bytes > MAX_UNSELECTED_RESIDENT_BYTES:
                raise RuntimeError("eager resident load exceeds 64 MiB aggregate unselected tensor limit")
        except BaseException:
            self._stack.close()
            raise

    def load(self, name: str, mx_module: Any) -> dict[str, Any]:
        if self.io_cache_mode == "buffered":
            return mx_module.load(str(self.paths[name]), format="safetensors")
        handle = self._files[name]
        try:
            handle.seek(0)
            # MLX's file-object binding evaluates every tensor before returning.
            return mx_module.load(handle, format="safetensors")
        finally:
            handle.close()

    def __enter__(self) -> "ResidentShardReader":
        return self

    def __exit__(self, *exc) -> None:
        self._stack.close()
