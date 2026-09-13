"""File-reader policy regressions without importing MLX or executing Metal."""

from __future__ import annotations

import ast
import errno
import fcntl
import json
import os
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _classes(platform):
    """Run the actual reader/cache classes, omitting the module's MLX import."""
    path = ROOT / "mtplx/ngram_row_cache.py"
    parsed = ast.parse(path.read_text())
    names = {"FileRowReader", "NGramRowCache", "_contiguous_runs"}
    nodes = [node for node in parsed.body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    scope = {"os": os, "sys": SimpleNamespace(platform=platform),
             "np": np, "OrderedDict": OrderedDict}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope["FileRowReader"], scope["NGramRowCache"]


def _bank(tmp_path):
    prefix = b"hdr"
    rows = bytes(range(256)) * 33  # 32 complete 264-byte records
    path = tmp_path / "rows.bin"
    path.write_bytes(prefix + rows)
    return path, prefix, rows


def test_macos_default_bypasses_file_cache_once_before_unaligned_reads(tmp_path, monkeypatch):
    Reader, Cache = _classes("darwin")
    path, prefix, rows = _bank(tmp_path)
    calls = []
    monkeypatch.setattr(fcntl, "F_NOCACHE", 48, raising=False)
    monkeypatch.setattr(fcntl, "fcntl", lambda fd, command, value: calls.append((fd, command, value)) or 0)
    with Reader(path, row_bytes=264, num_rows=32, data_offset=len(prefix)) as reader:
        assert calls == [(reader._fd, 48, 1)], "the bounded row LRU still fills the OS file cache"
        assert reader.io_cache_mode == "f-nocache"
        cache = Cache(reader, SimpleNamespace(row_bytes=264), cache_rows=2)
        assert cache.io_cache_mode == "f-nocache"
        for ids in ([7, 2, 7, 31], [0, 1, 2, 3], [7, 2]):
            got = cache.gather_bytes(ids)
            assert got.tobytes() == b"".join(rows[row * 264:(row + 1) * 264] for row in ids)
        cache.reset()
        assert cache.io_cache_mode == "f-nocache"
        assert len(calls) == 1, "file-cache policy must stay out of the read loop"


@pytest.mark.parametrize("platform,override", [("darwin", False), ("linux", None)])
def test_buffered_control_and_other_platform_default_are_preserved(tmp_path, monkeypatch, platform, override):
    Reader, _ = _classes(platform)
    path, prefix, rows = _bank(tmp_path)
    monkeypatch.setattr(fcntl, "fcntl", lambda *args: pytest.fail("buffered reader configured F_NOCACHE"))
    kwargs = {} if override is None else {"bypass_page_cache": override}
    with Reader(path, row_bytes=264, num_rows=32, data_offset=len(prefix), **kwargs) as reader:
        assert reader.io_cache_mode == "buffered"
        assert reader.read_run(3, 2) == rows[3 * 264:5 * 264]


@pytest.mark.parametrize("failure", ["missing", "refused", "unsupported"])
def test_requested_bypass_fails_closed_and_releases_descriptor(tmp_path, monkeypatch, failure):
    Reader, _ = _classes("linux" if failure == "unsupported" else "darwin")
    path, prefix, _ = _bank(tmp_path)
    if failure == "missing":
        monkeypatch.delattr(fcntl, "F_NOCACHE", raising=False)
    else:
        monkeypatch.setattr(fcntl, "F_NOCACHE", 48, raising=False)
    def refuse(*args):
        raise OSError(errno.ENOTSUP, "fixture refuses F_NOCACHE")
    monkeypatch.setattr(fcntl, "fcntl", refuse)
    closed = []
    close = os.close
    def track_close(fd):
        closed.append(fd)
        close(fd)
    monkeypatch.setattr(os, "close", track_close)
    reader = Reader.__new__(Reader)
    with pytest.raises(RuntimeError, match="F_NOCACHE"):
        reader.__init__(path, row_bytes=264, num_rows=32, data_offset=len(prefix),
                        bypass_page_cache=True)
    assert reader._fd is None
    assert len(closed) == 1
    reader.close()
    assert len(closed) == 1, "failed construction can close a reused FD during cleanup"


def test_short_file_failure_clears_descriptor_before_later_cleanup(tmp_path, monkeypatch):
    Reader, _ = _classes("linux")
    path, _, _ = _bank(tmp_path)
    closed = []
    close = os.close
    def track_close(fd):
        closed.append(fd)
        close(fd)
    monkeypatch.setattr(os, "close", track_close)
    reader = Reader.__new__(Reader)
    with pytest.raises(ValueError, match="required"):
        reader.__init__(path, row_bytes=264, num_rows=100)
    assert reader._fd is None
    reader.close()
    assert len(closed) == 1


@pytest.mark.skipif(os.uname().sysname != "Darwin", reason="macOS F_NOCACHE smoke")
def test_real_macos_bypass_preserves_unaligned_row_bytes(tmp_path):
    """OS-only real descriptor check; no MLX imports, arrays, or GPU execution."""
    Reader, _ = _classes("darwin")
    path, prefix, rows = _bank(tmp_path)
    with Reader(path, row_bytes=264, num_rows=32, data_offset=len(prefix)) as reader:
        assert reader.io_cache_mode == "f-nocache"
        assert reader.read_run(7, 3) == rows[7 * 264:10 * 264]


def _report_class():
    path = ROOT / "mtplx/resident_loader.py"
    parsed = ast.parse(path.read_text())
    report = next(node for node in parsed.body
                  if isinstance(node, ast.ClassDef) and node.name == "ResidentLoadReport")
    scope = {"dataclass": dataclass, "__name__": __name__}
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), report], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope["ResidentLoadReport"]


@pytest.mark.parametrize("modes", [{"1": "f-nocache", "14": "buffered"}, {}])
def test_resident_load_report_records_resolved_engram_reader_modes(modes):
    """Execute the real report tail without resident construction or MLX imports."""
    path = ROOT / "mtplx/models/deepseek_v41_loader.py"
    parsed = ast.parse(path.read_text())
    construct = next(node for node in parsed.body if isinstance(node, ast.FunctionDef)
                     and node.name == "construct_deepseek_v41_resident_model")
    start = next(i for i, node in enumerate(construct.body)
                 if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                 and node.target.id == "engram_layer_ids")
    model = SimpleNamespace(_engram_banks=[SimpleNamespace(layer_id=int(layer),
                            cache=SimpleNamespace(io_cache_mode=mode))
                            for layer, mode in modes.items()])
    construct.body = construct.body[start:]
    construct.args = ast.arguments(posonlyargs=[], args=[ast.arg(arg="report")], kwonlyargs=[],
                                   kw_defaults=[], defaults=[])
    construct.returns = None
    scope = {"model": model,
             "head_mode_pricing": None, "runtime": object(), "engram_bank_path": None,
             "engram_layer_ids": tuple(int(layer) for layer in modes), "config": {},
             "ResidentModel": SimpleNamespace, "replace": replace}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[construct], type_ignores=[])),
                 str(path), "exec"), scope)
    resident = scope[construct.name](_report_class()(1, 2, 3, 4, 5, True))
    # This is the real contract runtime.py and benchmark serializers consume.
    serialized = json.loads(json.dumps(resident.report.as_dict()))
    assert serialized["engram_io_cache_modes"] == modes
    assert model._mtplx_resident_load_report["engram_io_cache_modes"] == modes


def test_resident_report_omits_engram_metadata_for_other_families():
    report = _report_class()(1, 2, 3, 4, 5, True)
    assert report.as_dict() == {
        "shard_count": 1, "tensor_count": 2, "raw_tensor_bytes": 3,
        "evaluated_parameter_count": 4, "bound_sparse_layers": 5, "strict": True,
        "proj_quant": None, "proj_quantized_modules": 0,
        "proj_requant": None, "proj_requantized_modules": 0,
    }
