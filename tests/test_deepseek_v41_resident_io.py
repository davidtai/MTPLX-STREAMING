"""Resident I/O admission without MLX; real-array test requires the GPU guard."""
from __future__ import annotations

import ast
import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def darwin_io(monkeypatch):
    """Exercise Darwin admission on CPU-only hosts without real fcntl support."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(fcntl, "F_NOCACHE", 48, raising=False)
    monkeypatch.setattr(fcntl, "fcntl", lambda fd, command, flag: 0)


def _shard(root, name, *, extra_bytes=0, payload=b"\x80\x00\x34\x12"):
    header = {"kept": {"dtype": "U8", "shape": [len(payload)],
                       "data_offsets": [0, len(payload)]}}
    if extra_bytes:
        header["dropped"] = {"dtype": "U8", "shape": [extra_bytes],
                             "data_offsets": [len(payload), len(payload) + extra_bytes]}
    raw = json.dumps(header).encode()
    prefix = len(raw).to_bytes(8, "little") + raw
    path = root / name
    with path.open("wb") as handle:
        handle.write(prefix + payload)
        handle.truncate(len(prefix) + len(payload) + extra_bytes)
    tensor = SimpleNamespace(tensor="kept", shard=name, offset=len(prefix),
                             length=len(payload), dtype="U8", shape=(len(payload),))
    shard = SimpleNamespace(name=name, size=path.stat().st_size,
        header_bytes=len(prefix), header_sha256=hashlib.sha256(prefix).hexdigest(),
        kind="safetensors")
    return path, tensor, shard


def _loader():
    source = ROOT / "mtplx/models/deepseek_v41_loader.py"
    parsed = ast.parse(source.read_text())
    node = next(n for n in parsed.body if isinstance(n, ast.FunctionDef)
                and n.name == "load_text_only_resident_arrays")
    scope = {"__package__": "mtplx.models", "Path": Path, "sys": sys,
             "ResidentLoadError": RuntimeError, "resolve_artifact_member": lambda root, name: root / name,
             "_dtype_name": lambda value: value.dtype}
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), scope)
    return scope[node.name]


def _load(root, records, mx):
    tensors = tuple(record[1] for record in records)
    manifest = SimpleNamespace(shards=tuple(record[2] for record in records))
    partition = SimpleNamespace(kept=tensors, kept_count=len(tensors))
    return _loader()(root, manifest, partition=partition, mx_module=mx)


def _fake_mx(calls, *, expect_file=True):
    def load(source, *, format):
        assert format == "safetensors"
        if expect_file:
            assert not isinstance(source, str), "resident load still uses buffered path I/O"
            assert not source.closed
            calls.append(source.fileno())
            raw = source.read(8)
            header = json.loads(source.read(int.from_bytes(raw, "little")))
        else:
            assert isinstance(source, str)
            calls.append(source)
            with open(source, "rb") as handle:
                header = json.loads(handle.read(int.from_bytes(handle.read(8), "little")))
        return {name: SimpleNamespace(shape=tuple(info["shape"]), dtype=info["dtype"],
                                      nbytes=info["data_offsets"][1] - info["data_offsets"][0])
                for name, info in header.items()}
    return SimpleNamespace(load=load)


def test_darwin_resident_load_uses_live_uncached_file_then_closes(tmp_path, monkeypatch, darwin_io):
    monkeypatch.setattr(sys, "platform", "darwin")
    records = [_shard(tmp_path, "a.safetensors")]
    configured = []
    monkeypatch.setattr(fcntl, "F_NOCACHE", 48, raising=False)
    monkeypatch.setattr(fcntl, "fcntl", lambda fd, command, flag: configured.append((fd, command, flag)) or 0)
    calls = []
    result = _load(tmp_path, records, _fake_mx(calls))
    assert set(result) == {"kept"}
    assert configured == [(calls[0], 48, 1)]
    with pytest.raises(OSError):
        os.fstat(calls[0])


def test_all_shards_are_admitted_before_any_mlx_load(tmp_path, monkeypatch, darwin_io):
    monkeypatch.setattr(sys, "platform", "darwin")
    records = [_shard(tmp_path, "a.safetensors"),
               _shard(tmp_path, "z.safetensors", extra_bytes=64 * 1024**2 + 1)]
    mx = SimpleNamespace(load=lambda *args, **kwargs: pytest.fail("array allocation preceded full header admission"))
    with pytest.raises(RuntimeError, match="unselected"):
        _load(tmp_path, records, mx)


def test_changed_header_is_rejected_before_any_mlx_load(tmp_path, monkeypatch, darwin_io):
    monkeypatch.setattr(sys, "platform", "darwin")
    record = _shard(tmp_path, "a.safetensors")
    record[2].header_sha256 = "0" * 64
    mx = SimpleNamespace(load=lambda *args, **kwargs: pytest.fail("mismatched header was loaded"))
    with pytest.raises(RuntimeError, match="provenance"):
        _load(tmp_path, [record], mx)


def test_nocache_refusal_closes_descriptor_before_any_load(tmp_path, monkeypatch, darwin_io):
    monkeypatch.setattr(sys, "platform", "darwin")
    records = [_shard(tmp_path, "a.safetensors")]
    descriptors = []
    monkeypatch.setattr(fcntl, "F_NOCACHE", 48, raising=False)
    def refuse(fd, command, flag):
        descriptors.append(fd)
        raise OSError("fixture refuses uncached reads")
    monkeypatch.setattr(fcntl, "fcntl", refuse)
    mx = SimpleNamespace(load=lambda *args, **kwargs: pytest.fail("refused bypass reached mx.load"))
    with pytest.raises(RuntimeError, match="F_NOCACHE"):
        _load(tmp_path, records, mx)
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


def test_non_macos_retains_existing_path_load(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    calls = []
    record = _shard(tmp_path, "a.safetensors")
    result = _load(tmp_path, [record], _fake_mx(calls, expect_file=False))
    assert set(result) == {"kept"}
    assert calls == [str(record[0])]


def test_header_parser_does_not_close_or_seek_borrowed_descriptor(tmp_path):
    from mtplx.expert_manifest import _read_safetensors_header
    record = _shard(tmp_path, "a.safetensors")
    with record[0].open("rb", buffering=0) as handle:
        handle.seek(3)
        shard, tensors = _read_safetensors_header(record[0], relative_name=record[0].name,
                                                 fd=handle.fileno())
        assert handle.tell() == 3
        assert os.fstat(handle.fileno()).st_size == shard.size
        assert tensors[0].offset == record[1].offset


def test_large_header_is_rejected_before_json_inventory_allocation(tmp_path, darwin_io):
    from mtplx.resident_io import ResidentShardReader
    length = 1024**2 + 1
    header = b'{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}'
    path = tmp_path / "header.safetensors"
    path.write_bytes(length.to_bytes(8, "little") + header.ljust(length) + b"x")
    with pytest.raises(RuntimeError, match="header"):
        with ResidentShardReader({path.name: path}):
            pytest.fail("large header was admitted")


def test_dropped_limit_counts_all_touched_shards(tmp_path, monkeypatch, darwin_io):
    monkeypatch.setattr(sys, "platform", "darwin")
    records = [_shard(tmp_path, "a.safetensors", extra_bytes=33 * 1024**2),
               _shard(tmp_path, "b.safetensors", extra_bytes=33 * 1024**2)]
    mx = SimpleNamespace(load=lambda *args, **kwargs: pytest.fail("aggregate dropped budget was bypassed"))
    with pytest.raises(RuntimeError, match="unselected"):
        _load(tmp_path, records, mx)


@pytest.mark.parametrize("unused", ["unused.extra", "layers.14.engram.q_weight",
                                   "layers.1.engram.wkv.biases"])
def test_engram_rejects_unowned_payload_before_eager_load(tmp_path, monkeypatch, unused, darwin_io):
    monkeypatch.setattr(sys, "platform", "darwin")
    path = ROOT / "mtplx/engram_v41.py"
    parsed = ast.parse(path.read_text())
    helper = next(node for node in parsed.body if isinstance(node, ast.FunctionDef)
                  and node.name == "load_engram_resident_tensors")
    scope = {"__package__": "mtplx", "Path": Path,
             "mx": SimpleNamespace(load=lambda *args, **kwargs: pytest.fail("unowned sidecar payload loaded"))}
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), helper], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    size = 65 * 1024**2
    header = {unused: {"dtype": "U8", "shape": [size], "data_offsets": [0, size]}}
    raw = json.dumps(header).encode()
    sidecar = tmp_path / "engram-residents.safetensors"
    with sidecar.open("wb") as handle:
        handle.write(len(raw).to_bytes(8, "little") + raw)
        handle.truncate(8 + len(raw) + size)
    with pytest.raises(RuntimeError, match="unselected"):
        scope[helper.name](sidecar, layer_ids=(1,), mode="mxfp8")


def test_load_failure_closes_all_admitted_descriptors(tmp_path, monkeypatch, darwin_io):
    monkeypatch.setattr(sys, "platform", "darwin")
    records = [_shard(tmp_path, "a.safetensors"), _shard(tmp_path, "b.safetensors")]
    descriptors = []
    monkeypatch.setattr(fcntl, "F_NOCACHE", 48, raising=False)
    monkeypatch.setattr(fcntl, "fcntl", lambda fd, command, flag: descriptors.append(fd) or 0)
    def fail(source, **kwargs):
        source.read(8)
        raise OSError("fixture read failure")
    with pytest.raises(RuntimeError, match="fixture read failure"):
        _load(tmp_path, records, SimpleNamespace(load=fail))
    assert len(descriptors) == 2
    for fd in descriptors:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_standalone_engram_layer_keeps_existing_lazy_path(tmp_path):
    path = ROOT / "mtplx/engram_v41.py"
    parsed = ast.parse(path.read_text())
    loader = next(node for node in parsed.body if isinstance(node, ast.FunctionDef)
                  and node.name == "load_engram_residents")
    sidecar = tmp_path / "engram-residents.safetensors"
    sidecar.touch()
    tensors = {f"layers.1.engram.{name}": SimpleNamespace(shape=shape)
               for name, shape in (("wkv.weight", (2, 1)), ("wkv.scales", (2, 1)),
                                   ("wkv.biases", (2, 1)), ("q_weight", (1, 1)),
                                   ("k_weight", (1, 1)))}
    calls = []
    def lazy_load(source):
        assert isinstance(source, str)
        calls.append(source)
        return tensors
    scope = {"Path": Path, "json": json, "mx": SimpleNamespace(load=lazy_load),
             "EngramResidents": SimpleNamespace,
             "load_engram_resident_tensors": lambda *args, **kwargs: pytest.fail("standalone eagerly loads sibling layers")}
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), loader], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    result = scope[loader.name](tmp_path, 1)
    assert calls == [str(sidecar)]
    assert result.q_weight is tensors["layers.1.engram.q_weight"]


def test_engram_shared_load_accepts_snapshot_sidecar_symlink(tmp_path, darwin_io):
    path = ROOT / "mtplx/engram_v41.py"
    helper = next(node for node in ast.parse(path.read_text()).body
                  if isinstance(node, ast.FunctionDef) and node.name == "load_engram_resident_tensors")
    name = "layers.1.engram.q_weight"
    payload = bytes.fromhex("00000080")
    raw = json.dumps({name: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    blob = tmp_path / "blob"
    blob.write_bytes(len(raw).to_bytes(8, "little") + raw + payload)
    sidecar = tmp_path / "engram-residents.safetensors"
    sidecar.symlink_to(blob)
    value = object()
    def load(source, *, format):
        assert format == "safetensors"
        assert Path(source.name) == blob.resolve()
        assert source.read() == blob.read_bytes()
        return {name: value}
    scope = {"__package__": "mtplx", "Path": Path, "mx": SimpleNamespace(load=load)}
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), helper], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    result = scope[helper.name](sidecar, layer_ids=(1,), mode="mxfp8")
    assert result == {name: value}


def test_engram_attach_loads_one_shared_sidecar_for_declared_layers(tmp_path):
    path = ROOT / "mtplx/models/deepseek_v41.py"
    parsed = ast.parse(path.read_text())
    model_class = next(node for node in parsed.body if isinstance(node, ast.ClassDef)
                       and node.name == "Model")
    attach = next(node for node in model_class.body if isinstance(node, ast.FunctionDef)
                  and node.name == "attach_engram")
    attach.body = [node for node in attach.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    manifest = {"hashing": {"layer_ids": [1, 14]},
                "residents": {"file": "custom.safetensors", "quant": {"wkv": {"mode": "mxfp8"}}}}
    (tmp_path / "engram-manifest.json").write_text(json.dumps(manifest))
    shared = {1: object(), 14: object()}
    loads = []
    def load_tensors(sidecar, *, layer_ids, mode):
        assert layer_ids == (1, 14)
        assert mode == "mxfp8"
        loads.append(sidecar)
        return shared
    def load_residents(directory, layer_id, *, preloaded=None):
        assert preloaded is shared, "each Engram layer independently loads the sidecar"
        return SimpleNamespace(build_module=lambda **kwargs: SimpleNamespace(weight=preloaded[layer_id]))
    scope = {"_json": json, "_Path": Path,
        "EngramBank": SimpleNamespace(open=lambda directory, layer, **kwargs: SimpleNamespace(cache=layer)),
        "NgramHashState": SimpleNamespace(from_manifest=lambda *args: object()),
        "load_engram_residents": load_residents, "load_engram_resident_tensors": load_tensors}
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), attach], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace() for _ in range(15)]),
                            args=SimpleNamespace(rms_norm_eps=1e-6))
    assert scope[attach.name](model, tmp_path, tokenizer=object(), cache_bytes=264) == (1, 14)
    assert loads == [tmp_path / "custom.safetensors"]
    assert model.model.layers[1].engram_hook.weight is shared[1]
    assert model.model.layers[14].engram_hook.weight is shared[14]


@pytest.mark.parametrize("limit,extra", [("shard", 3 * 1024**3), ("tensor", 2**31)])
def test_oversized_resident_payload_is_rejected_before_loading(tmp_path, monkeypatch, limit, extra, darwin_io):
    monkeypatch.setattr(sys, "platform", "darwin")
    from mtplx.resident_io import ResidentShardReader
    record = _shard(tmp_path, "large.safetensors", extra_bytes=extra)
    with pytest.raises(RuntimeError, match=limit):
        with ResidentShardReader({record[0].name: record[0]}):
            pytest.fail("oversized file was admitted")


@pytest.mark.skipif(sys.platform != "darwin", reason="requires guarded MLX")
def test_real_uncached_resident_bytes_match_path_loading(tmp_path):
    """Must run under the exclusive GPU guard, including on mx.cpu."""
    import mlx.core as mx
    from mtplx.resident_io import ResidentShardReader

    payloads = {
        "bf16": ("BF16", bytes.fromhex("0080c17f0100807f"), [4]),
        "f32": ("F32", bytes.fromhex("000000804523c17f010000000000807f"), [4]),
        "u8": ("U8", bytes.fromhex("007f80ff"), [4]),
        "u32": ("U32", bytes.fromhex("00000000ffffffff78563412"), [3]),
    }
    header = {}; payload = b""
    for name, (dtype, data, shape) in payloads.items():
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [len(payload), len(payload) + len(data)]}
        payload += data
    raw = json.dumps(header).encode()
    path = tmp_path / "bits.safetensors"
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + payload)
    with ResidentShardReader({path.name: path}) as reader:
        uncached = reader.load(path.name, mx)
    reference = mx.load(str(path), format="safetensors")
    mx.eval(uncached, reference)
    for name, (_, expected, _) in payloads.items():
        assert uncached[name].dtype == reference[name].dtype
        assert uncached[name].shape == reference[name].shape
        assert memoryview(uncached[name]).tobytes() == expected
        assert memoryview(reference[name]).tobytes() == expected
