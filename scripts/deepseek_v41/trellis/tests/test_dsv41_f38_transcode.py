"""F38 transcode driver: CPU-only tests of the record layout, resumable dry-run chunks, manifest and link_rest."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import transcode_bank as tb  # noqa: E402

SRC_REC = {
    "layer": 0, "expert": 0, "logical_bytes": 18800640, "sha256": "",
    "segments": [
        {"component": "gate_proj.weight", "tensor": "t", "offset": 0, "length": 5898240, "dtype": "U32", "shape": [2304, 640]},
        {"component": "gate_proj.scales", "tensor": "t", "offset": 5898240, "length": 368640, "dtype": "U8", "shape": [2304, 160]},
        {"component": "up_proj.weight", "tensor": "t", "offset": 6266880, "length": 5898240, "dtype": "U32", "shape": [2304, 640]},
        {"component": "up_proj.scales", "tensor": "t", "offset": 12165120, "length": 368640, "dtype": "U8", "shape": [2304, 160]},
        {"component": "down_proj.weight", "tensor": "t", "offset": 12533760, "length": 5898240, "dtype": "U32", "shape": [5120, 288]},
        {"component": "down_proj.scales", "tensor": "t", "offset": 18432000, "length": 368640, "dtype": "U8", "shape": [5120, 72]},
    ],
}


def test_record_layout_matches_the_bank_geometry():
    segs = tb.record_layout(SRC_REC)
    assert [s["component"] for s in segs] == ["gate_proj.code", "gate_proj.rout", "up_proj.code", "up_proj.rout",
                                              "down_proj.code", "down_proj.rout"]
    assert segs[0]["shape"] == [320, 144, 48] and segs[4]["shape"] == [144, 320, 48]
    assert tb.record_bytes(segs) == 13_290_496
    assert segs[3]["offset"] == 2 * 4_423_680 + 4_608 and segs[5]["length"] == 5120 * 2


def _fake_src(tmp_path: Path, n: int) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    recs = []
    for i in range(n):
        r = json.loads(json.dumps(SRC_REC)); r["layer"], r["expert"] = i // 4, i % 4
        recs.append(r)
    m = {"format": "x", "quantization": {"bits": 4, "group_size": 32, "mode": "mxfp4"}, "artifact": "src", "records": recs}
    (src / "expert-manifest.json").write_text(json.dumps(m))
    (src / "config.json").write_text("{}")
    (src / "engram").mkdir()
    return src


def test_dry_run_chunks_resume_and_write_a_consistent_manifest(tmp_path):
    src = _fake_src(tmp_path, 5)
    out = tmp_path / "out"
    py = sys.executable
    script = str(HERE / "transcode_bank.py")
    subprocess.run([py, script, "--src-dir", str(src), "--out-dir", str(out), "--dry-run", "--max-records", "3",
                    "--progress-every", "1"], check=True, capture_output=True)
    prog = json.loads((out / "progress.json").read_text())
    assert prog["next"] == 3 and not (out / "expert-manifest.json").exists()
    subprocess.run([py, script, "--src-dir", str(src), "--out-dir", str(out), "--dry-run"], check=True, capture_output=True)
    prog = json.loads((out / "progress.json").read_text())
    m = json.loads((out / "expert-manifest.json").read_text())
    assert prog["next"] == 5 and len(m["records"]) == 5 and m["quantization"]["mode"] == "tcq3"
    rb = m["records"][0]["logical_bytes"]
    with open(out / "experts.bin", "rb") as f:
        for i, r in enumerate(m["records"]):
            f.seek(i * rb)
            assert hashlib.sha256(f.read(rb)).hexdigest() == r["sha256"] == prog["sha256"][i]
            assert r["segments"][0]["offset"] == i * rb
    # a re-run on a complete bank is a no-op
    subprocess.run([py, script, "--src-dir", str(src), "--out-dir", str(out), "--dry-run"], check=True, capture_output=True)
    assert json.loads((out / "progress.json").read_text())["next"] == 5


def test_link_rest_hard_links_files_and_symlinks_directories(tmp_path):
    src = _fake_src(tmp_path, 1)
    (src / "model-00001.safetensors").write_bytes(b"x" * 64)
    out = tmp_path / "out"
    out.mkdir()
    made = tb.link_rest(str(src), str(out))
    assert "expert-manifest.json" not in made and "config.json" in made and "engram" in made
    cfg = out / "config.json"
    assert cfg.is_file() and not cfg.is_symlink() and os.stat(cfg).st_ino == os.stat(src / "config.json").st_ino
    st = out / "model-00001.safetensors"
    assert st.is_file() and not st.is_symlink() and st.stat().st_size == 64
    assert (out / "engram").is_symlink()
    assert tb.link_rest(str(src), str(out)) == []          # idempotent
