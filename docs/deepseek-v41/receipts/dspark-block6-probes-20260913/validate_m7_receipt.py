"""Validate the observed M7 probe after its normal SystemExit(0); no MLX."""

import datetime
import fcntl
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

here = Path(__file__).resolve().parent
probe = json.loads((here / "verify-attn-real-m7.json").read_text())
bounds = json.loads((here / "verify-attn-real-m7.bounds.json").read_text())
guard = (here / "verify-attn-real-m7.guard.log").read_text()
observed = here / "run_verify_m7_real_shape.observed.py"
assert hashlib.sha256(observed.read_bytes()).hexdigest() == bounds["wrapper_sha256"]
assert probe["git_sha"] == bounds["source_commit"]
assert probe["device"] == "gpu" and probe["codec"] == "mxfp8"
assert probe["dims"]["T"] == 16384 and probe["rows"] == [6, 7]
assert set(probe["modes"]) == {"swa_only", "full", "reindex", "reuse"}
assert all(set(probe["results"][mode]) == {"6", "7"} for mode in probe["modes"])
assert bounds["mlx_active_peak_bytes"] < bounds["active_scope_bound_bytes"]
assert bounds["physical_peak_sampled_bytes"] < 110_000_000_000
assert bounds["swapouts_first"] == bounds["swapouts_last"]
assert "GPU step exited with code 0;" in guard
assert "background warmup ready" in guard and "released exclusive GPU lock:" in guard
with urlopen("http://127.0.0.1:8080/health", timeout=5) as response:
    health = json.load(response)
assert health["ok"] and health["model"] == "mtplx-flash-next-optimized-speed"
assert health["warmup"]["ran"] and not health["warmup"]["error"]
assert health["warmup"]["background"]["state"] == "done"
with open("/tmp/mtplx-gpu-exclusive.lock", "a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

validation = {
    "probe_completed_and_receipt_valid": True,
    "observed_wrapper_complete_field": bounds["complete"],
    "observed_wrapper_field_explanation": "The probe CLI raised SystemExit(0) after writing the receipt; the wrapper's finally block ran before post-run validation. Raw field preserved.",
    "source_commit": bounds["source_commit"],
    "rows": probe["rows"],
    "mode_count": len(probe["modes"]),
    "mlx_peak_bytes": bounds["mlx_active_peak_bytes"],
    "sampled_physical_peak_bytes": bounds["physical_peak_sampled_bytes"],
    "swapouts_unchanged": True,
    "guard_exit_code": 0,
    "exact_qwen_healthy_and_warm": True,
    "gpu_lock_free": True,
    "verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
(here / "verify-attn-real-m7.completion-validation.json").write_text(
    json.dumps(validation, indent=2) + "\n"
)
print(json.dumps(validation, indent=2))
