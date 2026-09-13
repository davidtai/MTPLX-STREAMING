"""Run the archived, bounded real-weight MTP head probe with block size six."""

from pathlib import Path

source_path = Path(
    "docs/deepseek-v41/receipts/dspark-head-peak-20260913/"
    "run_dspark_head_peak.py"
)
source = source_path.read_text()
replacements = {
    "OUT = Path('/tmp/dsv41-110-preflight/dspark-head-peak.json')":
        "OUT = Path('/tmp/dsv41-110-preflight/dspark-head-block6.json')",
    "probe.mtp = DSparkHead(args)":
        "args.dspark_block_size = 6\n"
        "    probe.mtp = DSparkHead(args)\n"
        "    assert probe.mtp.block_size == 6 and all(layer.block_size == 6 for layer in probe.mtp.layers)",
    "# extra 4 GiB exceeds the full main hidden + projected hidden + all KV copies.":
        "# extra 4 GiB exceeds the full main hidden + projected hidden + all KV copies at T6.",
}
for old, new in replacements.items():
    if source.count(old) != 1:
        raise RuntimeError(f"archived probe source changed or patch point ambiguous: {old}")
    source = source.replace(old, new)

exec(
    compile(source, str(source_path) + "[block6]", "exec"),
    {"__name__": "__main__", "__file__": str(Path(__file__).resolve())},
)
