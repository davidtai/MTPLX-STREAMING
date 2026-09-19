"""Paired 64-step native block-5 control using the block-6 diagnostic driver."""

from pathlib import Path

source_path = Path("/tmp/dsv41-110-preflight/run_dspark_block6_short.py")
source = source_path.read_text()
replacements = {
    "args.dspark_depth != 6": "args.dspark_depth != 5",
    "altered.dspark_block_size = 6": "altered.dspark_block_size = 5",
    "self.block_size != 6 or any(layer.block_size != 6":
        "self.block_size != 5 or any(layer.block_size != 5",
    "model.mtp.block_size != 6": "model.mtp.block_size != 5",
    "len(prompt_ids) + steps + 6 + 1": "len(prompt_ids) + steps + 5 + 1",
    "depth=6, stage_timing=False": "depth=5, stage_timing=False",
    '"target_verify_rows": 7': '"target_verify_rows": 6',
    '"target_max_assignments": 42': '"target_max_assignments": 36',
}
for old, new in replacements.items():
    if source.count(old) != 1:
        raise RuntimeError(f"source patch point changed: {old}")
    source = source.replace(old, new)
if "block6" not in source or "BLOCK6" not in source:
    raise RuntimeError("diagnostic output labels changed")
source = source.replace("block6", "block5").replace("BLOCK6", "BLOCK5")
exec(
    compile(source, str(source_path) + "[block5]", "exec"),
    {"__name__": "__main__", "__file__": str(Path(__file__).resolve())},
)
