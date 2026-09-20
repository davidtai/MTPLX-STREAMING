"""F23: build TWO other 16K Python cell prompts to validate the per-layer row rule.

CPU only (tokenizers + jinja2, NO MLX): the exact same schema
(``mtplx-server-cell-prompt-ids-v1``), chat rendering (``render_deepseek_v41_prompt``,
thinking OFF, generation prompt ON), the same exact-16,384-token sizing loop and the
same tail/BOS assertions as the original builder
(docs/deepseek-v41/receipts/memory-budget-110/build_coding_prompts.py). Deterministic:
seed-fixed rotation, no randomness, no timestamps in the ids file.

Two prompts, each written to
``/tmp/dsv41-110-stage/prompts/<name>/python-prompt-ids.json`` (+ python-<target>.txt /
python-<target>.rendered.txt):

  * ``io_module``     -- context = the pinned run worktree's ``mtplx/expert_io.py``,
    instruction = a generic unified-diff request (add one small, reusable, typed helper
    with a docstring; "Return only a unified diff"; no prose). Written generically; it
    does NOT mention the benchmark.
  * ``server_module`` -- context = the pinned run worktree's ``mtplx/server/openai.py``,
    instruction = the existing ``mtplx/benchmarks/prompts/long_code_uncapped.jsonl``
    prompt (continue the module with production-quality code).

Nothing here is tuned to a prompt after the fact; the two prompts are FIXED by the F23
spec. The context modules come from the read-only pinned run worktree; this module never
writes there (bytecode is disabled before any import from it).
"""
from __future__ import annotations

import sys

# Import nothing that would write bytecode into the read-only pinned run worktree, and
# nothing MLX. chat_encoding is import-clean (json + typing only); guard regardless.
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
from pathlib import Path

# render_deepseek_v41_prompt lives in THIS worktree's mtplx (the f2/next-layer-prefetch
# lineage), which the venv's editable mtplx (main) does not carry. Resolve it from this
# worktree so the builder does not depend on which mtplx the venv points at. Bytecode is
# disabled above; this worktree is writable regardless.
_WORKTREE_ROOT = str(Path(__file__).resolve().parents[3])
if _WORKTREE_ROOT not in sys.path:
    sys.path.insert(0, _WORKTREE_ROOT)

from tokenizers import Tokenizer
from jinja2 import Environment
from mtplx.chat_encoding import render_deepseek_v41_prompt

ARTIFACT = Path("/Users/davidtai/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4")
RUN_WORKTREE = Path(
    "/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/dsv41-run-d5f15e7a"
)
SEED = 20260829
# F23 uses only the 16,384-token cell (the runner and ab both select target_tokens==16384;
# the done-when requires exactly the 16K cell). The original builder also emitted a 1,024
# warm-up cell, which F23 does not use and which does not fit the longer instructions here.
TARGETS = (16384,)
DEFAULT_OUT_ROOT = Path("/tmp/dsv41-110-stage/prompts")

# io_module: a generic, prompt-agnostic unified-diff request in the STYLE of the
# benchmark's (add one small reusable, typed helper with a docstring; return only a
# unified diff; no prose). Deliberately generic -- no mention of any benchmark.
_IO_MODULE_INSTRUCTION = {
    "id": "f23_io_module_typed_helper_patch",
    "category": "coding",
    "max_tokens": 1024,
    "prompt": (
        "Please add one small, reusable helper function to the module above. It must be "
        "fully type-annotated, carry a concise one-line docstring, follow the module's "
        "existing style and imports, and must not duplicate behaviour that already "
        "exists. Do not restate the unchanged module and do not add prose. Return only a "
        "unified diff against the code above."
    ),
}


def _load_long_code_instruction() -> dict:
    """The server_module instruction is the existing long_code_uncapped.jsonl prompt."""
    path = RUN_WORKTREE / "mtplx/benchmarks/prompts/long_code_uncapped.jsonl"
    return json.loads(path.read_text().splitlines()[0])


def _prompt_specs() -> dict:
    return {
        "io_module": {
            "context_path": RUN_WORKTREE / "mtplx/expert_io.py",
            "instruction": _IO_MODULE_INSTRUCTION,
            # content markers that pin THIS prompt's identity (fixed by the spec)
            "require_substrings": ("Return only a unified diff",),
            "reject_substrings": ("No code.",),
        },
        "server_module": {
            "context_path": RUN_WORKTREE / "mtplx/server/openai.py",
            "instruction": _load_long_code_instruction(),
            "require_substrings": (
                "Continue this Python module with production-quality code",
            ),
            "reject_substrings": (),
        },
    }


def _digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def build_one(name: str, *, out_root: Path = DEFAULT_OUT_ROOT) -> dict:
    """Build one named prompt file and return its result dict (also written to disk).

    Mirrors the original builder byte-for-byte in method: same rotation, same sizing
    loop, same tail/BOS assertions; only the context module and instruction differ.
    """
    specs = _prompt_specs()
    if name not in specs:
        raise SystemExit(f"unknown prompt name {name!r}; choose from {sorted(specs)}")
    spec = specs[name]
    context_path = spec["context_path"]
    instruction = spec["instruction"]

    context = context_path.read_text()
    tokenizer = Tokenizer.from_file(str(ARTIFACT / "tokenizer.json"))
    template = Environment().from_string((ARTIFACT / "chat_template.jinja").read_text())

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False).ids

    lines = context.splitlines()
    offset = SEED % len(lines)
    base_text = "\n".join(lines[offset:] + lines[:offset]).rstrip() + "\n"
    # The original builder decodes a PREFIX of the tokenized context, so the module must
    # itself tokenize to >= the target. mtplx/expert_io.py is only ~15.3K tokens (< 16,384),
    # so it cannot fill a 16K cell on its own; tile the rotated module until it comfortably
    # exceeds the largest target. A large module (mtplx/server/openai.py) tiles zero times.
    # Content stays 100% the chosen module and the exact-token sizing loop below is unchanged.
    context_text = base_text
    while len(encode(context_text)) < max(TARGETS) + 256:
        context_text += base_text
    context_ids = encode(context_text)
    tiled_copies = context_text.count(base_text) if base_text else 1

    result = {
        "schema": "mtplx-server-cell-prompt-ids-v1",
        "model": str(ARTIFACT),
        "model_family": "deepseek-v41",
        "context_source": str(context_path),
        "context_sha256": _digest(context.encode()),
        "context_tiled_copies": tiled_copies,
        "instruction": instruction,
        "template_settings": {
            "add_generation_prompt": True,
            "enable_thinking": False,
            "add_special_tokens": False,
            "bos_token_id": 0,
        },
        "tokenizer_sha256": _digest((ARTIFACT / "tokenizer.json").read_bytes()),
        "template_sha256": _digest((ARTIFACT / "chat_template.jinja").read_bytes()),
        "prompts": [],
    }

    def _render_and_encode(text, *, check=True):
        msgs = [{"role": "user", "content": text}]
        rendered = render_deepseek_v41_prompt(
            msgs, enable_thinking=False, add_generation_prompt=True, drop_thinking=False,
        )
        if check:
            assert rendered == template.render(
                messages=msgs, enable_thinking=False,
                add_generation_prompt=True, preserve_thinking=True,
            )
        return text, rendered, encode(rendered)

    def _build_token_prefix(n):  # original method: decode a TOKEN prefix of the context
        return _render_and_encode(
            tokenizer.decode(context_ids[:n], skip_special_tokens=False).rstrip()
            + "\n\n" + instruction["prompt"]
        )

    def _build_char_prefix(m, pad, *, check):  # exact-fit method: CHAR prefix + token pad
        return _render_and_encode(
            context_text[:m].rstrip() + pad + "\n\n" + instruction["prompt"], check=check,
        )

    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    digests = {}
    for target in TARGETS:
        # (1) Original coarse budget loop (whole-token prefix). Converges when the tokenizer
        #     boundary lands exactly on target (e.g. io_module).
        budget = target - len(encode(instruction["prompt"])) - 4
        text = rendered = ids = None
        for _ in range(12):
            text, rendered, ids = _build_token_prefix(budget)
            if len(ids) == target:
                break
            budget += target - len(ids)
        # (2) Deterministic exact-fit fallback. Some contexts (mtplx/server/openai.py) skip
        #     target with BOTH whole-token and whole-char prefix steps (one boundary unit adds
        #     two tokens), so the coarse loop above can never land on 16,384 there. Take the
        #     largest char prefix that renders to <= target, then pad with "\n#" (a comment-
        #     line start that adds EXACTLY one token per copy, verified) to hit target exactly.
        #     The pad sits inside the user content (after rstrip), so BOS/tail are unaffected.
        if len(ids) != target:
            lo, hi = 0, len(context_text)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if len(_build_char_prefix(mid, "", check=False)[2]) <= target:
                    lo = mid
                else:
                    hi = mid - 1
            short = len(_build_char_prefix(lo, "", check=False)[2])
            if short > target:
                raise RuntimeError(f"exact-fit fallback: no prefix <= target for {name} {target}")
            text, rendered, ids = _build_char_prefix(lo, "\n#" * (target - short), check=True)
            if len(ids) != target:
                raise RuntimeError(
                    f"exact-fit fallback failed for {name} target {target}: got {len(ids)}"
                )

        # Identical STRUCTURAL assertions to the original builder.
        assert len(ids) == target and ids[0] == 0 and ids[-2:] == [
            tokenizer.token_to_id("<｜Assistant｜>"),
            tokenizer.token_to_id("</think>"),
        ], (len(ids), ids[-5:])
        # Per-prompt CONTENT assertions (identity pins fixed by the spec).
        for sub in spec["require_substrings"]:
            assert sub in text, f"{name}: required substring missing: {sub!r}"
        for sub in spec["reject_substrings"]:
            assert sub not in text, f"{name}: rejected substring present: {sub!r}"

        (out_dir / f"python-{target}.txt").write_text(text)
        (out_dir / f"python-{target}.rendered.txt").write_text(rendered)
        entry = {
            "cell": "sweep",
            "target_tokens": target,
            "seed": SEED,
            "text_sha256": _digest(text.encode()),
            "templated_tokens": len(ids),
            "input_tokens": len(ids),
            "token_ids_sha256": _digest(json.dumps(ids).encode()),
            "bos_id_prepended": True,
            "token_ids": ids,
        }
        result["prompts"].append(entry)
        digests[target] = entry["token_ids_sha256"]
        print(f"{name} {target} {entry['token_ids_sha256']}")

    (out_dir / "python-prompt-ids.json").write_text(json.dumps(result, indent=2) + "\n")
    result["_out_dir"] = str(out_dir)
    result["_digests"] = digests
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the F23 other-prompt cells (CPU only)")
    ap.add_argument(
        "--name",
        choices=("io_module", "server_module"),
        default=None,
        help="build only this prompt (default: build both)",
    )
    ap.add_argument(
        "--out-root",
        default=str(DEFAULT_OUT_ROOT),
        help="root dir for <name>/python-prompt-ids.json (default: %(default)s)",
    )
    args = ap.parse_args(argv)
    out_root = Path(args.out_root)
    names = [args.name] if args.name else ["io_module", "server_module"]
    for name in names:
        build_one(name, out_root=out_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
